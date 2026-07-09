import SimpleITK as sitk
import numpy as np

# 1. Load the CartiMorph prediction (native geometry)
pred_native = sitk.ReadImage('temp_cartimorph_io/output/oaizib_001.nii.gz')

# 2. Load the ground truth (already preprocessed to 0.5mm isotropic)
gt_iso = sitk.ReadImage('data/preprocessed/oaizib_001_0000_label.nii.gz')

# 3. Resample prediction to match the GT grid exactly
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(gt_iso)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
resampler.SetDefaultPixelValue(0)
resampler.SetTransform(sitk.Transform())
pred_iso = resampler.Execute(pred_native)

# 4. Convert to numpy for Dice calculation
pred_arr = sitk.GetArrayFromImage(pred_iso)
gt_arr = sitk.GetArrayFromImage(gt_iso)

print(f"Pred shape: {pred_arr.shape}")
print(f"GT shape: {gt_arr.shape}")

labels = {
    1: 'Femur',
    2: 'Femoral Cartilage',
    3: 'Tibia',
    4: 'Medial Tibial Cartilage',
    5: 'Lateral Tibial Cartilage',
    6: 'Patella',
    7: 'Patellar Cartilage'
}

for label_val, label_name in labels.items():
    p = (pred_arr == label_val)
    g = (gt_arr == label_val)
    
    intersection = np.sum(p & g)
    total = np.sum(p) + np.sum(g)
    
    if total == 0:
        dice = 1.0
    else:
        dice = 2.0 * intersection / total
        
    print(f"{label_name:25s} | Dice: {dice:.4f} | Pred Voxels: {np.sum(p):7d} | GT Voxels: {np.sum(g):7d}")

import matplotlib.pyplot as plt
z = pred_arr.shape[0] // 2
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("CartiMorph Resampled Pred")
plt.imshow(pred_arr[z], cmap='nipy_spectral', vmin=0, vmax=7)
plt.subplot(1, 2, 2)
plt.title("Ground Truth")
plt.imshow(gt_arr[z], cmap='nipy_spectral', vmin=0, vmax=7)
plt.savefig("cartimorph_dice_preview.png")
print("Saved cartimorph_dice_preview.png")
