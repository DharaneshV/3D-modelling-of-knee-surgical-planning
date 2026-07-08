import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

case = 'STS_006'
our_mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
ct_raw_path = f"data/ct_knee/case01_{case}_cropped.nii.gz"

pred_img = sitk.ReadImage(our_mask_path)
pred_arr = sitk.GetArrayFromImage(pred_img)

ct_raw = sitk.ReadImage(ct_raw_path)
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(pred_img)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(-1000)
ct_resampled = resampler.Execute(ct_raw)
ct_arr = sitk.GetArrayFromImage(ct_resampled)

z = 200
our_femur = (pred_arr == 1)

plt.figure(figsize=(8, 8))
plt.imshow(ct_arr[z, :, :], cmap='gray', vmin=-200, vmax=1000)

red = np.zeros((ct_arr.shape[1], ct_arr.shape[2], 4))
red[our_femur[z], 0] = 1.0 # R
red[our_femur[z], 3] = 0.5 # Alpha
plt.imshow(red)

plt.title(f"Axial Slice Z={z} - Red=Our Femur Mask")
plt.axis('off')
plt.savefig("STS_006_axial_Z200_overlay.png", dpi=300)
print("Saved axial overlay to STS_006_axial_Z200_overlay.png")
