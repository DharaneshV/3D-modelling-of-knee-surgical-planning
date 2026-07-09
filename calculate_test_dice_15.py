import SimpleITK as sitk
import numpy as np

cases = [f'oaizib_{i}' for i in range(405, 420)]

labels_dict = {
    1: 'Femur',
    2: 'Femoral Cartilage',
    3: 'Tibia',
    4: 'Medial Tibial Cartilage',
    5: 'Lateral Tibial Cartilage',
}

all_dices = {l: [] for l in labels_dict.values()}

for case in cases:
    try:
        pred_native = sitk.ReadImage(f'temp_cartimorph_io/output_test_15/{case}.nii.gz')
        gt_iso = sitk.ReadImage(f'temp_cartimorph_io/gt_test_15/{case}_label.nii.gz')
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(gt_iso)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(sitk.Transform())
        pred_iso = resampler.Execute(pred_native)
        
        pred_arr = sitk.GetArrayFromImage(pred_iso)
        gt_arr = sitk.GetArrayFromImage(gt_iso)
        
        for label_val, label_name in labels_dict.items():
            p = (pred_arr == label_val)
            g = (gt_arr == label_val)
            intersection = np.sum(p & g)
            total = np.sum(p) + np.sum(g)
            if total == 0:
                dice = 1.0
            else:
                dice = 2.0 * intersection / total
            all_dices[label_name].append(dice)
    except Exception as e:
        print(f"Failed on {case}: {e}")

print("\n=== Average Test Dice (15 Cases) ===")
for label_name, scores in all_dices.items():
    avg = np.mean(scores) if scores else 0
    std = np.std(scores) if scores else 0
    print(f"{label_name:25s} | Mean Dice: {avg:.4f} +/- {std:.4f}")
