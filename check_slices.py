import SimpleITK as sitk
import numpy as np

case = 'STS_006'
our_mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
ts_total_dir = f"data/ct_knee/{case}_totalseg_reference"

pred_img = sitk.ReadImage(our_mask_path)
pred_arr = sitk.GetArrayFromImage(pred_img)

import os
template_img = sitk.ReadImage(os.path.join(ts_total_dir, "femur_right.nii.gz"))
template_arr = sitk.GetArrayFromImage(template_img)
combined = np.zeros_like(template_arr, dtype=np.uint8)

for side in ['left', 'right']:
    p = os.path.join(ts_total_dir, f"femur_{side}.nii.gz")
    if os.path.exists(p):
        arr = sitk.GetArrayFromImage(sitk.ReadImage(p))
        combined[arr > 0] = 1

ts_img_raw = sitk.GetImageFromArray(combined)
ts_img_raw.CopyInformation(template_img)

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(pred_img)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
resampler.SetDefaultPixelValue(0)
ts_img = resampler.Execute(ts_img_raw)
ts_arr = sitk.GetArrayFromImage(ts_img)

our_femur = (pred_arr == 1)
ts_femur = (ts_arr == 1)

print("Slice-by-slice voxel counts (Z, Ours, TS, Ratio):")
for z in range(200, 294, 10):
    our_count = our_femur[z].sum()
    ts_count = ts_femur[z].sum()
    ratio = our_count / ts_count if ts_count > 0 else 0
    print(f"  Z={z}: Ours={our_count}, TS={ts_count}, Ratio={ratio:.2f}")

