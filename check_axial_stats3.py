import SimpleITK as sitk
import numpy as np
import sys
import os

sys.path.append(".")
from scripts.run_dice_evaluation import create_ts_mask

case = 'STS_006'
our_mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
ts_total_dir = f"data/ct_knee/{case}_totalseg_reference"
ts_bones_dir = f"data/ct_knee/{case}_totalseg_reference_bones"

ref_image = sitk.ReadImage(our_mask_path)
our_arr = sitk.GetArrayFromImage(ref_image)

ts_img = create_ts_mask(ts_total_dir, ts_bones_dir)
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(ref_image)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
ts_resampled_img = resampler.Execute(ts_img)
ts_arr = sitk.GetArrayFromImage(ts_resampled_img)

zs = [200, 230, 260, 290]

print("Slice Ratio Check (Femur, Label 1):")
for z in zs:
    our_area = np.sum(our_arr[z] == 1)
    ts_area = np.sum(ts_arr[z] == 1)
    ratio = our_area / ts_area if ts_area > 0 else 0
    print(f"Z={z}: Our Area = {our_area}, TS Area = {ts_area}, Ratio = {ratio:.2f}")

