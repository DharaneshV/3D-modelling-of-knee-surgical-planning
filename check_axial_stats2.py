import SimpleITK as sitk
import numpy as np
import sys
import os

case = 'STS_006'
our_mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"

ts_total_dir = f"data/ct_knee/case01_{case}_cropped_ts_total"
ts_bones_dir = f"data/ct_knee/case01_{case}_cropped_ts_bones"

ref_image = sitk.ReadImage(our_mask_path)
our_arr = sitk.GetArrayFromImage(ref_image)

combined = np.zeros_like(our_arr, dtype=np.uint8)

def resample(p, ref, label_val):
    if os.path.exists(p):
        img = sitk.ReadImage(p)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampled = resampler.Execute(img)
        arr = sitk.GetArrayFromImage(resampled)
        combined[arr > 0] = label_val

resample(os.path.join(ts_total_dir, "femur_right.nii.gz"), ref_image, 1)
resample(os.path.join(ts_total_dir, "femur_left.nii.gz"), ref_image, 1)

ts_arr = combined
zs = [200, 230, 260, 290]

print("Slice Ratio Check (Femur):")
for z in zs:
    our_area = np.sum(our_arr[z] == 1)
    ts_area = np.sum(ts_arr[z] == 1)
    ratio = our_area / ts_area if ts_area > 0 else 0
    print(f"Z={z}: Our Area = {our_area}, TS Area = {ts_area}, Ratio = {ratio:.2f}")

