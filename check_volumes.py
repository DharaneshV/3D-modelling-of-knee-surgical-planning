import os
import sys
import numpy as np
import SimpleITK as sitk

def check_volumes(case):
    print(f"\n[{case}] Volume check:")
    our_mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
    ts_total_dir = f"data/ct_knee/{case}_totalseg_reference"
    ts_bones_dir = f"data/ct_knee/{case}_totalseg_reference_bones"
    
    if not os.path.exists(our_mask_path) or not os.path.exists(ts_total_dir):
        print("Files missing")
        return

    pred_img = sitk.ReadImage(our_mask_path)
    pred_arr = sitk.GetArrayFromImage(pred_img)
    
    # Load template for TS
    template_img = None
    for side in ['left', 'right']:
        p = os.path.join(ts_total_dir, f"femur_{side}.nii.gz")
        if os.path.exists(p):
            template_img = sitk.ReadImage(p)
            break
    if not template_img:
        p = os.path.join(ts_bones_dir, "tibia.nii.gz")
        if os.path.exists(p):
            template_img = sitk.ReadImage(p)
            
    template_arr = sitk.GetArrayFromImage(template_img)
    combined = np.zeros_like(template_arr, dtype=np.uint8)

    # Femur 
    for side in ['left', 'right']:
        p = os.path.join(ts_total_dir, f"femur_{side}.nii.gz")
        if os.path.exists(p):
            arr = sitk.GetArrayFromImage(sitk.ReadImage(p))
            if arr.sum() > 0:
                combined[arr > 0] = 1
                
    # Tibia + Patella
    for roi, label_val in [('tibia', 2), ('patella', 3)]:
        p = os.path.join(ts_bones_dir, f"{roi}.nii.gz")
        if os.path.exists(p):
            arr = sitk.GetArrayFromImage(sitk.ReadImage(p))
            combined[arr > 0] = label_val
            
    gt_img_raw = sitk.GetImageFromArray(combined)
    gt_img_raw.CopyInformation(template_img)

    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(pred_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    gt_img = resampler.Execute(gt_img_raw)
    
    gt_arr = sitk.GetArrayFromImage(gt_img)
    
    for bone, label_val in [('Femur', 1), ('Tibia', 2), ('Patella', 3)]:
        our_vol = np.sum(pred_arr == label_val)
        ts_vol = np.sum(gt_arr == label_val)
        print(f"  {bone}: Our Voxel Count = {our_vol}, Resampled TS Count = {ts_vol}")

for case in ['STS_006', 'STS_035']:
    check_volumes(case)

