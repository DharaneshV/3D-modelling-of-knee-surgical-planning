import SimpleITK as sitk
import numpy as np
import os

def get_volume_stats(mask_path):
    img = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(img)
    
    femur_voxels = np.sum(arr == 1)
    tibia_voxels = np.sum(arr == 2)
    patella_voxels = np.sum(arr == 3)
    
    spacing = img.GetSpacing()
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    
    return {
        'femur_vox': int(femur_voxels),
        'tibia_vox': int(tibia_voxels),
        'patella_vox': int(patella_voxels),
        'femur_vol_mm3': femur_voxels * voxel_vol,
        'tibia_vol_mm3': tibia_voxels * voxel_vol
    }

print("=== Final Label Volumes ===")
for case in ['STS_006', 'STS_035', 'STS_043', 'STS_051']:
    mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
    if os.path.exists(mask_path):
        stats = get_volume_stats(mask_path)
        print(f"{case}: Femur = {stats['femur_vox']} vox ({stats['femur_vol_mm3']:.1f} mm3), Tibia = {stats['tibia_vox']} vox ({stats['tibia_vol_mm3']:.1f} mm3)")
