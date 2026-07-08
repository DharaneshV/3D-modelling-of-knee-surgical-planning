import SimpleITK as sitk
import numpy as np
import scipy.ndimage

cases = ['STS_006', 'STS_035', 'STS_043', 'STS_051']
for case in cases:
    img = sitk.ReadImage(f"data/ct_knee/case01_{case}_bone_mask.nii.gz")
    arr = sitk.GetArrayFromImage(img)
    f_z = scipy.ndimage.center_of_mass(arr == 1)[0]
    t_z = scipy.ndimage.center_of_mass(arr == 2)[0]
    print(f"{case} COM Z: Femur={f_z:.1f}, Tibia={t_z:.1f}")
