import SimpleITK as sitk
import numpy as np

case = 'STS_035'
img = sitk.ReadImage(f"data/ct_knee/case01_{case}_bone_mask.nii.gz")
arr = sitk.GetArrayFromImage(img)

f_mask = arr == 1
t_mask = arr == 2

f_z = np.where(np.any(f_mask, axis=(1,2)))[0]
t_z = np.where(np.any(t_mask, axis=(1,2)))[0]

print(f"{case} Femur Z: {f_z[0]} to {f_z[-1]}")
print(f"{case} Tibia Z: {t_z[0]} to {t_z[-1]}")

