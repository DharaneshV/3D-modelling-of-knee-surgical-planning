import SimpleITK as sitk
import numpy as np

case = 'STS_035'
img = sitk.ReadImage(f"data/ct_knee/case01_{case}_bone_mask.nii.gz")
arr = sitk.GetArrayFromImage(img)

f_mask = arr == 1
t_mask = arr == 2

f_x = np.where(np.any(f_mask, axis=(0,1)))[0]
f_y = np.where(np.any(f_mask, axis=(0,2)))[0]

t_x = np.where(np.any(t_mask, axis=(0,1)))[0]
t_y = np.where(np.any(t_mask, axis=(0,2)))[0]

print(f"{case} Femur X: {f_x[0]} to {f_x[-1]}")
print(f"{case} Tibia X: {t_x[0]} to {t_x[-1]}")

print(f"{case} Femur Y: {f_y[0]} to {f_y[-1]}")
print(f"{case} Tibia Y: {t_y[0]} to {t_y[-1]}")
