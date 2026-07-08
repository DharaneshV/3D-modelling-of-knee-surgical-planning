import SimpleITK as sitk
import numpy as np

case = 'STS_006'
our_mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
img = sitk.ReadImage(our_mask_path)
arr = sitk.GetArrayFromImage(img)

# Femur is label 1
femur_arr = (arr == 1).astype(np.uint8)

# Find Z range of femur
z_indices = np.where(np.any(femur_arr, axis=(1,2)))[0]
min_z = z_indices.min()
max_z = z_indices.max()
print(f"Femur Z range: {min_z} to {max_z}")

total_vol = 0
total_shaft_vol = 0
total_condyle_vol = 0

# Let's say Z=0 is the top (proximal) and Z=295 is the bottom (distal).
# Distal femur (condyles) has the highest area.
for z in range(min_z, max_z + 1):
    area = np.sum(femur_arr[z])
    total_vol += area
    if z < 100: # Distal part? Let's check which way is which.
        pass

# Let's just print area at intervals to see the profile
for z in range(min_z, max_z + 1, 20):
    print(f"Z={z}: Area = {np.sum(femur_arr[z])}")

print(f"Total voxels: {total_vol}")
