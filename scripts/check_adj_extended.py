import SimpleITK as sitk
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

img = sitk.ReadImage('meshes/a0810f6b-f014-44df-91ff-0428436c59e5/a0810f6b-f014-44df-91ff-0428436c59e5_mask.nii.gz')
arr = sitk.GetArrayFromImage(img)

femur_r = (arr == 2)
tibia_r = (arr == 4)

# 26-connected dilation
struct = generate_binary_structure(3, 3)
femur_dilated = binary_dilation(femur_r, structure=struct)
overlap = femur_dilated & tibia_r
print(f"26-connected overlap: {np.sum(overlap)} voxels")

# How many voxels are separated by exactly 1 voxel?
# Dilate by 2 iterations
femur_dilated_2 = binary_dilation(femur_r, structure=struct, iterations=2)
overlap_2 = femur_dilated_2 & tibia_r
print(f"2-voxel distance overlap: {np.sum(overlap_2)} voxels")
