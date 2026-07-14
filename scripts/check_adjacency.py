import SimpleITK as sitk
import numpy as np

# Let's run a quick script to check how many voxels overlap in the RAW TotalSegmentator outputs 
# before they are merged in run_ct_segmentation.py.
# Wait, the intermediate files are deleted unless --keep-intermediate is used.
# Let's just check the combined mask. How many voxels of Femur (2) are adjacent to Tibia (4)?

img = sitk.ReadImage('meshes/a0810f6b-f014-44df-91ff-0428436c59e5/a0810f6b-f014-44df-91ff-0428436c59e5_mask.nii.gz')
arr = sitk.GetArrayFromImage(img)

# Find adjacent voxels
from scipy.ndimage import binary_dilation

femur_r = (arr == 2)
tibia_r = (arr == 4)

# Dilate femur by 1 voxel
femur_dilated = binary_dilation(femur_r)

# Find intersection
overlap = femur_dilated & tibia_r
overlap_count = np.sum(overlap)

print(f"Right knee: {overlap_count} voxels of Tibia are directly adjacent to Femur")

femur_l = (arr == 1)
tibia_l = (arr == 3)
femur_l_dilated = binary_dilation(femur_l)
overlap_l = femur_l_dilated & tibia_l
print(f"Left knee: {np.sum(overlap_l)} voxels of Tibia are directly adjacent to Femur")
