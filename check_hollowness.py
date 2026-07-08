import SimpleITK as sitk
import numpy as np
from scipy import ndimage

case = 'STS_006'
our_mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
pred_img = sitk.ReadImage(our_mask_path)
pred_arr = sitk.GetArrayFromImage(pred_img)

our_femur = (pred_arr == 1)
z = 230
slice_mask = our_femur[z]

filled_slice = ndimage.binary_fill_holes(slice_mask)

original_area = np.sum(slice_mask)
filled_area = np.sum(filled_slice)

print(f"Z={z} Femur Area: {original_area}")
print(f"Z={z} Filled Area: {filled_area}")
print(f"Hollowness: {100 - (original_area/filled_area)*100:.1f}% empty interior")

