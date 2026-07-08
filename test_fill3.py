import SimpleITK as sitk
import os

case = "STS_035"
mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
ct_path = f"data/ct_knee/case01_{case}_cropped.nii.gz"

ct = sitk.ReadImage(ct_path, sitk.sitkFloat32)
binary_low = sitk.BinaryThreshold(ct, lowerThreshold=200, upperThreshold=4000, insideValue=1, outsideValue=0)
closed = sitk.BinaryMorphologicalClosing(binary_low, [2, 2, 2])
opened = sitk.BinaryMorphologicalOpening(closed, [1, 1, 1])

# Fill holes
fill_filter = sitk.BinaryFillholeImageFilter()
fill_filter.SetForegroundValue(1)
filled = fill_filter.Execute(opened)

# Check if joint is merged
import numpy as np
from scipy.ndimage import distance_transform_edt

mask_arr = sitk.GetArrayFromImage(filled)
dist = distance_transform_edt(mask_arr)
print(f"Max distance transform: {np.max(dist)}")

# Check if femur and tibia are merged in the raw (unfilled) vs filled Pass 2 mask
def check_merged(mask_img, name):
    components = sitk.ConnectedComponent(mask_img)
    comp_arr = sitk.GetArrayFromImage(components)
    
    # Get Pass 1 markers
    binary_high = sitk.BinaryThreshold(ct, lowerThreshold=400, upperThreshold=4000, insideValue=1, outsideValue=0)
    binary_high = sitk.BinaryMorphologicalClosing(binary_high, [2, 2, 2])
    binary_high = sitk.BinaryMorphologicalOpening(binary_high, [1, 1, 1])
    
    comp_high = sitk.ConnectedComponent(binary_high)
    stats_high = sitk.LabelShapeStatisticsImageFilter()
    stats_high.Execute(comp_high)
    sizes = {l: stats_high.GetNumberOfPixels(l) for l in stats_high.GetLabels()}
    top_labels = sorted(sizes, key=sizes.get, reverse=True)[:3]
    
    centroids = {l: stats_high.GetCentroid(l) for l in top_labels}
    top_labels.sort(key=lambda l: centroids[l][2], reverse=True)
    femur_label, tibia_label = top_labels[0], top_labels[1]
    
    comp_high_arr = sitk.GetArrayFromImage(comp_high)
    femur_pixels = comp_arr[comp_high_arr == femur_label]
    tibia_pixels = comp_arr[comp_high_arr == tibia_label]
    
    femur_comps = set(femur_pixels[femur_pixels > 0])
    tibia_comps = set(tibia_pixels[tibia_pixels > 0])
    
    merged = len(femur_comps.intersection(tibia_comps)) > 0
    print(f"[{name}] Merged: {merged}")

check_merged(opened, "Unfilled Pass 2 Mask")
check_merged(filled, "Filled Pass 2 Mask")
