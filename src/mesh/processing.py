import pyvista as pv

import scipy.ndimage
import SimpleITK as sitk
import numpy as np

def fill_label_gaps(label_volume: sitk.Image, labels: list, closing_radius_mm: float) -> sitk.Image:
    """
    Perform a multi-label safe gap filling.
    Applies simple 3D hole filling to remove internal voids (like marrow cavities),
    without expanding the exterior boundaries.
    
    Optimized: crops to tight bounding box per label before running binary_fill_holes
    to avoid processing the full volume (~109M voxels) for each label.
    """
    output_arr = sitk.GetArrayFromImage(label_volume).copy()
    
    for l in labels:
        if l == 0:
            continue
        
        # Find voxels for this label
        coords = np.argwhere(output_arr == l)
        if len(coords) == 0:
            continue  # Skip empty labels (e.g. unilateral cases)
        
        # Compute tight bounding box with 2-voxel padding
        pad = 2
        mins = np.maximum(coords.min(axis=0) - pad, 0)
        maxs = np.minimum(coords.max(axis=0) + pad + 1, output_arr.shape)
        
        # Extract ROI
        slices = tuple(slice(lo, hi) for lo, hi in zip(mins, maxs))
        roi = output_arr[slices]
        
        # Fill holes on the small ROI only
        mask_roi = (roi == l)
        filled_roi = scipy.ndimage.binary_fill_holes(mask_roi)
        
        # Only assign label to voxels that were filled (and weren't previously another label)
        roi[(filled_roi) & (roi == 0)] = l
        output_arr[slices] = roi
        
    result_img = sitk.GetImageFromArray(output_arr)
    result_img.CopyInformation(label_volume)
    
    return result_img

