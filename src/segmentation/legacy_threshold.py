"""
bone_segmentation.py

Week 1-2 combined: takes a raw downloaded CT volume, resamples it to a
standard isotropic spacing, then applies HU thresholding + morphological
cleanup + connected-component separation to isolate femur, tibia, patella.

Usage:
    python scripts/bone_segmentation.py --input data/ct_knee/temp_EAY131-5310722.nii.gz --output data/ct_knee/case01_bone_mask.nii.gz
"""

import argparse
import SimpleITK as sitk
import numpy as np

DEFAULT_HU_THRESHOLD = 200
DEFAULT_SPACING = (1.0, 1.0, 1.0)  # mm, isotropic
LABELS = {"femur": 1, "tibia": 2, "patella": 3}


# ---------- Week 1: preprocessing ----------

def resample_to_isotropic(image, target_spacing=DEFAULT_SPACING, is_label=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())

    # Nearest-neighbor for label maps, linear for continuous CT intensities
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)

    return resampler.Execute(image)


# ---------- Week 2: HU thresholding + connected components ----------

def threshold_bone(ct_image, hu_threshold=DEFAULT_HU_THRESHOLD):
    return sitk.BinaryThreshold(
        ct_image,
        lowerThreshold=hu_threshold,
        upperThreshold=4000,
        insideValue=1,
        outsideValue=0,
    )


def clean_mask(binary_mask):
    closed = sitk.BinaryMorphologicalClosing(binary_mask, [2, 2, 2])
    opened = sitk.BinaryMorphologicalOpening(closed, [1, 1, 1])
    return opened


def crop_to_leg(image, side="right"):
    import logging
    logger = logging.getLogger(__name__)
    direction = image.GetDirection()
    logger.info(f"Cropping volume to {side} leg... X-axis direction cosine: {direction[0]:.4f}")
    
    size = image.GetSize()
    spacing = image.GetSpacing()
    
    # If the physical width is less than 300mm, it's likely already cropped to a single leg
    if size[0] * spacing[0] < 300:
        logger.info(f"Image width is {size[0] * spacing[0]:.1f}mm (<300mm). Assuming already cropped to a single leg.")
        return image
        
    mid_x = size[0] // 2

    # In standard LPS, X goes Right to Left. If direction[0] > 0, lower X index is Right.
    # If direction[0] < 0, higher X index is Right.
    if direction[0] >= 0:
        if side.lower() == "right":
            extract_size = [mid_x, size[1], size[2]]
            extract_index = [0, 0, 0]
        elif side.lower() == "left":
            extract_size = [size[0] - mid_x, size[1], size[2]]
            extract_index = [mid_x, 0, 0]
        else:
            raise ValueError("side must be 'right' or 'left'")
    else:
        if side.lower() == "left":
            extract_size = [mid_x, size[1], size[2]]
            extract_index = [0, 0, 0]
        elif side.lower() == "right":
            extract_size = [size[0] - mid_x, size[1], size[2]]
            extract_index = [mid_x, 0, 0]
        else:
            raise ValueError("side must be 'right' or 'left'")

    extractor = sitk.ExtractImageFilter()
    extractor.SetSize(extract_size)
    extractor.SetIndex(extract_index)
    return extractor.Execute(image)


def separate_bones(binary_mask, min_component_voxels=500):
    """
    Finds and separates the bones using connected components.
    Returns all large components (no truncation here).
    """
    components = sitk.ConnectedComponent(binary_mask)
    relabeled = sitk.RelabelComponent(components, minimumObjectSize=min_component_voxels)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(relabeled)

    return relabeled, stats, list(stats.GetLabels())

def validate_anatomy(stats, all_labels, ct_leg):
    import logging
    logger = logging.getLogger(__name__)

    # Filter out minor noise components, keeping only major bone structures (>= 3000 mm3)
    major_labels = [l for l in all_labels if stats.GetPhysicalSize(l) >= 3000.0]
    logger.info(f"All components: {len(all_labels)}, Major components (>=3000 mm3): {len(major_labels)}")

    if len(major_labels) > 6:
        raise ValueError(f"Found {len(major_labels)} major components - expected <= 6 "
                         "(femur/tibia/patella/fibula/fragments). This strongly suggests wrong "
                         "anatomy (hand/foot/ribs) or an unfiltered bilateral crop.")
    
    if len(major_labels) < 3:
        raise ValueError(f"Found only {len(major_labels)} major components - expected at least 3.")

    # Calculate bounding box of all major bones combined to get total z-extent
    min_z, max_z = float('inf'), -float('inf')
    for l in major_labels:
        bbox = stats.GetBoundingBox(l)
        z_start = bbox[2]
        z_end = bbox[2] + bbox[5]
        min_z = min(min_z, z_start)
        max_z = max(max_z, z_end)
    
    z_extent_voxels = max_z - min_z
    spacing_z = ct_leg.GetSpacing()[2]
    z_extent_mm = z_extent_voxels * spacing_z

    if z_extent_mm < 50 or z_extent_mm > 450:
        raise ValueError(f"Bone z-extent {z_extent_mm:.1f}mm is way outside expected range (50-450mm). "
                         "Wrong body part or extreme pathology.")
    elif z_extent_mm < 100 or z_extent_mm > 400:
        logger.warning(f"WARNING: Bone z-extent {z_extent_mm:.1f}mm is borderline (expected 100-400mm). "
                       "Proceeding, but this may be abnormal geometry.")

    # Sort labels by volume (largest first)
    sorted_labels = sorted(major_labels, key=lambda l: stats.GetNumberOfPixels(l), reverse=True)
    
    # We expect Femur, Tibia, and Patella. But Fibula might be larger than Patella.
    # Check the top 4 components to find the Patella.
    top_candidates = sorted_labels[:min(4, len(sorted_labels))]
    
    femur_tibia = []
    patella = None
    
    for i, label in enumerate(top_candidates):
        bbox = stats.GetBoundingBox(label)
        x_len, y_len, z_len = bbox[3] * ct_leg.GetSpacing()[0], bbox[4] * ct_leg.GetSpacing()[1], bbox[5] * ct_leg.GetSpacing()[2]
        aspect_ratio = z_len / max(x_len, y_len)
        
        if len(femur_tibia) < 2:
            # Femur and Tibia should be long bones
            if aspect_ratio <= 1.5:
                raise ValueError(f"Bone {i+1} (expected femur/tibia) has low aspect ratio {aspect_ratio:.2f}. "
                                 f"Dimensions: {x_len:.1f}x{y_len:.1f}x{z_len:.1f}mm. "
                                 "Geometry distorted by pathology or wrong anatomy.")
            femur_tibia.append(label)
        elif patella is None and aspect_ratio <= 1.5:
            patella = label
            
    if patella is None:
        raise ValueError("Could not find a compact bone (aspect ratio <= 1.5) in the top 4 components to serve as the patella. "
                         "Likely caught the fibula but missed the patella.")
                         
    return femur_tibia + [patella]


def assign_anatomical_labels(component_mask, stats, top_labels):
    """
    NOTE: assumes standard orientation (superior = higher z). Verify on
    your first case in a viewer before trusting this across the batch.
    """
    centroids = {l: stats.GetCentroid(l) for l in top_labels}
    sizes = {l: stats.GetNumberOfPixels(l) for l in top_labels}

    patella_label = min(sizes, key=sizes.get)
    remaining = [l for l in top_labels if l != patella_label]
    remaining.sort(key=lambda l: centroids[l][2], reverse=True)
    femur_label, tibia_label = remaining[0], remaining[1]
    
    # Programmatic assertion: verify femur is superior to tibia in Z-index space
    # based on the image direction matrix.
    direction = component_mask.GetDirection()
    # direction[8] is the Z-component of the Z-axis (index 8 in 3x3 matrix row-major)
    # If direction[8] > 0, higher Z-index is Superior.
    # If direction[8] < 0, lower Z-index is Superior.
    
    # To get Z-index centroids, we need to convert physical centroids back to continuous index
    femur_idx = component_mask.TransformPhysicalPointToContinuousIndex(centroids[femur_label])
    tibia_idx = component_mask.TransformPhysicalPointToContinuousIndex(centroids[tibia_label])
    
    femur_z_idx = femur_idx[2]
    tibia_z_idx = tibia_idx[2]
    
    if direction[8] > 0:
        assert femur_z_idx > tibia_z_idx, f"Orientation failure: direction[8]={direction[8]:.2f}, but Femur Z-index ({femur_z_idx:.1f}) is not greater than Tibia Z-index ({tibia_z_idx:.1f})"
    elif direction[8] < 0:
        assert femur_z_idx < tibia_z_idx, f"Orientation failure: direction[8]={direction[8]:.2f}, but Femur Z-index ({femur_z_idx:.1f}) is not less than Tibia Z-index ({tibia_z_idx:.1f})"
        
    label_map = {
        femur_label: LABELS["femur"],
        tibia_label: LABELS["tibia"],
        patella_label: LABELS["patella"],
    }

    # Map selected labels, and map ALL other labels to 0
    full_change_map = {}
    for l in stats.GetLabels():
        if l in label_map:
            full_change_map[int(l)] = int(label_map[l])
        else:
            full_change_map[int(l)] = 0

    change_filter = sitk.ChangeLabelImageFilter()
    change_filter.SetChangeMap(full_change_map)
    labeled_markers = change_filter.Execute(sitk.Cast(component_mask, sitk.sitkUInt16))
    
    # Fill holes on the markers independently to ensure solid seeds for watershed
    labeled_markers_arr = sitk.GetArrayFromImage(labeled_markers)
    filled_markers_arr = np.zeros_like(labeled_markers_arr)
    
    for l in [1, 2, 3]:
        label_mask_arr = (labeled_markers_arr == l).astype(np.uint8)
        if np.sum(label_mask_arr) > 0:
            label_img = sitk.GetImageFromArray(label_mask_arr)
            filled_label_arr = np.zeros_like(label_mask_arr)
            for z in range(label_mask_arr.shape[0]):
                slice_img = sitk.GetImageFromArray(label_mask_arr[z].astype(np.uint8))
                filled_slice = sitk.BinaryFillhole(slice_img)
                filled_label_arr[z] = sitk.GetArrayFromImage(filled_slice)
            filled_markers_arr[filled_label_arr == 1] = l
            
    final_markers = sitk.GetImageFromArray(filled_markers_arr)
    final_markers.CopyInformation(component_mask)
    return final_markers


# ---------- Entry point ----------

def run(input_path, output_path, hu_threshold=DEFAULT_HU_THRESHOLD, hu_threshold_pass2=200, spacing=DEFAULT_SPACING, leg="right"):
    import logging
    logger = logging.getLogger(__name__)

    raw_ct = sitk.ReadImage(input_path, sitk.sitkFloat32)

    print(f"Original spacing: {raw_ct.GetSpacing()}, size: {raw_ct.GetSize()}")
    ct = resample_to_isotropic(raw_ct, spacing, is_label=False)
    print(f"Resampled to spacing: {ct.GetSpacing()}, size: {ct.GetSize()}")

    ct_leg = crop_to_leg(ct, side=leg)
    print(f"Cropped to {leg} leg, new size: {ct_leg.GetSize()}")

    binary = threshold_bone(ct_leg, hu_threshold)
    binary = clean_mask(binary)
    components, stats, all_labels = separate_bones(binary)
    
    try:
        top_3_labels = validate_anatomy(stats, all_labels, ct_leg)
    except ValueError as e:
        import logging
        logging.getLogger(__name__).error(f"Anatomical validation failed: {e}")
        raise

    markers = assign_anatomical_labels(components, stats, top_3_labels)
    
    print(f"Running Pass 2 (Low threshold) at HU={hu_threshold_pass2}...")
    binary_low = threshold_bone(ct_leg, hu_threshold_pass2)
    binary_low = clean_mask(binary_low)
    
    # Run a quick check on Pass 2 topology at the joint
    components_low = sitk.ConnectedComponent(binary_low)
    stats_low = sitk.LabelShapeStatisticsImageFilter()
    stats_low.Execute(components_low)
    
    # We use skimage watershed on the negative distance transform
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    from skimage.segmentation import watershed
    
    markers_arr = sitk.GetArrayFromImage(markers)
    mask_arr = sitk.GetArrayFromImage(binary_low)
    comp_arr = sitk.GetArrayFromImage(components_low)
    
    # Check if femur and tibia markers fall into the same component in low mask
    # marker labels: femur=1, tibia=2
    femur_pixels = comp_arr[markers_arr == 1]
    tibia_pixels = comp_arr[markers_arr == 2]
    femur_comps = set(femur_pixels[femur_pixels > 0])
    tibia_comps = set(tibia_pixels[tibia_pixels > 0])
    
    merged = len(femur_comps.intersection(tibia_comps)) > 0
    if merged:
        print("PASS2_STATUS: MERGED")
    else:
        print("PASS2_STATUS: SEPARATED")
        
    # Topography: distance to background. Deep valleys at center of bones.
    dist = distance_transform_edt(mask_arr)
    topography = -dist
    
    print("Applying marker-controlled watershed...")
    labels_arr = watershed(topography, markers_arr, mask=mask_arr)
    
    final_mask_arr = np.zeros_like(labels_arr, dtype=np.uint16)
    
    # Fill holes per-label to make bones solid
    for l in [1, 2, 3]:  # femur, tibia, patella
        # Extract binary mask for this label
        label_mask_arr = (labels_arr == l).astype(np.uint8)
        if np.sum(label_mask_arr) > 0:
            label_img = sitk.GetImageFromArray(label_mask_arr)
            filled_label_arr = np.zeros_like(label_mask_arr)
            for z in range(label_mask_arr.shape[0]):
                slice_img = sitk.GetImageFromArray(label_mask_arr[z].astype(np.uint8))
                filled_slice = sitk.BinaryFillhole(slice_img)
                filled_label_arr[z] = sitk.GetArrayFromImage(filled_slice)
            # Recombine into final mask
            final_mask_arr[filled_label_arr == 1] = l

    # Explicitly pad the image by 1 voxel on all sides to ensure closed meshes
    padded_arr = np.pad(final_mask_arr, pad_width=1, mode='constant', constant_values=0)
    final_mask = sitk.GetImageFromArray(padded_arr)
    final_mask.SetSpacing(ct_leg.GetSpacing())
    final_mask.SetDirection(ct_leg.GetDirection())
    
    # Adjust origin to account for the 1-voxel padding
    origin = ct_leg.GetOrigin()
    spacing = ct_leg.GetSpacing()
    direction = ct_leg.GetDirection()
    
    # Physical shift = -1 * spacing * direction matrix
    import numpy as np
    dir_mat = np.array(direction).reshape((3,3))
    shift = -1.0 * np.array(spacing)
    new_origin = np.array(origin) + dir_mat.dot(shift)
    final_mask.SetOrigin(new_origin.tolist())
    
    # Re-run anatomical validation on the Pass 2 result
    final_stats = sitk.LabelShapeStatisticsImageFilter()
    final_stats.Execute(final_mask)
    for l in [1, 2]: # Femur and Tibia
        if final_stats.HasLabel(l):
            bbox = final_stats.GetBoundingBox(l)
            x_len, y_len, z_len = bbox[3] * ct_leg.GetSpacing()[0], bbox[4] * ct_leg.GetSpacing()[1], bbox[5] * ct_leg.GetSpacing()[2]
            aspect_ratio = z_len / max(x_len, y_len)
            if aspect_ratio <= 1.0:
                raise ValueError(f"Pass 2 validation failed: Bone {l} (expected femur/tibia) has low aspect ratio {aspect_ratio:.2f}. "
                                 f"Dimensions: {x_len:.1f}x{y_len:.1f}x{z_len:.1f}mm.")
    
    sitk.WriteImage(final_mask, output_path)
    print(f"Saved two-pass multi-label bone mask to {output_path} "
          f"(labels: femur=1, tibia=2, patella=3)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample + HU threshold + connected components bone segmentation")
    parser.add_argument("--input", required=True, help="Path to raw downloaded CT volume (NIfTI/DICOM)")
    parser.add_argument("--output", required=True, help="Path to write multi-label bone mask")
    parser.add_argument("--hu-threshold", type=int, default=DEFAULT_HU_THRESHOLD,
                         help=f"HU cutoff for Pass 1 (seeds) (default: {DEFAULT_HU_THRESHOLD})")
    parser.add_argument("--hu-threshold-pass2", type=int, default=200,
                         help=f"HU cutoff for Pass 2 (boundaries) (default: 200)")
    parser.add_argument("--spacing", type=float, nargs=3, default=DEFAULT_SPACING,
                         help="Target isotropic spacing in mm, e.g. 1.0 1.0 1.0")
    parser.add_argument("--leg", type=str, choices=["right", "left"], default="right",
                         help="Which leg to isolate from a bilateral scan (default: right)")
    args = parser.parse_args()

    run(args.input, args.output, args.hu_threshold, args.hu_threshold_pass2, tuple(args.spacing), args.leg)
