"""
Bone segmentation from CT volumes using Hounsfield-unit thresholding.

Week 2–3 deliverable: femur, tibia, and patella masks from CT scans.
The implementation plan confirms that threshold-based segmentation is the
correct approach here — bone-CT contrast is high enough that deep learning
is unnecessary overhead for the POC.

Pipeline:
  1. HU thresholding (bone window)
  2. Connected-component analysis to isolate individual bones
  3. Morphological cleanup (fill holes, smooth boundaries)
  4. Label assignment (femur=1, tibia=2, patella=3)
"""

import logging
import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hounsfield Unit thresholds for cortical + trabecular bone
# ---------------------------------------------------------------------------
# These are standard clinical ranges. The plan notes that threshold values
# may need tuning per-scanner — expose them as parameters.

DEFAULT_BONE_HU_MIN = 200   # Lower bound: captures trabecular bone
DEFAULT_BONE_HU_MAX = 3000  # Upper bound: avoids metal artifacts
DEFAULT_CORTICAL_HU_MIN = 500  # Dense cortical bone (for tighter segmentation)


# ---------------------------------------------------------------------------
# 1. Threshold-based bone extraction
# ---------------------------------------------------------------------------

def threshold_bone(
    ct_image: sitk.Image,
    hu_min: int = DEFAULT_BONE_HU_MIN,
    hu_max: int = DEFAULT_BONE_HU_MAX,
) -> sitk.Image:
    """
    Extract bone voxels from a CT volume using Hounsfield-unit thresholding.

    Args:
        ct_image: Input CT volume (sitk.Image) in Hounsfield units.
        hu_min: Lower HU threshold (default 200 for trabecular bone).
        hu_max: Upper HU threshold (default 3000 to avoid metal artifacts).

    Returns:
        Binary mask of bone voxels (sitk.Image, uint8).
    """
    logger.info(f"Thresholding bone: HU range [{hu_min}, {hu_max}]")

    binary_mask = sitk.BinaryThreshold(
        ct_image,
        lowerThreshold=hu_min,
        upperThreshold=hu_max,
        insideValue=1,
        outsideValue=0,
    )
    binary_mask = sitk.Cast(binary_mask, sitk.sitkUInt8)

    return binary_mask


# ---------------------------------------------------------------------------
# 2. Morphological cleanup
# ---------------------------------------------------------------------------

def morphological_cleanup(
    mask: sitk.Image,
    closing_radius: int = 3,
    opening_radius: int = 1,
    fill_holes: bool = True,
) -> sitk.Image:
    """
    Clean up a binary bone mask with morphological operations.

    Steps:
      1. Binary closing — fills small gaps in bone cortex
      2. Binary opening — removes small noise islands
      3. Hole filling — fills internal cavities (marrow space)

    Args:
        mask: Binary bone mask (sitk.Image).
        closing_radius: Radius for closing operation (default 3).
        opening_radius: Radius for opening operation (default 1).
        fill_holes: Whether to fill internal holes (default True).

    Returns:
        Cleaned binary mask (sitk.Image).
    """
    logger.info("Running morphological cleanup...")

    # Closing: fills small gaps
    closed = sitk.BinaryMorphologicalClosing(mask, [closing_radius] * 3)

    # Opening: removes small noise
    opened = sitk.BinaryMorphologicalOpening(closed, [opening_radius] * 3)

    # Fill holes (marrow cavity)
    if fill_holes:
        opened = sitk.BinaryFillhole(opened)

    return opened


# ---------------------------------------------------------------------------
# 3. Connected-component labeling to isolate individual bones
# ---------------------------------------------------------------------------

def separate_bones(
    bone_mask: sitk.Image,
    min_volume_mm3: float = 5000.0,
) -> sitk.Image:
    """
    Use connected-component analysis to isolate individual bone structures
    (femur, tibia, patella) from a single binary bone mask.

    Bones are labeled by size (largest = femur, second = tibia, third = patella).
    This heuristic works well for knee CT because the femur is always the
    largest bone in the field of view, followed by the tibia.

    Args:
        bone_mask: Binary bone mask (sitk.Image).
        min_volume_mm3: Minimum component volume to keep (filters noise).

    Returns:
        Multi-label mask: femur=1, tibia=2, patella=3 (sitk.Image, uint8).
    """
    logger.info("Separating bones via connected-component analysis...")

    # Label connected components
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    labeled = cc_filter.Execute(bone_mask)
    num_components = cc_filter.GetObjectCount()
    logger.info(f"Found {num_components} connected components")

    # Measure component sizes
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labeled)

    # Get (label, physical_size) pairs, sorted by size descending
    components = []
    for label_val in stats.GetLabels():
        volume = stats.GetPhysicalSize(label_val)
        if volume >= min_volume_mm3:
            components.append((label_val, volume))

    components.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Retained {len(components)} components above {min_volume_mm3} mm³ threshold")

    # Assign anatomical labels by size order
    # Femur (largest) = 1, Tibia (second) = 2, Patella (third) = 3
    BONE_LABELS = {0: "femur", 1: "tibia", 2: "patella"}

    labeled_arr = sitk.GetArrayFromImage(labeled)
    output_arr = np.zeros_like(labeled_arr, dtype=np.uint8)

    for idx, (cc_label, volume) in enumerate(components[:3]):
        bone_name = BONE_LABELS.get(idx, f"bone_{idx}")
        anatomical_label = idx + 1
        output_arr[labeled_arr == cc_label] = anatomical_label
        logger.info(
            f"  {bone_name} (label={anatomical_label}): "
            f"volume={volume:.1f} mm³"
        )

    output_image = sitk.GetImageFromArray(output_arr)
    output_image.CopyInformation(bone_mask)

    return output_image


# ---------------------------------------------------------------------------
# 4. Full bone segmentation pipeline
# ---------------------------------------------------------------------------

def segment_bones(
    ct_image: sitk.Image,
    hu_min: int = DEFAULT_BONE_HU_MIN,
    hu_max: int = DEFAULT_BONE_HU_MAX,
    closing_radius: int = 3,
    opening_radius: int = 1,
    fill_holes: bool = True,
    min_volume_mm3: float = 5000.0,
) -> dict:
    """
    Full bone segmentation pipeline for knee CT.

    Pipeline:
      1. HU thresholding
      2. Morphological cleanup (closing → opening → hole fill)
      3. Connected-component separation into femur/tibia/patella

    Args:
        ct_image: Input CT volume in Hounsfield units (sitk.Image).
        hu_min: Lower HU threshold.
        hu_max: Upper HU threshold.
        closing_radius: Morphological closing radius.
        opening_radius: Morphological opening radius.
        fill_holes: Whether to fill internal holes.
        min_volume_mm3: Minimum component volume to retain.

    Returns:
        dict with keys:
            binary_mask: Raw thresholded bone mask.
            cleaned_mask: Morphologically cleaned mask.
            labeled_mask: Multi-label mask (femur=1, tibia=2, patella=3).
    """
    logger.info("=" * 60)
    logger.info("Starting bone segmentation pipeline")
    logger.info("=" * 60)

    # Step 1: Threshold
    binary_mask = threshold_bone(ct_image, hu_min=hu_min, hu_max=hu_max)

    # Step 2: Cleanup
    cleaned_mask = morphological_cleanup(
        binary_mask,
        closing_radius=closing_radius,
        opening_radius=opening_radius,
        fill_holes=fill_holes,
    )

    # Step 3: Separate into individual bones
    labeled_mask = separate_bones(cleaned_mask, min_volume_mm3=min_volume_mm3)

    logger.info("Bone segmentation pipeline complete.")
    return {
        "binary_mask": binary_mask,
        "cleaned_mask": cleaned_mask,
        "labeled_mask": labeled_mask,
    }
