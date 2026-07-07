"""
Preprocessing utilities for MRI and CT volumes.

Implements the Week 1 pipeline from the implementation plan:
  1. N4 Bias Field Correction (MRI)
  2. Resampling to isotropic spacing
  3. QA gating (slice spacing, artifact detection)
"""

import os
import glob
import logging
import SimpleITK as sitk
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. N4 Bias Field Correction
# ---------------------------------------------------------------------------

def n4_bias_field_correction(
    image: sitk.Image,
    shrink_factor: int = 4,
    num_iterations: list = None,
    convergence_threshold: float = 1e-6,
) -> sitk.Image:
    """
    Apply N4 Bias Field Correction to an MRI volume.

    Uses an Otsu-threshold mask to exclude background air before fitting the
    B-spline field, as recommended in the implementation plan.

    Args:
        image: Input MRI volume (sitk.Image).
        shrink_factor: Downsample factor for speed (default 4).
        num_iterations: Iterations per fitting level (default [50, 50, 30, 20]).
        convergence_threshold: Convergence threshold for the corrector.

    Returns:
        Bias-corrected MRI volume (sitk.Image).
    """
    if num_iterations is None:
        num_iterations = [50, 50, 30, 20]

    logger.info("Running N4 Bias Field Correction...")

    # Cast to float32 — N4 requires a real-valued pixel type
    input_image = sitk.Cast(image, sitk.sitkFloat32)

    # Create an Otsu-threshold mask to exclude background air
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    mask = otsu_filter.Execute(input_image)

    # Shrink for speed
    shrunk_image = sitk.Shrink(input_image, [shrink_factor] * input_image.GetDimension())
    shrunk_mask = sitk.Shrink(mask, [shrink_factor] * mask.GetDimension())

    # Configure and run the corrector
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(num_iterations)
    corrector.SetConvergenceThreshold(convergence_threshold)

    corrected_shrunk = corrector.Execute(shrunk_image, shrunk_mask)

    # Reconstruct the bias field at full resolution and apply it
    log_bias_field = corrector.GetLogBiasFieldAsImage(input_image)
    corrected_image = input_image / sitk.Exp(log_bias_field)

    logger.info("N4 Bias Field Correction complete.")
    return corrected_image


# ---------------------------------------------------------------------------
# 2. Resampling to Isotropic Spacing
# ---------------------------------------------------------------------------

def resample_to_isotropic(
    image: sitk.Image,
    target_spacing: float = 0.5,
    interpolator=sitk.sitkBSpline,
    default_pixel_value: float = 0.0,
) -> sitk.Image:
    """
    Resample a volume to isotropic voxel spacing.

    Uses B-spline interpolation for image volumes (use sitkNearestNeighbor
    for label masks to preserve discrete values).

    Args:
        image: Input volume (sitk.Image).
        target_spacing: Desired isotropic spacing in mm (default 0.5).
        interpolator: SimpleITK interpolator (default B-spline for images).
        default_pixel_value: Fill value for out-of-bounds voxels.

    Returns:
        Resampled volume (sitk.Image).
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_spacing = [target_spacing] * image.GetDimension()
    new_size = [
        int(round(osz * ospc / target_spacing))
        for osz, ospc in zip(original_size, original_spacing)
    ]

    logger.info(
        f"Resampling: {original_spacing} -> {tuple(new_spacing)}, "
        f"size {original_size} -> {tuple(new_size)}"
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetInterpolator(interpolator)

    resampled = resampler.Execute(image)
    logger.info("Resampling complete.")
    return resampled


def resample_label_to_isotropic(
    label: sitk.Image,
    target_spacing: float = 0.5,
) -> sitk.Image:
    """
    Convenience wrapper: resample a label mask using nearest-neighbor
    interpolation to preserve discrete label values.
    """
    return resample_to_isotropic(
        label,
        target_spacing=target_spacing,
        interpolator=sitk.sitkNearestNeighbor,
        default_pixel_value=0.0,
    )


# ---------------------------------------------------------------------------
# 3. QA Gating
# ---------------------------------------------------------------------------

MAX_SLICE_SPACING_MM = 1.5  # From the implementation plan


def qa_check(image: sitk.Image, filepath: str = "") -> dict:
    """
    Run quality-assurance checks on a volume.

    Checks performed (after N4 correction, as the plan recommends):
      - Slice spacing > MAX_SLICE_SPACING_MM
      - Intensity range sanity (detects blank or saturated volumes)
      - NaN / Inf detection

    Args:
        image: The (bias-corrected) volume to check.
        filepath: Optional path string for logging context.

    Returns:
        dict with keys:
            passed (bool): True if the volume passes all checks.
            flags (list[str]): List of human-readable issue descriptions.
    """
    flags = []
    spacing = image.GetSpacing()

    # --- Slice spacing check ---
    # In a 3D volume the third spacing dimension is the slice spacing
    if image.GetDimension() >= 3:
        slice_spacing = spacing[2]
        if slice_spacing > MAX_SLICE_SPACING_MM:
            flags.append(
                f"Slice spacing {slice_spacing:.3f} mm exceeds "
                f"threshold {MAX_SLICE_SPACING_MM} mm"
            )

    # --- Intensity sanity ---
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    img_min = stats.GetMinimum()
    img_max = stats.GetMaximum()
    img_mean = stats.GetMean()
    img_std = stats.GetSigma()

    if img_max - img_min < 1e-6:
        flags.append("Flat intensity — volume appears blank or constant")

    if img_std < 1e-6:
        flags.append("Near-zero standard deviation — possible corrupt volume")

    # --- NaN / Inf check ---
    arr = sitk.GetArrayFromImage(image)
    if np.any(np.isnan(arr)):
        flags.append("Volume contains NaN values")
    if np.any(np.isinf(arr)):
        flags.append("Volume contains Inf values")

    passed = len(flags) == 0
    status = "PASS" if passed else "FAIL"
    tag = filepath or "volume"
    logger.info(f"QA [{status}] {tag}: {flags if flags else 'all checks passed'}")

    return {"passed": passed, "flags": flags}


# ---------------------------------------------------------------------------
# 4. Full Preprocessing Pipeline
# ---------------------------------------------------------------------------

def preprocess_mri(
    image_path: str,
    label_path: str = None,
    output_dir: str = "data/preprocessed",
    target_spacing: float = 0.5,
    skip_n4: bool = False,
) -> dict:
    """
    Run the full Week 1 preprocessing pipeline on a single MRI case.

    Pipeline order (from the implementation plan):
      1. Load NIfTI / DICOM
      2. N4 Bias Field Correction
      3. Resample to isotropic spacing
      4. QA gating (post-correction, as the plan specifies)

    Args:
        image_path: Path to the input image (.nii.gz or DICOM directory).
        label_path: Optional path to the corresponding label mask.
        output_dir: Directory to write preprocessed outputs.
        target_spacing: Isotropic spacing in mm.
        skip_n4: If True, skip bias field correction (e.g. for CT volumes).

    Returns:
        dict with keys: image_out, label_out (paths), qa_result.
    """
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.splitext(os.path.basename(image_path))[0])[0]

    # --- Load ---
    if os.path.isdir(image_path):
        # DICOM directory
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(image_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        logger.info(f"Loaded DICOM series from {image_path} ({len(dicom_names)} slices)")
    else:
        image = sitk.ReadImage(image_path)
        logger.info(f"Loaded {image_path}")

    # --- N4 Correction ---
    if not skip_n4:
        image = n4_bias_field_correction(image)
    else:
        logger.info("Skipping N4 (CT or skip_n4=True)")

    # --- Resample ---
    image = resample_to_isotropic(image, target_spacing=target_spacing)

    # --- QA (post-correction as the plan recommends) ---
    qa_result = qa_check(image, filepath=image_path)

    # --- Save ---
    image_out = os.path.join(output_dir, f"{basename}_preprocessed.nii.gz")
    sitk.WriteImage(image, image_out)
    logger.info(f"Saved preprocessed image to {image_out}")

    label_out = None
    if label_path and os.path.exists(label_path):
        label = sitk.ReadImage(label_path)
        label = resample_label_to_isotropic(label, target_spacing=target_spacing)
        label_out = os.path.join(output_dir, f"{basename}_label.nii.gz")
        sitk.WriteImage(label, label_out)
        logger.info(f"Saved preprocessed label to {label_out}")

    return {
        "image_out": image_out,
        "label_out": label_out,
        "qa_result": qa_result,
    }


# ---------------------------------------------------------------------------
# 5. Batch Processing
# ---------------------------------------------------------------------------

def preprocess_oaizib_batch(
    images_dir: str,
    labels_dir: str = None,
    output_dir: str = "data/preprocessed",
    target_spacing: float = 0.5,
    max_cases: int = None,
) -> list:
    """
    Batch-preprocess all NIfTI files in the OAI-ZIB dataset directory.

    Args:
        images_dir: Path to the imagesTr or imagesTs directory.
        labels_dir: Path to the corresponding labelsTr or labelsTs directory.
        output_dir: Output directory for preprocessed files.
        target_spacing: Isotropic spacing in mm.
        max_cases: Limit the number of cases to process (for quick testing).

    Returns:
        List of result dicts from preprocess_mri().
    """
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    if max_cases:
        image_files = image_files[:max_cases]

    logger.info(f"Found {len(image_files)} cases in {images_dir}")
    results = []

    for img_path in image_files:
        # Match the label file by replacing imagesTr -> labelsTr in the path
        label_path = None
        if labels_dir:
            img_basename = os.path.basename(img_path)
            # OAI-ZIB convention: image is *_0000.nii.gz, label is *.nii.gz
            label_basename = img_basename.replace("_0000.nii.gz", ".nii.gz")
            candidate = os.path.join(labels_dir, label_basename)
            if os.path.exists(candidate):
                label_path = candidate

        result = preprocess_mri(
            image_path=img_path,
            label_path=label_path,
            output_dir=output_dir,
            target_spacing=target_spacing,
        )
        results.append(result)

    # --- Summary ---
    passed = sum(1 for r in results if r["qa_result"]["passed"])
    failed = len(results) - passed
    logger.info(f"Batch complete: {passed} passed, {failed} failed QA out of {len(results)} cases")

    return results
