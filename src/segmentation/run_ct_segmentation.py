"""
run_ct_segmentation.py — Production CT bone segmentation using TotalSegmentator

Replaces the HU-threshold + watershed pipeline (legacy_threshold.py) as the
primary production method. Uses TotalSegmentator models for all three bones:
  - task='total'              (Apache 2.0) → femur, tibia
  - task='appendicular_bones' (licensed)  → patella

Combines both task outputs into a single multi-label mask matching the
downstream label convention: 1=femur, 2=tibia, 3=patella.

Output is geometrically equivalent to the old bone_segmentation.py output
so that nothing downstream (meshing, clinical measurements) needs changes.

Usage:
    python src/segmentation/run_ct_segmentation.py \
        --input  data/raw/case01_STS_006.nii.gz \
        --output outputs/STS_006/masks/bone_mask.nii.gz \
        [--leg right|left|auto]
"""

import os
import sys
import shutil
import logging
import argparse
import subprocess
import tempfile
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# Suppress noisy nnU-Net / torch warnings
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Labels — must stay consistent with meshing, measurements, and report downstream
LABEL_FEMUR  = 1
LABEL_TIBIA  = 2
LABEL_PATELLA = 3

# TotalSegmentator ROI names for each task
# 'total' task covers femur on both sides
TOTAL_FEMUR_ROIS  = ["femur_left",  "femur_right"]

# 'appendicular_bones' task covers tibia and patella (no left/right suffix in V2)
APPENDICULAR_TIBIA_ROIS = ["tibia"]
APPENDICULAR_PATELLA_ROIS = ["patella"]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: detect which leg (left/right) is in a cropped volume
# ─────────────────────────────────────────────────────────────────────────────

def detect_leg_side(image: sitk.Image) -> str:
    """Return 'right' or 'left' based on image direction cosines."""
    direction = image.GetDirection()
    # In standard LPS, direction[0] > 0 means lower X-index is Right
    return "right" if direction[0] >= 0 else "left"


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing: resampling (TotalSegmentator handles its own resampling,
# but we resample to isotropic 1mm³ before passing in for consistency)
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_isotropic(image: sitk.Image, spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    orig_spacing = image.GetSpacing()
    orig_size    = image.GetSize()
    new_size = [
        int(round(sz * ospc / tspc))
        for sz, ospc, tspc in zip(orig_size, orig_spacing, spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1024)
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)


def crop_to_leg(image: sitk.Image, side: str = "auto") -> sitk.Image:
    """Crop bilateral scan to a single leg. Mirrors legacy logic exactly."""
    size    = image.GetSize()
    spacing = image.GetSpacing()
    width_mm = size[0] * spacing[0]

    if width_mm < 300:
        logger.info(f"Image width {width_mm:.1f}mm < 300mm — assumed single-leg, no crop.")
        return image

    if side == "auto":
        side = detect_leg_side(image)
        logger.info(f"Auto-detected leg side: {side}")

    direction = image.GetDirection()
    mid_x = size[0] // 2

    if direction[0] >= 0:
        # Lower X-index = Right side
        extract_index = [0, 0, 0]         if side == "right" else [mid_x, 0, 0]
        extract_size  = [mid_x, size[1], size[2]] if side == "right" else [size[0] - mid_x, size[1], size[2]]
    else:
        # Lower X-index = Left side
        extract_index = [0, 0, 0]         if side == "left" else [mid_x, 0, 0]
        extract_size  = [mid_x, size[1], size[2]] if side == "left" else [size[0] - mid_x, size[1], size[2]]

    extractor = sitk.ExtractImageFilter()
    extractor.SetSize(extract_size)
    extractor.SetIndex(extract_index)
    cropped = extractor.Execute(image)
    logger.info(f"Cropped to {side} leg: {cropped.GetSize()}")
    return cropped


# ─────────────────────────────────────────────────────────────────────────────
# TotalSegmentator inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_totalsegmentator(nifti_path: str, out_dir: str, task: str,
                          roi_subset: list | None = None):
    """
    Calls TotalSegmentator via subprocess (the console script entry point),
    not the Python API. This avoids the Windows multiprocessing pickling crash
    in nnUNet's prediction worker spawn — same pattern used in run_mri_segmentation.py.
    """
    # Use the venv TotalSegmentator console script (which is an .exe in v2)
    ts_exe = Path(sys.executable).parent / "TotalSegmentator.exe"
    cmd = [
        str(ts_exe),
        "-i", nifti_path,
        "-o", out_dir,
        "-ta", task,        # -ta is the CLI flag for --task
        "-nr", "1",         # nr_thr_resamp
        "-ns", "1",         # nr_thr_saving
        "-q",               # quiet
    ]
    if task == "total":
        cmd.append("-f")    # fast mode (1.5mm model) only supported on total task
    if roi_subset:
        cmd += ["-rs"] + roi_subset  # -rs is --roi_subset

    logger.info(f"  Running TotalSegmentator task='{task}'  roi_subset={roi_subset}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"TotalSegmentator CLI failed (task={task}):\n"
            f"STDOUT: {result.stdout[-800:]}\nSTDERR: {result.stderr[-800:]}"
        )
    logger.info(f"  task='{task}' complete.")


def _load_roi_as_binary(out_dir: str, roi_names: list) -> np.ndarray | None:
    """
    Load one or more per-ROI NIfTI files from TotalSegmentator output dir,
    combine them into a single binary array. Returns None if no files found.
    """
    combined = None
    ref_img  = None
    for roi in roi_names:
        roi_path = os.path.join(out_dir, f"{roi}.nii.gz")
        if not os.path.exists(roi_path):
            logger.warning(f"  ROI file not found: {roi_path}")
            continue
        img = sitk.ReadImage(roi_path)
        arr = sitk.GetArrayFromImage(img).astype(np.uint8)
        if combined is None:
            combined = arr
            ref_img  = img
        else:
            combined = np.maximum(combined, arr)
    return combined, ref_img


# ─────────────────────────────────────────────────────────────────────────────
# Anatomical validation (carried over from legacy — same hard gates)
# ─────────────────────────────────────────────────────────────────────────────

def validate_combined_mask(mask_arr: np.ndarray, spacing: tuple) -> None:
    """
    Confirms the combined mask has at least femur + tibia + patella populated.
    Raises ValueError with a clear message if not, so batch pipeline can log & skip.
    """
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    labels_present = []
    min_voxels = int(3000 / voxel_vol)  # 3000 mm³ threshold from legacy

    for label_val, name in [(LABEL_FEMUR, "femur"), (LABEL_TIBIA, "tibia"), (LABEL_PATELLA, "patella")]:
        count = int(np.sum(mask_arr == label_val))
        if count >= min_voxels:
            labels_present.append(name)
        else:
            logger.warning(f"  {name}: only {count} voxels (< {min_voxels} threshold) — treating as absent.")

    if len(labels_present) < 3:
        missing = [n for n in ["femur", "tibia", "patella"] if n not in labels_present]
        raise ValueError(
            f"Anatomical validation failed: missing bones after TotalSegmentator: {missing}. "
            f"This is likely a non-knee scan, or TotalSegmentator did not detect these structures."
        )

    # Volume sanity: femur volume should be larger than patella
    femur_vol = np.sum(mask_arr == LABEL_FEMUR)  * voxel_vol
    patella_vol= np.sum(mask_arr == LABEL_PATELLA)* voxel_vol
    if patella_vol > femur_vol:
        raise ValueError(
            f"Anatomical sanity check failed: patella volume ({patella_vol:.0f}mm³) > "
            f"femur volume ({femur_vol:.0f}mm³). Label assignment likely incorrect."
        )
    logger.info(f"  Validation passed: {labels_present} all present.")


# ─────────────────────────────────────────────────────────────────────────────
# Main segmentation entry point
# ─────────────────────────────────────────────────────────────────────────────

def segment(input_path: str, output_path: str, leg: str = "auto",
            intermediate_dir: str | None = None, keep_intermediate: bool = False):
    """
    Full production segmentation pipeline:
      1. Resample to isotropic 1mm³
      2. Crop to single leg
      3. Run TotalSegmentator (total task → femur, tibia)
      4. Run TotalSegmentator (appendicular_bones task → patella)
      5. Combine into single multi-label mask (1=femur, 2=tibia, 3=patella)
      6. Anatomical validation
      7. Write output mask

    Args:
        input_path:         Path to raw CT NIfTI volume.
        output_path:        Path to write the combined bone mask.
        leg:                'right', 'left', or 'auto' (auto-detects from direction cosines).
        intermediate_dir:   Where to write TotalSegmentator raw output. Defaults to a temp dir.
        keep_intermediate:  If False (default), intermediate dir is deleted after success.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load + resample
    logger.info(f"Loading: {input_path}")
    raw_ct = sitk.ReadImage(input_path, sitk.sitkFloat32)
    logger.info(f"  Original spacing={raw_ct.GetSpacing()}, size={raw_ct.GetSize()}")
    ct = resample_to_isotropic(raw_ct)
    logger.info(f"  Resampled: spacing={ct.GetSpacing()}, size={ct.GetSize()}")

    # ── 2. Crop to single leg
    ct_leg = crop_to_leg(ct, side=leg)

    # ── 3 & 4. Run TotalSegmentator — use a temp dir if none specified
    cleanup_intermediate = False
    if intermediate_dir is None:
        intermediate_dir = tempfile.mkdtemp(prefix="totalseg_")
        cleanup_intermediate = not keep_intermediate

    total_dir        = os.path.join(intermediate_dir, "total")
    appendicular_dir = os.path.join(intermediate_dir, "appendicular")
    os.makedirs(total_dir, exist_ok=True)
    os.makedirs(appendicular_dir, exist_ok=True)

    # Write cropped leg to temp NIfTI for TotalSegmentator input
    leg_nifti = os.path.join(intermediate_dir, "ct_leg.nii.gz")
    sitk.WriteImage(ct_leg, leg_nifti)

    try:
        # task='total' for femur (Apache 2.0, no license concern)
        _run_totalsegmentator(leg_nifti, total_dir, task="total",
                              roi_subset=TOTAL_FEMUR_ROIS)

        # task='appendicular_bones' for patella and tibia (requires license)
        # Note: TotalSegmentator V2 does not support roi_subset on this task,
        # so we run it for all bones and simply load only the ones we need.
        _run_totalsegmentator(leg_nifti, appendicular_dir, task="appendicular_bones")
    except Exception as e:
        # Clean up temp dir before re-raising
        if cleanup_intermediate:
            shutil.rmtree(intermediate_dir, ignore_errors=True)
        raise e

    logger.info("Combining TotalSegmentator labels...")
    ref_img  = ct_leg
    combined = np.zeros(sitk.GetArrayFromImage(ct_leg).shape, dtype=np.uint8)

    femur_arr,  femur_ref  = _load_roi_as_binary(total_dir,        TOTAL_FEMUR_ROIS)
    tibia_arr,  tibia_ref  = _load_roi_as_binary(appendicular_dir, APPENDICULAR_TIBIA_ROIS)
    patella_arr, pat_ref   = _load_roi_as_binary(appendicular_dir, APPENDICULAR_PATELLA_ROIS)

    if femur_arr  is not None: combined[femur_arr  > 0] = LABEL_FEMUR
    if tibia_arr  is not None: combined[tibia_arr  > 0] = LABEL_TIBIA
    # Patella written last — if there's any overlap with tibia near joint, patella wins
    if patella_arr is not None: combined[patella_arr > 0] = LABEL_PATELLA

    # ── 6. Anatomical validation
    validate_combined_mask(combined, ct_leg.GetSpacing())

    # ── 7. Write output
    mask_img = sitk.GetImageFromArray(combined)
    mask_img.CopyInformation(ct_leg)
    sitk.WriteImage(mask_img, output_path)
    logger.info(f"Mask written: {output_path}")

    # Cleanup intermediate files
    if cleanup_intermediate:
        shutil.rmtree(intermediate_dir, ignore_errors=True)
        logger.info("Intermediate TotalSegmentator outputs deleted.")

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TotalSegmentator-based CT bone segmentation (production)")
    parser.add_argument("--input",  required=True, help="Path to input CT NIfTI (.nii.gz)")
    parser.add_argument("--output", required=True, help="Path to write combined bone mask")
    parser.add_argument("--leg",    default="auto", choices=["auto", "right", "left"],
                        help="Which leg to isolate (default: auto-detect from direction cosines)")
    parser.add_argument("--keep-intermediate", action="store_true",
                        help="Keep raw TotalSegmentator per-ROI outputs for inspection")
    parser.add_argument("--intermediate-dir", default=None,
                        help="Explicit directory for intermediate outputs (implies --keep-intermediate if set)")
    args = parser.parse_args()

    try:
        segment(
            input_path=args.input,
            output_path=args.output,
            leg=args.leg,
            intermediate_dir=args.intermediate_dir,
            keep_intermediate=args.keep_intermediate or (args.intermediate_dir is not None),
        )
        print(f"SUCCESS: mask saved to {args.output}")
        sys.exit(0)
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
