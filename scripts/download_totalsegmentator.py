"""
Download and prepare CT data for the bone segmentation track (Track B).

STRATEGY (revised):
  The TotalSegmentator Zenodo dataset is a monolithic 23 GB archive — impractical
  to download for a POC. Instead, we use two approaches:

  Option A (recommended): Install the `totalsegmentator` pip package and run it
    on any publicly available knee CT DICOM to generate bone masks automatically.
    TotalSegmentator will segment femur, tibia, patella from any CT.

  Option B: Download a small public knee CT from the TCIA Knee Phantom collection
    or similar small, open-access CT dataset, then run TotalSegmentator on it.

  This script:
    1. Installs totalsegmentator if not present.
    2. Downloads 5-8 small public CT cases (TCIA knee phantom / other CC-licensed source).
    3. Runs TotalSegmentator inference on each to produce bone masks.
    4. Crops each volume to the knee bounding box using the resulting labels.

Usage:
    python scripts/download_totalsegmentator.py --input_dir data/ct_dicom --output_dir data/ct_knee
    python scripts/download_totalsegmentator.py --single path/to/ct.nii.gz
"""

import os
import argparse
import logging
import subprocess
import sys
import glob
import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# TotalSegmentator output label values for knee bones (v2)
LABEL_FEMUR_LEFT = 7
LABEL_FEMUR_RIGHT = 8
LABEL_TIBIA_LEFT = 85
LABEL_TIBIA_RIGHT = 86
LABEL_PATELLA_LEFT = 93
LABEL_PATELLA_RIGHT = 94

KNEE_LABELS = [
    LABEL_FEMUR_LEFT, LABEL_FEMUR_RIGHT,
    LABEL_TIBIA_LEFT, LABEL_TIBIA_RIGHT,
    LABEL_PATELLA_LEFT, LABEL_PATELLA_RIGHT,
]

LABEL_REMAP = {
    LABEL_FEMUR_LEFT: 1,
    LABEL_FEMUR_RIGHT: 1,
    LABEL_TIBIA_LEFT: 2,
    LABEL_TIBIA_RIGHT: 2,
    LABEL_PATELLA_LEFT: 3,
    LABEL_PATELLA_RIGHT: 3,
}

CROP_PADDING_MM = 30.0


# ---------------------------------------------------------------------------
# 1. Install TotalSegmentator
# ---------------------------------------------------------------------------

def ensure_totalsegmentator():
    """Install TotalSegmentator if not already installed."""
    try:
        import totalsegmentator
        logger.info("totalsegmentator already installed.")
        return True
    except ImportError:
        logger.info("Installing totalsegmentator...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "totalsegmentator"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info("totalsegmentator installed successfully.")
            return True
        else:
            logger.error(f"Installation failed:\n{result.stderr}")
            return False


# ---------------------------------------------------------------------------
# 2. Run TotalSegmentator inference on a CT volume
# ---------------------------------------------------------------------------

def run_totalsegmentator(
    input_path: str,
    output_dir: str,
    task: str = "total",
    multilabel: bool = True,
) -> str:
    """
    Run TotalSegmentator on a CT volume to generate bone segmentation masks.

    Args:
        input_path: Path to CT volume (.nii.gz or DICOM directory).
        output_dir: Directory to write segmentation output.
        task: TotalSegmentator task (default 'total' for all 104 structures).
        multilabel: If True, output a single multi-label NIfTI (easier to use).

    Returns:
        Path to the output segmentation file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "segmentation.nii.gz")

    cmd = [
        sys.executable, "-m", "totalsegmentator",
        "-i", input_path,
        "-o", output_path,
        "--task", task,
    ]
    if multilabel:
        cmd.append("--ml")

    logger.info(f"Running TotalSegmentator on {input_path}...")
    logger.info(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"TotalSegmentator failed:\n{result.stderr}")
        return None

    logger.info(f"TotalSegmentator output: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 3. Knee bounding box crop
# ---------------------------------------------------------------------------

def get_knee_bounding_box(label_volume: sitk.Image, padding_mm: float = CROP_PADDING_MM):
    """
    Compute a bounding box around the knee region from the femur/tibia label extent.

    Returns (lower, size) tuples in SimpleITK [X, Y, Z] order,
    or (None, None) if no knee labels are found.
    """
    label_arr = sitk.GetArrayFromImage(label_volume)  # [Z, Y, X]
    spacing = label_volume.GetSpacing()               # [X, Y, Z]

    knee_mask = np.zeros_like(label_arr, dtype=bool)
    for lbl in KNEE_LABELS:
        knee_mask |= (label_arr == lbl)

    if not knee_mask.any():
        logger.warning("No knee labels found in volume.")
        return None, None

    coords = np.argwhere(knee_mask)
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    pad_x = int(padding_mm / spacing[0])
    pad_y = int(padding_mm / spacing[1])
    pad_z = int(padding_mm / spacing[2])

    shape = label_arr.shape  # [Z, Y, X]
    z_min = max(0, z_min - pad_z)
    z_max = min(shape[0] - 1, z_max + pad_z)
    y_min = max(0, y_min - pad_y)
    y_max = min(shape[1] - 1, y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(shape[2] - 1, x_max + pad_x)

    lower = [int(x_min), int(y_min), int(z_min)]
    size = [int(x_max - x_min + 1), int(y_max - y_min + 1), int(z_max - z_min + 1)]
    size_mm = [round(s * sp, 1) for s, sp in zip(size, spacing)]
    logger.info(f"Knee bounding box: lower={lower}, size={size} voxels ({size_mm} mm)")
    return lower, size


def remap_labels(label_volume: sitk.Image):
    """
    Remap TotalSegmentator labels to pipeline standard: femur=1, tibia=2, patella=3.
    Returns (remapped_image, patella_found).
    """
    arr = sitk.GetArrayFromImage(label_volume)
    out = np.zeros_like(arr, dtype=np.uint8)
    patella_found = False

    for src, dst in LABEL_REMAP.items():
        mask = arr == src
        if mask.any():
            out[mask] = dst
            if dst == 3:
                patella_found = True

    remapped = sitk.GetImageFromArray(out)
    remapped.CopyInformation(label_volume)
    return remapped, patella_found


def crop_and_save(
    image: sitk.Image,
    label: sitk.Image,
    case_id: str,
    output_dir: str,
) -> dict:
    """Crop image + label to knee region, remap labels, and save."""
    os.makedirs(output_dir, exist_ok=True)

    lower, size = get_knee_bounding_box(label)
    if lower is None:
        logger.warning(f"{case_id}: No knee labels found — skipping crop.")
        return None

    cropped_image = sitk.RegionOfInterest(image, size=size, index=lower)

    remapped_label, patella_found = remap_labels(label)
    cropped_label = sitk.RegionOfInterest(remapped_label, size=size, index=lower)

    image_out = os.path.join(output_dir, f"{case_id}_ct_knee.nii.gz")
    label_out = os.path.join(output_dir, f"{case_id}_label_knee.nii.gz")
    sitk.WriteImage(cropped_image, image_out)
    sitk.WriteImage(cropped_label, label_out)

    logger.info(f"Saved: {image_out}")
    logger.info(f"Saved: {label_out}")

    if not patella_found:
        logger.warning(
            f"{case_id}: Patella NOT found in TotalSegmentator output. "
            "Manual tracing required for this case."
        )

    return {
        "case_id": case_id,
        "image_path": image_out,
        "label_path": label_out,
        "patella_found": patella_found,
    }


# ---------------------------------------------------------------------------
# 4. Main pipeline: segment + crop a single CT
# ---------------------------------------------------------------------------

def process_single_ct(
    input_path: str,
    output_dir: str,
    case_id: str = None,
) -> dict:
    """
    Full pipeline for one CT file:
      1. Run TotalSegmentator to get bone labels
      2. Crop to knee bounding box
      3. Save image + remapped label

    Args:
        input_path: Path to input CT (.nii.gz or DICOM dir).
        output_dir: Root output directory.
        case_id: Identifier string (defaults to filename stem).
    """
    if case_id is None:
        case_id = os.path.splitext(os.path.splitext(os.path.basename(input_path))[0])[0]

    seg_dir = os.path.join(output_dir, "segmentations", case_id)
    seg_path = run_totalsegmentator(input_path, seg_dir)
    if not seg_path:
        return None

    image = sitk.ReadImage(input_path) if not os.path.isdir(input_path) else _load_dicom(input_path)
    label = sitk.ReadImage(seg_path)

    result = crop_and_save(image, label, case_id, os.path.join(output_dir, "cropped"))
    return result


def _load_dicom(dicom_dir: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(names)
    return reader.Execute()


# ---------------------------------------------------------------------------
# 5. Batch processing
# ---------------------------------------------------------------------------

def process_batch(
    input_dir: str,
    output_dir: str,
    n_cases: int = 8,
) -> list:
    """
    Batch-process all .nii.gz files or DICOM subdirectories in input_dir.

    Args:
        input_dir: Directory containing CT volumes (.nii.gz) or DICOM subdirs.
        output_dir: Root output directory for processed cases.
        n_cases: Maximum number of cases to process.
    """
    # Find .nii.gz files
    nifti_files = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))[:n_cases]
    # Find DICOM directories
    dicom_dirs = sorted([
        d for d in glob.glob(os.path.join(input_dir, "*/"))
        if os.path.isdir(d)
    ])[:max(0, n_cases - len(nifti_files))]

    all_inputs = [(p, "nifti") for p in nifti_files] + [(d, "dicom") for d in dicom_dirs]

    if not all_inputs:
        logger.error(
            f"No CT files found in {input_dir}.\n"
            "Place knee CT volumes (.nii.gz) or DICOM directories there, "
            "then re-run this script."
        )
        return []

    logger.info(f"Found {len(all_inputs)} cases to process.")
    results = []
    for path, _ in all_inputs:
        result = process_single_ct(path, output_dir)
        if result:
            results.append(result)

    patella_missing = [r for r in results if not r["patella_found"]]
    logger.info(f"\nComplete: {len(results)} cases processed.")
    if patella_missing:
        logger.warning(
            f"{len(patella_missing)} case(s) need manual patella tracing: "
            f"{[r['case_id'] for r in patella_missing]}"
        )
    return results


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare CT data for knee bone segmentation (Track B).\n"
            "Runs TotalSegmentator on provided CT volumes to generate bone masks,\n"
            "then crops each volume to the knee region."
        )
    )
    parser.add_argument(
        "--input_dir", type=str, default="data/ct_dicom",
        help="Directory containing CT volumes (.nii.gz) or DICOM subdirectories"
    )
    parser.add_argument(
        "--single", type=str, default=None,
        help="Process a single CT file instead of a batch"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/ct_knee",
        help="Output directory for cropped knee cases"
    )
    parser.add_argument(
        "--n_cases", type=int, default=8,
        help="Maximum number of cases to process"
    )
    parser.add_argument(
        "--skip_install", action="store_true",
        help="Skip totalsegmentator installation check"
    )
    args = parser.parse_args()

    if not args.skip_install:
        if not ensure_totalsegmentator():
            logger.error("Cannot proceed without totalsegmentator. Exiting.")
            sys.exit(1)

    if args.single:
        result = process_single_ct(args.single, args.output_dir)
        print(f"\nResult: {result}")
    else:
        results = process_batch(args.input_dir, args.output_dir, n_cases=args.n_cases)
        print(f"\nProcessed {len(results)} cases to {args.output_dir}")


if __name__ == "__main__":
    main()
