"""
Download and crop TotalSegmentator CT cases for the bone segmentation track.

TotalSegmentator (CC-BY licensed) ships whole-body CT volumes with pre-existing
bone segmentation masks. We pull 5-8 cases, identify the knee region from the
femur/tibia label extent, and crop both image and mask to that ROI.

Usage:
    python scripts/download_totalsegmentator.py
    python scripts/download_totalsegmentator.py --n_cases 5 --output_dir data/ct_knee
"""

import os
import argparse
import logging
import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# TotalSegmentator label values for knee-relevant bones
# (These are the standard TotalSegmentator v2 label indices)
LABEL_FEMUR_LEFT = 7
LABEL_FEMUR_RIGHT = 8
LABEL_TIBIA_LEFT = 85
LABEL_TIBIA_RIGHT = 86
LABEL_PATELLA_LEFT = 93   # Added in later TotalSegmentator versions — may be absent
LABEL_PATELLA_RIGHT = 94

KNEE_LABELS = [
    LABEL_FEMUR_LEFT, LABEL_FEMUR_RIGHT,
    LABEL_TIBIA_LEFT, LABEL_TIBIA_RIGHT,
    LABEL_PATELLA_LEFT, LABEL_PATELLA_RIGHT,
]

# Remap to standard pipeline labels: femur=1, tibia=2, patella=3
LABEL_REMAP = {
    LABEL_FEMUR_LEFT: 1,
    LABEL_FEMUR_RIGHT: 1,
    LABEL_TIBIA_LEFT: 2,
    LABEL_TIBIA_RIGHT: 2,
    LABEL_PATELLA_LEFT: 3,
    LABEL_PATELLA_RIGHT: 3,
}

# Padding around the knee bounding box in mm
CROP_PADDING_MM = 30.0


def download_totalsegmentator_dataset(n_cases: int = 8, output_dir: str = "data/totalseg_raw"):
    """
    Download whole-body CT cases from the TotalSegmentator dataset via HuggingFace.

    Dataset: Wasserthal et al., 'TotalSegmentator: Robust segmentation of 104 anatomic
    structures in CT images.' Radiology: AI, 2023. CC-BY 4.0 license.
    HF repo: https://huggingface.co/datasets/wasserth/TotalSegmentator_dataset
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface-hub not installed. Run: pip install huggingface-hub")
        return

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Downloading TotalSegmentator dataset to {output_dir} ({n_cases} cases)...")

    try:
        snapshot_download(
            repo_id="wasserth/TotalSegmentator_dataset",
            repo_type="dataset",
            local_dir=output_dir,
            max_workers=4,
        )
        logger.info("TotalSegmentator download complete.")
    except Exception as e:
        logger.error(f"HuggingFace download failed: {e}")
        logger.info("Trying fallback: direct Zenodo download via requests...")
        _download_zenodo_fallback(n_cases, output_dir)


def _download_zenodo_fallback(n_cases: int, output_dir: str):
    """
    Fallback: download individual TotalSegmentator cases from Zenodo.
    TotalSegmentator v2 dataset is available at Zenodo DOI: 10.5281/zenodo.6802613
    """
    try:
        import requests
    except ImportError:
        logger.error("requests not installed.")
        return

    # Zenodo API to list files for this record
    zenodo_record_id = "6802613"
    api_url = f"https://zenodo.org/api/records/{zenodo_record_id}"
    logger.info(f"Querying Zenodo record {zenodo_record_id}...")

    try:
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        record = resp.json()
        files = record.get("files", [])

        # Filter to .zip or .tar files that look like individual cases
        case_files = [f for f in files if f["key"].endswith(".zip")][:n_cases]

        for f in case_files:
            url = f["links"]["self"]
            fname = f["key"]
            dest = os.path.join(output_dir, fname)
            if os.path.exists(dest):
                logger.info(f"Already exists: {fname}")
                continue
            logger.info(f"Downloading {fname} from Zenodo...")
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as fout:
                    for chunk in r.iter_content(chunk_size=8192):
                        fout.write(chunk)
            logger.info(f"Downloaded {fname}")
    except Exception as e:
        logger.error(f"Zenodo fallback failed: {e}")
        logger.info(
            "Manual download option: Visit https://zenodo.org/record/6802613 "
            "and download 5-8 case archives into data/totalseg_raw/"
        )


def get_knee_bounding_box(
    label_volume: sitk.Image,
    padding_mm: float = CROP_PADDING_MM,
) -> tuple:
    """
    Compute a bounding box around the knee region using the femur/tibia label extent.

    Args:
        label_volume: TotalSegmentator multi-label mask (sitk.Image).
        padding_mm: Padding around the bounding box in mm (default 30 mm).

    Returns:
        Tuple of (lower_index, upper_index) as lists — ready for sitk.RegionOfInterest.
        Returns (None, None) if no knee labels are found in the volume.
    """
    label_arr = sitk.GetArrayFromImage(label_volume)  # [Z, Y, X]
    spacing = label_volume.GetSpacing()  # [X, Y, Z] spacing

    # Mask for knee-relevant labels
    knee_mask = np.zeros_like(label_arr, dtype=bool)
    for lbl in KNEE_LABELS:
        knee_mask |= (label_arr == lbl)

    if not knee_mask.any():
        logger.warning("No knee labels found in volume (femur/tibia/patella missing).")
        return None, None

    # Get bounding box in voxel indices (Z, Y, X)
    coords = np.argwhere(knee_mask)
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    # Convert padding from mm to voxels
    pad_x = int(padding_mm / spacing[0])
    pad_y = int(padding_mm / spacing[1])
    pad_z = int(padding_mm / spacing[2])

    # Apply padding, clamp to volume bounds
    shape = label_arr.shape  # [Z, Y, X]
    z_min = max(0, z_min - pad_z)
    z_max = min(shape[0] - 1, z_max + pad_z)
    y_min = max(0, y_min - pad_y)
    y_max = min(shape[1] - 1, y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(shape[2] - 1, x_max + pad_x)

    # SimpleITK uses [X, Y, Z] order
    lower = [int(x_min), int(y_min), int(z_min)]
    size = [int(x_max - x_min + 1), int(y_max - y_min + 1), int(z_max - z_min + 1)]

    logger.info(
        f"Knee bounding box: lower={lower}, size={size} voxels "
        f"({[s * sp:.1f for s, sp in zip(size, spacing)]} mm)"
    )
    return lower, size


def remap_labels(label_volume: sitk.Image) -> tuple:
    """
    Remap TotalSegmentator label indices to pipeline-standard labels:
      femur=1, tibia=2, patella=3

    Also returns a boolean indicating whether patella labels were found
    (so the caller can warn about the manual-tracing requirement).

    Args:
        label_volume: TotalSegmentator multi-label mask.

    Returns:
        (remapped_mask: sitk.Image, patella_found: bool)
    """
    arr = sitk.GetArrayFromImage(label_volume)
    out = np.zeros_like(arr, dtype=np.uint8)

    patella_found = False
    for src_label, dst_label in LABEL_REMAP.items():
        voxels = (arr == src_label)
        if voxels.any():
            out[voxels] = dst_label
            if dst_label == 3:
                patella_found = True

    remapped = sitk.GetImageFromArray(out)
    remapped.CopyInformation(label_volume)
    return remapped, patella_found


def crop_to_knee(
    image: sitk.Image,
    label: sitk.Image,
    case_id: str,
    output_dir: str,
    padding_mm: float = CROP_PADDING_MM,
):
    """
    Crop a whole-body CT image and label to the knee region.

    Args:
        image: Whole-body CT volume.
        label: TotalSegmentator whole-body label mask.
        case_id: Identifier for logging/saving.
        output_dir: Directory to save cropped outputs.
        padding_mm: Padding around knee bounding box in mm.

    Returns:
        dict with image_path, label_path, patella_found.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get bounding box
    lower, size = get_knee_bounding_box(label, padding_mm=padding_mm)
    if lower is None:
        logger.warning(f"{case_id}: Skipping — no knee labels found.")
        return None

    # Crop image
    cropped_image = sitk.RegionOfInterest(image, size=size, index=lower)

    # Remap and crop label
    remapped_label, patella_found = remap_labels(label)
    cropped_label = sitk.RegionOfInterest(remapped_label, size=size, index=lower)

    # Save
    image_out = os.path.join(output_dir, f"{case_id}_ct_knee.nii.gz")
    label_out = os.path.join(output_dir, f"{case_id}_label_knee.nii.gz")
    sitk.WriteImage(cropped_image, image_out)
    sitk.WriteImage(cropped_label, label_out)
    logger.info(f"Saved cropped CT:    {image_out}")
    logger.info(f"Saved cropped label: {label_out}")

    if not patella_found:
        logger.warning(
            f"{case_id}: Patella labels NOT found in TotalSegmentator mask. "
            f"Manual tracing required for patella ground truth on this case."
        )

    return {
        "case_id": case_id,
        "image_path": image_out,
        "label_path": label_out,
        "patella_found": patella_found,
    }


def process_totalsegmentator_cases(
    raw_dir: str = "data/totalseg_raw",
    output_dir: str = "data/ct_knee",
    n_cases: int = 8,
):
    """
    Process TotalSegmentator cases: find CT + label pairs, crop to knee, save.

    Expects raw_dir to have per-case subdirectories, each containing:
        ct.nii.gz           -- the CT volume
        segmentations/      -- directory with per-label .nii.gz files, OR
        segmentation.nii.gz -- a single multi-label mask

    Args:
        raw_dir: Directory containing raw TotalSegmentator downloads.
        output_dir: Output directory for cropped knee cases.
        n_cases: Maximum number of cases to process.
    """
    import glob

    results = []
    case_dirs = sorted([
        d for d in glob.glob(os.path.join(raw_dir, "*/"))
        if os.path.isdir(d)
    ])[:n_cases]

    if not case_dirs:
        logger.error(
            f"No case directories found in {raw_dir}. "
            "Please download TotalSegmentator cases first."
        )
        return results

    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir.rstrip("/\\"))

        # Locate CT image
        ct_path = os.path.join(case_dir, "ct.nii.gz")
        if not os.path.exists(ct_path):
            logger.warning(f"{case_id}: ct.nii.gz not found, skipping.")
            continue

        # Locate segmentation mask (multi-label)
        seg_path = os.path.join(case_dir, "segmentation.nii.gz")
        if not os.path.exists(seg_path):
            # Some releases store it under a different name
            seg_candidates = glob.glob(os.path.join(case_dir, "*.nii.gz"))
            seg_candidates = [p for p in seg_candidates if "ct" not in os.path.basename(p)]
            seg_path = seg_candidates[0] if seg_candidates else None

        if not seg_path:
            logger.warning(f"{case_id}: No segmentation mask found, skipping.")
            continue

        # Load
        logger.info(f"Processing {case_id}...")
        image = sitk.ReadImage(ct_path)
        label = sitk.ReadImage(seg_path)

        result = crop_to_knee(image, label, case_id, output_dir)
        if result:
            results.append(result)

    # Summary
    found = [r for r in results if r["patella_found"]]
    missing = [r for r in results if not r["patella_found"]]
    logger.info(f"\nProcessed {len(results)} cases.")
    logger.info(f"  Patella found: {len(found)}")
    logger.info(f"  Patella missing (manual trace needed): {len(missing)}")
    if missing:
        logger.warning(f"  Cases needing patella tracing: {[r['case_id'] for r in missing]}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download and crop TotalSegmentator CT cases to knee region"
    )
    parser.add_argument("--n_cases", type=int, default=8, help="Number of cases to process")
    parser.add_argument("--raw_dir", type=str, default="data/totalseg_raw",
                        help="Directory for raw TotalSegmentator downloads")
    parser.add_argument("--output_dir", type=str, default="data/ct_knee",
                        help="Directory for cropped knee CT outputs")
    parser.add_argument("--download_only", action="store_true",
                        help="Only download, don't crop")
    parser.add_argument("--crop_only", action="store_true",
                        help="Skip download, only crop existing raw data")

    args = parser.parse_args()

    if not args.crop_only:
        download_totalsegmentator_dataset(n_cases=args.n_cases, output_dir=args.raw_dir)

    if not args.download_only:
        results = process_totalsegmentator_cases(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            n_cases=args.n_cases,
        )
        print(f"\nDone. {len(results)} cropped knee cases saved to {args.output_dir}")


if __name__ == "__main__":
    main()
