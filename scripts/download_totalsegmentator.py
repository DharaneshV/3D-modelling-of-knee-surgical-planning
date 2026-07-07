"""
Prepare CT data for the bone segmentation track (Track B).

Features:
  1. Automated Data Acquisition: Uses idc-index to query and download public
     knee-relevant CT scans from the NCI Imaging Data Commons (IDC) (such as
     extremity CTs from the soft_tissue_sarcoma collection). Downloads are
     accelerated using s5cmd under the hood.
  2. TotalSegmentator Inference: Runs TotalSegmentator on the downloaded DICOM
     directories to automatically generate femur, tibia, and patella masks.
  3. Knee-Region Cropping: Computes a bounding box around the knee joint based on
     the label extents, applies 30mm padding, and crops both the CT volume and the
     segmentation mask to the knee region.
  4. Label Remapping: Remaps TotalSegmentator bone outputs to pipeline-standard
     labels (femur=1, tibia=2, patella=3).

Usage:
    # Run the full automated pipeline (download 8 cases -> segment -> crop):
    python scripts/download_totalsegmentator.py

    # Process a single local file/directory instead:
    python scripts/download_totalsegmentator.py --single path/to/ct.nii.gz
"""

import os
import argparse
import logging
import subprocess
import sys
import glob
import numpy as np
import pandas as pd
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
# 1. Install & Verify TotalSegmentator
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
# 2. Automated Download from IDC (NCI Imaging Data Commons)
# ---------------------------------------------------------------------------

def download_idc_samples(n_cases: int = 8, download_dir: str = "data/ct_dicom") -> list:
    """
    Query and download knee-relevant CT scans from IDC.

    Queries the local IDC index for CT scans of the lower extremity/extremity/knee
    that are 3D volumes (high instance count). Downloads the DICOM files using s5cmd.

    Args:
        n_cases: Number of cases to download.
        download_dir: Local directory to save downloaded DICOM files.

    Returns:
        List of local DICOM folder paths.
    """
    try:
        from idc_index import IDCClient
    except ImportError:
        logger.error("idc-index is not installed. Run `pip install idc-index`.")
        return []

    os.makedirs(download_dir, exist_ok=True)
    logger.info("Querying local IDC metadata index for matching CT scans...")

    client = IDCClient.client()
    query = """
    SELECT 
        collection_id, 
        PatientID, 
        StudyInstanceUID, 
        SeriesInstanceUID, 
        instanceCount, 
        BodyPartExamined,
        SeriesDescription
    FROM index 
    WHERE Modality = 'CT' 
      AND (SeriesDescription LIKE '%KNEE%' 
           OR SeriesDescription LIKE '%LEG%' 
           OR SeriesDescription LIKE '%THIGH%' 
           OR SeriesDescription LIKE '%FEMUR%' 
           OR SeriesDescription LIKE '%TIBIA%'
           OR BodyPartExamined LIKE '%KNEE%')
      AND instanceCount > 80
    ORDER BY instanceCount ASC
    """
    try:
        results = client.sql_query(query)
        if not isinstance(results, pd.DataFrame):
            df = pd.DataFrame(results)
        else:
            df = results

        if df.empty:
            logger.error("No matching CT scans found in the IDC index.")
            return []

        # Deduplicate by PatientID to ensure we get unique patients
        df_unique = df.drop_duplicates(subset=["PatientID"]).head(n_cases)
        series_uids = df_unique["SeriesInstanceUID"].tolist()
        patient_ids = df_unique["PatientID"].tolist()

        logger.info(f"Selected {len(series_uids)} unique Patient cases from IDC:")
        for idx, row in df_unique.iterrows():
            logger.info(f"  Patient: {row['PatientID']} | Collection: {row['collection_id']} | Desc: {row['SeriesDescription']} | CT ({row['instanceCount']} images)")

        # Download using the client
        logger.info(f"Downloading {len(series_uids)} series via idc-index to {download_dir}...")
        
        # We specify a directory template to organize files by PatientID
        client.download_from_selection(
            seriesInstanceUID=series_uids,
            downloadDir=download_dir,
            dirTemplate="%PatientID"
        )
        
        # Verify and return paths to the patient directories
        downloaded_dirs = []
        for pid in patient_ids:
            p_dir = os.path.join(download_dir, pid)
            if os.path.exists(p_dir) and os.listdir(p_dir):
                downloaded_dirs.append(p_dir)
                
        logger.info(f"Successfully downloaded {len(downloaded_dirs)} cases.")
        return downloaded_dirs

    except Exception as e:
        logger.error(f"Error querying/downloading from IDC: {e}")
        return []


# ---------------------------------------------------------------------------
# 3. Run TotalSegmentator
# ---------------------------------------------------------------------------

def run_totalsegmentator(
    input_path: str,
    output_dir: str,
    task: str = "total",
    multilabel: bool = True,
) -> str:
    """Run TotalSegmentator on a CT volume to generate bone segmentation masks."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "segmentation.nii.gz")

    # Locate the TotalSegmentator executable in the virtual environment's bin/Scripts folder
    bin_dir = os.path.dirname(sys.executable)
    exe_names = ["TotalSegmentator.exe", "TotalSegmentator", "totalsegmentator.exe", "totalsegmentator"]
    totalseg_bin = "TotalSegmentator"  # default fallback
    for name in exe_names:
        candidate = os.path.join(bin_dir, name)
        if os.path.exists(candidate):
            totalseg_bin = candidate
            break

    # We must squeeze the input image if it's 4D to prevent nnUNet/nibabel crashes
    squeezed_input_path = input_path
    if not os.path.isdir(input_path):
        try:
            img = sitk.ReadImage(input_path)
            squeezed_img = check_and_squeeze_3d(img)
            if squeezed_img.GetDimension() < img.GetDimension():
                # Write back squeezed image to a temp file
                squeezed_input_path = os.path.join(os.path.dirname(output_path), f"squeezed_{os.path.basename(input_path)}")
                sitk.WriteImage(squeezed_img, squeezed_input_path)
                logger.info(f"Wrote squeezed 3D image to {squeezed_input_path}")
        except Exception as e:
            logger.error(f"Error checking image dimensions: {e}")

    if not totalseg_bin.lower().endswith(".exe") and os.name == "nt":
        cmd_prefix = [sys.executable, totalseg_bin]
    else:
        cmd_prefix = [totalseg_bin]

    cmd = cmd_prefix + [
        "-i", squeezed_input_path,
        "-o", output_path,
        "--task", task,
        "--fast",
    ]
    if multilabel:
        cmd.append("--ml")

    logger.info(f"Running TotalSegmentator on {squeezed_input_path}...")
    logger.info(f"Command: {' '.join(cmd)}")

    # Set environment variable to bypass PyTorch 2.6 default weights_only unpickling error
    env = os.environ.copy()
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    # Clean up squeezed temp file if created
    if squeezed_input_path != input_path and os.path.exists(squeezed_input_path):
        os.remove(squeezed_input_path)

    if result.returncode != 0:
        logger.error(f"TotalSegmentator failed:\n{result.stderr}")
        return None

    logger.info(f"TotalSegmentator output: {output_path}")
    return output_path


def check_and_squeeze_3d(image: sitk.Image) -> sitk.Image:
    """If image has 4 dimensions (e.g. 3D+time), collapse the 4th dimension to get 3D."""
    if image.GetDimension() == 4:
        size = list(image.GetSize())
        if size[3] == 1:
            size[3] = 0
            logger.info("Extracting 3D volume from 4D image.")
            image = sitk.Extract(image, size, [0, 0, 0, 0])
    return image


# ---------------------------------------------------------------------------
# 4. Knee-region cropping and remapping
# ---------------------------------------------------------------------------

def get_knee_bounding_box(label_volume: sitk.Image, padding_mm: float = CROP_PADDING_MM):
    """
    Compute a bounding box around the knee region from the femur/tibia label extent.
    Returns (lower, size) tuples in SimpleITK [X, Y, Z] order.
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
# 5. Main pipelines
# ---------------------------------------------------------------------------

def process_single_ct(
    input_path: str,
    output_dir: str,
    case_id: str = None,
) -> dict:
    """Full pipeline for one CT file: Segment -> Crop -> Remap."""
    # Strip trailing slashes to prevent empty case_id when directories have trailing slashes
    input_path = input_path.rstrip("/\\")
    if case_id is None:
        case_id = os.path.splitext(os.path.splitext(os.path.basename(input_path))[0])[0]

    # Convert DICOM directory to temporary NIfTI if it is a directory.
    # SimpleITK is extremely robust and avoids dicom2nifti spacing crashes.
    is_temp_nifti = False
    if os.path.isdir(input_path):
        logger.info(f"Converting DICOM directory {input_path} to temporary NIfTI via SimpleITK...")
        temp_nifti_path = os.path.join(output_dir, f"temp_{case_id}.nii.gz")
        os.makedirs(output_dir, exist_ok=True)
        try:
            image = _load_dicom(input_path)
            sitk.WriteImage(image, temp_nifti_path)
            totalseg_input = temp_nifti_path
            is_temp_nifti = True
            logger.info(f"Temporary NIfTI written to {temp_nifti_path}")
        except Exception as e:
            logger.error(f"Failed to convert DICOM to temporary NIfTI: {e}")
            return None
    else:
        totalseg_input = input_path

    seg_dir = os.path.join(output_dir, "segmentations", case_id)
    seg_path = run_totalsegmentator(totalseg_input, seg_dir)
    
    # Clean up temporary NIfTI if we created it
    if is_temp_nifti and os.path.exists(totalseg_input):
        os.remove(totalseg_input)
        logger.info(f"Cleaned up temporary NIfTI: {totalseg_input}")

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


def process_batch(
    input_dir: str,
    output_dir: str,
    n_cases: int = 8,
) -> list:
    """Batch-process all .nii.gz files or DICOM subdirectories in input_dir."""
    nifti_files = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))[:n_cases]
    dicom_dirs = sorted([
        d for d in glob.glob(os.path.join(input_dir, "*/"))
        if os.path.isdir(d)
    ])[:max(0, n_cases - len(nifti_files))]

    all_inputs = nifti_files + dicom_dirs

    if not all_inputs:
        logger.error(f"No CT files found in {input_dir}.")
        return []

    logger.info(f"Found {len(all_inputs)} cases to process.")
    results = []
    for path in all_inputs:
        result = process_single_ct(path, output_dir)
        if result:
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare CT data for knee bone segmentation (Track B).\n"
            "Automatically downloads public knee CTs from IDC (via idc-index),\n"
            "runs TotalSegmentator, crops to knee joint, and remaps labels."
        )
    )
    parser.add_argument(
        "--input_dir", type=str, default="data/ct_dicom",
        help="Directory containing CT volumes or DICOM directories (download target)"
    )
    parser.add_argument(
        "--single", type=str, default=None,
        help="Process a single local CT file instead of downloading/batching"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/ct_knee",
        help="Output directory for cropped knee cases"
    )
    parser.add_argument(
        "--n_cases", type=int, default=8,
        help="Number of cases to query/process (default: 8)"
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
        # Check if local input dir already has cases
        local_cases = glob.glob(os.path.join(args.input_dir, "*.nii.gz")) + [
            d for d in glob.glob(os.path.join(args.input_dir, "*/")) if os.path.isdir(d)
        ]
        
        # If no local cases, automatically download from IDC first
        if not local_cases:
            logger.info("No local CT files found in input_dir. Starting automated IDC download...")
            local_cases = download_idc_samples(n_cases=args.n_cases, download_dir=args.input_dir)
            
        results = process_batch(args.input_dir, args.output_dir, n_cases=args.n_cases)
        print(f"\nProcessed {len(results)} cases to {args.output_dir}")


if __name__ == "__main__":
    main()
