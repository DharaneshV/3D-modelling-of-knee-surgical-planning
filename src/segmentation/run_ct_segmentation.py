"""
run_ct_segmentation.py — Production CT bone segmentation using TotalSegmentator

Uses TotalSegmentator models for all three bones:
  - task='total'              (Apache 2.0) → femur_left, femur_right
  - task='appendicular_bones' (licensed)  → tibia, patella (unlateralized)

Handles bilateral (500mm FOV) scans natively. Uses femur centroids to assign 
tibia and patella fragments to left/right sides. Output is a 6-label mask:
1=Femur_L, 2=Femur_R, 3=Tibia_L, 4=Tibia_R, 5=Patella_L, 6=Patella_R

Usage:
    python src/segmentation/run_ct_segmentation.py \
        --input  data/raw/case01_STS_006.nii.gz \
        --output outputs/STS_006/masks/bone_mask.nii.gz
"""

import os
import sys
import shutil
import logging
import argparse
import subprocess
import tempfile
import json
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, center_of_mass
from pathlib import Path

# Suppress noisy nnU-Net / torch warnings
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 6-Label Convention
LABEL_FEMUR_L   = 1
LABEL_FEMUR_R   = 2
LABEL_TIBIA_L   = 3
LABEL_TIBIA_R   = 4
LABEL_PATELLA_L = 5
LABEL_PATELLA_R = 6

# TotalSegmentator ROI names for each task
TOTAL_FEMUR_ROIS  = ["femur_left", "femur_right"]
APPENDICULAR_ROIS = ["tibia", "patella"]

# Minimum voxel thresholds to ignore noise fragments during L/R assignment
MIN_VOXELS_TIBIA = 500
MIN_VOXELS_PATELLA = 200

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_isotropic(image: sitk.Image, spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkBSpline) -> sitk.Image:
    orig_spacing = image.GetSpacing()
    orig_size    = image.GetSize()
    
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(orig_size, orig_spacing, spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    # Use -1024 (air) for CT images, 0 for masks
    default_val = -1024.0 if interpolator != sitk.sitkNearestNeighbor else 0.0
    resampler.SetDefaultPixelValue(default_val)
    resampler.SetInterpolator(interpolator)
    
    return resampler.Execute(image)


# ─────────────────────────────────────────────────────────────────────────────
# TotalSegmentator inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_totalsegmentator(nifti_path: str, out_dir: str, task: str,
                          roi_subset: list | None = None):
    ts_exe = Path(sys.executable).parent / "TotalSegmentator.exe"
    cmd = [
        str(ts_exe),
        "-i", nifti_path,
        "-o", out_dir,
        "-ta", task,
        "-nr", "1",
        "-ns", "1",
        "-d", "gpu",
        "-q",
    ]
    if roi_subset:
        cmd += ["-rs"] + roi_subset

    logger.info(f"  Running TotalSegmentator task='{task}'  roi_subset={roi_subset}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"TotalSegmentator CLI failed (task={task}):\n"
            f"STDOUT: {result.stdout[-800:]}\nSTDERR: {result.stderr[-800:]}"
        )
    logger.info(f"  task='{task}' complete.")


def _load_roi_as_binary(out_dir: str, roi_name: str) -> np.ndarray | None:
    roi_path = os.path.join(out_dir, f"{roi_name}.nii.gz")
    if not os.path.exists(roi_path):
        return None
    img = sitk.ReadImage(roi_path)
    return sitk.GetArrayFromImage(img).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Anatomical Component Filtering
# ─────────────────────────────────────────────────────────────────────────────

def keep_largest_component(binary_arr: np.ndarray | None) -> np.ndarray | None:
    """
    Isolates the single largest connected component in a binary mask.
    ASSUMPTION: True bone anatomy in this dataset (soft tissue sarcoma CT) is always a 
    single connected blob. This assumes no severe post-surgical fragmentation or shattered 
    fractures where a single bone is physically split into multiple true pieces.
    """
    if binary_arr is None or np.sum(binary_arr) == 0:
        return binary_arr
        
    labeled_arr, num_features = label(binary_arr > 0)
    if num_features <= 1:
        return binary_arr
        
    largest_cc = 0
    max_voxels = 0
    for i in range(1, num_features + 1):
        voxel_count = np.sum(labeled_arr == i)
        if voxel_count > max_voxels:
            max_voxels = voxel_count
            largest_cc = i
            
    filtered_arr = np.zeros_like(binary_arr)
    filtered_arr[labeled_arr == largest_cc] = 1
    return filtered_arr


# ─────────────────────────────────────────────────────────────────────────────
# Anatomical Side Assignment (Tibia / Patella)
# ─────────────────────────────────────────────────────────────────────────────

def get_centroid(binary_arr: np.ndarray) -> tuple | None:
    if np.sum(binary_arr) == 0:
        return None
    return center_of_mass(binary_arr)

def assign_sides(binary_mask: np.ndarray, femur_l_centroid: tuple | None, 
                 femur_r_centroid: tuple | None, min_voxels: int, 
                 bone_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits a single unlateralized mask into Left and Right based on proximity 
    of each connected component to the given femur centroids.
    Returns (left_mask, right_mask).
    """
    left_mask = np.zeros_like(binary_mask)
    right_mask = np.zeros_like(binary_mask)
    
    if binary_mask is None or np.sum(binary_mask) == 0:
        return left_mask, right_mask
        
    labeled_arr, num_features = label(binary_mask > 0)
    
    dropped_noise = 0
    for i in range(1, num_features + 1):
        comp_mask = (labeled_arr == i)
        voxel_count = np.sum(comp_mask)
        
        if voxel_count < min_voxels:
            dropped_noise += 1
            continue
            
        comp_centroid = center_of_mass(comp_mask)
        
        # Calculate distances
        dist_l = float('inf')
        dist_r = float('inf')
        
        if femur_l_centroid:
            dist_l = np.linalg.norm(np.array(comp_centroid) - np.array(femur_l_centroid))
        if femur_r_centroid:
            dist_r = np.linalg.norm(np.array(comp_centroid) - np.array(femur_r_centroid))
            
        logger.info(f"    {bone_name} component {i} ({voxel_count} voxels): dist_L={dist_l:.1f}, dist_R={dist_r:.1f}")
        
        if dist_l == float('inf') and dist_r == float('inf'):
            logger.warning(f"    No femurs found, dropping {bone_name} component {i}.")
            continue
            
        if dist_l <= dist_r:
            logger.info(f"    -> Assigned to LEFT")
            left_mask[comp_mask] = 1
        else:
            logger.info(f"    -> Assigned to RIGHT")
            right_mask[comp_mask] = 1
            
    if dropped_noise > 0:
        logger.info(f"    Dropped {dropped_noise} noise fragments (< {min_voxels} voxels) for {bone_name}.")
        
    return left_mask, right_mask


# ─────────────────────────────────────────────────────────────────────────────
# Anatomical validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_combined_mask(mask_arr: np.ndarray, spacing: tuple) -> dict:
    """
    Confirms the combined mask has at least a complete triplet (femur, tibia, patella)
    on at least one side. Returns a laterality summary dict.
    """
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    min_voxels = int(3000 / voxel_vol)  # 3000 mm³ threshold
    
    counts = {}
    labels_map = {
        LABEL_FEMUR_L: "femur_left", LABEL_FEMUR_R: "femur_right",
        LABEL_TIBIA_L: "tibia_left", LABEL_TIBIA_R: "tibia_right",
        LABEL_PATELLA_L: "patella_left", LABEL_PATELLA_R: "patella_right"
    }
    
    for label_val, name in labels_map.items():
        count = int(np.sum(mask_arr == label_val))
        if count >= min_voxels:
            counts[name] = count
        else:
            if count > 0:
                logger.warning(f"  {name}: only {count} voxels (< {min_voxels} threshold) — treating as absent.")
            counts[name] = 0

    has_left = counts["femur_left"] > 0 and counts["tibia_left"] > 0 and counts["patella_left"] > 0
    has_right = counts["femur_right"] > 0 and counts["tibia_right"] > 0 and counts["patella_right"] > 0

    if not has_left and not has_right:
        raise ValueError(
            "Anatomical validation failed: neither side has a complete femur+tibia+patella triplet."
        )

    sides_present = []
    if has_left: sides_present.append("left")
    if has_right: sides_present.append("right")
    
    laterality = "bilateral" if len(sides_present) == 2 else sides_present[0]
    logger.info(f"  Validation passed: laterality={laterality}")

    return {
        "laterality": laterality,
        "sides_present": sides_present,
        "voxel_counts": counts
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main segmentation entry point
# ─────────────────────────────────────────────────────────────────────────────

def segment(input_path: str, output_path: str,
            intermediate_dir: str | None = None, keep_intermediate: bool = False):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load + resample
    logger.info(f"Loading: {input_path}")
    raw_ct = sitk.ReadImage(input_path, sitk.sitkFloat32)
    logger.info(f"  Original spacing={raw_ct.GetSpacing()}, size={raw_ct.GetSize()}")
    ct = resample_to_isotropic(raw_ct)
    logger.info(f"  Resampled: spacing={ct.GetSpacing()}, size={ct.GetSize()}")

    # ── 2 & 3. Run TotalSegmentator — use a temp dir if none specified
    cleanup_intermediate = False
    if intermediate_dir is None:
        intermediate_dir = tempfile.mkdtemp(prefix="totalseg_")
        cleanup_intermediate = not keep_intermediate

    total_dir        = os.path.join(intermediate_dir, "total")
    appendicular_dir = os.path.join(intermediate_dir, "appendicular")
    os.makedirs(total_dir, exist_ok=True)
    os.makedirs(appendicular_dir, exist_ok=True)

    ct_nifti = os.path.join(intermediate_dir, "ct_full.nii.gz")
    sitk.WriteImage(ct, ct_nifti)

    try:
        # task='total' for femur (Apache 2.0, no license concern)
        _run_totalsegmentator(ct_nifti, total_dir, task="total", roi_subset=TOTAL_FEMUR_ROIS)

        # task='appendicular_bones' for patella and tibia
        _run_totalsegmentator(ct_nifti, appendicular_dir, task="appendicular_bones")
    except Exception as e:
        if cleanup_intermediate:
            shutil.rmtree(intermediate_dir, ignore_errors=True)
        raise e

    logger.info("Combining TotalSegmentator labels and assigning sides...")
    
    # Load Femurs
    femur_l_arr = _load_roi_as_binary(total_dir, "femur_left")
    femur_r_arr = _load_roi_as_binary(total_dir, "femur_right")
    
    femur_l_centroid = get_centroid(femur_l_arr) if femur_l_arr is not None else None
    femur_r_centroid = get_centroid(femur_r_arr) if femur_r_arr is not None else None
    
    # Load and split Tibia
    tibia_raw = _load_roi_as_binary(appendicular_dir, "tibia")
    logger.info("Splitting Tibia...")
    tibia_l_arr, tibia_r_arr = assign_sides(
        tibia_raw, femur_l_centroid, femur_r_centroid, MIN_VOXELS_TIBIA, "Tibia"
    )

    # Load and split Patella
    patella_raw = _load_roi_as_binary(appendicular_dir, "patella")
    logger.info("Splitting Patella...")
    patella_l_arr, patella_r_arr = assign_sides(
        patella_raw, femur_l_centroid, femur_r_centroid, MIN_VOXELS_PATELLA, "Patella"
    )

    # Apply single-largest-component filter to drop disconnected noise (e.g. fibula leaks)
    femur_l_arr = keep_largest_component(femur_l_arr)
    femur_r_arr = keep_largest_component(femur_r_arr)
    tibia_l_arr = keep_largest_component(tibia_l_arr)
    tibia_r_arr = keep_largest_component(tibia_r_arr)
    patella_l_arr = keep_largest_component(patella_l_arr)
    patella_r_arr = keep_largest_component(patella_r_arr)

    # Merge into 6-label mask
    combined = np.zeros(sitk.GetArrayFromImage(ct).shape, dtype=np.uint8)
    
    if femur_l_arr is not None: combined[femur_l_arr > 0] = LABEL_FEMUR_L
    if femur_r_arr is not None: combined[femur_r_arr > 0] = LABEL_FEMUR_R
    if tibia_l_arr is not None: combined[tibia_l_arr > 0] = LABEL_TIBIA_L
    if tibia_r_arr is not None: combined[tibia_r_arr > 0] = LABEL_TIBIA_R
    
    # Patella written last — patella wins at joint boundary
    if patella_l_arr is not None: combined[patella_l_arr > 0] = LABEL_PATELLA_L
    if patella_r_arr is not None: combined[patella_r_arr > 0] = LABEL_PATELLA_R

    # ── 4. Anatomical validation & summary
    summary_dict = validate_combined_mask(combined, ct.GetSpacing())
    
    summary_path = os.path.join(Path(output_path).parent, "laterality_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_dict, f, indent=4)
    logger.info(f"Laterality summary written: {summary_path}")

    # ── 5. Write output mask
    logger.info("Writing output mask...")
    mask_img = sitk.GetImageFromArray(combined)
    mask_img.CopyInformation(ct)
    
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
    parser.add_argument("--keep-intermediate", action="store_true",
                        help="Keep raw TotalSegmentator per-ROI outputs for inspection")
    parser.add_argument("--intermediate-dir", default=None,
                        help="Explicit directory for intermediate outputs (implies --keep-intermediate if set)")
    args = parser.parse_args()

    try:
        segment(
            input_path=args.input,
            output_path=args.output,
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
