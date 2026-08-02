"""
Validation metrics for segmentation accuracy.

Implements the Week 4 hard gate from the implementation plan, but built
early (as the plan recommends) so we can run Dice/Hausdorff on bone
output as soon as it exists.

Metrics:
  - Dice Similarity Coefficient (DSC)
  - Hausdorff Distance (HD)
  - Average Symmetric Surface Distance (ASSD)

Uses SimpleITK's LabelOverlapMeasuresImageFilter and
HausdorffDistanceImageFilter as specified in the plan (Section 8.2–8.3).
"""

import logging
import pandas as pd
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# QA gate thresholds from the implementation plan (Section 8.5)
BONE_DICE_THRESHOLD = 0.90
SOFT_TISSUE_DICE_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Per-label metrics
# ---------------------------------------------------------------------------

def compute_dice(
    prediction: sitk.Image,
    ground_truth: sitk.Image,
    label: int = 1,
) -> float:
    """
    Compute Dice Similarity Coefficient for a specific label.

    Args:
        prediction: Predicted segmentation mask.
        ground_truth: Ground truth segmentation mask.
        label: The label value to evaluate (default 1).

    Returns:
        Dice coefficient (float, 0.0–1.0).
    """
    pred_binary = sitk.Equal(prediction, label)
    gt_binary = sitk.Equal(ground_truth, label)

    overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_filter.Execute(pred_binary, gt_binary)

    return overlap_filter.GetDiceCoefficient()


def compute_hausdorff(
    prediction: sitk.Image,
    ground_truth: sitk.Image,
    label: int = 1,
) -> dict:
    """
    Compute Hausdorff Distance for a specific label.

    Args:
        prediction: Predicted segmentation mask.
        ground_truth: Ground truth segmentation mask.
        label: The label value to evaluate.

    Returns:
        dict with hausdorff_distance and average_hausdorff_distance.
    """
    pred_binary = sitk.Cast(sitk.Equal(prediction, label), sitk.sitkUInt8)
    gt_binary = sitk.Cast(sitk.Equal(ground_truth, label), sitk.sitkUInt8)

    hausdorff_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_filter.Execute(pred_binary, gt_binary)

    return {
        "hausdorff_distance": hausdorff_filter.GetHausdorffDistance(),
        "average_hausdorff_distance": hausdorff_filter.GetAverageHausdorffDistance(),
    }


def compute_surface_distance(
    prediction: sitk.Image,
    ground_truth: sitk.Image,
    label: int = 1,
) -> float:
    """
    Compute Average Symmetric Surface Distance (ASSD) for a specific label.

    The plan recommends logging ASSD in addition to Dice — it's more
    sensitive to boundary jaggedness and gives an early signal on mesh
    quality before Week 5.

    Args:
        prediction: Predicted segmentation mask.
        ground_truth: Ground truth segmentation mask.
        label: The label value to evaluate.

    Returns:
        ASSD in mm (float). Returns -1 if not supported by the SimpleITK version.
    """
    pred_binary = sitk.Cast(sitk.Equal(prediction, label), sitk.sitkUInt8)
    gt_binary = sitk.Cast(sitk.Equal(ground_truth, label), sitk.sitkUInt8)

    try:
        # Get surface distances using signed Maurer distance maps
        pred_distance = sitk.SignedMaurerDistanceMap(pred_binary, squaredDistance=False)
        gt_distance = sitk.SignedMaurerDistanceMap(gt_binary, squaredDistance=False)

        # Extract surface voxels (boundary)
        pred_surface = sitk.LabelContour(pred_binary)
        gt_surface = sitk.LabelContour(gt_binary)

        # Sample distances at surface points
        pred_surface_arr = sitk.GetArrayFromImage(pred_surface).astype(bool)
        gt_surface_arr = sitk.GetArrayFromImage(gt_surface).astype(bool)

        gt_dist_arr = sitk.GetArrayFromImage(gt_distance)
        pred_dist_arr = sitk.GetArrayFromImage(pred_distance)

        # Distances from pred surface to gt, and gt surface to pred
        pred_to_gt = abs(gt_dist_arr[pred_surface_arr])
        gt_to_pred = abs(pred_dist_arr[gt_surface_arr])

        if len(pred_to_gt) == 0 or len(gt_to_pred) == 0:
            return -1.0

        import numpy as np
        assd = (np.mean(pred_to_gt) + np.mean(gt_to_pred)) / 2.0
        return float(assd)

    except Exception as e:
        logger.warning(f"ASSD computation failed: {e}")
        return -1.0


# ---------------------------------------------------------------------------
# Multi-label evaluation
# ---------------------------------------------------------------------------

# CartiMorph / OAI-ZIB label convention — matches the MRI track's segmentation
# output and data/oaizib/labelsTs ground truth. Pass an explicit `labels` dict
# for any other track (e.g. CT bone, which is laterality-split).
LABEL_NAMES = {
    1: "femur",
    2: "femoral_cartilage",
    3: "tibia",
    4: "medial_tibial_cartilage",
    5: "lateral_tibial_cartilage",
}


def evaluate_segmentation(
    prediction: sitk.Image,
    ground_truth: sitk.Image,
    labels: dict = None,
    case_id: str = "",
) -> list:
    """
    Evaluate a multi-label segmentation against ground truth.

    Computes Dice, Hausdorff, and ASSD for each label.

    Args:
        prediction: Predicted multi-label mask.
        ground_truth: Ground truth multi-label mask.
        labels: Dict mapping label_value -> label_name.
                Defaults to LABEL_NAMES.
        case_id: Identifier string for logging.

    Returns:
        List of dicts, one per label, with all metrics.
    """
    if labels is None:
        labels = LABEL_NAMES

    results = []

    for label_val, label_name in labels.items():
        dice = compute_dice(prediction, ground_truth, label=label_val)
        hausdorff = compute_hausdorff(prediction, ground_truth, label=label_val)
        assd = compute_surface_distance(prediction, ground_truth, label=label_val)

        result = {
            "case_id": case_id,
            "structure": label_name,
            "label": label_val,
            "dice": round(dice, 4),
            "hausdorff_mm": round(hausdorff["hausdorff_distance"], 4),
            "avg_hausdorff_mm": round(hausdorff["average_hausdorff_distance"], 4),
            "assd_mm": round(assd, 4) if assd >= 0 else "N/A",
        }
        results.append(result)

        logger.info(
            f"[{case_id}] {label_name}: "
            f"Dice={result['dice']:.4f}, "
            f"HD={result['hausdorff_mm']:.2f}mm, "
            f"ASSD={result['assd_mm']}"
        )

    return results


# ---------------------------------------------------------------------------
# QA Gate enforcement
# ---------------------------------------------------------------------------

def enforce_qa_gate(
    results: list,
    bone_threshold: float = BONE_DICE_THRESHOLD,
    soft_tissue_threshold: float = SOFT_TISSUE_DICE_THRESHOLD,
) -> dict:
    """
    Enforce the accuracy gate from the implementation plan.

    Flags any structure below 0.90 (bone) or 0.75 (soft tissue) Dice.
    The plan says to enforce this literally in code, not just rely on
    someone reading a CSV.

    Args:
        results: List of metric dicts from evaluate_segmentation().
        bone_threshold: Dice threshold for bone structures (default 0.90).
        soft_tissue_threshold: Dice threshold for soft tissue (default 0.75).

    Returns:
        dict with: passed (bool), failures (list of failed structures).
    """
    bone_structures = {"femur", "tibia", "patella"}
    failures = []

    for r in results:
        structure = r["structure"]
        dice = r["dice"]

        if structure in bone_structures:
            threshold = bone_threshold
        else:
            threshold = soft_tissue_threshold

        if dice < threshold:
            failures.append({
                "case_id": r["case_id"],
                "structure": structure,
                "dice": dice,
                "threshold": threshold,
                "gap": round(threshold - dice, 4),
            })

    passed = len(failures) == 0

    if not passed:
        logger.warning(f"QA GATE FAILED — {len(failures)} structure(s) below threshold:")
        for f in failures:
            logger.warning(
                f"  {f['case_id']} / {f['structure']}: "
                f"Dice={f['dice']:.4f} < {f['threshold']:.2f} "
                f"(gap={f['gap']:.4f})"
            )
    else:
        logger.info("QA GATE PASSED — all structures above threshold")

    return {"passed": passed, "failures": failures}


# ---------------------------------------------------------------------------
# Batch evaluation + CSV export
# ---------------------------------------------------------------------------

def results_to_dataframe(results: list) -> pd.DataFrame:
    """Convert a list of metric dicts to a pandas DataFrame."""
    return pd.DataFrame(results)


def save_results_csv(results: list, output_path: str = "measurement_validation.csv"):
    """
    Save evaluation results to CSV (the spec's deliverable format).

    Args:
        results: List of metric dicts.
        output_path: Output CSV path.
    """
    df = results_to_dataframe(results)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved validation results to {output_path}")
    return df
