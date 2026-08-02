"""
Segmentation accuracy validation on the OAI-ZIB test set (spec Section 8).

Compares the MRI pipeline's CartiMorph output against the dataset's ground-truth
masks and produces results/dice_scores_summary.csv — Dice, Hausdorff (max + avg),
and ASSD per structure per case, plus the Section 8.5 QA gate.

Coordinate spaces differ between prediction and ground truth: the pipeline
reorients to RAI and resamples to 0.5mm isotropic before inference, while the
ground truth stays in its native RAS anisotropic grid. Predictions are therefore
resampled onto the ground-truth grid (nearest-neighbour, labels must not be
interpolated) so metrics are computed in the reference standard's own space.

Usage:
    python scripts/run_oaizib_validation.py [--limit N]
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.segmentation.metrics import (
    evaluate_segmentation,
    enforce_qa_gate,
    save_results_csv,
)

ROOT = Path(__file__).resolve().parent.parent
GT_DIR = ROOT / "data" / "oaizib" / "labelsTs"
MESHES_DIR = ROOT / "meshes"
OUTPUT_CSV = ROOT / "results" / "dice_scores_summary.csv"

# CartiMorph / OAI-ZIB convention — identical in prediction and ground truth.
OAIZIB_LABELS = {
    1: "femur",
    2: "femoral_cartilage",
    3: "tibia",
    4: "medial_tibial_cartilage",
    5: "lateral_tibial_cartilage",
}


def resample_to_reference(moving: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Resample a label mask onto the reference grid without interpolating labels."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(moving)


def find_prediction(case_id: str) -> Path | None:
    """Locate the pipeline's predicted mask for a case, if it exists."""
    candidate = MESHES_DIR / case_id / f"{case_id}_mask.nii.gz"
    return candidate if candidate.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N cases (for timing/smoke runs)")
    args = parser.parse_args()

    gt_files = sorted(glob.glob(str(GT_DIR / "*.nii.gz")))
    if args.limit:
        gt_files = gt_files[:args.limit]
    print(f"Found {len(gt_files)} ground-truth masks in {GT_DIR}")

    all_results = []
    missing = []

    for i, gt_path in enumerate(gt_files, 1):
        case_id = os.path.basename(gt_path).replace(".nii.gz", "")
        pred_path = find_prediction(case_id)

        if pred_path is None:
            missing.append(case_id)
            print(f"[{i}/{len(gt_files)}] {case_id}: no prediction found — skipping")
            continue

        print(f"[{i}/{len(gt_files)}] {case_id}: evaluating...")
        gt_img = sitk.ReadImage(gt_path)
        pred_img = sitk.ReadImage(str(pred_path))

        # Bring the prediction into the ground truth's space before comparing.
        if (pred_img.GetSize() != gt_img.GetSize()
                or pred_img.GetSpacing() != gt_img.GetSpacing()
                or pred_img.GetDirection() != gt_img.GetDirection()
                or pred_img.GetOrigin() != gt_img.GetOrigin()):
            pred_img = resample_to_reference(pred_img, gt_img)

        all_results.extend(
            evaluate_segmentation(
                prediction=pred_img,
                ground_truth=gt_img,
                labels=OAIZIB_LABELS,
                case_id=case_id,
            )
        )

    if not all_results:
        print("No cases evaluated — nothing to report.")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    save_results_csv(all_results, str(OUTPUT_CSV))

    # --- Summary ---
    df = pd.DataFrame(all_results)
    n_cases = df["case_id"].nunique()

    print("\n" + "=" * 82)
    print(f"SEGMENTATION ACCURACY - OAI-ZIB test set ({n_cases} cases)")
    print("=" * 82)
    print(f"{'Structure':<28} | {'Dice (mean +/- sd)':<20} | {'HD95':<7} | {'MaxHD':<7} | {'ASSD':<7} | Gate")
    print("-" * 96)

    bone = {"femur", "tibia", "patella"}
    for name in OAIZIB_LABELS.values():
        sub = df[df["structure"] == name]
        if sub.empty:
            continue
        assd = pd.to_numeric(sub["assd_mm"], errors="coerce")
        hd95 = pd.to_numeric(sub["hd95_mm"], errors="coerce")
        threshold = 0.90 if name in bone else 0.75
        verdict = "PASS" if sub["dice"].mean() >= threshold else "FAIL"
        print(f"{name:<28} | "
              f"{sub['dice'].mean():.4f} +/- {sub['dice'].std():.4f}  | "
              f"{hd95.mean():<7.2f} | "
              f"{sub['hausdorff_mm'].mean():<7.2f} | "
              f"{assd.mean():<7.3f} | {verdict} (>={threshold:.2f})")

    # Max Hausdorff is reported for continuity but is a weak gate on thin
    # structures — see compute_surface_metrics in src/segmentation/metrics.py.
    n_hd_high = int((df["hausdorff_mm"] > 5).sum())
    passing = df.apply(
        lambda r: r["dice"] >= (0.90 if r["structure"] in bone else 0.75), axis=1)
    print(f"\nMax HD > 5mm in {n_hd_high}/{len(df)} evaluations, of which "
          f"{int((passing & (df['hausdorff_mm'] > 5)).sum())} still pass their Dice gate "
          f"(corr with Dice: {df['hausdorff_mm'].corr(df['dice']):.2f}).")

    if missing:
        print(f"\nNote: {len(missing)} case(s) had no prediction on disk: "
              f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")

    gate = enforce_qa_gate(all_results)
    n_struct = len(all_results)
    if gate["passed"]:
        print(f"\nQA GATE PASSED — all {n_struct} structure evaluations met threshold.")
    else:
        print(f"\nQA GATE: {len(gate['failures'])}/{n_struct} structure evaluations below threshold.")

    print(f"\nFull per-case results: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
