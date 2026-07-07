"""
CLI entry point for bone segmentation on CT volumes.

Usage:
    # Single CT volume
    python -m src.segmentation.run_bone_segmentation --input path/to/ct.nii.gz --output_dir results/bone

    # With ground truth validation
    python -m src.segmentation.run_bone_segmentation --input ct.nii.gz --ground_truth gt.nii.gz

    # Adjust HU thresholds per-scanner
    python -m src.segmentation.run_bone_segmentation --input ct.nii.gz --hu_min 250 --hu_max 2500
"""

import os
import argparse
import glob
import logging
import SimpleITK as sitk

from src.segmentation.bone_segmentation import segment_bones
from src.segmentation.metrics import (
    evaluate_segmentation,
    enforce_qa_gate,
    save_results_csv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def process_single(
    input_path: str,
    output_dir: str,
    ground_truth_path: str = None,
    hu_min: int = 200,
    hu_max: int = 3000,
) -> dict:
    """Process a single CT volume through the bone segmentation pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.splitext(os.path.basename(input_path))[0])[0]

    # Load CT
    ct_image = sitk.ReadImage(input_path)
    logger.info(f"Loaded CT: {input_path}")

    # Segment
    result = segment_bones(ct_image, hu_min=hu_min, hu_max=hu_max)

    # Save labeled mask
    output_path = os.path.join(output_dir, f"{basename}_bone_labels.nii.gz")
    sitk.WriteImage(result["labeled_mask"], output_path)
    logger.info(f"Saved bone labels to {output_path}")

    # Save binary mask too (useful for mesh generation later)
    binary_path = os.path.join(output_dir, f"{basename}_bone_binary.nii.gz")
    sitk.WriteImage(result["binary_mask"], binary_path)

    # Validate against ground truth if provided
    metrics = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        gt = sitk.ReadImage(ground_truth_path)
        bone_labels = {1: "femur", 2: "tibia", 3: "patella"}
        metrics = evaluate_segmentation(
            result["labeled_mask"], gt, labels=bone_labels, case_id=basename
        )
        gate = enforce_qa_gate(metrics)
        if not gate["passed"]:
            logger.warning(f"Case {basename} FAILED the QA gate!")

    return {
        "output_path": output_path,
        "binary_path": binary_path,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Bone segmentation from CT via HU thresholding (Week 2-3)"
    )
    parser.add_argument("--input", type=str, required=True, help="Input CT volume (.nii.gz)")
    parser.add_argument("--output_dir", type=str, default="results/bone", help="Output directory")
    parser.add_argument("--ground_truth", type=str, default=None, help="Ground truth mask for validation")
    parser.add_argument("--hu_min", type=int, default=200, help="Lower HU threshold (default 200)")
    parser.add_argument("--hu_max", type=int, default=3000, help="Upper HU threshold (default 3000)")
    parser.add_argument("--save_csv", type=str, default=None, help="Path to save metrics CSV")

    args = parser.parse_args()

    result = process_single(
        input_path=args.input,
        output_dir=args.output_dir,
        ground_truth_path=args.ground_truth,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
    )

    if result["metrics"] and args.save_csv:
        save_results_csv(result["metrics"], args.save_csv)

    print(f"\nBone segmentation complete. Output: {result['output_path']}")


if __name__ == "__main__":
    main()
