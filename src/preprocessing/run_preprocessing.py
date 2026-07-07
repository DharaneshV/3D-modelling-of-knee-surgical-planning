"""
CLI entry point to run the preprocessing pipeline on the OAI-ZIB dataset.

Usage:
    python -m src.preprocessing.run_preprocessing --images_dir data/oaizib/imagesTr --labels_dir data/oaizib/labelsTr
    python -m src.preprocessing.run_preprocessing --images_dir data/oaizib/imagesTr --max_cases 3  # quick test
"""

import argparse
from src.preprocessing.data_utils import preprocess_oaizib_batch, preprocess_mri


def main():
    parser = argparse.ArgumentParser(
        description="Run the Week 1 preprocessing pipeline (N4 + resample + QA)"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="data/oaizib/imagesTr",
        help="Path to the images directory (e.g. data/oaizib/imagesTr)",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=None,
        help="Path to the labels directory (e.g. data/oaizib/labelsTr)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/preprocessed",
        help="Output directory for preprocessed files",
    )
    parser.add_argument(
        "--target_spacing",
        type=float,
        default=0.5,
        help="Isotropic spacing in mm (default: 0.5)",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Limit number of cases to process (for quick testing)",
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Preprocess a single file instead of a batch",
    )
    parser.add_argument(
        "--skip_n4",
        action="store_true",
        help="Skip N4 bias field correction (e.g. for CT volumes)",
    )

    args = parser.parse_args()

    if args.single:
        result = preprocess_mri(
            image_path=args.single,
            output_dir=args.output_dir,
            target_spacing=args.target_spacing,
            skip_n4=args.skip_n4,
        )
        print(f"\nResult: {result}")
    else:
        results = preprocess_oaizib_batch(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            output_dir=args.output_dir,
            target_spacing=args.target_spacing,
            max_cases=args.max_cases,
        )
        print(f"\nProcessed {len(results)} cases.")
        passed = sum(1 for r in results if r["qa_result"]["passed"])
        print(f"QA: {passed}/{len(results)} passed")


if __name__ == "__main__":
    main()
