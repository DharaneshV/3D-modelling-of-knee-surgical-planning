import os
import glob
import SimpleITK as sitk

from src.segmentation.metrics import (
    evaluate_segmentation,
    enforce_qa_gate,
    save_results_csv
)

def run_evaluation(data_dir, predictions_dir, output_csv="results/accuracy_report.csv"):
    """
    Evaluates predictions against ground truth masks in data_dir/labelsTs.
    """
    labels_dir = os.path.join(data_dir, "labelsTs")
    
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Ground truth labels not found at {labels_dir}")
        
    if not os.path.exists(predictions_dir):
        raise FileNotFoundError(f"Predictions not found at {predictions_dir}")

    # Label map for OAI-ZIB dataset
    labels_map = {
        1: "Femur",
        2: "Femoral Cartilage",
        3: "Tibia",
        4: "Medial Tibial Cartilage",
        5: "Lateral Tibial Cartilage",
    }
    
    gt_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))
    print(f"Found {len(gt_files)} ground truth files.")
    
    all_results = []
    
    for gt_path in gt_files:
        basename = os.path.basename(gt_path)
        pred_path = os.path.join(predictions_dir, basename)
        
        if not os.path.exists(pred_path):
            print(f"Warning: No prediction found for {basename}")
            continue
            
        print(f"Evaluating {basename}...")
        gt_img = sitk.ReadImage(gt_path)
        pred_img = sitk.ReadImage(pred_path)
        
        case_id = basename.replace(".nii.gz", "")
        
        case_results = evaluate_segmentation(
            prediction=pred_img,
            ground_truth=gt_img,
            case_id=case_id,
            labels=labels_map
        )
        all_results.extend(case_results)

    if not all_results:
        print("No results to evaluate.")
        return

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    save_results_csv(all_results, output_csv)
    
    # Run the hard gate check
    gate_status = enforce_qa_gate(all_results)
    
    if gate_status["passed"]:
        print(f"\nSUCCESS: All {len(all_results)} segmentations passed the QA gate!")
    else:
        print(f"\nFAILED: {len(gate_status['failures'])} segmentations failed the QA gate.")
