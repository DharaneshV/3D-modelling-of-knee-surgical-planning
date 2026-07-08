import os
import sys
import numpy as np
import SimpleITK as sitk
from totalsegmentator.python_api import totalsegmentator

sys.path.append(".")
from src.segmentation.metrics import compute_dice

TS_LABEL_MAP = {'femur': 1, 'tibia': 2, 'patella': 3}
CASES = ['STS_006', 'STS_035', 'STS_043', 'STS_051']

def create_ts_mask(ts_out_dir, ref_image):
    ref_arr = sitk.GetArrayFromImage(ref_image)
    combined = np.zeros_like(ref_arr, dtype=np.uint8)
    for roi, label_val in TS_LABEL_MAP.items():
        roi_path = os.path.join(ts_out_dir, f"{roi}.nii.gz")
        if os.path.exists(roi_path):
            roi_img = sitk.ReadImage(roi_path)
            roi_arr = sitk.GetArrayFromImage(roi_img)
            combined[roi_arr > 0] = label_val
    out_img = sitk.GetImageFromArray(combined)
    out_img.CopyInformation(ref_image)
    return out_img

def main():
    results = []
    # Set this to avoid the weights pickling error during model load
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    
    for case in CASES:
        print(f"\nProcessing {case}...")
        cropped_img_path = f"data/ct_knee/case01_{case}_cropped.nii.gz"
        our_mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
        ts_out_dir = f"data/ct_knee/{case}_totalseg_reference"
        
        if not os.path.exists(ts_out_dir):
            os.makedirs(ts_out_dir)
            
        if not all(os.path.exists(os.path.join(ts_out_dir, f"{roi}.nii.gz")) for roi in TS_LABEL_MAP.keys()):
            try:
                print("Running TotalSegmentator...")
                totalsegmentator(cropped_img_path, ts_out_dir, task="total", fast=True, nr_thr_resamp=1, nr_thr_saving=1) 
            except Exception as e:
                print(f"Error running TotalSegmentator programmatically: {e}")
                
        pred_img = sitk.ReadImage(our_mask_path)
        gt_img = create_ts_mask(ts_out_dir, pred_img)
        
        case_scores = {'Case': case}
        for bone, label_val in TS_LABEL_MAP.items():
            dice = compute_dice(pred_img, gt_img, label=label_val)
            case_scores[bone.capitalize()] = f"{dice:.4f}"
        results.append(case_scores)
        
    print("\n\n=== Dice Comparison (TotalSegmentator Reference) ===")
    print("| Case | Femur | Tibia | Patella |")
    print("|---|---|---|---|")
    for r in results:
        print(f"| {r['Case']} | {r['Femur']} | {r['Tibia']} | {r['Patella']} |")

if __name__ == "__main__":
    main()
