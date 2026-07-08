# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import SimpleITK as sitk

sys.path.append(".")
from src.segmentation.metrics import compute_dice

CASES = ['STS_006', 'STS_035', 'STS_043', 'STS_051']


def create_ts_mask(ts_total_dir, ts_bones_dir):
    """Combine femur (from 'total' task, split left/right) with tibia/patella
    (from 'appendicular_bones' task, unified) into our standard 1/2/3 label mask."""
    # Find a template image to match TS output dimensions
    template_img = None
    for side in ['left', 'right']:
        p = os.path.join(ts_total_dir, f"femur_{side}.nii.gz")
        if os.path.exists(p):
            template_img = sitk.ReadImage(p)
            break
    if not template_img:
        p = os.path.join(ts_bones_dir, "tibia.nii.gz")
        if os.path.exists(p):
            template_img = sitk.ReadImage(p)
            
    if not template_img:
        raise FileNotFoundError("Could not find any TotalSegmentator output files to use as a template.")
        
    template_arr = sitk.GetArrayFromImage(template_img)
    combined = np.zeros_like(template_arr, dtype=np.uint8)

    # Femur (label 1) - from 'total' task, may be split left/right
    femur_found = False
    for side in ['left', 'right']:
        p = os.path.join(ts_total_dir, f"femur_{side}.nii.gz")
        if os.path.exists(p):
            arr = sitk.GetArrayFromImage(sitk.ReadImage(p))
            if arr.sum() > 0:
                combined[arr > 0] = 1
                femur_found = True
    if not femur_found:
        print(f"  WARNING: no femur voxels found in {ts_total_dir}")

    # Tibia (label 2) and Patella (label 3) - from 'appendicular_bones' task
    for roi, label_val in [('tibia', 2), ('patella', 3)]:
        p = os.path.join(ts_bones_dir, f"{roi}.nii.gz")
        if os.path.exists(p):
            arr = sitk.GetArrayFromImage(sitk.ReadImage(p))
            if arr.sum() == 0:
                print(f"  WARNING: {roi}.nii.gz exists but has zero voxels")
            combined[arr > 0] = label_val
        else:
            print(f"  WARNING: {p} not found")

    out_img = sitk.GetImageFromArray(combined)
    out_img.CopyInformation(template_img)
    return out_img


def check_grid_match(img_a, img_b, case):
    """Assert both masks share identical geometry before computing Dice."""
    ok = True
    if img_a.GetSize() != img_b.GetSize():
        print(f"  [{case}] SIZE MISMATCH: {img_a.GetSize()} vs {img_b.GetSize()}")
        ok = False
    if img_a.GetSpacing() != img_b.GetSpacing():
        print(f"  [{case}] SPACING MISMATCH: {img_a.GetSpacing()} vs {img_b.GetSpacing()}")
        ok = False
    if img_a.GetOrigin() != img_b.GetOrigin():
        print(f"  [{case}] ORIGIN MISMATCH: {img_a.GetOrigin()} vs {img_b.GetOrigin()}")
        ok = False
    return ok


def main():
    results = []

    for case in CASES:
        print(f"\nProcessing {case}...")
        our_mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
        ts_total_dir = f"data/ct_knee/{case}_totalseg_reference"
        ts_bones_dir = f"data/ct_knee/{case}_totalseg_reference_bones"

        if not os.path.exists(our_mask_path):
            print(f"  Missing {our_mask_path}, skipping.")
            continue
        if not os.path.exists(ts_total_dir) or not os.path.exists(ts_bones_dir):
            print(f"  Missing TotalSegmentator output dirs for {case}, skipping.")
            continue

        pred_img = sitk.ReadImage(our_mask_path)
        gt_img_raw = create_ts_mask(ts_total_dir, ts_bones_dir)

        # Resample GT image to prediction's physical space
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(pred_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        gt_img = resampler.Execute(gt_img_raw)

        if not check_grid_match(pred_img, gt_img, case):
            print(f"  Skipping Dice for {case} due to grid mismatch - investigate before trusting any score.")
            continue

        case_scores = {'Case': case}
        for bone, label_val in [('Femur', 1), ('Tibia', 2), ('Patella', 3)]:
            dice = compute_dice(pred_img, gt_img, label=label_val)
            case_scores[bone] = f"{dice:.4f}"
        results.append(case_scores)

    print("\n\n=== Dice Comparison (TotalSegmentator Reference: total + appendicular_bones) ===")
    print("| Case | Femur | Tibia | Patella |")
    print("|---|---|---|---|")
    for r in results:
        print(f"| {r['Case']} | {r.get('Femur','-')} | {r.get('Tibia','-')} | {r.get('Patella','-')} |")


if __name__ == "__main__":
    main()
