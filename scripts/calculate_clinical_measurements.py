import os
import sys
import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import SimpleITK as sitk
import pandas as pd
import json

def calculate_anatomic_axis(vertices):
    centered = vertices - np.mean(vertices, axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    primary_axis = eigvecs[:, np.argmax(eigvals)]
    if primary_axis[2] < 0:
        primary_axis = -primary_axis
    return primary_axis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="outputs", help="Base outputs directory to read from (e.g., outputs or outputs_bilateral)")
    args = parser.parse_args()
    
    log_path = 'data/build_log.json'
    if not os.path.exists(log_path):
        print(f"Build log not found at {log_path}")
        return
        
    with open(log_path, 'r') as f:
        build_log = json.load(f)

    passing_cases = [case_id for case_id, info in build_log.items() if info.get('status') == 'pass']
    results = []

    for case in sorted(passing_cases):
        print(f"Processing {case}...")
        
        mask_path = f"{args.base_dir}/{case}/masks/bone_mask_closed.nii.gz"
        if not os.path.exists(mask_path):
            mask_path = f"{args.base_dir}/{case}/masks/bone_mask.nii.gz"
            
        summary_path = f"{args.base_dir}/{case}/masks/laterality_summary.json"
        
        if not os.path.exists(mask_path):
            print(f"  Mask missing for {case}, skipping.")
            continue
            
        # Determine sides
        sides = ["left", "right"]
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                sides = summary.get("sides_present", sides)
        
        img = sitk.ReadImage(mask_path)
        arr = sitk.GetArrayFromImage(img)
        spacing = img.GetSpacing()
        voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
        
        for side in sides:
            f_mesh_path = f"{args.base_dir}/{case}/meshes/femur_{side}_decimated.obj"
            t_mesh_path = f"{args.base_dir}/{case}/meshes/tibia_{side}_decimated.obj"
            p_mesh_path = f"{args.base_dir}/{case}/meshes/patella_{side}_decimated.obj"
            
            if not os.path.exists(f_mesh_path) or not os.path.exists(t_mesh_path):
                print(f"  Meshes missing for {case} ({side}), skipping side.")
                continue
                
            femur = trimesh.load(f_mesh_path)
            tibia = trimesh.load(t_mesh_path)
            
            case_res = {'Case': case, 'Side': side}
            
            f_label = 1 if side == 'left' else 2
            t_label = 3 if side == 'left' else 4
            p_label = 5 if side == 'left' else 6
            
            # Volumes (cm3)
            case_res['Femur Vol (cm3)'] = round(float(np.sum(arr == f_label) * voxel_vol_mm3 / 1000.0), 1)
            case_res['Tibia Vol (cm3)'] = round(float(np.sum(arr == t_label) * voxel_vol_mm3 / 1000.0), 1)
            case_res['Patella Vol (cm3)'] = round(float(np.sum(arr == p_label) * voxel_vol_mm3 / 1000.0), 1)
            
            # Z-Length (mm) from mask
            for label_name, val in [('Femur', f_label), ('Tibia', t_label)]:
                mask = (arr == val)
                if mask.any():
                    z_indices = np.where(np.any(mask, axis=(1,2)))[0]
                    z_length_mm = (z_indices[-1] - z_indices[0] + 1) * spacing[2]
                    case_res[f'{label_name} Z-Length (mm)'] = round(float(z_length_mm), 1)
                else:
                    case_res[f'{label_name} Z-Length (mm)'] = 0.0
                    
            # Sizing (AP/ML)
            case_res['Femur ML (mm)'] = round(float(np.ptp(femur.vertices[:, 0])), 1)
            case_res['Femur AP (mm)'] = round(float(np.ptp(femur.vertices[:, 1])), 1)
            case_res['Tibia ML (mm)'] = round(float(np.ptp(tibia.vertices[:, 0])), 1)
            case_res['Tibia AP (mm)'] = round(float(np.ptp(tibia.vertices[:, 1])), 1)
            
            # Anatomic Axis Angle via PCA
            f_axis = calculate_anatomic_axis(femur.vertices)
            t_axis = calculate_anatomic_axis(tibia.vertices)
            cos_theta = np.dot(f_axis, t_axis) / (np.linalg.norm(f_axis) * np.linalg.norm(t_axis))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            case_res['Anatomic Axis Angle (deg)'] = round(float(np.degrees(np.arccos(cos_theta))), 1)
            
            # JSW Calculation
            joint_z = np.percentile(tibia.vertices[:, 2], 99)
            f_mask = (femur.vertices[:, 2] >= joint_z - 20) & (femur.vertices[:, 2] <= joint_z + 40)
            t_mask = (tibia.vertices[:, 2] >= joint_z - 40) & (tibia.vertices[:, 2] <= joint_z + 20)
            
            f_pts = femur.vertices[f_mask]
            t_pts = tibia.vertices[t_mask]
            
            jsw = 0.0
            if len(f_pts) > 0 and len(t_pts) > 0:
                tree = cKDTree(t_pts)
                dists, _ = tree.query(f_pts)
                jsw = np.min(dists)
            case_res['JSW (mm)'] = round(float(jsw), 2)
            
            results.append(case_res)

    if not results:
        print("No cases processed successfully.")
        return

    df = pd.DataFrame(results)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/measurement_validation.csv', index=False)
    print("\nGenerated data/measurement_validation.csv successfully!")
    print(df.head())

if __name__ == "__main__":
    main()
