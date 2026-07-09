import os
import sys
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

# Load passing cases from build_log.json
log_path = 'data/build_log.json'
with open(log_path, 'r') as f:
    build_log = json.load(f)

passing_cases = [case_id for case_id, info in build_log.items() if info.get('status') == 'pass']

results = []

for case in sorted(passing_cases):
    print(f"Processing {case}...")
    
    # Try closed mask first, fallback to raw
    mask_path = f"outputs/{case}/masks/bone_mask_closed.nii.gz"
    if not os.path.exists(mask_path):
        mask_path = f"outputs/{case}/masks/bone_mask.nii.gz"
        
    f_mesh_path = f"outputs/{case}/meshes/femur_decimated.obj"
    t_mesh_path = f"outputs/{case}/meshes/tibia_decimated.obj"
    p_mesh_path = f"outputs/{case}/meshes/patella_decimated.obj"
    
    if not os.path.exists(mask_path) or not os.path.exists(f_mesh_path) or not os.path.exists(t_mesh_path):
        print(f"  Files missing for {case}, skipping.")
        continue
        
    img = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    
    femur = trimesh.load(f_mesh_path)
    tibia = trimesh.load(t_mesh_path)
    
    case_res = {'Case': case}
    
    # Volumes (cm3)
    case_res['Femur Vol (cm3)'] = round(float(np.sum(arr == 1) * voxel_vol_mm3 / 1000.0), 1)
    case_res['Tibia Vol (cm3)'] = round(float(np.sum(arr == 2) * voxel_vol_mm3 / 1000.0), 1)
    case_res['Patella Vol (cm3)'] = round(float(np.sum(arr == 3) * voxel_vol_mm3 / 1000.0), 1)
    
    # Z-Length (mm) from mask
    for label_name, val in [('Femur', 1), ('Tibia', 2)]:
        mask = (arr == val)
        if mask.any():
            z_indices = np.where(np.any(mask, axis=(1,2)))[0]
            z_length_mm = (z_indices[-1] - z_indices[0] + 1) * spacing[2]
            case_res[f'{label_name} Z-Length (mm)'] = round(float(z_length_mm), 1)
        else:
            case_res[f'{label_name} Z-Length (mm)'] = 0.0
            
    # Sizing (AP/ML)
    f_ml = np.ptp(femur.vertices[:, 0])
    f_ap = np.ptp(femur.vertices[:, 1])
    t_ml = np.ptp(tibia.vertices[:, 0])
    t_ap = np.ptp(tibia.vertices[:, 1])
    
    case_res['Femur ML (mm)'] = round(float(f_ml), 1)
    case_res['Femur AP (mm)'] = round(float(f_ap), 1)
    case_res['Tibia ML (mm)'] = round(float(t_ml), 1)
    case_res['Tibia AP (mm)'] = round(float(t_ap), 1)
    
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

df = pd.DataFrame(results)
os.makedirs('data', exist_ok=True)
df.to_csv('data/measurement_validation.csv', index=False)
print("\nGenerated data/measurement_validation.csv successfully!")
print(df.head())
