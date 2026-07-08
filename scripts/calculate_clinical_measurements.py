import os
import sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import SimpleITK as sitk

CASES = ['STS_006', 'STS_035', 'STS_043', 'STS_051']
LABELS = {'Femur': 1, 'Tibia': 2, 'Patella': 3}

results = []

def check_bounds(name, val, min_v, max_v):
    if val < min_v or val > max_v:
        print(f"  WARNING: {name} ({val:.1f}) is outside physiological bounds ({min_v}-{max_v})!")

for case in CASES:
    print(f"\nProcessing {case}...")
    mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
    f_mesh_path = f"data/ct_knee/{case}_meshes/femur_decimated.obj"
    t_mesh_path = f"data/ct_knee/{case}_meshes/tibia_decimated.obj"
    
    if not os.path.exists(mask_path) or not os.path.exists(f_mesh_path) or not os.path.exists(t_mesh_path):
        continue
        
    img = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    
    femur = trimesh.load(f_mesh_path)
    tibia = trimesh.load(t_mesh_path)
    
    case_res = {'Case': case}
    
    # VOLUMES
    for bone, val in LABELS.items():
        mask = (arr == val)
        if mask.any():
            vol_cm3 = (np.sum(mask) * voxel_vol_mm3) / 1000.0
            case_res[f'{bone} Vol (cm3)'] = f"{vol_cm3:.1f}"
            
            z_indices = np.where(np.any(mask, axis=(1,2)))[0]
            z_length_mm = (z_indices[-1] - z_indices[0] + 1) * spacing[2]
            star = "*" if bone in ['Femur', 'Tibia'] else ""
            case_res[f'{bone} Z-Length (mm)'] = f"{z_length_mm:.1f}{star}"
        else:
            case_res[f'{bone} Vol (cm3)'] = "-"
            case_res[f'{bone} Z-Length (mm)'] = "-"
            
    # DIMENSIONS
    f_width = np.ptp(femur.vertices[:, 0])
    check_bounds("Femur Bicondylar Width", f_width, 70, 105)
    
    t_width = np.ptp(tibia.vertices[:, 0])
    t_depth = np.ptp(tibia.vertices[:, 1])
    check_bounds("Tibia Plateau Width", t_width, 60, 90)
    check_bounds("Tibia Plateau Depth", t_depth, 40, 70)
    
    case_res['Femur Width'] = f"{f_width:.1f}"
    case_res['Tibia Width'] = f"{t_width:.1f}"
    case_res['Tibia Depth'] = f"{t_depth:.1f}"
    
    # JSW
    joint_z = np.percentile(tibia.vertices[:, 2], 99)
    f_mask = (femur.vertices[:, 2] >= joint_z - 20) & (femur.vertices[:, 2] <= joint_z + 40)
    t_mask = (tibia.vertices[:, 2] >= joint_z - 40) & (tibia.vertices[:, 2] <= joint_z + 20)
    f_pts = femur.vertices[f_mask]
    t_pts = tibia.vertices[t_mask]
    
    jsw = -1
    if len(f_pts) > 0 and len(t_pts) > 0:
        tree = cKDTree(t_pts)
        dists, _ = tree.query(f_pts)
        jsw = np.min(dists)
        check_bounds("Joint Space Width", jsw, 2.0, 6.0)
    
    case_res['JSW'] = f"{jsw:.2f}"
    results.append(case_res)

print("\n")
headers = ["Case", "Femur Vol (cm3)", "Femur Z-Length (mm)", "Femur Width", "Tibia Vol (cm3)", "Tibia Z-Length (mm)", "Tibia Width", "Tibia Depth", "JSW"]
print("| " + " | ".join(headers) + " |")
print("|" + "|".join(["---"] * len(headers)) + "|")
for r in results:
    row = [r.get(h, "-") for h in headers]
    print("| " + " | ".join(row) + " |")

print("\n* Denotes a measurement constrained by the synthetic cropping boundary (not true anatomical length).")
