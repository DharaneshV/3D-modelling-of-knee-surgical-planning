import os
import SimpleITK as sitk
import numpy as np

CASES = ['STS_006', 'STS_035', 'STS_043', 'STS_051']
LABELS = {'Femur': 1, 'Tibia': 2, 'Patella': 3}

results = []

for case in CASES:
    mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
    if not os.path.exists(mask_path):
        continue
        
    img = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing() # (x, y, z)
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    
    case_res = {'Case': case}
    
    for bone, val in LABELS.items():
        mask = (arr == val)
        if not mask.any():
            case_res[f'{bone} Vol (cm3)'] = 0
            case_res[f'{bone} Length (mm)'] = 0
            continue
            
        # Volume
        voxel_count = np.sum(mask)
        vol_cm3 = (voxel_count * voxel_vol_mm3) / 1000.0
        
        # Length in Z (inferior-superior)
        z_indices = np.where(np.any(mask, axis=(1,2)))[0]
        z_length_mm = (z_indices[-1] - z_indices[0] + 1) * spacing[2]
        
        case_res[f'{bone} Vol (cm3)'] = f"{vol_cm3:.1f}"
        
        # Mark truncated bones
        if bone in ['Femur', 'Tibia']:
            case_res[f'{bone} Length (mm)'] = f"{z_length_mm:.1f}*"
        else:
            case_res[f'{bone} Length (mm)'] = f"{z_length_mm:.1f}"
            
    results.append(case_res)

# Print markdown table manually
headers = ["Case", "Femur Vol (cm3)", "Femur Z-Length (mm)", "Tibia Vol (cm3)", "Tibia Z-Length (mm)", "Patella Vol (cm3)", "Patella Z-Length (mm)"]
print("| " + " | ".join(headers) + " |")
print("|" + "|".join(["---"] * len(headers)) + "|")
for r in results:
    row = [
        r['Case'],
        r.get('Femur Vol (cm3)', '-'), r.get('Femur Length (mm)', '-'),
        r.get('Tibia Vol (cm3)', '-'), r.get('Tibia Length (mm)', '-'),
        r.get('Patella Vol (cm3)', '-'), r.get('Patella Length (mm)', '-')
    ]
    print("| " + " | ".join(row) + " |")

print("\n* Denotes a measurement constrained by the synthetic cropping boundary (not true anatomical length).")
