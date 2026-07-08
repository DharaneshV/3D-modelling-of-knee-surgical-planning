import argparse
import json
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt
from pathlib import Path

def calculate_mri_measurements(label_path: str, output_json: str):
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file {label_path} not found.")
        
    img = sitk.ReadImage(str(label_path))
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    voxel_vol_cm3 = voxel_vol_mm3 / 1000.0
    
    # OAI-ZIB Labels
    # 1: Femur, 2: Femoral Cartilage
    # 3: Tibia, 4: Medial Tibial Cartilage, 5: Lateral Tibial Cartilage
    
    measurements = {}
    
    # Volumes
    vols = {
        'Femoral_Cartilage_Vol_cm3': float(np.sum(arr == 2) * voxel_vol_cm3),
        'Medial_Tibial_Cartilage_Vol_cm3': float(np.sum(arr == 4) * voxel_vol_cm3),
        'Lateral_Tibial_Cartilage_Vol_cm3': float(np.sum(arr == 5) * voxel_vol_cm3)
    }
    measurements.update(vols)
    
    # Thickness using Distance Transform from Bone
    # Voxel physical spacing for EDT
    sampling = [spacing[2], spacing[1], spacing[0]] # z, y, x
    
    # Femoral Cartilage Thickness
    femur_mask = (arr == 1)
    if np.any(femur_mask):
        # EDT computes distance to background (0). So we want distance to bone (1).
        # We invert the bone mask so bone is 0 and everything else is 1.
        edt_femur = distance_transform_edt(~femur_mask, sampling=sampling)
        fc_mask = (arr == 2)
        if np.any(fc_mask):
            mean_dist = np.mean(edt_femur[fc_mask])
            # The mean distance inside the volume is ~half the thickness
            measurements['Femoral_Cartilage_Mean_Thickness_mm'] = float(mean_dist * 2.0)
        else:
            measurements['Femoral_Cartilage_Mean_Thickness_mm'] = 0.0
            
    # Tibial Cartilage Thickness
    tibia_mask = (arr == 3)
    if np.any(tibia_mask):
        edt_tibia = distance_transform_edt(~tibia_mask, sampling=sampling)
        
        mtc_mask = (arr == 4)
        if np.any(mtc_mask):
            measurements['Medial_Tibial_Cartilage_Mean_Thickness_mm'] = float(np.mean(edt_tibia[mtc_mask]) * 2.0)
        else:
            measurements['Medial_Tibial_Cartilage_Mean_Thickness_mm'] = 0.0
            
        ltc_mask = (arr == 5)
        if np.any(ltc_mask):
            measurements['Lateral_Tibial_Cartilage_Mean_Thickness_mm'] = float(np.mean(edt_tibia[ltc_mask]) * 2.0)
        else:
            measurements['Lateral_Tibial_Cartilage_Mean_Thickness_mm'] = 0.0
            
    # Save to JSON
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(measurements, f, indent=4)
        
    print(f"Saved MRI measurements to {output_json}")
    for k, v in measurements.items():
        print(f"  {k}: {v:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MRI Cartilage Measurements")
    parser.add_argument("-i", "--input", required=True, help="Input segmentation mask (0.5mm isotropic)")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file path")
    args = parser.parse_args()
    
    calculate_mri_measurements(args.input, args.output)
