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
    
    from skimage.measure import marching_cubes, mesh_surface_area
    
    # Thickness using Volume / (Surface Area / 2)
    sampling = [spacing[2], spacing[1], spacing[0]] # z, y, x
    
    def calc_thickness(mask, vol):
        if not np.any(mask): return 0.0
        padded = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
        verts, faces, normals, values = marching_cubes(padded, level=0.5, spacing=sampling)
        sa_total = mesh_surface_area(verts, faces)
        sa_interface = sa_total / 2.0
        return vol / sa_interface

    measurements['Femoral_Cartilage_Mean_Thickness_mm'] = float(calc_thickness(arr == 2, measurements['Femoral_Cartilage_Vol_cm3'] * 1000))
    measurements['Medial_Tibial_Cartilage_Mean_Thickness_mm'] = float(calc_thickness(arr == 4, measurements['Medial_Tibial_Cartilage_Vol_cm3'] * 1000))
    measurements['Lateral_Tibial_Cartilage_Mean_Thickness_mm'] = float(calc_thickness(arr == 5, measurements['Lateral_Tibial_Cartilage_Vol_cm3'] * 1000))
            
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
