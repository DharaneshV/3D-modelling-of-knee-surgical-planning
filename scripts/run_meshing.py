import argparse
import os

from src.mesh.surface_nets import generate_multilabel_mesh
from src.mesh.processing import apply_taubin_smoothing, decimate_mesh

# Pre-defined label maps for different tracks
LABEL_MAPS = {
    "ct_bone": {
        "femur": 1, 
        "tibia": 2, 
        "patella": 3
    },
    "mri_cartilage": {
        "femoral_cartilage": 2, 
        "medial_tibial_cartilage": 4, 
        "lateral_tibial_cartilage": 5
    }
}

def main():
    parser = argparse.ArgumentParser(description="3D Mesh Generation pipeline using VTK SurfaceNets3D.")
    parser.add_argument("--input", type=str, required=True, help="Path to input segmentation mask (.nii.gz)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the meshes")
    parser.add_argument("--track", type=str, choices=["ct_bone", "mri_cartilage"], required=True, 
                        help="Which anatomical track to process (dictates the label map)")
    parser.add_argument("--decimation_target", type=float, default=0.5, help="Target reduction for decimation")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.basename(args.input).replace(".nii.gz", "").replace(".nii", "")
    
    label_map = LABEL_MAPS[args.track]
    print(f"Running mesh generation for track: {args.track}")
    print(f"Using label map: {label_map}")
    
    # 1. Generate multi-label mesh via Surface Nets
    print("Extracting meshes using vtkSurfaceNets3D...")
    raw_mesh = generate_multilabel_mesh(args.input, label_map)
    
    # Save the RAW multi-label mesh for reference
    raw_mesh_path = os.path.join(args.output_dir, f"{base_name}_{args.track}_raw.obj")
    raw_mesh.save(raw_mesh_path)
    
    # Extract individual labels, then smooth, then decimate
    import pyvista as pv
    import numpy as np
    for label_name, label_id in label_map.items():
        if 'BoundaryLabels' not in raw_mesh.cell_data:
            print(f"Warning: BoundaryLabels not found in raw mesh for {label_name}. Skipping extraction.")
            continue
            
        labels = raw_mesh.cell_data['BoundaryLabels']
        mask = (labels[:, 0] == label_id) | (labels[:, 1] == label_id)
        
        # extract cells for this label
        sub_mesh = raw_mesh.extract_cells(mask)
        # extract_surface to get clean PolyData
        surf = sub_mesh.extract_surface(algorithm='dataset_surface')
        surf = surf.clean()
        
        # apply smoothing
        print(f"Smoothing {label_name}...")
        smoothed_comp = apply_taubin_smoothing(surf)
        
        # apply decimation
        print(f"Decimating {label_name}...")
        decimated_comp = decimate_mesh(smoothed_comp, target_reduction=args.decimation_target)
        
        comp_path = os.path.join(args.output_dir, f"{label_name}_decimated.obj")
        decimated_comp.save(comp_path)
        print(f"Saved individual decimated mesh {label_name} to {comp_path}")
        
if __name__ == "__main__":
    main()
