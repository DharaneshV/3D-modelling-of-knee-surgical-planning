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
    parser.add_argument("--decimation_target", type=float, default=0.9, help="Target reduction for decimation")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.basename(args.input).replace(".nii.gz", "").replace(".nii", "")
    
    label_map = LABEL_MAPS[args.track]
    print(f"Running mesh generation for track: {args.track}")
    print(f"Using label map: {label_map}")
    
    # 1. Generate multi-label mesh via Surface Nets
    print("Extracting meshes using vtkSurfaceNets3D...")
    raw_mesh = generate_multilabel_mesh(args.input, label_map)
    
    # 2. Taubin Smoothing
    print("Applying Taubin smoothing...")
    smoothed_mesh = apply_taubin_smoothing(raw_mesh)
    
    # Save the PRE-decimation mesh for Week 6 validation
    pre_decimation_path = os.path.join(args.output_dir, f"{base_name}_{args.track}_smoothed_pre_decimation.obj")
    smoothed_mesh.save(pre_decimation_path)
    print(f"Saved pre-decimation mesh to {pre_decimation_path}")
    
    # 3. Decimation
    print(f"Decimating mesh (target reduction {args.decimation_target})...")
    decimated_mesh = decimate_mesh(smoothed_mesh, target_reduction=args.decimation_target)
    
    # Save the POST-decimation mesh for AR delivery
    post_decimation_path = os.path.join(args.output_dir, f"{base_name}_{args.track}_final.obj")
    decimated_mesh.save(post_decimation_path)
    print(f"Saved final decimated mesh to {post_decimation_path}")
    
if __name__ == "__main__":
    main()
