import argparse
import os
import sys
from pathlib import Path

# Ensure the root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from src.mesh.surface_nets import extract_multilabel_surface
from src.mesh.processing import fill_label_gaps
import SimpleITK as sitk
# boolean_resolution import removed: unified SurfaceNets produces topologically-consistent
# bone/cartilage boundaries by construction; no independent-mesh boolean clip needed.
import subprocess

# Pre-defined label maps for different tracks
LABEL_MAPS = {
    "ct_bone": {
        "femur_left": 1, "femur_right": 2, 
        "tibia_left": 3, "tibia_right": 4, 
        "patella_left": 5, "patella_right": 6
    },
    "mri_cartilage": {
        "femur_unknown": 1,          # CartiMorph label 1 — femur bone (native MRI geometry)
        "femoral_cartilage": 2,
        "tibia_unknown": 3,          # CartiMorph label 3 — tibia bone (native MRI geometry)
        "medial_tibial_cartilage": 4,
        "lateral_tibial_cartilage": 5
    }
}

LABEL_COLORS = {
    'femur_left': '#e74c3c', 'femur_right': '#e74c3c', 'femur_unknown': '#e74c3c',
    'tibia_left': '#2ecc71', 'tibia_right': '#2ecc71', 'tibia_unknown': '#2ecc71',
    'patella_left': '#f1c40f', 'patella_right': '#f1c40f',
    'femoral_cartilage': '#ff9f43',
    'medial_tibial_cartilage': '#00d2d3',
    'lateral_tibial_cartilage': '#54a0ff'
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
    
    # 0. Pre-cleanup: fill gaps to prevent overlap
    print("Running fill_label_gaps...")
    label_img = sitk.ReadImage(args.input)
    filled_img = fill_label_gaps(label_img, list(label_map.values()), closing_radius_mm=2.0)
    
    filled_path = os.path.join(args.output_dir, f"{base_name}_{args.track}_filled.nii.gz")
    sitk.WriteImage(filled_img, filled_path)
    
    # 1. Generate multi-label mesh via Surface Nets
    print("Extracting meshes using vtkSurfaceNets3D...")
    raw_mesh = extract_multilabel_surface(filled_path, label_map)
    
    # Save the RAW multi-label mesh for reference
    raw_mesh_path = os.path.join(args.output_dir, f"{base_name}_{args.track}_raw.obj")
    raw_mesh.save(raw_mesh_path)
    
    # Extract individual labels, then smooth, then decimate
    import pyvista as pv
    import numpy as np
    
    pl = pv.Plotter(off_screen=True)
    
    for label_name, label_id in label_map.items():
        if 'BoundaryLabels' not in raw_mesh.cell_data:
            print(f"Warning: BoundaryLabels not found in raw mesh for {label_name}. Skipping extraction.")
            continue
            
        labels = raw_mesh.cell_data['BoundaryLabels']
        mask = (labels[:, 0] == label_id) | (labels[:, 1] == label_id)
        
        if not np.any(mask):
            print(f"Skipping {label_name} — no voxels/cells found in mesh (likely unilateral).")
            continue
            
        # extract cells for this label
        sub_mesh = raw_mesh.extract_cells(mask)
        # extract_surface to get clean PolyData
        surf = sub_mesh.extract_surface(algorithm='dataset_surface')
        surf = surf.clean()
        
        # Extract largest connected component to drop any noise pinched off during meshing
        surf = surf.connectivity(extraction_mode='largest')
        
        # Decimate the mesh to prevent VTK from hanging during boolean operations on 500k+ triangles
        surf = surf.decimate(0.9)
        
        # Keep volume preservation natively through SurfaceNets
        # instead of independent label smoothing.
        
        comp_path = os.path.join(args.output_dir, f"{label_name}.obj")
        surf.save(comp_path)
        print(f"Saved individual mesh {label_name} to {comp_path}")
        
        # Add to plotter for colored export
        # We will add to plotter AFTER boolean resolution if applicable.
        pass

    # Re-build GLTF scene with potentially modified meshes
    for label_name in label_map.keys():
        comp_path = os.path.join(args.output_dir, f"{label_name}.obj")
        if os.path.exists(comp_path):
            surf = pv.read(comp_path)
            color = LABEL_COLORS.get(label_name, '#ffffff')
            pl.add_mesh(surf, color=color)

    gltf_path = os.path.join(args.output_dir, f"{base_name}_{args.track}_colored_scene.gltf")
    pl.export_gltf(gltf_path)
    print(f"Saved colored GLTF scene to {gltf_path}")

    # QA Inter-Region Overlap Check
    print("Running QA Inter-Region Overlap Check...")
    
    # Identify pairs to check based on track
    pairs_to_check = []
    quarantine_log = []
    if args.track == 'ct_bone':
        if 'femur_left' in label_map and 'tibia_left' in label_map:
            pairs_to_check.append(('femur_left', 'tibia_left'))
        if 'femur_right' in label_map and 'tibia_right' in label_map:
            pairs_to_check.append(('femur_right', 'tibia_right'))
        if 'femur_left' in label_map and 'patella_left' in label_map:
            pairs_to_check.append(('femur_left', 'patella_left'))
        if 'femur_right' in label_map and 'patella_right' in label_map:
            pairs_to_check.append(('femur_right', 'patella_right'))
    elif args.track == 'mri_cartilage':
        pairs_to_check.append(('femur_unknown', 'tibia_unknown'))
            
    for label1, label2 in pairs_to_check:
        mesh1_path = os.path.join(args.output_dir, f"{label1}.obj")
        mesh2_path = os.path.join(args.output_dir, f"{label2}.obj")
        
        if os.path.exists(mesh1_path) and os.path.exists(mesh2_path):
            m1 = pv.read(mesh1_path)
            m2 = pv.read(mesh2_path)
            
            try:
                col, n_contacts = m1.collision(m2)
                if n_contacts > 13000:
                    print(f"QA FAILED: Intersection between {label1} and {label2} has {n_contacts} intersecting faces (> 13000 threshold).")
                    print(f"Deleting corrupted meshes to prevent downstream usage.")
                    if os.path.exists(mesh1_path): os.remove(mesh1_path)
                    if os.path.exists(mesh2_path): os.remove(mesh2_path)
                    quarantine_log.append({
                        "bones": [label1, label2],
                        "intersection_faces": n_contacts,
                        "threshold": 13000,
                        "status": "deleted"
                    })
                elif n_contacts > 6000:
                    print(f"QA WARNING: Intersection between {label1} and {label2} has {n_contacts} intersecting faces (borderline flush contact).")
                    quarantine_log.append({
                        "bones": [label1, label2],
                        "intersection_faces": n_contacts,
                        "threshold": 6000,
                        "status": "warning"
                    })
                else:
                    print(f"QA PASSED: {label1} and {label2} intersect by {n_contacts} faces (acceptable margin)")
            except Exception as e:
                print(f"Warning: Could not compute collision for QA check: {e}")

    if quarantine_log:
        q_path = os.path.join(args.output_dir, "quarantine_log.json")
        if os.path.exists(q_path):
            with open(q_path, "r") as f:
                existing = json.load(f)
            quarantine_log = existing + quarantine_log
        with open(q_path, "w") as f:
            json.dump(quarantine_log, f, indent=4)

    # AR-ready export (mm->m + LPS->glTF Y-up transform, per-part color).
    # Runs after the QA overlap check so quarantined/deleted parts are excluded.
    from src.mesh.export_ar_glb import export_ar_glb
    ar_glb_path = export_ar_glb(args.output_dir, label_map, LABEL_COLORS, base_name, args.track)
    if ar_glb_path:
        print(f"Saved AR-ready GLB to {ar_glb_path}")
    else:
        print("Skipped AR-ready GLB export — no label parts found.")

if __name__ == "__main__":
    main()
