import os
import sys
from pathlib import Path
# Ensure root is in PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import logging
import argparse
import numpy as np
import pyvista as pv
import vtk
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRATCH_DIR = Path('d:/knee surgery model/scratch')
VISUAL_MODEL_DIR = Path('d:/knee surgery model/models/visual')

def align_and_scale_bone(case_dir: Path, bone_type: str) -> Path:
    """
    Fits a generic reference bone to the patient's native cartilage for illustrative visualization.
    Uses a similarity transform (uniform scale derived from bounding box + rigid ICP).
    """
    logger.info(f"\n--- Visual Fitting {bone_type} for {case_dir.name} ---")
    
    mesh_dir = case_dir / "meshes" if (case_dir / "meshes").exists() else case_dir
    
    # Check laterality
    summary_path = case_dir / "laterality_summary.json"
    if not summary_path.exists() and (mesh_dir / "laterality_summary.json").exists():
        summary_path = mesh_dir / "laterality_summary.json"
    side = "right"
    if summary_path.exists():
        with open(summary_path) as f:
            side = json.load(f).get("laterality", "right")
    is_left = side == "left"
    
    # 1. Load Generic Bone and Generic Cartilage Proxy
    generic_bone_path = VISUAL_MODEL_DIR / f"generic_{bone_type}.obj"
    generic_proxy_path = VISUAL_MODEL_DIR / f"generic_{bone_type}_proxy.obj"
    
    if not generic_bone_path.exists() or not generic_proxy_path.exists():
        raise FileNotFoundError("Visual reference models missing. Run generate_visual_reference.py first.")
        
    generic_bone = pv.read(str(generic_bone_path))
    generic_proxy = pv.read(str(generic_proxy_path))
    
    # 2. Load Patient's Native Cartilage
    # Always load femoral cartilage to compute a stable global scale factor
    femoral_cart_path = mesh_dir / "femoral_cartilage.obj"
    if not femoral_cart_path.exists():
        logger.warning(f"Native femoral cartilage not found. Cannot determine scale. Skipping.")
        return None
    femoral_cart = pv.read(str(femoral_cart_path))
    
    # Load the target cartilage for ICP
    if bone_type == "femur":
        native_cart = femoral_cart
    else:
        medial_path = mesh_dir / "medial_tibial_cartilage.obj"
        lateral_path = mesh_dir / "lateral_tibial_cartilage.obj"
        
        cart_meshes = []
        if medial_path.exists():
            cart_meshes.append(pv.read(str(medial_path)))
        if lateral_path.exists():
            cart_meshes.append(pv.read(str(lateral_path)))
            
        if not cart_meshes:
            logger.warning("No native tibia cartilages found. Skipping.")
            return None
            
        native_cart = cart_meshes[0]
        if len(cart_meshes) > 1:
            native_cart = native_cart.merge(cart_meshes[1])
            
    # 3. Handle Laterality Reflection
    # Generic models are based on DU02 (RIGHT knee).
    # If patient is LEFT knee, reflect patient cartilage to RIGHT space for fitting.
    if is_left:
        logger.info("Reflecting native cartilage to RIGHT space for fitting...")
        native_cart.points[:, 0] = -native_cart.points[:, 0]
        femoral_cart.points[:, 0] = -femoral_cart.points[:, 0]
        
    # 4. Uniform Scale Factor
    # Always compute from Femur to avoid issues with missing lateral tibia cartilage
    generic_femur_proxy_path = VISUAL_MODEL_DIR / "generic_femur_proxy.obj"
    generic_femur_proxy = pv.read(str(generic_femur_proxy_path))
    
    generic_width = generic_femur_proxy.bounds[1] - generic_femur_proxy.bounds[0]
    native_width = femoral_cart.bounds[1] - femoral_cart.bounds[0]
    scale_factor = native_width / generic_width if generic_width > 0 else 1.0
    logger.info(f"Derived global scale factor (from femur): {scale_factor:.3f}")
    
    # Apply scale to generic bone and proxy (centered around proxy centroid)
    generic_centroid = np.mean(generic_proxy.points, axis=0)
    
    scale_transform = vtk.vtkTransform()
    scale_transform.Translate(generic_centroid)
    scale_transform.Scale(scale_factor, scale_factor, scale_factor)
    scale_transform.Translate(-generic_centroid[0], -generic_centroid[1], -generic_centroid[2])
    
    scaled_bone = generic_bone.copy()
    scaled_bone.transform(scale_transform, inplace=True)
    scaled_proxy = generic_proxy.copy()
    scaled_proxy.transform(scale_transform, inplace=True)
    
    # 5. Coarse Pre-Alignment (Centroid Translation)
    scaled_centroid = np.mean(scaled_proxy.points, axis=0)
    native_centroid = np.mean(native_cart.points, axis=0)
    translation = native_centroid - scaled_centroid
    
    coarse_transform = vtk.vtkTransform()
    coarse_transform.Translate(translation)
    
    scaled_proxy.transform(coarse_transform, inplace=True)
    
    # 6. Rigid ICP (Fine Alignment)
    logger.info("Running Rigid ICP (Scaled Proxy -> Native Cartilage)...")
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(scaled_proxy)
    icp.SetTarget(native_cart)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(100)
    icp.SetMaximumMeanDistance(1e-5)
    icp.Update()
    
    icp_transform = vtk.vtkTransform()
    icp_transform.SetMatrix(icp.GetMatrix())
    
    # The total transform mapping original generic geometry to native cartilage
    total_transform = vtk.vtkTransform()
    total_transform.PostMultiply()
    total_transform.Concatenate(scale_transform)
    total_transform.Concatenate(coarse_transform)
    total_transform.Concatenate(icp_transform)
    
    # 7. Apply total transform to Generic Bone
    final_bone = generic_bone.copy()
    final_bone.transform(total_transform, inplace=True)
    
    # 7b. Clip bone to its relevant articular half using native cartilage centroid as the clip plane.
    # This prevents the two full-length generic bones from visually merging into a single green blob —
    # we only need the distal femur (condyles) and proximal tibia (plateau) for visualization.
    # The clip plane is set at the centroid of the cartilage that was used for ICP.
    cart_centroid = np.mean(native_cart.points, axis=0)
    if bone_type == "femur":
        # Keep the distal (lower-Y) half — the condylar end — clip away the shaft above the cartilage
        final_bone = final_bone.clip(normal=(0, 1, 0), origin=cart_centroid, invert=False)
    else:
        # Keep the proximal (upper-Y) half — the plateau end — clip away the shaft below the cartilage
        final_bone = final_bone.clip(normal=(0, -1, 0), origin=cart_centroid, invert=False)
    
    if final_bone.n_cells == 0:
        logger.warning(f"Half-bone clip produced empty mesh for {bone_type} — falling back to full bone.")
        final_bone = generic_bone.copy()
        final_bone.transform(total_transform, inplace=True)
    else:
        logger.info(f"Half-bone clip retained {final_bone.n_cells} cells (articular end only).")
    
    # 8. Visual Boolean Clip (bone vs cartilage interface)
    logger.info("Running visual boolean clipping...")
    from backend.mesh_processing.boolean_resolution import resolve_bone_cartilage_boundary
    final_bone, route = resolve_bone_cartilage_boundary(final_bone, native_cart)
    logger.info(f"Clipping finished via: {route}")
    
    # If left, reflect back to patient's true LEFT space!
    if is_left:
        logger.info("Reflecting aligned bone back to LEFT space...")
        final_bone.points[:, 0] = -final_bone.points[:, 0]
        
    out_path = mesh_dir / f"visual_{bone_type}.obj"
    final_bone.save(str(out_path))
    logger.info(f"Successfully saved visual mesh to {out_path.name}")
    logger.warning("CLINICAL USAGE CONSTRAINT: This mesh is an illustrative proxy and MUST NOT be used for quantitative analysis or surgical planning.")
    
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Generic Bone Visual Fitting Tool")
    parser.add_argument("case_id", help="The case ID to process (e.g., DU03)")
    args = parser.parse_args()
    
    case_dir = SCRATCH_DIR / args.case_id
    if not case_dir.exists():
        logger.error(f"Case directory {case_dir} does not exist.")
        sys.exit(1)
        
    align_and_scale_bone(case_dir, "femur")
    align_and_scale_bone(case_dir, "tibia")
    logger.info("\nVisual fitting complete for both bones.")

if __name__ == "__main__":
    main()
