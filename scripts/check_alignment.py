import pyvista as pv
from pathlib import Path
import sys

def check_alignment(case_name):
    print(f"\n--- Alignment Check for {case_name} ---")
    mesh_dir = Path("meshes") / case_name
    bone_path = mesh_dir / "femur_unknown.obj"
    cartilage_path = mesh_dir / "femoral_cartilage.obj"
    
    if not bone_path.exists() or not cartilage_path.exists():
        print("Meshes not found.")
        return
        
    bone = pv.read(str(bone_path))
    cartilage = pv.read(str(cartilage_path))
    
    print(f"Bone Bounds: {bone.bounds}")
    print(f"Cartilage Bounds: {cartilage.bounds}")
    
    bone_center = bone.center
    cart_center = cartilage.center
    
    print(f"Bone Center: {bone_center}")
    print(f"Cartilage Center: {cart_center}")
    
    dist = ((bone_center[0] - cart_center[0])**2 + 
            (bone_center[1] - cart_center[1])**2 + 
            (bone_center[2] - cart_center[2])**2)**0.5
    print(f"Distance between centers: {dist:.2f} mm")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_alignment(sys.argv[1])
    else:
        check_alignment("DU03")
        check_alignment("DU06")
