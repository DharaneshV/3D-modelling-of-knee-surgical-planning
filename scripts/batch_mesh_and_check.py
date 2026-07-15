import os
import subprocess
import sys
import trimesh

def main():
    import glob
    case_paths = glob.glob('data/ct_knee/case01_STS_*_bone_mask.nii.gz')
    cases = [os.path.basename(p).replace('case01_', '').replace('_bone_mask.nii.gz', '') for p in case_paths]
    
    for case in cases:
        print(f"\n{'='*50}\nMeshing {case}\n{'='*50}")
        mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
        out_dir = f"data/ct_knee/{case}_meshes"
        
        cmd = [
            sys.executable, "scripts/run_meshing.py",
            "--input", mask_path,
            "--output_dir", out_dir,
            "--track", "ct_bone"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to mesh {case}:\n{result.stderr}")
            continue
            
        print(f"Meshing complete for {case}. Output:\n{result.stdout}")
        
        # Check output meshes
        for bone in ['femur_left', 'tibia_left', 'patella_left', 'femur_right', 'tibia_right', 'patella_right']:
            obj_path = os.path.join(out_dir, f"{bone}.obj")
            if not os.path.exists(obj_path):
                print(f"Warning: {obj_path} not found.")
                continue
                
            mesh = trimesh.load(obj_path)
            is_watertight = mesh.is_watertight
            print(f"  {bone.capitalize()}: Watertight = {is_watertight} (Faces: {len(mesh.faces)}, Vertices: {len(mesh.vertices)})")
            
            if not is_watertight:
                print(f"  WARNING: {bone} mesh is NOT watertight!")

if __name__ == "__main__":
    main()
