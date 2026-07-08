import os
import subprocess
import sys
import trimesh

def main():
    cases = ['STS_006', 'STS_035', 'STS_043', 'STS_051']
    os.environ['PYTHONPATH'] = '.'
    
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
            
        print(f"Meshing complete for {case}. Checking watertightness...")
        
        # Check output meshes
        for bone in ['femur', 'tibia', 'patella']:
            obj_path = os.path.join(out_dir, f"{bone}_decimated.obj")
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
