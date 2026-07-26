import os
import subprocess
import sys
from pathlib import Path

def run_case(case_name, input_file):
    print(f"========== RUNNING {case_name} ==========")
    task_mesh_dir = Path("meshes") / case_name
    task_mesh_dir.mkdir(exist_ok=True, parents=True)
    mask_output = task_mesh_dir / f"{case_name}_mask.nii.gz"
    
    python_exe = sys.executable
    
    # 1. Segment
    print(f"Segmenting {case_name}...")
    seg_cmd = [python_exe, "src/segmentation/run_mri_segmentation.py", "--input", str(input_file), "--output", str(mask_output)]
    res_seg = subprocess.run(seg_cmd, capture_output=True, text=True)
    if res_seg.returncode != 0:
        print(f"Segmentation failed for {case_name}:\n{res_seg.stderr}")
        return
    print(f"Segmentation done.")
    
    # 2. Meshing
    print(f"Meshing {case_name}...")
    mesh_cmd = [python_exe, "scripts/run_meshing.py", "--input", str(mask_output), "--output_dir", str(task_mesh_dir), "--track", "mri_cartilage"]
    res_mesh = subprocess.run(mesh_cmd, capture_output=True, text=True)
    
    print(f"Meshing STDOUT for {case_name}:")
    print(res_mesh.stdout)
    if res_mesh.stderr:
        print(f"Meshing STDERR for {case_name}:")
        print(res_mesh.stderr)
    print("=========================================\n")

if __name__ == "__main__":
    run_case("DU03", "data/mri&ct/DU03.nii.gz")
    run_case("DU06", "data/mri&ct/DU06.nii.gz")
