import os
import subprocess
import sys
from pathlib import Path

def run_ood_case(input_file):
    case_name = "test_ood_mri"
    print(f"========== RUNNING {case_name} ==========")
    task_mesh_dir = Path("meshes") / case_name
    task_mesh_dir.mkdir(exist_ok=True, parents=True)
    mask_output = task_mesh_dir / f"{case_name}_mask.nii.gz"
    
    python_exe = sys.executable
    
    # Run the meshing directly on the QA gate, or actually through pipeline_runner
    print(f"Running Modality Detector on {case_name}...")
    detect_cmd = [python_exe, "-c", f"from backend.modality_detector import detect_modality; print(detect_modality('{input_file}'))"]
    res_detect = subprocess.run(detect_cmd, capture_output=True, text=True)
    
    print(f"Modality Detection STDOUT for {case_name}:")
    print(res_detect.stdout)
    if res_detect.stderr:
        print(f"Modality Detection STDERR for {case_name}:")
        print(res_detect.stderr)
    print("=========================================\n")

if __name__ == "__main__":
    run_ood_case("data/PCIR_Knee_MRI.nii.gz")
