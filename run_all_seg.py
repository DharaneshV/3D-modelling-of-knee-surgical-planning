import subprocess
import os

cases = ['STS_006', 'STS_035', 'STS_043', 'STS_051']
python_exe = r".\venv\Scripts\python"

for case in cases:
    print(f"Running segmentation for {case}...")
    in_file = f"data/ct_knee/case01_{case}.nii.gz"
    
    cmd = [python_exe, "-m", "src.segmentation.run_bone_segmentation", "--input", in_file, "--output_dir", "data/ct_knee"]
    subprocess.run(cmd, check=True)
    print(f"Completed {case}")

