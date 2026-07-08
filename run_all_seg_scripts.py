import subprocess
import os

cases = ['STS_006', 'STS_035', 'STS_043', 'STS_051']
python_exe = r".\venv\Scripts\python"

for case in cases:
    print(f"Running segmentation for {case}...")
    ct_path = f"data/ct_knee/case01_{case}_cropped.nii.gz"
    mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
    
    cmd = [
        python_exe, "scripts/bone_segmentation.py",
        "--input", ct_path,
        "--output", mask_path,
        "--hu-threshold", "400",
        "--hu-threshold-pass2", "200"
    ]
    subprocess.run(cmd, check=True)
    print(f"Completed {case}")

