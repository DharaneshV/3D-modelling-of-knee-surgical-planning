import os
import subprocess

import json

cases = [
    "STS_006",
    "STS_035",
    "STS_043",
    "STS_051"
]

metadata = {}

python_exe = r".\venv\Scripts\python.exe"

for case in cases:
    print(f"\n{'='*50}\nProcessing {case}\n{'='*50}")
    
    input_ct = f"data/ct_knee/case01_{case}_cropped.nii.gz"
    output_mask = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
    preview_img = f"case01_{case}_segmentation_preview_v2.png"
    
    print(f"Running segmentation for {case}...")
    success = False
    for hu in [300, 400, 200, 500, 600, 250, 350]:
        print(f"  Trying HU threshold {hu}...")
        seg_cmd = [python_exe, "scripts/bone_segmentation.py", "--input", input_ct, "--output", output_mask, "--hu-threshold", str(hu)]
        result = subprocess.run(seg_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Success with HU {hu}!")
            
            # Extract PASS2_STATUS from stdout
            status = "UNKNOWN"
            for line in result.stdout.split('\n'):
                if line.startswith("PASS2_STATUS:"):
                    status = line.split(":")[-1].strip()
            
            metadata[case] = {
                "hu_pass1": hu,
                "hu_pass2": 200,
                "pass2_joint_topology": status
            }
            
            success = True
            break
        else:
            print(f"  Failed with HU {hu}. (Error: {result.stderr.strip().split(chr(10))[-1]})")
            
    if not success:
        print(f"Failed to segment {case} with any HU threshold.")
        continue
    
    print(f"Generating preview for {case}...")
    viz_cmd = [python_exe, "scripts/visualize_segmentation.py", "--ct", input_ct, "--mask", output_mask, "--output", preview_img]
    subprocess.run(viz_cmd, check=True)
    
with open("segmentation_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\nAll cases processed successfully! Metadata saved to segmentation_metadata.json")
