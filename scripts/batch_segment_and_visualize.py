import os
import subprocess
import sys

def main():
    cases = ['STS_006', 'STS_035', 'STS_043', 'STS_051']
    thresholds = [300, 350, 400, 450, 500, 550, 600]
    
    for case in cases:
        print(f"\n{'='*50}\nProcessing {case}\n{'='*50}")
        ct_path = f"data/ct_knee/case01_{case}_cropped.nii.gz"
        mask_path = f"data/ct_knee/case01_{case}_bone_mask.nii.gz"
        preview_path = fr"C:\Users\dhara\.gemini\antigravity-ide\brain\785fa445-04f0-4094-bfcb-cc61a87bf792\case01_{case}_segmentation_preview_v4.png"
        
        success = False
        for thresh in thresholds:
            print(f"Trying Pass 1 threshold: {thresh} HU")
            cmd = [
                sys.executable, "scripts/bone_segmentation.py",
                "--input", ct_path,
                "--output", mask_path,
                "--hu-threshold", str(thresh),
                "--hu-threshold-pass2", "200"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"SUCCESS with Pass 1 threshold {thresh} HU.")
                success = True
                break
            else:
                print(f"Failed with {thresh} HU. Error snippet:")
                # print last 3 lines of stderr
                print("\n".join(result.stderr.strip().split('\n')[-3:]))
                
        if not success:
            print(f"FAILED to segment {case} at any threshold.")
            continue
            
        print(f"Generating visualization for {case}...")
        vis_cmd = [
            sys.executable, "scripts/visualize_segmentation.py",
            "--ct", ct_path,
            "--mask", mask_path,
            "--output", preview_path
        ]
        vis_result = subprocess.run(vis_cmd, capture_output=True, text=True)
        if vis_result.returncode != 0:
            print(f"Failed to visualize {case}:\n{vis_result.stderr}")
        else:
            print(f"Saved visualization to {preview_path}")

if __name__ == "__main__":
    main()
