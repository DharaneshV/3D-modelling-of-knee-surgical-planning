import os
import sys
import glob
import subprocess
from pathlib import Path

def run_batch():
    raw_dir = Path("data/raw")
    out_dir = Path("outputs_bilateral")
    log_dir = Path("logs")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    cases = sorted(glob.glob(str(raw_dir / "case01_STS_*.nii.gz")))
    total = len(cases)
    
    print(f"Starting Phase 6 Batch Run for {total} cases...")
    
    for i, ct_path in enumerate(cases, 1):
        case_id = Path(ct_path).name.replace("case01_", "").replace(".nii.gz", "")
        print(f"[{i}/{total}] Processing {case_id}...")
        
        mask_path = out_dir / case_id / "masks" / "bone_mask.nii.gz"
        mesh_dir = out_dir / case_id / "meshes"
        log_path = log_dir / f"{case_id}.log"
        
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_dir.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, "w") as log_f:
            def run_and_stream(cmd, title):
                print(f"--- {title} ---")
                log_f.write(f"--- {title} ---\n")
                log_f.flush()
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_f.write(line)
                    log_f.flush()
                process.wait()
                return process.returncode

            cmd_seg = [
                sys.executable, "src/segmentation/run_ct_segmentation.py",
                "--input", ct_path,
                "--output", str(mask_path)
            ]
            
            if run_and_stream(cmd_seg, "SEGMENTATION") != 0:
                print(f"  -> Segmentation failed for {case_id}. See {log_path}")
                continue
                
            cmd_mesh = [
                sys.executable, "scripts/run_meshing.py",
                "--input", str(mask_path),
                "--output_dir", str(mesh_dir),
                "--track", "ct_bone"
            ]
            
            if run_and_stream(cmd_mesh, "MESHING") != 0:
                print(f"  -> Meshing failed for {case_id}. See {log_path}")
            else:
                print(f"  -> Success.")
                
    print("\nBatch processing complete. Running QA Report...")
    subprocess.run([
        sys.executable, "scripts/batch_qa_report.py",
        "--outputs-dir", str(out_dir),
        "--qa-json", str(out_dir / "batch_qa.json")
    ])

if __name__ == "__main__":
    run_batch()
