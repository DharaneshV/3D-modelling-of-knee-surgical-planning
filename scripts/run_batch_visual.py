"""
DEPRECATED as of PIPELINE_VERSION v3.2.

This script invokes src/synthesis/bone_from_mri.py (now archived) and reads
visual_femur.obj / visual_tibia.obj — neither of which are produced by the
updated pipeline. Bone is now meshed natively from CartiMorph labels 1/3 as
femur_unknown.obj / tibia_unknown.obj. Use run_batch_qa_mri.py instead.

--- Original docstring below ---
Batch runner: for each OAIZIB case in labelsTs
  1. Run run_meshing.py to extract cartilage meshes (if not already done)
  2. Run bone_from_mri.align_and_scale_bone for femur + tibia  [ARCHIVED]
  3. Report femur/tibia bounding-box gap and clipping route     [ARCHIVED]

Progress is written to batch_visual_run.jsonl so it can be resumed on crash.
"""
import subprocess
import sys
import json
import time
from pathlib import Path

PYTHON = sys.executable
ROOT = Path("d:/knee surgery model")
LABELS_DIR = ROOT / "data/oaizib/labelsTs"
SCRATCH_DIR = ROOT / "scratch"
LOG_FILE = ROOT / "batch_visual_run.jsonl"

# Load completed cases
done = set()
if LOG_FILE.exists():
    for line in LOG_FILE.read_text().splitlines():
        try:
            r = json.loads(line)
            if r.get("status") in ("ok", "failed"):
                done.add(r["case_id"])
        except:
            pass

cases = sorted([f.stem.replace(".nii", "") for f in LABELS_DIR.glob("oaizib_*.nii.gz")])
remaining = [c for c in cases if c not in done]
print(f"Total cases: {len(cases)}, already done: {len(done)}, remaining: {len(remaining)}")

def log(record):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

for i, case_id in enumerate(remaining):
    case_dir = SCRATCH_DIR / case_id
    mesh_dir = case_dir / "meshes"
    label_path = LABELS_DIR / f"{case_id}.nii.gz"
    
    print(f"\n[{i+1}/{len(remaining)}] {case_id}")
    t0 = time.time()
    
    # Step 1: Meshing (if cartilage mesh doesn't exist yet)
    if not (mesh_dir / "femoral_cartilage.obj").exists():
        r = subprocess.run(
            [PYTHON, "scripts/run_meshing.py",
             "--input", str(label_path),
             "--output_dir", str(mesh_dir),
             "--track", "mri_cartilage"],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        if r.returncode != 0:
            print(f"  MESHING FAILED: {r.stderr[-500:]}")
            log({"case_id": case_id, "status": "failed", "step": "meshing", "err": r.stderr[-300:]})
            continue
        print(f"  Meshing OK ({time.time()-t0:.1f}s)")
    else:
        print(f"  Meshing already done, skipping.")
    
    # Step 2: Visual bone fitting
    r2 = subprocess.run(
        [PYTHON, "src/synthesis/bone_from_mri.py", case_id],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    if r2.returncode != 0:
        print(f"  BONE FIT FAILED: {r2.stderr[-400:]}")
        log({"case_id": case_id, "status": "failed", "step": "bone_fit", "err": r2.stderr[-300:]})
        continue
    
    # Parse clipping routes from stdout
    routes = [line for line in r2.stdout.splitlines() if "Clipping finished" in line]
    
    # Step 3: Quick bone-gap check (bounding-box Y proximity)
    try:
        import pyvista as pv
        import numpy as np
        f = pv.read(str(mesh_dir / "visual_femur.obj"))
        t = pv.read(str(mesh_dir / "visual_tibia.obj"))
        femur_y_min = f.bounds[2]
        tibia_y_max = t.bounds[3]
        gap_mm = femur_y_min - tibia_y_max
    except Exception as e:
        gap_mm = None
        print(f"  Gap check failed: {e}")
    
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — gap={gap_mm:.1f}mm — routes: {routes}")
    log({
        "case_id": case_id,
        "status": "ok",
        "elapsed_s": round(elapsed, 1),
        "bone_gap_mm": round(gap_mm, 1) if gap_mm is not None else None,
        "clipping_routes": routes
    })

print("\nBatch run complete.")
