"""
profile_segmentation.py - Timed TotalSegmentator inference with GPU monitoring.
Runs segmentation on STS_006 and reports wall-clock time + captures stdout/stderr
for GPU engagement confirmation.
"""
import subprocess
import sys
import time
import os
from pathlib import Path

INPUT = "data/raw/case01_STS_006.nii.gz"
OUTPUT = "profile_seg_output/bone_mask.nii.gz"

os.makedirs("profile_seg_output", exist_ok=True)

python_exe = sys.executable

print("=" * 60)
print("PROFILING: TotalSegmentator inference (GPU mode)")
print(f"Input: {INPUT}")
print("=" * 60)

# Pre-run nvidia-smi snapshot
print("\n--- GPU state BEFORE inference ---")
r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
                     "--format=csv,noheader"], capture_output=True, text=True)
print(f"  {r.stdout.strip()}")

# Run TotalSegmentator with timing
print(f"\nStarting TotalSegmentator... (time.perf_counter)")
t0 = time.perf_counter()

result = subprocess.run(
    [python_exe, "src/segmentation/run_ct_segmentation.py",
     "--input", INPUT,
     "--output", OUTPUT],
    capture_output=True, text=True
)

elapsed = time.perf_counter() - t0

print(f"\n--- TotalSegmentator completed in {elapsed:.1f}s ---")
print(f"Return code: {result.returncode}")

# Print stdout/stderr for GPU engagement evidence
if result.stdout.strip():
    print(f"\nSTDOUT (last 2000 chars):\n{result.stdout[-2000:]}")
if result.stderr.strip():
    print(f"\nSTDERR (last 2000 chars):\n{result.stderr[-2000:]}")

# Post-run nvidia-smi snapshot
print("\n--- GPU state AFTER inference ---")
r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
                     "--format=csv,noheader"], capture_output=True, text=True)
print(f"  {r.stdout.strip()}")

# Verify output
if os.path.exists(OUTPUT):
    import SimpleITK as sitk
    import numpy as np
    img = sitk.ReadImage(OUTPUT)
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    
    print(f"\n--- Output verification ---")
    print(f"  Shape: {arr.shape}")
    print(f"  Labels present: {np.unique(arr)}")
    for label_id, name in [(1, "femur_left"), (2, "femur_right"), (3, "tibia_left"), 
                            (4, "tibia_right"), (5, "patella_left"), (6, "patella_right")]:
        count = np.sum(arr == label_id)
        vol_cm3 = count * voxel_vol / 1000.0
        print(f"  Label {label_id} ({name}): {count} voxels, {vol_cm3:.1f} cm3")
else:
    print(f"\nERROR: Output file {OUTPUT} not created!")

print(f"\n{'=' * 60}")
print(f"TOTAL SEGMENTATION TIME: {elapsed:.1f}s")
print(f"{'=' * 60}")
