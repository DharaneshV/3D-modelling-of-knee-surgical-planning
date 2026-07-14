"""
profile_pipeline.py — Per-stage timing of the full post-segmentation pipeline.

Runs on an existing segmentation mask (skips TotalSegmentator) and reports
wall-clock time for every sub-stage: fill_label_gaps, SurfaceNets extraction,
per-bone label splitting, OBJ I/O, QA collision checks, report generation
(volumes, JSW, alignment, sizing), and PDF generation.

Usage:
    python profile_pipeline.py --mask outputs/STS_006/masks/bone_mask.nii.gz --track ct_bone
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import SimpleITK as sitk
import numpy as np

def timed(label):
    """Context manager that prints elapsed time for a block."""
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start
            timings.append((label, self.elapsed))
            print(f"  [{self.elapsed:7.2f}s] {label}")
    return Timer()

timings = []

LABEL_MAPS = {
    "ct_bone": {
        "femur_left": 1, "femur_right": 2,
        "tibia_left": 3, "tibia_right": 4,
        "patella_left": 5, "patella_right": 6
    },
    "mri_cartilage": {
        "femur_unknown": 1,
        "femoral_cartilage": 2,
        "tibia_unknown": 3,
        "medial_tibial_cartilage": 4,
        "lateral_tibial_cartilage": 5
    }
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True, help="Path to segmentation mask")
    parser.add_argument("--track", required=True, choices=["ct_bone", "mri_cartilage"])
    parser.add_argument("--scan", default=None, help="Path to original scan (for report gen)")
    args = parser.parse_args()

    output_dir = "profile_output"
    os.makedirs(output_dir, exist_ok=True)
    label_map = LABEL_MAPS[args.track]
    base_name = os.path.basename(args.mask).replace(".nii.gz", "").replace(".nii", "")

    pipeline_start = time.perf_counter()
    print("=" * 60)
    print(f"PROFILING PIPELINE: {args.mask} (track={args.track})")
    print("=" * 60)

    # ── Stage 1: Read mask ──────────────────────────────────────
    with timed("1. Read segmentation mask (sitk.ReadImage)"):
        label_img = sitk.ReadImage(args.mask)
        arr = sitk.GetArrayFromImage(label_img)
        print(f"     Shape: {arr.shape}, Labels present: {np.unique(arr)}")

    # ── Stage 2: fill_label_gaps ────────────────────────────────
    with timed("2. fill_label_gaps (binary_fill_holes)"):
        from src.mesh.processing import fill_label_gaps
        filled_img = fill_label_gaps(label_img, list(label_map.values()), closing_radius_mm=2.0)

    # ── Stage 3: Write filled mask ──────────────────────────────
    filled_path = os.path.join(output_dir, f"{base_name}_{args.track}_filled.nii.gz")
    with timed("3. Write filled mask to disk"):
        sitk.WriteImage(filled_img, filled_path)

    # ── Stage 4: SurfaceNets extraction ─────────────────────────
    with timed("4. SurfaceNets3D extraction (VTK)"):
        from src.mesh.surface_nets import extract_multilabel_surface
        raw_mesh = extract_multilabel_surface(filled_path, label_map)
        print(f"     Total cells: {raw_mesh.n_cells}, Total points: {raw_mesh.n_points}")

    # ── Stage 5: Write raw unified mesh ─────────────────────────
    raw_mesh_path = os.path.join(output_dir, f"{base_name}_{args.track}_raw.obj")
    with timed("5. Write raw unified OBJ to disk"):
        raw_mesh.save(raw_mesh_path)
        fsize = os.path.getsize(raw_mesh_path) / (1024 * 1024)
        print(f"     File size: {fsize:.1f} MB")

    # ── Stage 6: Per-bone label splitting ───────────────────────
    import pyvista as pv

    bone_meshes = {}
    with timed("6. Per-bone label splitting (total)"):
        for label_name, label_id in label_map.items():
            t0 = time.perf_counter()
            if 'BoundaryLabels' not in raw_mesh.cell_data:
                continue
            labels = raw_mesh.cell_data['BoundaryLabels']
            mask = (labels[:, 0] == label_id) | (labels[:, 1] == label_id)
            if not np.any(mask):
                print(f"     {label_name}: skipped (no cells)")
                continue
            sub_mesh = raw_mesh.extract_cells(mask)
            surf = sub_mesh.extract_surface(algorithm='dataset_surface')
            surf = surf.clean()
            surf = surf.connectivity(extraction_mode='largest')
            bone_meshes[label_name] = surf
            dt = time.perf_counter() - t0
            print(f"     {label_name}: {surf.n_cells} faces, {dt:.2f}s")

    # ── Stage 7: Write individual OBJ files ─────────────────────
    with timed("7. Write individual OBJ files (total)"):
        for label_name, surf in bone_meshes.items():
            t0 = time.perf_counter()
            path = os.path.join(output_dir, f"{label_name}.obj")
            surf.save(path)
            dt = time.perf_counter() - t0
            fsize = os.path.getsize(path) / (1024 * 1024)
            print(f"     {label_name}: {fsize:.1f} MB, {dt:.2f}s")

    # ── Stage 8: QA collision checks ────────────────────────────
    pairs = []
    if args.track == 'ct_bone':
        for side in ['left', 'right']:
            if f'femur_{side}' in bone_meshes and f'tibia_{side}' in bone_meshes:
                pairs.append((f'femur_{side}', f'tibia_{side}'))
    elif args.track == 'mri_cartilage':
        if 'femur_unknown' in bone_meshes and 'tibia_unknown' in bone_meshes:
            pairs.append(('femur_unknown', 'tibia_unknown'))

    with timed("8. QA collision checks (total)"):
        for l1, l2 in pairs:
            t0 = time.perf_counter()
            m1 = pv.read(os.path.join(output_dir, f"{l1}.obj"))
            m2 = pv.read(os.path.join(output_dir, f"{l2}.obj"))
            try:
                col, n = m1.collision(m2)
                dt = time.perf_counter() - t0
                print(f"     {l1} vs {l2}: {n} contacts, {dt:.2f}s")
            except Exception as e:
                dt = time.perf_counter() - t0
                print(f"     {l1} vs {l2}: error ({e}), {dt:.2f}s")

    # ── Stage 9: Report generation (clinical metrics) ───────────
    with timed("9. Report generation (metrics + JSON)"):
        from backend.report_generator import generate_report_data
        scan_path = args.scan or args.mask  # fallback
        data_dict = generate_report_data("PROFILE_TEST", "CT", scan_path, output_dir, args.mask)

    # ── Stage 10: PDF generation ────────────────────────────────
    with timed("10. PDF generation"):
        from backend.report_generator import generate_pdf
        generate_pdf(data_dict, output_dir, "PROFILE_TEST")

    # ── Summary ─────────────────────────────────────────────────
    total = time.perf_counter() - pipeline_start
    print("\n" + "=" * 60)
    print("TIMING SUMMARY (post-segmentation pipeline)")
    print("=" * 60)
    for label, elapsed in timings:
        pct = (elapsed / total) * 100
        bar = "#" * int(pct / 2)
        print(f"  {elapsed:7.2f}s ({pct:5.1f}%) | {bar:<50} | {label}")
    print(f"  {'─' * 7}  {'─' * 7}")
    print(f"  {total:7.2f}s (100.0%) | TOTAL (excludes TotalSegmentator)")
    print("=" * 60)

if __name__ == "__main__":
    main()
