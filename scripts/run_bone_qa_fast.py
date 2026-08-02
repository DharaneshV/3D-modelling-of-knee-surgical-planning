"""
Fast-path bone QA (v3.2): validates restored label 1/3 (femur_unknown / tibia_unknown)
meshing against the 103 OAIZIB segmentation masks cached in meshes/oaizib_*/

Scope:
  - Only processes meshes/oaizib_* directories (103 cases, exactly the OAIZIB test set)
  - Each directory must have a *_mask.nii.gz (pre-existing CartiMorph output)
  - Runs ONLY run_meshing.py (no re-segmentation, no WSL inference)
  - Validates the restored label 1/3 path, not segmentation quality

Cache validity reasoning (confirmed before running):
  - Segmentation script changed between 4507f8a and HEAD only in:
    (1) temp dir naming (race-condition fix, no output effect)
    (2) RAI orientation pre-processing (introduced at 49a7f11, BEFORE synthesis commits)
  - All cached masks in meshes/oaizib_* were produced AFTER the RAI fix was in place
  - PIPELINE_VERSION bumped to v3.2, so cache entries (all tagged v3 or v3.1) are
    invalidated for any future full-pipeline runs -- this QA is mask-reuse only

Outputs: bone_qa_oaizib_v32.json + printed summary table
"""
import sys
import json
import subprocess
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path("d:/knee surgery model")
MESHES_DIR = ROOT / "meshes"
PYTHON = sys.executable

MIN_VERTICES = 500
MIN_BBOX_DIAG_MM = 40.0
MAX_BBOX_DIAG_MM = 250.0   # anything above 250mm is almost certainly a CT mask run through wrong track


def get_bone_metrics(mesh_path: Path) -> dict:
    if not mesh_path.exists():
        return {"exists": False, "vertices": 0, "bbox_diag_mm": 0.0, "status": "MISSING"}
    if mesh_path.stat().st_size == 0:
        return {"exists": True, "vertices": 0, "bbox_diag_mm": 0.0, "status": "EMPTY_FILE"}
    try:
        import pyvista as pv
        mesh = pv.read(str(mesh_path))
        n_verts = mesh.n_points
        bounds = mesh.bounds
        diag = np.sqrt(
            (bounds[1]-bounds[0])**2 +
            (bounds[3]-bounds[2])**2 +
            (bounds[5]-bounds[4])**2
        )
        status = "OK"
        if n_verts < MIN_VERTICES:
            status = f"DEGENERATE ({n_verts} verts)"
        elif diag < MIN_BBOX_DIAG_MM:
            status = f"TINY_BBOX ({diag:.1f}mm)"
        elif diag > MAX_BBOX_DIAG_MM:
            status = f"OVERSIZED ({diag:.1f}mm) -- possible wrong-track mask"
        return {"exists": True, "vertices": n_verts, "bbox_diag_mm": round(diag, 1), "status": status}
    except Exception as e:
        return {"exists": True, "vertices": 0, "bbox_diag_mm": 0.0, "status": f"READ_ERROR: {e}"}


def main():
    # Scope: only oaizib_* task directories with a cached mask
    oaizib_dirs = sorted(d for d in MESHES_DIR.iterdir()
                         if d.is_dir() and d.name.startswith("oaizib_"))

    mask_pairs = []
    for case_dir in oaizib_dirs:
        masks = list(case_dir.glob("*_mask.nii.gz"))
        if masks:
            mask_pairs.append((case_dir.name, masks[0]))

    print(f"OAIZIB bone QA — {len(mask_pairs)} cached masks (scope: oaizib_* dirs only)")
    print(f"Segmentation: unchanged (CartiMorph, RAI-corrected, post-49a7f11)")
    print(f"Pipeline version: v3.2 | Cache entries: all tagged v3/v3.1 (all stale, won't interfere)")
    print("=" * 80)

    results = []

    for case_id, mask_path in mask_pairs:
        qa_out_dir = MESHES_DIR / case_id / "qa_v32"
        qa_out_dir.mkdir(exist_ok=True)

        mesh_cmd = [
            PYTHON, str(ROOT / "scripts/run_meshing.py"),
            "--input", str(mask_path),
            "--output_dir", str(qa_out_dir),
            "--track", "mri_cartilage"
        ]
        result = subprocess.run(mesh_cmd, capture_output=True, text=True)
        # VTK writes INFO to stderr; treat as OK if output files exist
        meshing_ok = (qa_out_dir / "femur_unknown.obj").exists()

        femur_qa = get_bone_metrics(qa_out_dir / "femur_unknown.obj")
        tibia_qa  = get_bone_metrics(qa_out_dir / "tibia_unknown.obj")
        bone_qa_pass = femur_qa["status"] == "OK" and tibia_qa["status"] == "OK"

        entry = {
            "case_id": case_id,
            "meshing_ok": meshing_ok,
            "femur_vertices":      femur_qa["vertices"],
            "femur_bbox_diag_mm":  femur_qa["bbox_diag_mm"],
            "femur_status":        femur_qa["status"],
            "tibia_vertices":      tibia_qa["vertices"],
            "tibia_bbox_diag_mm":  tibia_qa["bbox_diag_mm"],
            "tibia_status":        tibia_qa["status"],
            "bone_qa":             "Pass" if bone_qa_pass else "FAIL",
        }
        results.append(entry)

        flag = "" if bone_qa_pass else "  *** FAIL ***"
        print(f"{case_id:<15}  F:{femur_qa['vertices']:>6} verts {femur_qa['bbox_diag_mm']:>6.1f}mm [{femur_qa['status']:<8}]  "
              f"T:{tibia_qa['vertices']:>6} verts {tibia_qa['bbox_diag_mm']:>6.1f}mm [{tibia_qa['status']:<8}]{flag}")

    # Summary
    total   = len(results)
    fails   = [r for r in results if r["bone_qa"] != "Pass"]
    f_diags = [r["femur_bbox_diag_mm"] for r in results if r["femur_bbox_diag_mm"] > 0]
    t_diags = [r["tibia_bbox_diag_mm"] for r in results if r["tibia_bbox_diag_mm"] > 0]

    print("\n" + "=" * 80)
    print(f"SUMMARY  ({total} OAIZIB cases)  Passed: {total - len(fails)}  Failed: {len(fails)}")
    if fails:
        print("FAILURES:")
        for f in fails:
            print(f"  {f['case_id']}: femur={f['femur_status']} | tibia={f['tibia_status']}")
    if f_diags:
        print(f"Femur bbox diagonal:  min={min(f_diags):.1f}  max={max(f_diags):.1f}  "
              f"mean={np.mean(f_diags):.1f}  p5={np.percentile(f_diags,5):.1f}  p95={np.percentile(f_diags,95):.1f} mm")
    if t_diags:
        print(f"Tibia bbox diagonal:  min={min(t_diags):.1f}  max={max(t_diags):.1f}  "
              f"mean={np.mean(t_diags):.1f}  p5={np.percentile(t_diags,5):.1f}  p95={np.percentile(t_diags,95):.1f} mm")

    out_path = ROOT / "bone_qa_oaizib_v32.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {out_path.name}")


if __name__ == "__main__":
    main()
