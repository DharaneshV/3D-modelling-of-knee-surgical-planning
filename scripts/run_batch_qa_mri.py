import os
import sys
import glob
import json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.pipeline_runner import _execute_pipeline

# Bone mesh sanity floors (native CartiMorph label 1/3 geometry, v3.2+)
MIN_VERTICES = 500          # fewer than this suggests a degenerate/collapsed label
MIN_BBOX_DIAG_MM = 40.0    # femur/tibia bounding-box diagonal should be > 40 mm

def get_bone_metrics(mesh_path: Path) -> dict:
    """Return vertex count and bounding-box diagonal (mm) for a mesh, or error info."""
    if not mesh_path.exists():
        return {"exists": False, "vertices": 0, "bbox_diag_mm": 0.0, "status": "MISSING"}
    if mesh_path.stat().st_size == 0:
        return {"exists": True, "vertices": 0, "bbox_diag_mm": 0.0, "status": "EMPTY_FILE"}
    try:
        import pyvista as pv
        mesh = pv.read(str(mesh_path))
        n_verts = mesh.n_points
        bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        diag = np.sqrt(
            (bounds[1]-bounds[0])**2 +
            (bounds[3]-bounds[2])**2 +
            (bounds[5]-bounds[4])**2
        )
        bone_status = "OK"
        if n_verts < MIN_VERTICES:
            bone_status = f"DEGENERATE (only {n_verts} verts)"
        elif diag < MIN_BBOX_DIAG_MM:
            bone_status = f"TINY_BBOX ({diag:.1f}mm diagonal)"
        return {"exists": True, "vertices": n_verts, "bbox_diag_mm": round(diag, 1), "status": bone_status}
    except Exception as e:
        return {"exists": True, "vertices": 0, "bbox_diag_mm": 0.0, "status": f"READ_ERROR: {e}"}

def main():
    oaizib_files = sorted(glob.glob(r"d:\knee surgery model\data\oaizib\imagesTs\*.nii.gz"))
    
    results = []
    
    for file in oaizib_files:
        case_id = Path(file).name.split("_0000.nii.gz")[0]
        if case_id == Path(file).name:
            case_id = Path(file).name.replace(".nii.gz", "")
            
        print(f"==================================================")
        print(f"Running pipeline on {case_id}...")
        
        pipeline_status = "Pass"
        error_msg = ""
        try:
            _execute_pipeline(case_id, file, "MRI")
        except Exception as e:
            pipeline_status = "Fail"
            error_msg = str(e)
            
        # Geometry QA for native MRI bone meshes (labels 1 and 3)
        mesh_dir = Path(r"d:\knee surgery model\meshes") / case_id
        
        femur_qa = get_bone_metrics(mesh_dir / "femur_unknown.obj")
        tibia_qa = get_bone_metrics(mesh_dir / "tibia_unknown.obj")
        
        # Overall bone QA: pass only if both meshes exist, have enough verts, and reasonable size
        bone_qa_pass = (
            femur_qa["status"] == "OK" and
            tibia_qa["status"] == "OK"
        )
        
        results.append({
            "case_id": case_id,
            "pipeline_status": pipeline_status,
            "error": error_msg,
            "femur_vertices": femur_qa["vertices"],
            "femur_bbox_diag_mm": femur_qa["bbox_diag_mm"],
            "femur_bone_status": femur_qa["status"],
            "tibia_vertices": tibia_qa["vertices"],
            "tibia_bbox_diag_mm": tibia_qa["bbox_diag_mm"],
            "tibia_bone_status": tibia_qa["status"],
            "bone_qa": "Pass" if bone_qa_pass else "FAIL",
        })
        
        print(f"  Femur: {femur_qa['status']} | {femur_qa['vertices']} verts | {femur_qa['bbox_diag_mm']}mm bbox diag")
        print(f"  Tibia: {tibia_qa['status']} | {tibia_qa['vertices']} verts | {tibia_qa['bbox_diag_mm']}mm bbox diag")
        print(f"  Pipeline: {pipeline_status} | Bone QA: {'Pass' if bone_qa_pass else 'FAIL'}")
        
    # --- Summary Report ---
    total = len(results)
    pipeline_fails = sum(1 for r in results if r["pipeline_status"] != "Pass")
    bone_qa_fails  = sum(1 for r in results if r["bone_qa"] != "Pass")

    print("\n\n" + "="*100)
    print("BATCH QA REPORT (v3.2 — Native MRI Bone, Labels 1/3)")
    print("="*100)
    col = f"{'Case ID':<20} | {'Pipeline':<8} | {'Bone QA':<8} | {'F-Verts':<8} | {'F-BBox':<8} | {'T-Verts':<8} | {'T-BBox':<8} | Notes"
    print(col)
    print("-" * 100)
    for res in results:
        notes = ""
        if res["femur_bone_status"] != "OK":
            notes += f"FEMUR:{res['femur_bone_status']} "
        if res["tibia_bone_status"] != "OK":
            notes += f"TIBIA:{res['tibia_bone_status']} "
        if res["error"]:
            notes += f"ERR:{res['error'][:40]}"
        print(
            f"{res['case_id']:<20} | {res['pipeline_status']:<8} | {res['bone_qa']:<8} | "
            f"{res['femur_vertices']:<8} | {res['femur_bbox_diag_mm']:<8} | "
            f"{res['tibia_vertices']:<8} | {res['tibia_bbox_diag_mm']:<8} | {notes}"
        )
    
    print(f"\nSummary: {total} cases | {pipeline_fails} pipeline failures | {bone_qa_fails} bone QA failures")

    # Bounding box distribution summary
    valid_f_diags = [r["femur_bbox_diag_mm"] for r in results if r["femur_bbox_diag_mm"] > 0]
    valid_t_diags = [r["tibia_bbox_diag_mm"] for r in results if r["tibia_bbox_diag_mm"] > 0]
    if valid_f_diags:
        print(f"Femur bbox diagonal: min={min(valid_f_diags):.1f}mm  max={max(valid_f_diags):.1f}mm  mean={np.mean(valid_f_diags):.1f}mm")
    if valid_t_diags:
        print(f"Tibia bbox diagonal: min={min(valid_t_diags):.1f}mm  max={max(valid_t_diags):.1f}mm  mean={np.mean(valid_t_diags):.1f}mm")
        
    with open("batch_qa_mri_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nFull results saved to batch_qa_mri_results.json")
        
if __name__ == "__main__":
    main()
