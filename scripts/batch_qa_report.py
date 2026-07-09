"""
batch_qa_report.py — Automated QA across all processed cases in outputs/

Runs per-case checks (watertight, volume sanity) and generates a summary table
suitable for the methodology section of the project report.

Usage:
    python scripts/batch_qa_report.py [--outputs-dir outputs] [--qa-json outputs/batch_qa.json]
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import SimpleITK as sitk
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# QA Thresholds (Expanded for TotalSegmentator V2 full-shaft inclusion)
# ─────────────────────────────────────────────────────────────────────────────
BONE_LABELS = {1: "femur", 2: "tibia", 3: "patella"}
VOLUME_BOUNDS = {
    "femur":   (80_000,  800_000),  # V2 can capture a lot of the shaft
    "tibia":   (60_000,  600_000),  # V2 can capture a lot of the shaft
    "patella": (5_000,   50_000),
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_symbol(is_pass: bool) -> str:
    # Use ASCII fallback for cross to avoid Windows cp1252 charmap crashes
    return "✓" if is_pass else "fail"

def check_watertight(obj_path: str) -> tuple[bool, int, int, int]:
    """Returns (is_watertight, boundary_edges, vertex_count, face_count)."""
    verts, faces = 0, []
    with open(obj_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if parts[0] == 'v': verts += 1
            elif parts[0] == 'f':
                face = [int(p.split('/')[0]) - 1 for p in parts[1:4]]
                faces.append(face)

    edge_count = {}
    for face in faces:
        for i in range(3):
            e = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary = sum(1 for c in edge_count.values() if c == 1)
    return boundary == 0, boundary, verts, len(faces)


# ─────────────────────────────────────────────────────────────────────────────
# Per-case QA
# ─────────────────────────────────────────────────────────────────────────────

def qa_case(case_dir: Path) -> dict:
    case_id = case_dir.name
    result  = {"case_id": case_id, "pass": True, "flags": [], "bones": {}, "z_extent_mm": 0}

    # 1. Mask-level volumetric check
    mask_path = case_dir / "masks" / "bone_mask.nii.gz"
    if not mask_path.exists():
        result["pass"] = False
        result["flags"].append("MISSING: bone_mask.nii.gz")
        return result

    try:
        mask_img  = sitk.ReadImage(str(mask_path))
        mask_arr  = sitk.GetArrayFromImage(mask_img)
        spacing   = mask_img.GetSpacing()
        size      = mask_img.GetSize()
        voxel_vol = spacing[0] * spacing[1] * spacing[2]
        
        result["z_extent_mm"] = round(size[2] * spacing[2])

        for label_val, bone_name in BONE_LABELS.items():
            voxels = int(np.sum(mask_arr == label_val))
            vol_mm3 = round(voxels * voxel_vol)
            lo, hi = VOLUME_BOUNDS[bone_name]
            in_range = lo <= vol_mm3 <= hi
            flag = None if in_range else f"VOLUME_FLAG: {bone_name} {vol_mm3}mm³ outside [{lo},{hi}]mm³"
            if flag:
                result["flags"].append(flag)
                result["pass"] = False
            result["bones"][bone_name] = {"volume_mm3": vol_mm3, "in_range": in_range}

    except Exception as e:
        result["flags"].append(f"MASK_ERROR: {e}")
        result["pass"] = False

    # 2. Mesh-level watertight check
    mesh_dir = case_dir / "meshes"
    if not mesh_dir.exists():
        result["flags"].append("MISSING: meshes/ directory")
        result["pass"] = False
        return result

    obj_files = sorted(mesh_dir.glob("*_decimated.obj"))
    if not obj_files:
        result["flags"].append("MISSING: no *_decimated.obj files")
        result["pass"] = False
        return result

    result["meshes"] = {}
    for obj_path in obj_files:
        bone_key = obj_path.stem.replace("_decimated", "")
        try:
            is_watertight, boundary_edges, verts, faces = check_watertight(str(obj_path))
            result["meshes"][bone_key] = {
                "watertight": is_watertight,
                "boundary_edges": boundary_edges,
                "vertices": verts,
                "faces": faces,
            }
            if not is_watertight:
                result["flags"].append(f"NOT_WATERTIGHT: {obj_path.name} ({boundary_edges} boundary edges)")
                result["pass"] = False
        except Exception as e:
            result["flags"].append(f"MESH_ERROR: {obj_path.name}: {e}")
            result["pass"] = False

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary table printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: list[dict]):
    print("\n" + "="*95)
    print(f"{'Case':15} {'Pass':6} {'FOV-Z(mm)':10} {'Femur(mm³)':12} {'Tibia(mm³)':12} {'Patella(mm³)':13} {'Watertight':10} Flags")
    print("-"*95)
    for r in results:
        bones  = r.get("bones", {})
        meshes = r.get("meshes", {})
        femur  = bones.get("femur",   {}).get("volume_mm3", "N/A")
        tibia  = bones.get("tibia",   {}).get("volume_mm3", "N/A")
        patella= bones.get("patella", {}).get("volume_mm3", "N/A")
        
        is_wt = all(m.get("watertight", False) for m in meshes.values()) if meshes else False
        wt_str = "Yes" if is_wt and meshes else "No"
        
        status = "✓" if r["pass"] else "✗"
        z_ext  = str(r.get("z_extent_mm", "N/A"))
        flags  = "; ".join(r["flags"]) if r["flags"] else "—"
        print(f"{r['case_id']:15} {status:6} {z_ext:10} {str(femur):12} {str(tibia):12} {str(patella):13} {wt_str:10} {flags}")
    print("="*95)
    passed = sum(1 for r in results if r["pass"])
    print(f"\nSummary: {passed}/{len(results)} cases passed all QA checks.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch QA across all processed cases")
    parser.add_argument("--outputs-dir", default="outputs",
                        help="Root outputs directory (default: outputs/)")
    parser.add_argument("--qa-json", default="outputs/batch_qa.json",
                        help="Where to write the JSON QA summary")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        print(f"ERROR: outputs directory not found: {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    case_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("STS_")])
    if not case_dirs:
        print(f"No STS_* case directories found under {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    logger.info(f"Running QA on {len(case_dirs)} cases...")
    results = []
    for case_dir in case_dirs:
        logger.info(f"  Checking {case_dir.name}...")
        result = qa_case(case_dir)
        results.append(result)

    # Write JSON
    Path(args.qa_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.qa_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"QA JSON written: {args.qa_json}")

    # Print human-readable table
    print_summary_table(results)

    all_pass = all(r["pass"] for r in results)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
