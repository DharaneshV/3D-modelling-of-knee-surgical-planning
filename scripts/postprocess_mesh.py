"""
postprocess_mesh.py — Independent post-processing step for bone segmentation masks and meshes.

Two independent stages applied in order:
  1. Morphological closing (3D, 1–2 voxel radius) on the binary mask before meshing.
     Closes small surface pits without altering gross anatomy.
     Logs voxel-count delta per bone so corrections are visible in QA, not silent.

  2. Taubin smoothing on the mesh after marching cubes/SurfaceNets.
     Volume-preserving (unlike naive Laplacian which shrinks the mesh).
     Re-runs watertight/boundary-edge check after smoothing to confirm topology.

Usage:
    python scripts/postprocess_mesh.py \
        --mask   outputs/STS_006/masks/bone_mask.nii.gz \
        --meshes outputs/STS_006/meshes/ \
        [--closing-radius 1] \
        [--taubin-iterations 30]
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BONE_LABELS = {1: "femur", 2: "tibia", 3: "patella"}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Morphological closing on the mask
# ─────────────────────────────────────────────────────────────────────────────

def apply_morphological_closing(mask_path: str, output_path: str,
                                 radius: int = 1) -> dict:
    """
    Applies per-bone 3D binary morphological closing to fill small surface pits.
    Returns a dict of voxel-count deltas per bone for QA logging.
    """
    logger.info(f"Morphological closing (radius={radius}) on: {mask_path}")
    mask_img = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_img).astype(np.uint8)

    deltas = {}
    output_arr = np.zeros_like(mask_arr)

    for label_val, bone_name in BONE_LABELS.items():
        bone_binary = (mask_arr == label_val).astype(np.uint8)
        before_count = int(np.sum(bone_binary))

        if before_count == 0:
            logger.warning(f"  {bone_name}: not present in mask, skipping.")
            deltas[bone_name] = {"before": 0, "after": 0, "delta": 0}
            continue

        # 3D morphological closing
        bone_sitk = sitk.GetImageFromArray(bone_binary)
        bone_sitk.CopyInformation(mask_img)
        closed = sitk.BinaryMorphologicalClosing(bone_sitk, [radius, radius, radius])
        closed_arr = sitk.GetArrayFromImage(closed).astype(np.uint8)

        after_count = int(np.sum(closed_arr))
        delta = after_count - before_count
        logger.info(f"  {bone_name}: {before_count} → {after_count} voxels (+{delta} filled)")

        deltas[bone_name] = {"before": before_count, "after": after_count, "delta": delta}
        # Merge into output (later labels overwrite earlier ones at overlaps — patella wins at joint)
        output_arr[closed_arr > 0] = label_val

    output_img = sitk.GetImageFromArray(output_arr)
    output_img.CopyInformation(mask_img)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output_img, output_path)
    logger.info(f"Closed mask saved: {output_path}")
    return deltas


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Taubin smoothing + watertight validation on meshes
# ─────────────────────────────────────────────────────────────────────────────

def taubin_smooth(vertices: np.ndarray, faces: np.ndarray,
                  iterations: int = 30, lambda_: float = 0.5, mu: float = -0.53) -> np.ndarray:
    """
    Taubin smoothing (two-step Laplacian per iteration): volume-preserving.
    lambda_ > 0 (shrink), mu < 0 (expand). |mu| slightly > lambda_ to correct
    for volume loss at each full iteration.
    Returns smoothed vertex positions.
    """
    verts = vertices.copy()

    # Build adjacency list
    adj = [set() for _ in range(len(verts))]
    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[f[i]].add(f[j])

    for _ in range(iterations):
        for step_lambda in [lambda_, mu]:
            new_verts = verts.copy()
            for i, neighbors in enumerate(adj):
                if neighbors:
                    neighbor_verts = verts[list(neighbors)]
                    laplacian = neighbor_verts.mean(axis=0) - verts[i]
                    new_verts[i] = verts[i] + step_lambda * laplacian
            verts = new_verts

    return verts


def check_watertight(faces: np.ndarray, n_verts: int) -> tuple[bool, int]:
    """
    Check mesh watertightness by counting boundary edges (edges with exactly
    one adjacent face). A watertight mesh has 0 boundary edges.
    Returns (is_watertight, boundary_edge_count).
    """
    edge_count = {}
    for f in faces:
        for i in range(3):
            e = tuple(sorted([f[i], f[(i + 1) % 3]]))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = sum(1 for count in edge_count.values() if count == 1)
    return boundary_edges == 0, boundary_edges


def load_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse .obj file into numpy vertex and face arrays."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                verts.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'f':
                face = [int(p.split('/')[0]) - 1 for p in parts[1:4]]
                faces.append(face)
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


def save_obj(path: str, vertices: np.ndarray, faces: np.ndarray):
    """Write vertices and faces back to .obj file."""
    with open(path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def process_mesh_file(mesh_path: str, iterations: int = 30) -> dict:
    """
    Apply Taubin smoothing to a single .obj file in-place.
    Validates watertightness before and after.
    Returns a QA result dict.
    """
    logger.info(f"  Processing: {Path(mesh_path).name}")
    verts, faces = load_obj(mesh_path)

    before_watertight, before_boundary = check_watertight(faces, len(verts))
    logger.info(f"    Before: {len(verts)} verts, boundary_edges={before_boundary} "
                f"({'watertight' if before_watertight else 'NOT watertight'})")

    # Apply Taubin smoothing
    smoothed_verts = taubin_smooth(verts, faces, iterations=iterations)

    # Re-check watertightness post-smoothing (topology shouldn't change but verify)
    after_watertight, after_boundary = check_watertight(faces, len(smoothed_verts))
    logger.info(f"    After:  {len(smoothed_verts)} verts, boundary_edges={after_boundary} "
                f"({'watertight' if after_watertight else 'NOT watertight'})")

    if not after_watertight:
        logger.warning(f"    WARNING: {Path(mesh_path).name} is NOT watertight after smoothing. "
                       "This should not happen with Taubin smoothing — check source mesh.")

    # Overwrite in-place
    save_obj(mesh_path, smoothed_verts, faces)

    return {
        "file": Path(mesh_path).name,
        "vertices": len(smoothed_verts),
        "faces": len(faces),
        "before_boundary_edges": before_boundary,
        "after_boundary_edges":  after_boundary,
        "watertight_before": before_watertight,
        "watertight_after":  after_watertight,
    }


def smooth_meshes_in_dir(mesh_dir: str, iterations: int = 30) -> list[dict]:
    """
    Find all *_decimated.obj files in mesh_dir and apply Taubin smoothing.
    Returns list of per-mesh QA results.
    """
    mesh_files = sorted(Path(mesh_dir).glob("*_decimated.obj"))
    if not mesh_files:
        logger.warning(f"No *_decimated.obj files found in {mesh_dir}")
        return []

    results = []
    for mesh_path in mesh_files:
        result = process_mesh_file(str(mesh_path), iterations=iterations)
        results.append(result)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mask closing + Taubin mesh smoothing")
    parser.add_argument("--mask",   required=True,  help="Input bone mask NIfTI path")
    parser.add_argument("--meshes", required=True,  help="Directory containing *_decimated.obj files")
    parser.add_argument("--mask-output", default=None,
                        help="Output path for closed mask (defaults to overwriting input)")
    parser.add_argument("--closing-radius", type=int, default=1,
                        help="Morphological closing radius in voxels (default: 1)")
    parser.add_argument("--taubin-iterations", type=int, default=30,
                        help="Taubin smoothing iterations (default: 30)")
    parser.add_argument("--qa-output", default=None,
                        help="Path to write JSON QA summary")
    args = parser.parse_args()

    mask_output = args.mask_output or args.mask

    # Stage 1: Morphological closing
    closing_deltas = apply_morphological_closing(
        args.mask, mask_output, radius=args.closing_radius
    )

    # Stage 2: Taubin smoothing
    mesh_results = smooth_meshes_in_dir(args.meshes, iterations=args.taubin_iterations)

    # QA summary
    qa = {
        "morphological_closing": closing_deltas,
        "mesh_smoothing": mesh_results,
        "all_watertight": all(r["watertight_after"] for r in mesh_results),
    }

    if args.qa_output:
        Path(args.qa_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.qa_output, "w") as f:
            json.dump(qa, f, indent=2)
        logger.info(f"QA summary written: {args.qa_output}")

    # Exit code 1 if any mesh is not watertight post-smoothing
    if not qa["all_watertight"]:
        print("WARNING: one or more meshes are NOT watertight after smoothing.", file=sys.stderr)
        sys.exit(1)

    print("Post-processing complete. All meshes watertight.")


if __name__ == "__main__":
    main()
