import os
import numpy as np
import trimesh

# Combined mm->m scale + LPS->glTF (Y-up, right-handed) axis remap, derived and
# visually verified in scratch/minimal_gltf.py:
#   glTF_X = -LPS_X, glTF_Y = LPS_Z (Superior->Up), glTF_Z = LPS_Y (Posterior->Backward)
_TRANSFORM = np.array([
    [-0.001, 0.0,    0.0,   0.0],
    [0.0,    0.0,    0.001, 0.0],
    [0.0,    0.001,  0.0,   0.0],
    [0.0,    0.0,    0.0,   1.0],
])


def _hex_to_rgba(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return np.array([r, g, b, 255], dtype=np.uint8)


# Mobile AR renderers will not sustain the full-resolution meshes: a five-part
# knee arrives at ~525k triangles and Scene Viewer/WebXR simply fails to load it.
# The original spec puts the budget at ~100k polygons for AR.
AR_TRIANGLE_BUDGET = 100_000


def _fit_triangle_budget(meshes: dict, budget: int = AR_TRIANGLE_BUDGET) -> dict:
    """
    Reduce meshes proportionally so the scene fits an AR triangle budget.

    This affects the AR export only. Clinical measurements are taken from the
    full-resolution .obj files, which this never touches, so decimating here
    cannot move a reported dimension — the spec's warning about validating
    measurements after decimation applies to the pipeline's own decimation
    step, not to this one.
    """
    import pyvista as pv

    total = sum(len(m.faces) for m in meshes.values())
    if total <= budget:
        return meshes

    keep = budget / total
    reduced = {}
    for name, mesh in meshes.items():
        faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).astype(np.int64)
        surface = pv.PolyData(np.asarray(mesh.vertices), faces)
        try:
            simplified = surface.decimate(1.0 - keep)
            tri = simplified.faces.reshape(-1, 4)[:, 1:]
            reduced[name] = trimesh.Trimesh(vertices=np.asarray(simplified.points),
                                            faces=tri, process=False)
        except Exception:
            reduced[name] = mesh  # a part that will not decimate is better than no part
    return reduced


def export_ar_glb_from_meshes(meshes: dict, out_path: str, label_colors: dict) -> str | None:
    """
    Export already-loaded meshes as one colored, AR-ready GLB (mm/LPS -> m/Y-up).

    `meshes` maps label name -> trimesh.Trimesh. Meshes are copied before the
    transform is applied so the caller's geometry is left in pipeline coordinates.
    """
    scene = trimesh.Scene()

    for label_name, mesh in _fit_triangle_budget(meshes).items():
        m = mesh.copy()
        m.apply_transform(_TRANSFORM)

        color = _hex_to_rgba(label_colors.get(label_name, '#ffffff'))
        m.visual.vertex_colors = np.tile(color, (len(m.vertices), 1))

        scene.add_geometry(m, node_name=label_name)

    if not scene.geometry:
        return None

    scene.export(out_path)
    return out_path


def export_ar_glb(output_dir: str, label_map: dict, label_colors: dict, base_name: str, track: str) -> str | None:
    """Combine per-label OBJ parts into one colored, AR-ready GLB (mm/LPS -> m/Y-up)."""
    meshes = {}
    for label_name in label_map.keys():
        part_path = os.path.join(output_dir, f"{label_name}.obj")
        if os.path.exists(part_path):
            meshes[label_name] = trimesh.load_mesh(part_path, process=False)

    if not meshes:
        return None

    glb_path = os.path.join(output_dir, f"{base_name}_{track}_ar.glb")
    return export_ar_glb_from_meshes(meshes, glb_path, label_colors)
