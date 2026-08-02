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


def export_ar_glb(output_dir: str, label_map: dict, label_colors: dict, base_name: str, track: str) -> str | None:
    """Combine per-label OBJ parts into one colored, AR-ready GLB (mm/LPS -> m/Y-up)."""
    scene = trimesh.Scene()
    found_any = False

    for label_name in label_map.keys():
        part_path = os.path.join(output_dir, f"{label_name}.obj")
        if not os.path.exists(part_path):
            continue

        mesh = trimesh.load_mesh(part_path, process=False)
        mesh.apply_transform(_TRANSFORM)

        color = _hex_to_rgba(label_colors.get(label_name, '#ffffff'))
        mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))

        scene.add_geometry(mesh, node_name=label_name)
        found_any = True

    if not found_any:
        return None

    glb_path = os.path.join(output_dir, f"{base_name}_{track}_ar.glb")
    scene.export(glb_path)
    return glb_path
