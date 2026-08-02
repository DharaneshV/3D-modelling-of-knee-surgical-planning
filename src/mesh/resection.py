"""
Bone resection planes for TKA planning.

Computes the standard distal-femoral and proximal-tibial cuts, slices the bone
mesh with a capped plane, and measures the resulting cut surface.

Sizing is deliberately measured *on the resection surface* rather than at the
joint surface. That is how a component is sized surgically, and it avoids the
failure mode of measuring the tibial intercondylar eminence instead of the
plateau: the spines sit above the plateau, so a "top 10% of the tibia" ROI
reports roughly half the true mediolateral width.

Coordinate frame is the pipeline's native LPS millimetres (+Z superior). The
cut is defined relative to the supplied longitudinal axis; note that a knee-only
field of view has no hip or ankle centre, so that axis is a limb-axis proxy and
not a true mechanical axis.
"""
import numpy as np
import trimesh

# Resection depth is measured from the most distal (femur) or most proximal
# (tibia) point of the bone, along the longitudinal axis.
DEFAULT_FEMUR_DEPTH_MM = 9.0
DEFAULT_TIBIA_DEPTH_MM = 10.0


def limb_axis(femur_vertices: np.ndarray, tibia_vertices: np.ndarray) -> np.ndarray:
    """
    Longitudinal reference direction, pointing superior (femur-ward).

    The femur->tibia centroid vector is stable on cropped knee FOVs where PCA of
    the vertex cloud is not — see backend.report_generator.calculate_anatomic_axis.
    """
    axis = np.mean(femur_vertices, axis=0) - np.mean(tibia_vertices, axis=0)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.array([0.0, 0.0, 1.0])
    return axis / norm


def _rotate(vec: np.ndarray, about: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate `vec` about the `about` axis by `degrees` (Rodrigues)."""
    if degrees == 0:
        return vec
    k = about / np.linalg.norm(about)
    th = np.radians(degrees)
    return (vec * np.cos(th)
            + np.cross(k, vec) * np.sin(th)
            + k * np.dot(k, vec) * (1 - np.cos(th)))


def resection_plane(mesh: trimesh.Trimesh, axis: np.ndarray, bone: str,
                    depth_mm: float, varus_deg: float = 0.0,
                    slope_deg: float = 0.0) -> tuple:
    """
    Build the cut plane for a bone.

    Args:
        mesh:      the bone mesh, LPS mm.
        axis:      longitudinal reference direction, pointing superior.
        bone:      'femur' (cut the distal end) or 'tibia' (cut the proximal end).
        depth_mm:  cut depth from the joint-most extreme of the bone.
        varus_deg: coronal angulation, rotating the plane about the AP axis.
                   Positive tilts the lateral side of the cut proximally.
        slope_deg: sagittal angulation, rotating about the ML axis. Conventionally
                   used for tibial posterior slope.

    Returns:
        (origin, normal) with `normal` pointing towards the fragment that is
        removed by the cut.
    """
    if bone not in ("femur", "tibia"):
        raise ValueError(f"bone must be 'femur' or 'tibia', got {bone!r}")

    axis = axis / np.linalg.norm(axis)
    ml = np.array([1.0, 0.0, 0.0])          # LPS +X, mediolateral
    ap = np.cross(axis, ml)                  # anteroposterior, orthogonal to both
    ap /= np.linalg.norm(ap)
    ml = np.cross(ap, axis)                  # re-orthogonalise

    # Superior-pointing plane normal, angulated as requested.
    superior = _rotate(axis, ap, varus_deg)
    superior = _rotate(superior, ml, slope_deg)
    superior /= np.linalg.norm(superior)

    # Locate the cut along that direction. The femur is cut at its distal end,
    # the tibia at its proximal end, each measured from the joint-most extreme.
    proj = mesh.vertices @ superior
    if bone == "femur":
        cut_at = proj.min() + depth_mm
        discard = -superior          # the distal fragment is removed
    else:
        cut_at = proj.max() - depth_mm
        discard = superior           # the proximal fragment is removed

    # Any point p with p . superior == cut_at lies on the plane; superior is a
    # unit vector, so superior * cut_at is the nearest such point to the origin.
    origin = superior * cut_at
    return origin, discard


def ensure_watertight(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Best-effort close of small defects so volume can be computed.

    Most pipeline bone meshes are very slightly open — a handful of broken faces
    out of ~25k — which is enough to make trimesh refuse to report a volume.
    fill_holes only closes boundary loops, so it does not rescue the common case
    here (non-manifold edges); those keep a volume of None rather than being
    aggressively remeshed, which would risk moving the surfaces the cut and the
    sizing are measured from. Sizing does not depend on watertightness.

    The repair is done on a copy and kept only if it actually achieves
    watertightness, so a genuinely broken mesh is left alone rather than being
    silently patched into something that merely looks valid.
    """
    if mesh.is_watertight:
        return mesh
    patched = mesh.copy()
    try:
        trimesh.repair.fill_holes(patched)
    except Exception:
        return mesh
    return patched if patched.is_watertight else mesh


def resect(mesh: trimesh.Trimesh, origin: np.ndarray, normal: np.ndarray) -> trimesh.Trimesh:
    """
    Slice `mesh` with the plane, keeping the retained fragment and capping the
    cut so the result stays a closed solid (needed for volume and for an implant
    to seat against a real surface rather than a hole).
    """
    return trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal=-normal, plane_origin=origin, cap=True
    )


def cut_surface_dimensions(resected: trimesh.Trimesh, normal: np.ndarray,
                           tolerance_deg: float = 10.0) -> dict:
    """
    Measure the resection surface: the flat face created by the cut.

    Faces whose normal is within `tolerance_deg` of the cut normal are taken as
    the cut face; its extents along the in-plane axes give the ML and AP
    dimensions used for component sizing.
    """
    normal = normal / np.linalg.norm(normal)
    face_normals = resected.face_normals
    aligned = face_normals @ normal >= np.cos(np.radians(tolerance_deg))

    if not np.any(aligned):
        return {"ml_mm": 0.0, "ap_mm": 0.0, "area_mm2": 0.0, "found": False}

    verts = resected.vertices[np.unique(resected.faces[aligned])]
    area = float(resected.area_faces[aligned].sum())

    # In-plane basis
    ml = np.array([1.0, 0.0, 0.0])
    ap = np.cross(normal, ml)
    ap /= np.linalg.norm(ap)
    ml = np.cross(ap, normal)

    return {
        "ml_mm": round(float(np.ptp(verts @ ml)), 1),
        "ap_mm": round(float(np.ptp(verts @ ap)), 1),
        "area_mm2": round(area, 1),
        "found": True,
    }


def plan_resection(mesh: trimesh.Trimesh, axis: np.ndarray, bone: str,
                   depth_mm: float = None, varus_deg: float = 0.0,
                   slope_deg: float = 0.0) -> dict:
    """
    Full resection for one bone: plane, cut mesh, and measurements.

    Returns a dict with the resected mesh plus the numbers a surgeon would want
    — cut-surface ML/AP for component sizing, and how much bone is removed.
    """
    if depth_mm is None:
        depth_mm = DEFAULT_FEMUR_DEPTH_MM if bone == "femur" else DEFAULT_TIBIA_DEPTH_MM

    mesh = ensure_watertight(mesh)
    origin, normal = resection_plane(mesh, axis, bone, depth_mm, varus_deg, slope_deg)
    resected = resect(mesh, origin, normal)

    if resected is None or len(resected.faces) == 0:
        raise ValueError(f"Resection of {bone} at {depth_mm}mm removed the entire mesh")

    dims = cut_surface_dimensions(resected, normal)
    original_vol = float(mesh.volume) if mesh.is_watertight else None
    resected_vol = float(resected.volume) if resected.is_watertight else None

    return {
        "bone": bone,
        "depth_mm": depth_mm,
        "varus_deg": varus_deg,
        "slope_deg": slope_deg,
        "plane_origin": [round(float(x), 3) for x in origin],
        "plane_normal": [round(float(x), 4) for x in normal],
        "cut_surface": dims,
        "resected_volume_mm3": round(resected_vol, 1) if resected_vol else None,
        "removed_volume_mm3": (round(original_vol - resected_vol, 1)
                               if original_vol and resected_vol else None),
        "mesh": resected,
    }
