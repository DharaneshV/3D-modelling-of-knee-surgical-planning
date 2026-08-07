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


def normalize_winding(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Make face winding consistent and outward-facing before any volume is taken.

    Meshes generated before the topology repair landed (src/mesh/topology.py)
    carry the per-label-pair winding inconsistency that vtkSurfaceNets3D
    produces, which makes trimesh report volumes 1.4x-2.1x too large — a femur
    measured at 325 cm3 against a true 156 cm3. Repaired meshes are already
    consistent, so this is a no-op on them.

    Index-only and idempotent: no vertex coordinate is created, moved or removed,
    so the surfaces the cut and the sizing are measured from do not shift.

    EXPENSIVE — trimesh.repair.fix_winding walks face adjacency across the whole
    mesh: measured at 19s for a 145k-face tibia and 43s for a 209k-face femur.
    Only call it where a *signed volume* is actually needed. Verified directly
    that cut-surface ML/AP, area and centroid are bit-identical with and without
    it (only .volume differs), so the geometry path must not pay this cost.
    """
    fixed = mesh.copy()
    trimesh.repair.fix_winding(fixed)
    if fixed.volume < 0:
        fixed.invert()
    return fixed


def _volume_of(mesh: trimesh.Trimesh) -> float | None:
    """
    Correctly-signed volume, paying for a winding fix only when one is needed.

    trimesh's .volume is a signed sum over faces, so it is only meaningful once
    winding is consistent — but normalize_winding is very slow (see its
    docstring), so it must not run speculatively. A wrongly-wound closed mesh
    reports a volume that is too large or negative, never a plausible-looking
    correct one, so: take the raw volume, sanity-check it against the convex
    hull (computed from the point cloud, so winding-independent — a solid can
    never enclose more than its own hull), and only correct when it fails.

    Callers that need volumes from BOTH a mesh and something cut from it should
    normalize the source first via normalized_for_volume() — a mesh cut from a
    correctly-wound one inherits correct winding, so that pays the cost once
    instead of once per derived mesh.

    Returns None for a mesh that isn't closed, where no volume is defined.
    """
    if not mesh.is_watertight:
        return None

    raw = float(mesh.volume)
    if raw > 0:
        try:
            if raw <= float(mesh.convex_hull.volume):
                return raw
        except Exception:
            return raw

    return float(normalize_winding(mesh).volume)


# Winding normalization is pure input->output on mesh topology and costs 19-43s
# per bone, while the usual workflow re-plans the same bones repeatedly as the
# depth/angle sliders are adjusted. Cache by the mesh's identity so that cost is
# paid once per bone per process rather than once per request.
_winding_cache: dict = {}


def normalized_for_volume(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Winding-normalized view of `mesh`, cached and reused across calls.

    Keyed on the actual vertex/face data (not object identity) so a mesh
    re-loaded from the same file on a later request still hits the cache.
    Meshes already correctly wound skip the expensive pass entirely.
    """
    try:
        key = (mesh.vertices.tobytes(), mesh.faces.tobytes())
    except Exception:
        return normalize_winding(mesh)

    cached = _winding_cache.get(key)
    if cached is not None:
        return cached

    # Already-correct meshes (anything produced since the topology repair
    # landed) need no work — the hull check is orders of magnitude cheaper
    # than fix_winding.
    if mesh.is_watertight:
        raw = float(mesh.volume)
        try:
            if 0 < raw <= float(mesh.convex_hull.volume):
                _winding_cache[key] = mesh
                return mesh
        except Exception:
            pass

    fixed = normalize_winding(mesh)
    _winding_cache[key] = fixed
    return fixed


def femoral_box_planes(mesh: trimesh.Trimesh, axis: np.ndarray, box: dict,
                       distal_depth_mm: float = DEFAULT_FEMUR_DEPTH_MM,
                       posterior: np.ndarray = None) -> list:
    """
    The five planes of a standard femoral box preparation.

    A femoral component does not seat on a single flat cut. It caps the distal
    femur across five surfaces — distal, posterior, anterior, and a chamfer
    joining each of those to the distal — and the box dimensions come from the
    chosen component size, since the cuts are made to fit the implant rather
    than measured freely.

    `box` supplies the internal dimensions: anterior_mm and posterior_mm from
    the femoral axis, chamfer_mm, and the cut heights.

    Returns [(origin, normal), ...] with each normal pointing at the fragment
    that cut removes, matching resection_plane's convention.
    """
    axis = axis / np.linalg.norm(axis)
    ml = np.array([1.0, 0.0, 0.0])
    ap = np.cross(axis, ml)
    ap /= np.linalg.norm(ap)

    # LPS +Y is posterior; make `ap` point anteriorly so the box is unambiguous.
    if posterior is None:
        posterior = np.array([0.0, 1.0, 0.0])
    anterior = -posterior
    if ap @ anterior < 0:
        ap = -ap

    verts = mesh.vertices
    distal_level = (verts @ axis).min() + distal_depth_mm
    # Reference the AP cuts to the bone so the box lands on the condyles rather
    # than at an arbitrary distance from the world origin.
    ap_proj = verts @ ap
    anterior_face = ap_proj.max() - box["anterior_inset_mm"]
    posterior_face = ap_proj.min() + box["posterior_inset_mm"]

    c = box["chamfer_mm"]
    planes = [
        # distal cut — removes everything below it
        (axis * distal_level, -axis),
        # posterior cut — removes the posterior condyles behind it
        (ap * posterior_face, -ap),
        # anterior cut — trims the anterior cortex
        (ap * anterior_face, ap),
    ]

    # Chamfers at 45 degrees, offset so they meet the distal and AP cuts.
    post_normal = -(axis + ap) / np.sqrt(2.0)
    post_origin = axis * (distal_level + c) + ap * (posterior_face + c)
    planes.append((post_origin, post_normal))

    ant_normal = -(axis - ap) / np.sqrt(2.0)
    ant_origin = axis * (distal_level + c) + ap * (anterior_face - c)
    planes.append((ant_origin, ant_normal))

    return planes


def resect_femoral_box(mesh: trimesh.Trimesh, planes: list) -> trimesh.Trimesh:
    """Apply the five box cuts in turn, keeping the retained fragment each time."""
    cut = mesh
    for origin, normal in planes:
        cut = resect(cut, np.asarray(origin, dtype=float), np.asarray(normal, dtype=float))
        if cut is None or len(cut.faces) == 0:
            raise ValueError("Femoral box preparation removed the entire mesh")
    return cut


def ensure_watertight(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Best-effort close of small defects so volume can be computed.

    Bone meshes generated since the topology repair landed (src/mesh/topology.py)
    arrive closed and outward-wound, so this is a no-op on them. It stays as a
    guard for meshes produced by older pipeline runs, where a handful of
    non-manifold edges out of ~33k made trimesh refuse to report a volume.
    fill_holes only closes boundary loops, so it never rescued that case; such a
    mesh keeps a volume of None rather than being aggressively remeshed, which
    would risk moving the surfaces the cut and the sizing are measured from.
    Sizing does not depend on watertightness.

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
                           origin: np.ndarray = None,
                           tolerance_deg: float = 10.0,
                           plane_tol_mm: float = 0.5) -> dict:
    """
    Measure the resection surface: the flat face created by the cut.

    A face belongs to the cut if its normal is within `tolerance_deg` of the cut
    normal *and* it actually lies in the cut plane, within `plane_tol_mm`. The
    angular test alone is not enough: any part of the bone that happens to face
    the same way qualifies, and on a proximal tibia that pulled in surface up to
    55 mm off the plane, inflating AP by as much as 12 mm. It also made the
    result depend on face winding, so it moved when the winding was corrected.

    `origin` is any point on the cut plane. It is optional only so that older
    callers keep working; without it the angular test runs alone, as before.
    """
    normal = normal / np.linalg.norm(normal)
    face_normals = resected.face_normals
    aligned = face_normals @ normal >= np.cos(np.radians(tolerance_deg))

    if origin is not None:
        on_plane = aligned & (
            np.abs((resected.triangles_center - origin) @ normal) <= plane_tol_mm)
        # Keep the angular-only selection if nothing survives, so a degenerate
        # cut still reports something rather than silently collapsing to zero.
        if np.any(on_plane):
            aligned = on_plane

    if not np.any(aligned):
        return {"ml_mm": 0.0, "ap_mm": 0.0, "area_mm2": 0.0,
                "centroid": None, "found": False}

    verts = resected.vertices[np.unique(resected.faces[aligned])]
    area = float(resected.area_faces[aligned].sum())

    # In-plane basis
    ml = np.array([1.0, 0.0, 0.0])
    ap = np.cross(normal, ml)
    ap /= np.linalg.norm(ap)
    ml = np.cross(ap, normal)

    # Area-weighted centre of the cut face, which is where an implant seats.
    # Weighted rather than a plain vertex mean so a densely tessellated corner
    # does not drag the seating point off-centre.
    centroid = (resected.triangles_center[aligned]
                * resected.area_faces[aligned][:, None]).sum(axis=0) / area

    return {
        "ml_mm": round(float(np.ptp(verts @ ml)), 1),
        "ap_mm": round(float(np.ptp(verts @ ap)), 1),
        "area_mm2": round(area, 1),
        "centroid": [round(float(c), 3) for c in centroid],
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

    # Normalize the SOURCE mesh's winding once (cached across calls): the mesh
    # cut from it then inherits correct winding, so both volumes below are free.
    # Doing it lazily per-mesh instead would pay the 19-43s cost twice, once for
    # the original and again for the cut. Winding does not affect the cut
    # geometry either way — verified that ML/AP/area/centroid are bit-identical
    # with and without it — so this is purely to make the volumes meaningful.
    mesh = ensure_watertight(normalized_for_volume(mesh))
    origin, normal = resection_plane(mesh, axis, bone, depth_mm, varus_deg, slope_deg)
    resected = resect(mesh, origin, normal)

    if resected is None or len(resected.faces) == 0:
        raise ValueError(f"Resection of {bone} at {depth_mm}mm removed the entire mesh")

    dims = cut_surface_dimensions(resected, normal, origin)
    original_vol = _volume_of(mesh)
    resected_vol = _volume_of(resected)

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
