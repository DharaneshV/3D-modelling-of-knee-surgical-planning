"""
Generic parametric TKA tibial tray, sized from the resection surface.

This is NOT a specific commercial implant. No manufacturer publishes CAD for
their components, so this is an idealised generic tray whose *dimensions* come
from a discrete size table in the range real systems occupy. The claim it
supports is "a component of this size fits this patient's resection", not "this
is the component that will be implanted".

Sizing follows surgical practice: real trays come in discrete sizes and the
surgeon picks one, so the measured cut surface is snapped to the largest size
that does not overhang. Overhang is the thing to avoid — a tray wider than the
resection irritates soft tissue at the rim — so coverage is maximised subject to
staying inside the bone.

Geometry is built at the origin with +Z as the proximal direction, then placed
onto the cut plane. All dimensions are millimetres, matching the pipeline's
native LPS frame.
"""
import numpy as np
import trimesh

# Representative of the range commercial tibial trays span. Values are the
# baseplate's mediolateral width and anteroposterior depth.
TIBIAL_TRAY_SIZES = [
    {"size": 1, "ml_mm": 58.0, "ap_mm": 39.0},
    {"size": 2, "ml_mm": 62.0, "ap_mm": 41.0},
    {"size": 3, "ml_mm": 66.0, "ap_mm": 43.0},
    {"size": 4, "ml_mm": 70.0, "ap_mm": 46.0},
    {"size": 5, "ml_mm": 74.0, "ap_mm": 48.0},
    {"size": 6, "ml_mm": 78.0, "ap_mm": 51.0},
    {"size": 7, "ml_mm": 82.0, "ap_mm": 54.0},
]

BASEPLATE_THICKNESS_MM = 3.0
STEM_LENGTH_MM = 35.0
STEM_DIAMETER_MM = 12.0


# Overhang beyond this is treated as not fitting. Tibial component overhang is
# associated with soft-tissue irritation and post-operative pain, with ~3mm the
# threshold commonly cited as clinically meaningful, so a size that exceeds it
# is rejected in favour of a smaller one.
MAX_OVERHANG_MM = 3.0

# A surgeon seats the tray where it covers best, not at the geometric centre of
# the cut. This is the search window, in millimetres, about that centre.
POSITION_SEARCH_MM = 6.0
POSITION_STEP_MM = 1.0


def cut_outline(bone_mesh, origin, normal, tolerance_deg: float = 10.0,
                plane_tol_mm: float = 0.5):
    """
    Real 2D outline of the resection surface, in millimetres relative to the
    cut centroid, on the (mediolateral, anteroposterior) basis.

    A concave hull is used rather than a convex one: the proximal tibia is not
    convex, and a convex hull would overstate the available bone and so hide
    exactly the overhang this is meant to detect.

    Returns (polygon, ml_axis, ap_axis, centroid_3d) or None if no cut face.
    """
    from shapely import concave_hull
    from shapely.geometry import MultiPoint

    origin = np.asarray(origin, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    on_cut = (
        (bone_mesh.face_normals @ normal >= np.cos(np.radians(tolerance_deg)))
        & (np.abs((bone_mesh.triangles_center - origin) @ normal) <= plane_tol_mm)
    )
    if not np.any(on_cut):
        return None

    ml = np.array([1.0, 0.0, 0.0])
    ap = np.cross(normal, ml)
    ap /= np.linalg.norm(ap)
    ml = np.cross(ap, normal)

    verts = bone_mesh.vertices[np.unique(bone_mesh.faces[on_cut])]
    area = bone_mesh.area_faces[on_cut]
    centroid = (bone_mesh.triangles_center[on_cut] * area[:, None]).sum(0) / area.sum()

    local = verts - centroid
    pts = np.column_stack([local @ ml, local @ ap])
    hull = concave_hull(MultiPoint([tuple(p) for p in pts]), ratio=0.35)
    return hull, ml, ap, centroid


def _overhang(size: dict, outline, offset=(0.0, 0.0)) -> tuple:
    """(overhang area mm2, max overhang distance mm) for a size against an outline."""
    from shapely.affinity import translate
    from shapely.geometry import Polygon, Point

    poly = Polygon(np.asarray(_footprint(size["ml_mm"], size["ap_mm"]).exterior.coords))
    if offset != (0.0, 0.0):
        poly = translate(poly, xoff=offset[0], yoff=offset[1])
    outside = poly.difference(outline)
    if outside.is_empty:
        return 0.0, 0.0
    worst = max(Point(p).distance(outline)
                for p in poly.exterior.coords if not outline.contains(Point(p)))
    return float(outside.area), float(worst)


def _best_position(size: dict, outline) -> tuple:
    """
    Find where this size sits best on the resection.

    A tray centred on the cut centroid overhangs a real plateau, because the
    plateau is asymmetric and the tray is not. Surgeons position the component
    for coverage; this is the equivalent — a coarse search for the offset with
    the least overhang, tie-broken on overhang area.

    Returns (offset, overhang_area_mm2, max_overhang_mm).
    """
    steps = np.arange(-POSITION_SEARCH_MM, POSITION_SEARCH_MM + 1e-9, POSITION_STEP_MM)
    best = None
    for dx in steps:
        for dy in steps:
            area, worst = _overhang(size, outline, (float(dx), float(dy)))
            key = (round(worst, 3), round(area, 1))
            if best is None or key < best[0]:
                best = (key, (float(dx), float(dy)), area, worst)
            if worst == 0.0 and area == 0.0:
                break
    _, offset, area, worst = best
    return offset, area, worst


def select_tray_size(ml_mm: float, ap_mm: float, outline=None,
                     sizes: list = None) -> dict:
    """
    Largest tray that fits the resection without overhanging.

    With `outline` (from cut_outline) the test is true 2D containment of the
    tray footprint within the actual resection boundary. Without it, the test
    falls back to comparing bounding boxes, which is not sufficient: two shapes
    can share a bounding box and still overhang badly. On a real proximal tibia
    the bounding-box rule picked a size that stood 5.7mm proud over 8% of its
    footprint.

    `fit` is one of:
      "fitted"    — sits within the resection (to MAX_OVERHANG_MM)
      "undersize" — even the smallest tray overhangs; returned anyway, flagged
    """
    sizes = sizes or TIBIAL_TRAY_SIZES

    if outline is not None:
        scored = [(s,) + _best_position(s, outline) for s in sizes]
        fitting = [r for r in scored if r[3] <= MAX_OVERHANG_MM]
        if fitting:
            s, offset, area, worst = max(fitting, key=lambda r: r[0]["ml_mm"] * r[0]["ap_mm"])
            chosen = dict(s)
            chosen["fit"] = "fitted"
        else:
            s, offset, area, worst = min(scored, key=lambda r: r[3])
            chosen = dict(s)
            chosen["fit"] = "undersize"
        chosen["overhang_area_mm2"] = round(area, 1)
        chosen["max_overhang_mm"] = round(worst, 2)
        chosen["offset_mm"] = [round(offset[0], 1), round(offset[1], 1)]
    else:
        fitting = [s for s in sizes if s["ml_mm"] <= ml_mm and s["ap_mm"] <= ap_mm]
        chosen = dict(max(fitting, key=lambda s: s["ml_mm"] * s["ap_mm"])
                      if fitting else sizes[0])
        chosen["fit"] = "fitted" if fitting else "undersize"
        chosen["overhang_area_mm2"] = None
        chosen["max_overhang_mm"] = None
        chosen["offset_mm"] = [0.0, 0.0]

    chosen["ml_margin_mm"] = round(ml_mm - chosen["ml_mm"], 1)
    chosen["ap_margin_mm"] = round(ap_mm - chosen["ap_mm"], 1)
    return chosen


def _footprint(ml_mm: float, ap_mm: float, n: int = 96):
    """
    Tray outline in the cut plane: an ellipse flattened posteriorly, with a
    posterior midline notch for the PCL.

    +Y is posterior here, matching LPS. The notch makes the anterior/posterior
    orientation of the placed tray visually unambiguous, which matters when
    checking that placement is correct.
    """
    from shapely.geometry import Polygon

    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = (ml_mm / 2.0) * np.cos(theta)
    y = (ap_mm / 2.0) * np.sin(theta)

    # Flatten the posterior third — real trays are not a plain ellipse there.
    posterior = y > ap_mm * 0.25
    y[posterior] = ap_mm * 0.25 + (y[posterior] - ap_mm * 0.25) * 0.55

    # Flattening shortens the outline, so rescale back to the nominal depth.
    # Without this a "70 x 46" tray measures 70 x 41, and both the no-overhang
    # guarantee and the reported coverage would be computed against a size the
    # geometry does not actually have.
    y = (y - y.min()) * (ap_mm / np.ptp(y)) - ap_mm / 2.0

    outline = Polygon(np.column_stack([x, y]))

    # PCL notch: a bite out of the posterior midline.
    notch_w, notch_d = ml_mm * 0.16, ap_mm * 0.13
    notch = Polygon([
        (-notch_w / 2, ap_mm), (notch_w / 2, ap_mm),
        (notch_w / 2, ap_mm / 2 - notch_d), (-notch_w / 2, ap_mm / 2 - notch_d),
    ])
    return outline.difference(notch)


def build_tibial_tray(size: dict,
                      thickness_mm: float = BASEPLATE_THICKNESS_MM,
                      stem_length_mm: float = STEM_LENGTH_MM,
                      stem_diameter_mm: float = STEM_DIAMETER_MM) -> trimesh.Trimesh:
    """
    Build the tray at the origin: baseplate occupying z in [0, thickness],
    stem descending to z = -stem_length.

    z = 0 is the seating face, so placing the tray is a matter of mapping the
    origin onto the cut plane.
    """
    baseplate = trimesh.creation.extrude_polygon(
        _footprint(size["ml_mm"], size["ap_mm"]), height=thickness_mm)

    stem = trimesh.creation.cylinder(
        radius=stem_diameter_mm / 2.0, height=stem_length_mm, sections=48)
    # cylinder is centred on the origin; drop it so it hangs below the seating face
    stem.apply_translation([0.0, 0.0, -stem_length_mm / 2.0])

    tray = trimesh.util.concatenate([baseplate, stem])
    tray.merge_vertices()
    return tray


def place_tray(tray: trimesh.Trimesh, seat_point: np.ndarray,
               proximal: np.ndarray, posterior: np.ndarray) -> trimesh.Trimesh:
    """
    Move a tray built at the origin onto the resection.

    Args:
        seat_point: where the tray's seating face should sit — the centroid of
                    the cut surface.
        proximal:   unit normal of the cut plane pointing away from the retained
                    bone, i.e. the direction the baseplate rises into.
        posterior:  in-plane unit vector pointing posteriorly, so the PCL notch
                    ends up at the back rather than at a random rotation.
    """
    z = proximal / np.linalg.norm(proximal)
    y = posterior - (posterior @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)

    transform = np.eye(4)
    transform[:3, 0] = x
    transform[:3, 1] = y
    transform[:3, 2] = z
    transform[:3, 3] = seat_point

    placed = tray.copy()
    placed.apply_transform(transform)
    return placed


def fit_tibial_tray(resection: dict, bone_mesh: trimesh.Trimesh,
                    posterior: np.ndarray = None) -> dict:
    """
    Select and place a tray for a completed tibial resection.

    Args:
        resection: the dict returned by resection.plan_resection for the tibia.
        bone_mesh: the resected tibia, used to recover the true cut outline so
                   sizing can test containment rather than bounding boxes.
        posterior: in-plane posterior direction; defaults to LPS +Y projected
                   into the plane, which orients the PCL notch.

    Returns the chosen size, the placed mesh, and the coverage achieved.
    """
    cut = resection["cut_surface"]
    if not cut.get("found"):
        raise ValueError("Resection has no identifiable cut surface to seat a tray on")

    # The plane normal points at the discarded fragment, which for a proximal
    # tibial cut is the proximal direction the tray rises into.
    proximal = np.asarray(resection["plane_normal"], dtype=float)
    origin = np.asarray(resection["plane_origin"], dtype=float)

    geometry = cut_outline(bone_mesh, origin, proximal)
    if geometry is None:
        raise ValueError("Could not recover the cut outline from the resected mesh")
    outline, _ml, _ap, centroid = geometry

    size = select_tray_size(cut["ml_mm"], cut["ap_mm"], outline=outline)
    tray = build_tibial_tray(size)

    if posterior is None:
        posterior = np.array([0.0, 1.0, 0.0])  # LPS +Y

    # Seat where coverage is best rather than at the geometric centre. The
    # offset is in the same (ml, ap) frame cut_outline measured it in.
    dx, dy = size["offset_mm"]
    seat = centroid + dx * _ml + dy * _ap
    placed = place_tray(tray, seat, proximal, posterior)

    tray_area = size["ml_mm"] * size["ap_mm"]
    return {
        "size": size["size"],
        "ml_mm": size["ml_mm"],
        "ap_mm": size["ap_mm"],
        "fit": size["fit"],
        "ml_margin_mm": size["ml_margin_mm"],
        "ap_margin_mm": size["ap_margin_mm"],
        "overhang_area_mm2": size["overhang_area_mm2"],
        "max_overhang_mm": size["max_overhang_mm"],
        "resection_ml_mm": cut["ml_mm"],
        "resection_ap_mm": cut["ap_mm"],
        "coverage_pct": round(100.0 * tray_area / (cut["ml_mm"] * cut["ap_mm"]), 1),
        "mesh": placed,
    }
