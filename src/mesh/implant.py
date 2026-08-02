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


# --------------------------------------------------------------------------
# Femoral component
# --------------------------------------------------------------------------

# Femoral components are sized primarily on the anteroposterior dimension: too
# large notches the anterior cortex, too small leaves the flexion gap loose.
# Ranges here bracket the distal femora measured on this dataset (ML 70-87mm,
# AP 55-64mm).
FEMORAL_SIZES = [
    {"size": 1, "ml_mm": 58.0, "ap_mm": 52.0},
    {"size": 2, "ml_mm": 62.0, "ap_mm": 55.0},
    {"size": 3, "ml_mm": 66.0, "ap_mm": 58.0},
    {"size": 4, "ml_mm": 70.0, "ap_mm": 61.0},
    {"size": 5, "ml_mm": 74.0, "ap_mm": 64.0},
    {"size": 6, "ml_mm": 78.0, "ap_mm": 67.0},
]

FEMORAL_WALL_MM = 9.0          # distal and posterior thickness
FEMORAL_CHAMFER_MM = 8.0
FEMORAL_ANTERIOR_RISE_MM = 22.0   # how far the anterior flange runs proximally
FEMORAL_POSTERIOR_RISE_MM = 20.0


def femoral_box(size: dict) -> dict:
    """Internal box dimensions the given component size requires."""
    return {
        "anterior_inset_mm": FEMORAL_WALL_MM * 0.55,   # flange is thinner than the condyles
        "posterior_inset_mm": FEMORAL_WALL_MM,
        "chamfer_mm": FEMORAL_CHAMFER_MM,
        "anterior_rise_mm": FEMORAL_ANTERIOR_RISE_MM,
        "posterior_rise_mm": FEMORAL_POSTERIOR_RISE_MM,
    }


def select_femoral_size(ap_mm: float, ml_mm: float, sizes: list = None) -> dict:
    """
    Largest component that does not exceed the femur's AP depth.

    AP governs: an oversized component notches the anterior femoral cortex, a
    stress riser associated with periprosthetic fracture, so AP is never
    rounded up. ML is reported for reference and flagged if it overhangs.
    """
    sizes = sizes or FEMORAL_SIZES
    fitting = [s for s in sizes if s["ap_mm"] <= ap_mm]
    chosen = dict(max(fitting, key=lambda s: s["ap_mm"]) if fitting else sizes[0])
    chosen["fit"] = "fitted" if fitting else "undersize"
    chosen["ap_margin_mm"] = round(ap_mm - chosen["ap_mm"], 1)
    chosen["ml_margin_mm"] = round(ml_mm - chosen["ml_mm"], 1)
    chosen["ml_overhang"] = bool(chosen["ml_mm"] > ml_mm)
    return chosen


def _femoral_profile(size: dict, box: dict):
    """
    Sagittal cross-section of the component, in (anterior, proximal) mm with the
    origin at the centre of the distal cut.

    The inner boundary traces the five cut surfaces; the outer is that boundary
    offset by the wall thickness, which rounds the distal-posterior corner into
    the curved articular surface. This is a swept constant section — a real
    component's condylar radius varies through flexion and it carries an
    intercondylar notch, neither of which is modelled here.
    """
    from shapely.geometry import LineString, Polygon

    # A quoted femoral size is the component's external anteroposterior
    # dimension, so the bone box it seats on is inset by the wall thickness on
    # each side. Getting this backwards would cut a box the size of the implant
    # and leave the implant standing proud of the bone by its own thickness.
    a = size["ap_mm"] / 2.0 - FEMORAL_WALL_MM
    p = -a
    c = box["chamfer_mm"]

    # Open path along the cut surfaces, anterior top round to posterior top.
    # Deliberately not closed: buffering a closed ring and subtracting leaves an
    # annulus enclosing a hole, whereas the component is a C that opens
    # proximally where there is no cut to seat against.
    inner_path = [
        (a, box["anterior_rise_mm"]),
        (a, c),
        (a - c, 0.0),
        (p + c, 0.0),
        (p, c),
        (p, box["posterior_rise_mm"]),
    ]

    # Positive offset is to the left of travel, which along this path is away
    # from the bone — the side the component's material occupies.
    outer_path = LineString(inner_path).offset_curve(FEMORAL_WALL_MM,
                                                     join_style=1, quad_segs=16)
    outer = list(outer_path.coords)
    # offset_curve does not guarantee direction; align it with the inner path so
    # the two join end-to-end instead of crossing over themselves.
    if np.linalg.norm(np.array(outer[0]) - np.array(inner_path[0])) > \
       np.linalg.norm(np.array(outer[-1]) - np.array(inner_path[0])):
        outer = outer[::-1]

    shell = Polygon(inner_path + outer[::-1])
    if not shell.is_valid:
        shell = shell.buffer(0)
    if shell.geom_type == "MultiPolygon":
        shell = max(shell.geoms, key=lambda g: g.area)
    return shell


def build_femoral_component(size: dict) -> trimesh.Trimesh:
    """
    Component at the origin: sagittal shell swept across the mediolateral width,
    with +Y anterior and +Z proximal, seating face on the distal cut at z = 0.
    """
    profile = _femoral_profile(size, femoral_box(size))
    solid = trimesh.creation.extrude_polygon(profile, height=size["ml_mm"])

    # extrude_polygon lays the profile in XY (x = anterior, y = proximal) and
    # sweeps along +Z. Permute so the sweep becomes mediolateral and the profile
    # stands in the sagittal plane: (x, y, z) -> (z, x, y).
    permute = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    solid.apply_transform(permute)
    solid.apply_translation([-solid.bounds[0][0] - size["ml_mm"] / 2.0, 0.0, 0.0])
    return solid


def place_femoral_component(component: trimesh.Trimesh, distal_centre: np.ndarray,
                            proximal: np.ndarray, anterior: np.ndarray) -> trimesh.Trimesh:
    """Map the canonical component onto the prepared femur."""
    z = proximal / np.linalg.norm(proximal)
    y = anterior - (anterior @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)

    transform = np.eye(4)
    transform[:3, 0] = x
    transform[:3, 1] = y
    transform[:3, 2] = z
    transform[:3, 3] = distal_centre

    placed = component.copy()
    placed.apply_transform(transform)
    return placed


def fit_femoral_component(femur: trimesh.Trimesh, axis: np.ndarray,
                          distal_depth_mm: float = 9.0,
                          posterior: np.ndarray = None) -> dict:
    """
    Size a femoral component to the distal femur, cut the box it needs, and
    place it.

    Unlike the tibial tray, the bone preparation depends on the implant: the
    five box cuts are made to fit the chosen component, so the size is selected
    first and the cuts follow from it.

    Returns the size, the placed component, and the prepared femur.
    """
    from src.mesh.resection import femoral_box_planes, normalize_winding, resect_femoral_box

    # Same reason as plan_resection: meshes predating the topology repair carry
    # inconsistent winding, which inflates every volume taken from them.
    femur = normalize_winding(femur)

    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    if posterior is None:
        posterior = np.array([0.0, 1.0, 0.0])  # LPS +Y
    anterior = -posterior

    ml_axis = np.array([1.0, 0.0, 0.0])
    ap_axis = np.cross(axis, ml_axis)
    ap_axis /= np.linalg.norm(ap_axis)
    if ap_axis @ anterior < 0:
        ap_axis = -ap_axis
    ml_axis = np.cross(ap_axis, axis)

    # Measure over the condylar region the component caps, not the whole shaft.
    proj = femur.vertices @ axis
    distal = femur.vertices[proj <= proj.min() + 0.25 * np.ptp(proj)]
    femur_ap = float(np.ptp(distal @ ap_axis))
    femur_ml = float(np.ptp(distal @ ml_axis))

    size = select_femoral_size(femur_ap, femur_ml)
    planes = femoral_box_planes(femur, axis, femoral_box(size),
                                distal_depth_mm=distal_depth_mm,
                                posterior=posterior)
    prepared = resect_femoral_box(femur, planes)

    # Seat on the centre of the distal cut face.
    distal_level = proj.min() + distal_depth_mm
    on_distal = np.abs((prepared.triangles_center @ axis) - distal_level) <= 0.5
    if np.any(on_distal):
        area = prepared.area_faces[on_distal]
        centre = (prepared.triangles_center[on_distal] * area[:, None]).sum(0) / area.sum()
    else:
        centre = prepared.vertices.mean(axis=0)

    component = build_femoral_component(size)
    placed = place_femoral_component(component, centre, axis, ap_axis)

    return {
        "size": size["size"],
        "ml_mm": size["ml_mm"],
        "ap_mm": size["ap_mm"],
        "fit": size["fit"],
        "ap_margin_mm": size["ap_margin_mm"],
        "ml_margin_mm": size["ml_margin_mm"],
        "ml_overhang": size["ml_overhang"],
        "femur_ap_mm": round(femur_ap, 1),
        "femur_ml_mm": round(femur_ml, 1),
        "mesh": placed,
        "prepared_femur": prepared,
    }


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
