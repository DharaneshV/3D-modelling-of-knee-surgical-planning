"""
Tests for the generic parametric tibial tray.

The properties worth protecting are the clinical ones: a tray must actually be
the size the table claims, must not be chosen if it stands proud of the
resection, and must seat with its baseplate above the cut and its stem below it.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("shapely")

from src.mesh.implant import (  # noqa: E402
    BASEPLATE_THICKNESS_MM,
    MAX_OVERHANG_MM,
    STEM_LENGTH_MM,
    TIBIAL_TRAY_SIZES,
    build_tibial_tray,
    place_tray,
    select_tray_size,
)


@pytest.mark.parametrize("size", TIBIAL_TRAY_SIZES, ids=lambda s: f"size{s['size']}")
def test_geometry_matches_nominal_size(size):
    """A "66 x 43" tray must actually measure 66 x 43, or the no-overhang
    guarantee is computed against dimensions the geometry does not have."""
    tray = build_tibial_tray(size)
    plate = tray.vertices[(tray.vertices[:, 2] >= -1e-6)
                          & (tray.vertices[:, 2] <= BASEPLATE_THICKNESS_MM + 1e-6)]

    assert np.ptp(plate[:, 0]) == pytest.approx(size["ml_mm"], abs=0.5)
    assert np.ptp(plate[:, 1]) == pytest.approx(size["ap_mm"], abs=0.5)


@pytest.mark.parametrize("size", TIBIAL_TRAY_SIZES, ids=lambda s: f"size{s['size']}")
def test_tray_is_a_closed_solid(size):
    """Volume is reported for the implant, and an open mesh has none."""
    tray = build_tibial_tray(size)
    assert tray.is_watertight
    assert tray.volume > 0


def test_stem_below_seating_face_and_plate_above():
    tray = build_tibial_tray(TIBIAL_TRAY_SIZES[3])
    z = tray.vertices[:, 2]
    assert z.max() == pytest.approx(BASEPLATE_THICKNESS_MM, abs=1e-6)
    assert z.min() == pytest.approx(-STEM_LENGTH_MM, abs=1e-6)


def test_larger_resection_takes_at_least_as_large_a_tray():
    """Sizing must be monotonic — a bigger plateau never gets a smaller tray."""
    previous = 0
    for ml, ap in [(58, 39), (66, 43), (74, 48), (95, 65)]:
        size = select_tray_size(ml, ap)
        assert size["size"] >= previous
        previous = size["size"]


def test_bounding_box_fit_never_exceeds_the_resection():
    for ml, ap in [(60, 40), (71.5, 46.3), (80, 52)]:
        size = select_tray_size(ml, ap)
        assert size["ml_mm"] <= ml
        assert size["ap_mm"] <= ap
        assert size["fit"] == "fitted"


def test_resection_smaller_than_every_tray_is_flagged():
    size = select_tray_size(40.0, 25.0)
    assert size["fit"] == "undersize"
    assert size["ml_margin_mm"] < 0


def test_containment_sizing_rejects_overhang():
    """
    Against a real outline the choice is containment, not bounding boxes. A
    circle circumscribing the tray's bounding box admits it; one inscribed
    within it must not.
    """
    from shapely.geometry import Point

    size = TIBIAL_TRAY_SIZES[3]
    half_diagonal = 0.5 * np.hypot(size["ml_mm"], size["ap_mm"])

    generous = select_tray_size(90, 60, outline=Point(0, 0).buffer(half_diagonal + 2))
    assert generous["max_overhang_mm"] <= MAX_OVERHANG_MM

    tight = select_tray_size(90, 60, outline=Point(0, 0).buffer(size["ap_mm"] / 2))
    assert tight["size"] < size["size"] or tight["fit"] == "undersize"


def test_placement_puts_plate_along_the_given_normal():
    tray = build_tibial_tray(TIBIAL_TRAY_SIZES[2])
    seat = np.array([10.0, -5.0, 42.0])
    proximal = np.array([0.0, 0.0, 1.0])

    placed = place_tray(tray, seat, proximal, np.array([0.0, 1.0, 0.0]))
    height = (placed.vertices - seat) @ proximal

    assert height.max() == pytest.approx(BASEPLATE_THICKNESS_MM, abs=1e-6)
    assert height.min() == pytest.approx(-STEM_LENGTH_MM, abs=1e-6)
    # seating face lands on the seat point, not somewhere near it
    assert np.abs(height).min() == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# Femoral component
# --------------------------------------------------------------------------

from src.mesh.implant import (  # noqa: E402
    FEMORAL_NOTCH_WIDTH_FRACTION,
    FEMORAL_SIZES,
    FEMORAL_WALL_MM,
    _articular_curve,
    build_femoral_component,
    select_femoral_size,
)


@pytest.mark.parametrize("size", FEMORAL_SIZES, ids=lambda s: f"fem{s['size']}")
def test_femoral_geometry_matches_nominal_size(size):
    """A quoted femoral size is the component's external dimension. If the box
    were cut to the nominal instead, the implant would stand proud of the bone
    by its own wall thickness."""
    comp = build_femoral_component(size)
    extent = np.ptp(comp.vertices, axis=0)

    assert extent[0] == pytest.approx(size["ml_mm"], abs=0.5)   # mediolateral
    assert extent[1] == pytest.approx(size["ap_mm"], abs=0.5)   # anteroposterior


@pytest.mark.parametrize("size", FEMORAL_SIZES, ids=lambda s: f"fem{s['size']}")
def test_femoral_component_is_a_closed_shell(size):
    comp = build_femoral_component(size)
    assert comp.is_watertight
    assert comp.volume > 0


def test_femoral_component_extends_distal_to_the_cut():
    """The component adds thickness below the distal cut; if it sat entirely
    above z=0 the offset would have been applied into the bone."""
    comp = build_femoral_component(FEMORAL_SIZES[2])
    assert comp.bounds[0][2] == pytest.approx(-FEMORAL_WALL_MM, abs=0.5)
    assert comp.bounds[1][2] > 0


@pytest.mark.parametrize("size", FEMORAL_SIZES, ids=lambda s: f"fem{s['size']}")
def test_femoral_component_has_an_intercondylar_notch(size):
    """
    The condyles must be separated, not a solid block. Probed rather than
    inferred from volume, because a notch that failed to cut, cut off-centre,
    or cut the wrong width would all still leave a plausible-looking volume.
    """
    comp = build_femoral_component(size)
    half_ml = size["ml_mm"] / 2.0

    # Distal and posterior — between the condyles, where the notch is open.
    # Stepped finely so the measured width is not limited by probe spacing.
    step = 0.25
    xs = np.arange(-half_ml + 1.0, half_ml - 1.0, step)
    probes = np.column_stack([
        xs,
        np.full(len(xs), -size["ap_mm"] * 0.25),
        np.full(len(xs), -FEMORAL_WALL_MM * 0.4),
    ])
    gap = xs[~comp.contains(probes)]

    assert len(gap), "no notch: the component is solid between the condyles"
    assert np.ptp(gap) + step == pytest.approx(
        size["ml_mm"] * FEMORAL_NOTCH_WIDTH_FRACTION, abs=1.0)
    assert abs(gap.mean()) < 2.0, "notch is not centred between the condyles"


@pytest.mark.parametrize("size", FEMORAL_SIZES, ids=lambda s: f"fem{s['size']}")
def test_notch_stops_short_of_the_anterior_flange(size):
    """The flange is what holds the two condyles together. If the notch ran
    through it the component would be two loose pieces, which would still be
    watertight per-body and would still export — but could not be implanted."""
    comp = build_femoral_component(size)
    half_ml = size["ml_mm"] / 2.0

    xs = np.linspace(-half_ml + 1.0, half_ml - 1.0, 41)
    probes = np.column_stack([
        xs,
        np.full(41, size["ap_mm"] * 0.42),   # well anterior, up the flange
        np.full(41, 8.0),
    ])

    assert comp.contains(probes).all(), "notch has breached the anterior flange"
    assert comp.body_count == 1, "component has split into separate pieces"


def test_articular_surface_is_a_j_curve_not_a_constant_radius():
    """A real condyle's radius decreases from distal (extension) to posterior
    (deep flexion). A constant-radius sweep — what this replaced — would give a
    ratio of 1.0 and the wrong contact mechanics through the flexion arc."""
    curve = np.asarray(_articular_curve(FEMORAL_SIZES[2]))

    # Circumscribed-circle radius of each consecutive vertex triple.
    radii = []
    for p0, p1, p2 in zip(curve, curve[1:], curve[2:]):
        sides = [np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1),
                 np.linalg.norm(p2 - p0)]
        u, v = p1 - p0, p2 - p0
        area = abs(u[0] * v[1] - u[1] * v[0]) / 2.0
        if area > 1e-12:
            radii.append(np.prod(sides) / (4.0 * area))
    radii = np.asarray(radii)

    mid = len(radii) // 2
    distal = radii[mid - 5:mid + 5].mean()
    posterior = radii[-8:].mean()

    assert posterior < distal, "radius does not tighten towards flexion"
    assert distal / posterior > 2.0


def test_femoral_sizing_never_rounds_ap_up():
    """Oversizing in AP notches the anterior cortex, so AP is a hard ceiling."""
    for ap in [52.0, 57.9, 61.0, 70.0]:
        size = select_femoral_size(ap, 80.0)
        if size["fit"] == "fitted":
            assert size["ap_mm"] <= ap


def test_femoral_sizing_flags_a_femur_smaller_than_every_size():
    size = select_femoral_size(45.0, 50.0)
    assert size["fit"] == "undersize"
    assert size["ap_margin_mm"] < 0
