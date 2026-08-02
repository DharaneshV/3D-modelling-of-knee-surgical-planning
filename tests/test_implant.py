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
