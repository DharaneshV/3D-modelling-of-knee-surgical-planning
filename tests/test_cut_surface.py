"""
The resection surface must be measured on the cut plane, not merely on faces
that point the same way as it. Without the plane test, any parallel surface
elsewhere on the bone joins the measurement and inflates AP.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

trimesh = pytest.importorskip("trimesh")

from src.mesh.resection import cut_surface_dimensions

NORMAL = np.array([0.0, 0.0, 1.0])


def _bone_with_a_parallel_decoy():
    """A cut face 40x30 at z=+5, plus a wider parallel face 50 mm away."""
    cut = trimesh.creation.box(extents=[40.0, 30.0, 10.0])
    decoy = trimesh.creation.box(extents=[80.0, 60.0, 10.0])
    decoy.apply_translation([0.0, 0.0, 50.0])
    return trimesh.util.concatenate([cut, decoy])


def test_plane_filter_measures_only_the_cut():
    dims = cut_surface_dimensions(_bone_with_a_parallel_decoy(), NORMAL,
                                  origin=np.array([0.0, 0.0, 5.0]))

    assert dims["found"]
    assert dims["ml_mm"] == pytest.approx(40.0)
    assert dims["ap_mm"] == pytest.approx(30.0)
    assert dims["area_mm2"] == pytest.approx(1200.0)


def test_without_the_plane_filter_the_decoy_is_included():
    """Documents the behaviour the origin argument exists to fix."""
    dims = cut_surface_dimensions(_bone_with_a_parallel_decoy(), NORMAL)

    assert dims["ml_mm"] == pytest.approx(80.0)
    assert dims["ap_mm"] == pytest.approx(60.0)


def test_plane_filter_falls_back_rather_than_reporting_zero():
    """An origin nowhere near the cut must not collapse the measurement."""
    dims = cut_surface_dimensions(_bone_with_a_parallel_decoy(), NORMAL,
                                  origin=np.array([0.0, 0.0, 500.0]))

    assert dims["found"]
    assert dims["ml_mm"] > 0
