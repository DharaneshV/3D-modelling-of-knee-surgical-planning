"""
Tests for the AR export's triangle budget.

This path runs on every /api/resect and /api/ar call and had no test at all.
It exists because the AR icon appeared on a real Android device but the model
never loaded: the scene was 524,796 triangles at 10.5MB, roughly five times the
~100k the spec budgets for AR, and mobile renderers simply fail on that.

The property worth protecting hardest is SCALE. AR places the model at 1:1
real-world size, so a decimation pass that shifted the bounding box would be
showing a surgeon differently-sized anatomy. That was checked by hand when the
budget was introduced; it is asserted here so it stays checked.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("pyvista")

from src.mesh.export_ar_glb import (  # noqa: E402
    AR_TRIANGLE_BUDGET,
    _TRANSFORM,
    _fit_triangle_budget,
    export_ar_glb_from_meshes,
)

COLORS = {"a": "#e74c3c", "b": "#2ecc71"}


def _dense_sphere(subdivisions=6, radius=40.0):
    """A mesh well over the budget on its own (~80k faces at subdiv 6)."""
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


def test_over_budget_scene_is_reduced_to_the_budget():
    meshes = {"a": _dense_sphere(), "b": _dense_sphere()}
    before = sum(len(m.faces) for m in meshes.values())
    assert before > AR_TRIANGLE_BUDGET, "fixture must actually exceed the budget"

    reduced = _fit_triangle_budget(meshes)
    after = sum(len(m.faces) for m in reduced.values())

    assert after <= AR_TRIANGLE_BUDGET
    assert set(reduced) == set(meshes), "no part may be dropped to meet the budget"


def test_under_budget_scene_is_left_untouched():
    """Decimation is lossy, so a scene that already fits must not be degraded."""
    meshes = {"a": trimesh.creation.icosphere(subdivisions=2, radius=40.0)}
    assert sum(len(m.faces) for m in meshes.values()) < AR_TRIANGLE_BUDGET

    reduced = _fit_triangle_budget(meshes)

    assert len(reduced["a"].faces) == len(meshes["a"].faces)
    assert reduced["a"] is meshes["a"], "should be the same object, not a copy"


def test_decimation_preserves_real_world_scale():
    """
    The property that matters clinically. AR renders at 1:1, so the model's
    physical extent must survive decimation — a shifted bounding box means a
    differently-sized knee in front of the surgeon.
    """
    meshes = {"a": _dense_sphere(), "b": _dense_sphere()}
    before_extents = {k: np.ptp(m.vertices, axis=0) for k, m in meshes.items()}

    reduced = _fit_triangle_budget(meshes)

    for key, before in before_extents.items():
        after = np.ptp(reduced[key].vertices, axis=0)
        # Decimation moves the hull inward very slightly; a millimetre of
        # tolerance on a 80mm sphere is well inside anything clinically
        # meaningful, while still catching a unit-scale or transform error.
        assert np.allclose(after, before, atol=1.0), (
            f"{key} extent changed from {before} to {after}")


def test_export_applies_mm_to_metre_scale(tmp_path):
    """
    glTF/WebXR are metres, the pipeline is millimetres. Getting this wrong is
    the difference between a knee-sized model and one 1000x too large, and it
    is invisible in a desktop preview where the camera just reframes.
    """
    radius_mm = 40.0
    meshes = {"a": trimesh.creation.icosphere(subdivisions=3, radius=radius_mm)}
    out = tmp_path / "scene.glb"

    result = export_ar_glb_from_meshes(meshes, str(out), COLORS)

    assert result is not None and out.exists()
    scene = trimesh.load(str(out))
    extent = np.ptp(np.vstack([g.vertices for g in scene.geometry.values()]), axis=0)
    # 80mm diameter -> 0.08m
    assert np.allclose(extent, 0.08, atol=0.005), f"expected ~0.08m, got {extent}"


def test_export_returns_none_for_an_empty_scene():
    assert export_ar_glb_from_meshes({}, "unused.glb", COLORS) is None


def test_export_leaves_no_temp_file_behind(tmp_path):
    """The export writes to a temp path then os.replace()s it into place; a
    leftover .tmp would mean the rename never happened."""
    meshes = {"a": trimesh.creation.icosphere(subdivisions=2, radius=40.0)}
    out = tmp_path / "scene.glb"

    export_ar_glb_from_meshes(meshes, str(out), COLORS)

    assert out.exists()
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert leftovers == [], f"temp files left behind: {leftovers}"


def test_transform_is_a_pure_scale_and_axis_swap():
    """
    Guards the LPS->glTF matrix against an accidental edit. It must not
    introduce shear or a non-uniform scale: each row carries exactly one
    non-zero of magnitude 0.001 (the mm->m factor).
    """
    linear = _TRANSFORM[:3, :3]

    for row in linear:
        nonzero = row[row != 0]
        assert len(nonzero) == 1, f"row {row} is not a pure axis mapping"
        assert abs(abs(float(nonzero[0])) - 0.001) < 1e-12

    assert np.allclose(_TRANSFORM[:3, 3], 0.0), "transform must not translate"
