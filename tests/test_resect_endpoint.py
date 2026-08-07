"""
Integration tests for GET /api/resect/{task_id}.

The newest and most complex route had no test coverage at all: only the
lower-level geometry functions were tested, never the endpoint's own
validation, error handling, or response shape. Two bugs that shipped in this
area would have been caught here — the femoral box entry omitting `cut_surface`
(which crashed the frontend's table renderer) and the response's overall shape
changing without anything asserting it.

Uses synthetic box meshes rather than a real case: the real ones are 7-10MB and
take ~17-70s to resect, which is far too slow for a test suite, and none of the
behaviour under test depends on realistic anatomy.
"""

import json
import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("shapely")

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402
from backend.pipeline_runner import MESHES_DIR  # noqa: E402

client = TestClient(app)

TASK_ID = "11111111-2222-3333-4444-555555555555"
MISSING_TASK_ID = "99999999-8888-7777-6666-555555555555"


def _bone(width_ml, depth_ap, height_si, z_centre):
    """A box standing in for a bone, positioned along the superoinferior axis.

    Boxes are watertight and correctly wound, so volumes are well-defined and
    the winding-normalization path is skipped — keeping these tests fast.
    """
    mesh = trimesh.creation.box(extents=[width_ml, depth_ap, height_si])
    mesh.apply_translation([0.0, 0.0, z_centre])
    return mesh


@pytest.fixture(autouse=True)
def synthetic_task():
    task_dir = MESHES_DIR / TASK_ID
    task_dir.mkdir(parents=True, exist_ok=True)

    # Femur above, tibia below, so limb_axis (femur centroid - tibia centroid)
    # comes out superior, matching real anatomy.
    _bone(70.0, 55.0, 90.0, z_centre=60.0).export(str(task_dir / "femur_unknown.obj"))
    _bone(70.0, 50.0, 70.0, z_centre=-40.0).export(str(task_dir / "tibia_unknown.obj"))

    (task_dir / "manifest.json").write_text(json.dumps({
        "task_id": TASK_ID,
        "modality": "MRI",
        "laterality": "unknown",
        "ar_glb": None,
        "parts": [
            {"file": "femur_unknown.obj", "label": "Femur Bone", "color": "#e74c3c"},
            {"file": "tibia_unknown.obj", "label": "Tibia Bone", "color": "#2ecc71"},
        ],
    }))

    yield task_dir

    shutil.rmtree(task_dir, ignore_errors=True)


def test_resect_returns_expected_top_level_shape():
    res = client.get(f"/api/resect/{TASK_ID}")
    assert res.status_code == 200, res.text
    data = res.json()

    for key in ("task_id", "limb_axis", "axis_note", "resections",
                "implants", "implant_note", "ar_glb"):
        assert key in data, f"missing {key}"

    assert data["task_id"] == TASK_ID
    assert len(data["limb_axis"]) == 3
    # The caveats are not decoration: they are what stops a reader treating
    # these as mechanical-axis measurements or a real product selection.
    assert "mechanical axis" in data["axis_note"]
    assert "not specific commercial" in data["implant_note"]


def test_every_resection_entry_is_renderable_by_the_frontend():
    """
    Guards the bug that shipped: the femoral five-cut box entry has no
    cut_surface (five faces, so one ML/AP pair would be meaningless) while a
    single-plane cut does. The frontend reads r.cut_surface.ml_mm, so it threw
    a TypeError on the femur row and silently aborted the whole handler.
    Every entry must be renderable by one of the two known shapes.
    """
    data = client.get(f"/api/resect/{TASK_ID}").json()
    assert data["resections"], "expected at least one resection entry"

    for bone, entry in data["resections"].items():
        single_plane = "cut_surface" in entry and "depth_mm" in entry
        box_prep = "preparation" in entry and "distal_depth_mm" in entry
        assert single_plane or box_prep, (
            f"{bone} entry matches neither known shape: {sorted(entry)}")

        if single_plane:
            cut = entry["cut_surface"]
            for field in ("ml_mm", "ap_mm", "area_mm2", "centroid", "found"):
                assert field in cut, f"{bone}.cut_surface missing {field}"


def test_implants_report_their_governing_risk_number():
    data = client.get(f"/api/resect/{TASK_ID}").json()

    for kind, imp in data["implants"].items():
        assert kind in ("femoral", "tibial")
        for field in ("size", "ml_mm", "ap_mm", "fit"):
            assert field in imp
        assert imp["fit"] in ("fitted", "undersize")
        # Each component's own failure mode must be reported, not just a size.
        if kind == "tibial":
            assert "coverage_pct" in imp and "max_overhang_mm" in imp
        else:
            assert "ap_margin_mm" in imp and "ml_overhang" in imp


def test_resect_writes_the_meshes_it_reports(synthetic_task):
    data = client.get(f"/api/resect/{TASK_ID}").json()

    assert (synthetic_task / "tibia_resected.obj").exists()
    assert (synthetic_task / "femur_resected.obj").exists()
    assert (synthetic_task / data["ar_glb"]).exists()

    # An implant is only claimed if its mesh actually landed — the frontend
    # decides what to display from this, and a stale file must not be implied.
    if "tibial" in data["implants"]:
        assert (synthetic_task / "tibial_tray.obj").exists()
    if "femoral" in data["implants"]:
        assert (synthetic_task / "femoral_component.obj").exists()

    leftovers = [p.name for p in synthetic_task.iterdir() if ".tmp" in p.name]
    assert leftovers == [], f"atomic write left temp files: {leftovers}"


@pytest.mark.parametrize("params,reason", [
    ({"femur_depth_mm": 0}, "zero depth"),
    ({"femur_depth_mm": 40}, "depth at the exclusive upper bound"),
    ({"femur_depth_mm": 100}, "depth far past the bound"),
    ({"tibia_depth_mm": -5}, "negative depth"),
    ({"varus_deg": 20}, "angulation past +/-15"),
    ({"slope_deg": -20}, "negative angulation past bound"),
])
def test_out_of_range_parameters_are_rejected(params, reason):
    res = client.get(f"/api/resect/{TASK_ID}", params=params)
    assert res.status_code == 400, f"{reason} should be rejected, got {res.status_code}"


def test_parameters_at_the_edge_of_the_valid_range_are_accepted():
    res = client.get(f"/api/resect/{TASK_ID}",
                     params={"varus_deg": 15, "slope_deg": -15})
    assert res.status_code == 200, res.text


def test_missing_meshes_gives_404_not_500():
    task_dir = MESHES_DIR / MISSING_TASK_ID
    task_dir.mkdir(parents=True, exist_ok=True)
    try:
        res = client.get(f"/api/resect/{MISSING_TASK_ID}")
        assert res.status_code == 404
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)


def test_malformed_task_id_is_rejected_before_any_path_is_built():
    res = client.get("/api/resect/not-a-uuid")
    assert res.status_code == 400
