"""
Regression tests for the per-label surface repair.

The synthetic volume below reproduces both defects the real masks produce:
a label that abuts a second label (so vtkSurfaceNets3D winds part of the
surface the other way round) and a checkerboard voxel junction (so the dual
mesh pinches at a single non-manifold edge).
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

trimesh = pytest.importorskip("trimesh")

from src.mesh.surface_nets import extract_multilabel_surface
from src.mesh.topology import repair_label_surface, split_non_manifold_edges

LABEL_MAP = {"block": 1, "neighbour": 2}
SPACING = 0.5


@pytest.fixture(scope="module")
def label_surface():
    """The label-1 surface, sliced out of the shared multi-label mesh."""
    volume = np.zeros((24, 24, 24), dtype=np.uint8)   # (z, y, x)
    volume[4:16, 4:12, 4:12] = 1                      # main block
    volume[4:16, 12:16, 12:16] = 1                    # touches it only along an edge
    volume[4:16, 4:12, 12:16] = 2                     # face-adjacent second label

    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((SPACING, SPACING, SPACING))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "labels.nii.gz")
        sitk.WriteImage(image, path)
        raw = extract_multilabel_surface(path, LABEL_MAP)

    labels = raw.cell_data["BoundaryLabels"]
    selection = (labels[:, 0] == 1) | (labels[:, 1] == 1)
    surface = (raw.extract_cells(selection)
                  .extract_surface(algorithm="dataset_surface")
                  .clean()
                  .connectivity(extraction_mode="largest"))
    voxel_volume = float((volume == 1).sum()) * SPACING ** 3
    return surface, voxel_volume


def _to_trimesh(surface):
    triangulated = surface.triangulate()
    return trimesh.Trimesh(
        vertices=np.asarray(triangulated.points),
        faces=triangulated.faces.reshape(-1, 4)[:, 1:],
        process=False,
    )


def test_raw_label_surface_shows_both_defects(label_surface):
    """Guards the diagnosis: without repair the surface is unusable for volume."""
    mesh = _to_trimesh(label_surface[0])
    assert not mesh.is_winding_consistent
    assert not mesh.is_watertight


def test_repair_makes_surface_watertight_and_outward(label_surface):
    surface, voxel_volume = label_surface
    mesh = _to_trimesh(repair_label_surface(surface))

    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0
    # Surface Nets rounds the corners off this deliberately tiny block, so the
    # tolerance is loose; the point is that the volume lands on the voxelised
    # truth at all instead of the 1.4x-2.2x a mis-wound surface reports. On real
    # bones, where the surface-to-volume ratio is far lower, it lands within 0.3%.
    assert mesh.volume == pytest.approx(voxel_volume, rel=0.25)
    assert _to_trimesh(surface).volume > 1.3 * voxel_volume


def test_repair_does_not_move_any_vertex(label_surface):
    """The whole point of the fix: it is index-only."""
    surface = label_surface[0]
    before = np.asarray(surface.triangulate().points)
    after = np.asarray(repair_label_surface(surface).triangulate().points)

    assert len(after) >= len(before)
    # Every original coordinate survives untouched, and any added vertex is an
    # exact duplicate of one that was already there.
    assert np.array_equal(after[:len(before)], before)
    extra = after[len(before):]
    if len(extra):
        assert all((before == point).all(axis=1).any() for point in extra)


def test_split_leaves_a_clean_mesh_alone():
    sphere = trimesh.creation.icosphere(subdivisions=2)
    vertices, faces, duplicated, unresolved = split_non_manifold_edges(
        sphere.vertices, sphere.faces)

    assert duplicated == 0
    assert unresolved == 0
    assert np.array_equal(faces, np.asarray(sphere.faces))
    assert np.array_equal(vertices, np.asarray(sphere.vertices))
