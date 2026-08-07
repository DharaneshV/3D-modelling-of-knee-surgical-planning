"""
Regression tests for the anatomic-axis PCA rejection.

This guards a bug that already shipped once. calculate_anatomic_axis took the
largest-eigenvalue PCA eigenvector as the bone's long axis, which only holds
when the bone is imaged long enough to be its own longest dimension. A knee-only
MRI FOV crops the tibia to roughly 60mm of length against a ~73mm-wide plateau,
so the principal component came back MEDIOLATERAL and the reported "Anatomic
Axis Angle" ranged from 13 to 133 degrees across cases where the true figure is
about 5-7 - physiologically impossible numbers printed on a clinical report.

The fix accepts the PCA result only when it broadly agrees with an independent
reference direction (the femur->tibia centroid vector), and substitutes the
reference otherwise. calculate_anatomic_axis was imported by the test suite but
never actually called by any test, so nothing would have caught a regression
here. The synthetic point clouds below reproduce both the failing shape (wider
than it is long) and the working one.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.report_generator import (  # noqa: E402
    AXIS_AGREEMENT_THRESHOLD,
    calculate_anatomic_axis,
)

SUPERIOR = np.array([0.0, 0.0, 1.0])


def _box_cloud(ml_mm, ap_mm, si_mm, n=4000, seed=0):
    """Uniform point cloud in a box, sized (mediolateral, AP, superoinferior)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(
        low=[-ml_mm / 2, -ap_mm / 2, -si_mm / 2],
        high=[ml_mm / 2, ap_mm / 2, si_mm / 2],
        size=(n, 3),
    )


def test_pca_rejected_when_bone_is_wider_than_it_is_long():
    """
    The real failure: a knee-only tibia, ~73mm wide against ~60mm of imaged
    length. PCA locks onto the mediolateral direction, so it must be rejected
    in favour of the reference.
    """
    plateau = _box_cloud(ml_mm=73.0, ap_mm=50.0, si_mm=60.0)

    axis, pca_trusted = calculate_anatomic_axis(plateau, reference_dir=SUPERIOR)

    assert pca_trusted is False, "PCA locked onto ML here and must not be trusted"
    assert np.allclose(axis, SUPERIOR), "rejected PCA must fall back to the reference"


def test_pca_trusted_when_bone_is_genuinely_longer_than_it_is_wide():
    """A full-length bone (e.g. the CT track's larger FOV) — PCA is reliable
    here and must NOT be discarded, or the fix would throw away good data."""
    shaft = _box_cloud(ml_mm=40.0, ap_mm=40.0, si_mm=400.0)

    axis, pca_trusted = calculate_anatomic_axis(shaft, reference_dir=SUPERIOR)

    assert pca_trusted is True
    assert abs(float(np.dot(axis, SUPERIOR))) > 0.99, "should recover the shaft axis"


def test_returned_axis_points_along_the_reference_not_against_it():
    """Sign matters: the axis is used for distal-vs-proximal decisions (which
    end of the bone gets cut), so a flipped axis would resect the wrong end."""
    shaft = _box_cloud(ml_mm=40.0, ap_mm=40.0, si_mm=400.0)

    for reference in (SUPERIOR, -SUPERIOR):
        axis, _ = calculate_anatomic_axis(shaft, reference_dir=reference)
        assert float(np.dot(axis, reference)) > 0, "axis must not oppose the reference"


def test_threshold_boundary_behaviour():
    """
    Directly exercise the accept/reject decision either side of
    AXIS_AGREEMENT_THRESHOLD, rather than trusting a shape to land where
    intended. A cloud elongated along a known tilted direction lets the angle
    between PCA and the reference be set precisely.
    """
    for angle_deg, expect_trusted in [(20.0, True), (70.0, False)]:
        theta = np.radians(angle_deg)
        long_dir = np.array([np.sin(theta), 0.0, np.cos(theta)])

        rng = np.random.default_rng(1)
        t = rng.uniform(-200, 200, size=4000)[:, None]
        cloud = t * long_dir + rng.normal(scale=5.0, size=(4000, 3))

        axis, pca_trusted = calculate_anatomic_axis(cloud, reference_dir=SUPERIOR)

        assert pca_trusted is expect_trusted, (
            f"at {angle_deg} deg from the reference, "
            f"cos={np.cos(theta):.2f} vs threshold {AXIS_AGREEMENT_THRESHOLD}")
        if not pca_trusted:
            assert np.allclose(axis, SUPERIOR)


def test_no_reference_direction_falls_back_to_positive_z():
    """The reference is optional; without one the function orients to +Z and
    reports the PCA result as trusted (there is nothing to check it against)."""
    shaft = _box_cloud(ml_mm=40.0, ap_mm=40.0, si_mm=400.0)

    axis, pca_trusted = calculate_anatomic_axis(shaft, reference_dir=None)

    assert pca_trusted is True
    assert axis[2] > 0, "with no reference the axis is oriented to +Z"
