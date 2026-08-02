import numpy as np
import pytest
import SimpleITK as sitk

from src.mesh.processing import fill_label_gaps

@pytest.fixture
def spacing_mm():
    return (1.0, 1.0, 1.0)

@pytest.fixture
def single_voxel_pit_volume(spacing_mm):
    """
    A single solid label block with one deliberate 1-voxel interior pit
    (background voxel fully surrounded by the same label).
    """
    arr = np.ones((5, 10, 10), dtype=np.uint8)  # entirely label 1
    arr[2, 5, 5] = 0  # carve a 1-voxel pit in the center

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing_mm)
    return img

def test_single_pit_is_filled(single_voxel_pit_volume):
    """Verify that binary_fill_holes correctly patches a 1-voxel hole."""
    result = fill_label_gaps(
        single_voxel_pit_volume,
        labels=[1],
        closing_radius_mm=1.0,
    )
    result_arr = sitk.GetArrayFromImage(result)

    assert result_arr[2, 5, 5] == 1, "Pit was not filled."
    assert np.all(result_arr == 1), "Entire array should be 1."