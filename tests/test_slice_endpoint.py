import pytest
import os
import shutil
from fastapi.testclient import TestClient
from PIL import Image
import io
import json

from backend.main import app
from backend.pipeline_runner import UPLOADS_DIR, TASKS_DIR

client = TestClient(app)

TASK_ID = "test_task_id_999"

@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup
    fixture_path = "tests/fixtures/dummy.nii.gz"
    if not os.path.exists(fixture_path):
        raise RuntimeError("Fixture dummy.nii.gz not found")
        
    upload_dest = UPLOADS_DIR / f"{TASK_ID}_dummy.nii.gz"
    shutil.copy(fixture_path, upload_dest)
    
    # Create dummy status to bypass modality unknown issue
    task_dir = TASKS_DIR / TASK_ID
    task_dir.mkdir(parents=True, exist_ok=True)
    with open(task_dir / "status.json", "w") as f:
        json.dump({"modality": "CT"}, f)
        
    yield
    
    # Teardown
    if upload_dest.exists():
        upload_dest.unlink()
    if task_dir.exists():
        shutil.rmtree(task_dir)

def test_volume_info():
    res = client.get(f"/api/volume-info/{TASK_ID}")
    assert res.status_code == 200
    data = res.json()
    assert data["num_slices"]["sagittal"] == 50
    assert data["num_slices"]["coronal"] == 40
    assert data["num_slices"]["axial"] == 30
    assert data["modality"] == "CT"

def test_slice_endpoint():
    # Get a mid-axial slice
    res = client.get(f"/api/slices/{TASK_ID}/axial/15")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    
    content = res.content
    assert len(content) > 0 # Non-zero byte size
    
    # Verify dimensions match the expected metadata
    # The dummy array is (Z=30, Y=40, X=50) -> sitk Size is [50, 40, 30]
    # An axial slice (Z=15) should have dimensions [X=50, Y=40]
    img = Image.open(io.BytesIO(content))
    assert img.size == (50, 40)
    
def test_slice_endpoint_out_of_bounds():
    res = client.get(f"/api/slices/{TASK_ID}/axial/999")
    assert res.status_code == 400
    
def test_slice_endpoint_invalid_plane():
    res = client.get(f"/api/slices/{TASK_ID}/diagonal/15")
    assert res.status_code == 400
    
def test_slice_endpoint_invalid_task():
    res = client.get("/api/slices/DOES_NOT_EXIST_123/axial/15")
    assert res.status_code == 404
