"""
Regression tests for the upload path-traversal fix.

file.filename in a multipart upload is fully attacker-controlled — a client
library sets it to whatever string it likes. Unsanitized, backend/main.py used
to build the destination path as UPLOADS_DIR / f"{task_id}_{file.filename}",
and a filename with enough "../" (or "..\\") segments resolved outside
UPLOADS_DIR entirely: verified during the audit that '../../../backend/main.py'
resolved straight to the live source file, and one more level escaped the
drive. Combined with FastAPI serving frontend/ as static files from the same
app, that was a path to overwriting served JS with attacker content.

The fix is Path(file.filename).name at the top of the upload handler, which
discards every directory component. These tests assert the write is contained
regardless of which traversal style is used, on both single-request and
filesystem-visible-effect grounds — not just checking the response code, which
would not catch a version of the bug where the file still escaped but the
endpoint returned success anyway.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.pipeline_runner import UPLOADS_DIR

client = TestClient(app)

# Deliberately not a real image. sitk.ReadImage then fails regardless of
# filename/extension, which sends detect_modality down its filename-heuristic
# fallback -> UNKNOWN (none of the names below contain "ct"/"sts"/"mri"/
# "oaizib") -> /api/process returns its early 400, before ever reaching
# check_cache/run_pipeline_async. These tests are about the upload handler's
# own filesystem behavior, not segmentation, so never touching the real
# pipeline keeps them fast and — importantly — avoids racing a background
# pipeline thread against this file's test cleanup.
GARBAGE_CONTENT = b"KneeTwin test payload - not a real medical image"

# A marker name distinctive enough that finding it anywhere outside
# UPLOADS_DIR unambiguously means the traversal escaped containment.
MARKER_NAME = "kneetwin_traversal_probe.marker"

TRAVERSAL_FILENAMES = [
    f"../../../{MARKER_NAME}",
    f"..\\..\\..\\{MARKER_NAME}",              # backslash — also a separator on Windows
    f"....//....//{MARKER_NAME}",               # double-dot-slash trick
    f"C:\\Windows\\{MARKER_NAME}",               # absolute path disguised as a filename
]


def _cleanup_marker():
    """Remove the marker from every place it could have landed, safe or not."""
    for base in (Path.cwd(), Path.cwd().parent, UPLOADS_DIR, UPLOADS_DIR.parent):
        for p in base.glob(f"*{MARKER_NAME}*"):
            p.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def teardown():
    yield
    _cleanup_marker()


@pytest.mark.parametrize("malicious_name", TRAVERSAL_FILENAMES)
def test_upload_filename_cannot_escape_uploads_dir(malicious_name):
    response = client.post(
        "/api/process",
        files={"file": (malicious_name, GARBAGE_CONTENT, "application/octet-stream")},
    )
    task_id = response.json().get("task_id")
    assert task_id, "endpoint should still assign a task_id even when modality is UNKNOWN"

    # The traversal must not have landed anywhere outside UPLOADS_DIR.
    for base in (Path.cwd(), Path.cwd().parent, UPLOADS_DIR.parent):
        escaped = list(base.glob(f"*{MARKER_NAME}*"))
        assert escaped == [], f"upload escaped containment: found {escaped}"

    # And it should have landed safely inside UPLOADS_DIR, under the sanitized
    # basename only — confirms containment is a side effect of correct
    # behavior, not an accident of the attack simply failing to write anything.
    safe_matches = list(UPLOADS_DIR.glob(f"{task_id}_*{MARKER_NAME}*"))
    assert len(safe_matches) == 1
    assert safe_matches[0].parent == UPLOADS_DIR


def test_upload_normal_filename_still_works():
    """A filename with no directory components must survive sanitization
    unchanged — Path(...).name is a no-op on an already-safe basename."""
    response = client.post(
        "/api/process",
        files={"file": ("normal_scan.nii.gz", GARBAGE_CONTENT, "application/gzip")},
    )
    # Garbage content -> UNKNOWN modality -> the documented early 400. This
    # test is about filename handling, not modality detection, so the 400 is
    # expected and irrelevant to what's being asserted.
    task_id = response.json()["task_id"]

    expected = UPLOADS_DIR / f"{task_id}_normal_scan.nii.gz"
    assert expected.exists()
    expected.unlink()


def test_upload_oversized_file_rejected(monkeypatch):
    # Shrink the cap for this test rather than generating a real 500MB+ file.
    import backend.main as main_module
    monkeypatch.setattr(main_module, "MAX_UPLOAD_BYTES", 1024)

    oversized_content = b"0" * (1024 * 4)
    response = client.post(
        "/api/process",
        files={"file": ("too_big.nii.gz", oversized_content, "application/gzip")},
    )
    assert response.status_code == 413

    # No partial file should be left behind.
    leftovers = list(UPLOADS_DIR.glob("*_too_big.nii.gz"))
    assert leftovers == []
