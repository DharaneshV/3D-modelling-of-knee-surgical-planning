"""
Tests for the shared-key API authentication.

The property that matters is that patient data is unreachable without the key.
These assert it against the routes that actually serve that data, rather than
only against a representative one, since a route added later that forgets to
authenticate is exactly the failure this is meant to prevent — the middleware
covers /api/ wholesale precisely so a new route cannot opt out by omission.

Auth is off when KNEETWIN_API_KEY is unset, so the rest of the suite (and local
development) is unaffected; these tests set it explicitly via monkeypatch.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from backend import auth as auth_module  # noqa: E402
from backend.main import app  # noqa: E402

KEY = "test-key-do-not-use-in-production"

# Every route that returns patient-derived data. A caller without the key must
# get 401 from all of them.
PROTECTED_PATHS = [
    "/api/status/11111111-2222-3333-4444-555555555555",
    "/api/manifest/11111111-2222-3333-4444-555555555555",
    "/api/results/11111111-2222-3333-4444-555555555555",
    "/api/mesh/11111111-2222-3333-4444-555555555555/femur_unknown.obj",
    "/api/ar/11111111-2222-3333-4444-555555555555",
    "/api/resect/11111111-2222-3333-4444-555555555555",
    "/api/volume-info/11111111-2222-3333-4444-555555555555",
    "/api/slices/11111111-2222-3333-4444-555555555555/axial/10",
    "/api/report/11111111-2222-3333-4444-555555555555/pdf",
    "/api/report/11111111-2222-3333-4444-555555555555/data",
]


@pytest.fixture
def secured(monkeypatch):
    monkeypatch.setenv(auth_module.API_KEY_ENV, KEY)
    return TestClient(app)


@pytest.fixture
def unsecured(monkeypatch):
    monkeypatch.delenv(auth_module.API_KEY_ENV, raising=False)
    return TestClient(app)


@pytest.mark.parametrize("path", PROTECTED_PATHS)
def test_protected_paths_reject_anonymous_callers(secured, path):
    res = secured.get(path)
    assert res.status_code == 401, f"{path} served data without a key"
    assert res.json().get("auth_required") is True


@pytest.mark.parametrize("path", PROTECTED_PATHS)
def test_protected_paths_accept_the_api_key_header(secured, path):
    """Scripts and curl authenticate by header. A 401 here would mean the key
    was rejected; anything else means it got past auth (404/400 from the
    missing task is fine — this is not testing those routes' own behaviour)."""
    res = secured.get(path, headers={auth_module.API_KEY_HEADER: KEY})
    assert res.status_code != 401


def test_wrong_key_is_rejected(secured):
    res = secured.get(PROTECTED_PATHS[0],
                      headers={auth_module.API_KEY_HEADER: "wrong-key"})
    assert res.status_code == 401


def test_auth_endpoint_exchanges_the_key_for_a_session_cookie(secured):
    res = secured.post("/api/auth", json={"api_key": KEY})
    assert res.status_code == 200
    assert res.json()["authenticated"] is True

    cookie = res.cookies.get(auth_module.SESSION_COOKIE)
    assert cookie, "no session cookie set"
    # The raw key must not be handed to the browser — a leaked cookie should
    # not hand over the credential other clients are also using.
    assert cookie != KEY
    assert "httponly" in res.headers.get("set-cookie", "").lower()


def test_session_cookie_authenticates_subsequent_requests(secured):
    """
    The cookie is what makes <img>, <model-viewer> and the PDF link work —
    none of them can send a custom header, so if the cookie did not
    authenticate, every slice image and AR model would 401 in the browser
    while the JSON calls succeeded.
    """
    secured.post("/api/auth", json={"api_key": KEY})   # client retains the cookie

    for path in PROTECTED_PATHS:
        res = secured.get(path)
        assert res.status_code != 401, f"{path} rejected a valid session cookie"


def test_auth_endpoint_rejects_a_bad_key(secured):
    res = secured.post("/api/auth", json={"api_key": "nope"})
    assert res.status_code == 401
    assert res.cookies.get(auth_module.SESSION_COOKIE) is None


def test_status_endpoint_reports_whether_a_prompt_is_needed(secured):
    res = secured.get("/api/auth/status")
    assert res.status_code == 200
    assert res.json() == {"auth_required": True, "authenticated": False}

    secured.post("/api/auth", json={"api_key": KEY})
    assert secured.get("/api/auth/status").json()["authenticated"] is True


def test_frontend_stays_reachable_so_the_key_prompt_can_render(secured):
    """Static files are deliberately public: the page must load in order to
    ask for a key, and it carries no patient data."""
    assert secured.get("/index.html").status_code == 200


def test_auth_disabled_by_default_leaves_the_api_open(unsecured):
    """Unset key means unauthenticated — preserving local development and the
    rest of the suite. The startup warning is what makes this a deliberate
    choice rather than a silent one."""
    assert auth_module.auth_enabled() is False
    res = unsecured.get("/api/auth/status")
    assert res.json() == {"auth_required": False, "authenticated": True}
