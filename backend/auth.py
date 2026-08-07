"""
Single shared-key authentication for the API.

Scope, stated plainly: this stops an anonymous stranger who can reach the port
from reading patient scans, meshes and reports. It is NOT multi-user access
control — there are no accounts, no per-task ownership, and anyone holding the
key sees everything. That is the appropriate level for a local/demo tool; a
hospital deployment needs real identity, which is a different piece of work.

Two ways to present the key, because the browser cannot send headers on every
request the app makes:

  - `X-API-Key` header — for curl, scripts, tests.
  - A session cookie — for the browser. `<img src>` (slice images),
    `<model-viewer src>` (the AR model) and the PDF download link are all plain
    URL fetches with no way to attach a custom header, so a header-only scheme
    would authenticate the JSON calls and then 401 every image and model. The
    cookie is sent automatically on all of them.

The cookie holds an HMAC of the key rather than the key itself, so the raw
secret never sits in the browser, and it is HttpOnly so page scripts cannot
read it back out.

If KNEETWIN_API_KEY is unset, authentication is DISABLED and a warning is
logged on startup. That keeps the default local-development experience working
(and every existing test passing) rather than silently breaking a workflow that
has no key configured — but it means the protection is opt-in, so the warning
is deliberately loud.
"""

import hashlib
import hmac
import os
import secrets

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

API_KEY_ENV = "KNEETWIN_API_KEY"
SESSION_COOKIE = "kneetwin_session"
API_KEY_HEADER = "X-API-Key"

# Paths reachable without a key. The frontend itself is public so the key
# prompt has something to render in; everything that returns patient data is
# under /api/ and is protected. /api/auth is the endpoint that exchanges a key
# for a session, so it necessarily precedes having one.
PUBLIC_API_PATHS = {"/api/auth", "/api/auth/status"}


def get_api_key() -> str | None:
    """The configured key, or None when auth is disabled."""
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


def auth_enabled() -> bool:
    return get_api_key() is not None


def _session_token(api_key: str) -> str:
    """
    Cookie value: an HMAC of the key, not the key itself.

    Only someone who already holds the key can produce this, so it is no weaker
    as a credential, but a leaked cookie does not hand over the key that other
    clients and scripts are also using.
    """
    return hmac.new(api_key.encode(), b"kneetwin-session-v1", hashlib.sha256).hexdigest()


def issue_session_token() -> str | None:
    key = get_api_key()
    return _session_token(key) if key else None


def _valid_key(candidate: str | None) -> bool:
    key = get_api_key()
    if not key or not candidate:
        return False
    # Constant-time: a plain == leaks how much of the key matched via timing.
    return secrets.compare_digest(candidate, key)


def _valid_session(candidate: str | None) -> bool:
    key = get_api_key()
    if not key or not candidate:
        return False
    return secrets.compare_digest(candidate, _session_token(key))


def request_is_authenticated(request: Request) -> bool:
    if not auth_enabled():
        return True
    if _valid_key(request.headers.get(API_KEY_HEADER)):
        return True
    return _valid_session(request.cookies.get(SESSION_COOKIE))


async def auth_middleware(request: Request, call_next):
    """
    Gate /api/ on a valid key or session.

    Deliberately does not gate static files: the frontend must load in order to
    prompt for a key, and it contains no patient data. Everything that does is
    served from /api/.
    """
    path = request.url.path

    needs_auth = (
        auth_enabled()
        and path.startswith("/api/")
        and path not in PUBLIC_API_PATHS
    )

    if needs_auth and not request_is_authenticated(request):
        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication required",
                     "auth_required": True},
        )

    return await call_next(request)


def require_auth(request: Request) -> None:
    """Dependency form, for anywhere the middleware is bypassed."""
    if not request_is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required")
