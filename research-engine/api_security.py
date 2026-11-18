"""Small security helpers for the local CrimeVision research API."""

from __future__ import annotations

import ipaddress
import os
import re
from typing import Any
from urllib.parse import urlsplit

DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)
CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0:0:0:0:0:0:0:1"}


def parse_allowed_origins(raw: str | None) -> list[str]:
    """Parse a comma/semicolon separated, explicit HTTP(S) origin allowlist."""
    if raw is None or not raw.strip():
        return list(DEFAULT_ALLOWED_ORIGINS)
    values = [part.strip().rstrip("/") for part in re.split(r"[,;]", raw) if part.strip()]
    if not values:
        return list(DEFAULT_ALLOWED_ORIGINS)
    normalized: list[str] = []
    for origin in values:
        parsed = urlsplit(origin)
        if (
            origin == "*"
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path
            or parsed.query
            or parsed.fragment
            or "*" in origin
        ):
            raise ValueError(f"Invalid CORS origin {origin!r}; provide an explicit HTTP(S) origin without a path.")
        if origin not in normalized:
            normalized.append(origin)
    return normalized


def get_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def get_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean (1/0, true/false, yes/no, on/off)")


def assert_loopback_bind(host: str, allow_non_loopback: bool) -> None:
    """Refuse to bind the unauthenticated research API on a non-loopback host.

    The research API has no authentication or per-case authorization. Until
    that changes, allowing it to listen on 0.0.0.0 is effectively publishing
    every case and evidence hash to the network. Callers must explicitly opt
    in with ALLOW_NON_LOOPBACK=1, ideally only inside a trusted gateway.
    """
    if allow_non_loopback:
        return
    normalized = (host or "").strip().lower()
    if normalized in _LOOPBACK_HOSTS:
        return
    try:
        if ipaddress.ip_address(normalized).is_loopback:
            return
    except ValueError:
        pass
    raise RuntimeError(
        f"Refusing to start the research API on non-loopback host {host!r}. "
        "It has no authentication. Set ALLOW_NON_LOOPBACK=1 only if an "
        "authenticated reverse proxy sits in front of it."
    )


def validate_case_id(value: str) -> bool:
    return bool(value and CASE_ID_PATTERN.fullmatch(value))


def validate_upload_filename(filename: str) -> bool:
    if not filename or len(filename) > 255:
        return False
    if filename in {".", ".."} or "/" in filename or "\\" in filename:
        return False
    return not any(ord(character) < 32 or ord(character) == 127 for character in filename)


async def read_upload_limited(upload: Any, limit_bytes: int) -> bytes:
    """Read at most limit+1 bytes so oversized uploads can be rejected safely."""
    if limit_bytes <= 0:
        raise ValueError("Upload limit must be positive.")
    data = await upload.read(limit_bytes + 1)
    if len(data) > limit_bytes:
        raise ValueError(f"Uploaded file exceeds the configured {limit_bytes}-byte limit.")
    return data