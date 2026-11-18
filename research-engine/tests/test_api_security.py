import asyncio

import pytest

from api_security import (
    DEFAULT_ALLOWED_ORIGINS,
    get_positive_int,
    assert_loopback_bind,
    parse_allowed_origins,
    read_upload_limited,
    validate_case_id,
    validate_upload_filename,
)


def test_default_cors_allowlist_is_local_only():
    assert parse_allowed_origins(None) == list(DEFAULT_ALLOWED_ORIGINS)


def test_cors_allowlist_parses_and_deduplicates_explicit_origins():
    origins = parse_allowed_origins("http://localhost:3000/;http://127.0.0.1:3000,http://localhost:3000")
    assert origins == ["http://localhost:3000", "http://127.0.0.1:3000"]


@pytest.mark.parametrize("value", ["*", "https://example.com/path", "ftp://example.com", "http:///missing-host"])
def test_cors_rejects_wildcard_or_non_origin_values(value):
    with pytest.raises(ValueError):
        parse_allowed_origins(value)


def test_positive_int_env_uses_default_and_validates(monkeypatch):
    monkeypatch.delenv("TEST_LIMIT", raising=False)
    assert get_positive_int("TEST_LIMIT", 123) == 123
    monkeypatch.setenv("TEST_LIMIT", "456")
    assert get_positive_int("TEST_LIMIT", 123) == 456
    monkeypatch.setenv("TEST_LIMIT", "0")
    with pytest.raises(RuntimeError):
        get_positive_int("TEST_LIMIT", 123)


def test_case_id_validation_blocks_path_separators_and_bad_characters():
    assert validate_case_id("CASE-2026-001")
    assert not validate_case_id("../CASE-001")
    assert not validate_case_id("CASE 001")
    assert not validate_case_id("x" * 129)


def test_filename_validation_blocks_traversal_and_controls():
    assert validate_upload_filename("camera_01.mp4")
    assert not validate_upload_filename("../secret.jpg")
    assert not validate_upload_filename(r"..\\secret.jpg")
    assert not validate_upload_filename("bad\nname.jpg")
    assert not validate_upload_filename("x" * 256)


class FakeUpload:
    def __init__(self, payload: bytes):
        self.payload = payload

    async def read(self, size: int = -1) -> bytes:
        return self.payload if size < 0 else self.payload[:size]


def test_upload_limited_accepts_data_within_limit():
    result = asyncio.run(read_upload_limited(FakeUpload(b"1234"), 4))
    assert result == b"1234"


def test_upload_limited_rejects_data_over_limit():
    with pytest.raises(ValueError, match="configured 4-byte limit"):
        asyncio.run(read_upload_limited(FakeUpload(b"12345"), 4))


def test_research_api_bind_guard_allows_only_loopback_by_default():
    for host in ("127.0.0.1", "localhost", "::1"):
        assert_loopback_bind(host, allow_non_loopback=False)
    with pytest.raises(RuntimeError, match="Refusing to start"):
        assert_loopback_bind("0.0.0.0", allow_non_loopback=False)

def test_research_api_bind_guard_requires_explicit_override_for_container_network():
    assert_loopback_bind("0.0.0.0", allow_non_loopback=True)
