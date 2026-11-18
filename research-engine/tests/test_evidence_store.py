from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from evidence_store import (
    EvidenceIntegrityError,
    EvidenceNotFoundError,
    EvidenceStore,
    EvidenceStoreError,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_round_trip_preserves_bytes_and_hash(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence")
    payload = b"\x00\x01hello evidence bytes"
    saved = store.save("CASE-A", _sha(payload), ".jpg", payload)

    assert saved.size_bytes == len(payload)
    assert saved.extension == ".jpg"
    assert saved.sha256 == _sha(payload)
    assert saved.absolute_path.is_file()

    data, restored = store.read("CASE-A", saved.sha256)
    assert data == payload
    assert restored.sha256 == saved.sha256


def test_save_rejects_mismatched_declared_hash(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence")
    wrong_sha = "0" * 64
    with pytest.raises(EvidenceIntegrityError, match="does not match declared"):
        store.save("CASE-A", wrong_sha, ".jpg", b"payload")


def test_read_detects_tampering(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence")
    payload = b"original"
    saved = store.save("CASE-A", _sha(payload), ".jpg", payload)
    # Tamper with the file on disk.
    saved.absolute_path.write_bytes(b"tampered")
    with pytest.raises(EvidenceIntegrityError, match="hash mismatch"):
        store.read("CASE-A", saved.sha256)


def test_read_without_verification_returns_tampered_bytes(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence")
    saved = store.save("CASE-A", _sha(b"orig"), ".jpg", b"orig")
    saved.absolute_path.write_bytes(b"tampered")
    data, _ = store.read("CASE-A", saved.sha256, verify=False)
    assert data == b"tampered"


def test_case_id_with_path_separator_is_rejected(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence")
    with pytest.raises(ValueError, match="path separators"):
        store.save("../evil", _sha(b"x"), ".jpg", b"x")


def test_unsafe_extension_is_rejected(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence")
    with pytest.raises(ValueError, match="Unsafe file extension"):
        store.save("CASE-A", _sha(b"x"), "../etc/passwd", b"x")


def test_manifest_lists_all_files_for_a_case(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence")
    store.save("CASE-A", _sha(b"one"), ".jpg", b"one")
    store.save("CASE-A", _sha(b"two"), ".mp4", b"two")
    store.save("CASE-B", _sha(b"three"), ".png", b"three")

    manifest = store.manifest("CASE-A")
    assert {entry["sha256"] for entry in manifest} == {_sha(b"one"), _sha(b"two")}
    assert all(entry["size_bytes"] > 0 for entry in manifest)


def test_find_returns_file_without_known_extension(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence")
    saved = store.save("CASE-A", _sha(b"x"), ".mp4", b"x")
    found = store.find("CASE-A", saved.sha256)
    assert found is not None
    assert found.suffix == ".mp4"
    assert store.find("CASE-A", "0" * 64) is None


def test_missing_file_raises_not_found(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence")
    with pytest.raises(EvidenceNotFoundError):
        store.read("CASE-A", "a" * 64, extension=".jpg")


def test_relative_root_is_rejected(tmp_path: Path):
    with pytest.raises(EvidenceStoreError, match="must be absolute"):
        EvidenceStore("relative/path")


def test_save_is_idempotent_for_same_content(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence")
    payload = b"same"
    first = store.save("CASE-A", _sha(payload), ".jpg", payload)
    second = store.save("CASE-A", _sha(payload), ".jpg", payload)
    assert first.absolute_path == second.absolute_path
    assert first.absolute_path.read_bytes() == payload