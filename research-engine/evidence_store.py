"""Filesystem evidence preservation with SHA-256 integrity verification.

Design goals:
- Preserve the exact bytes that were analysed (chain-of-custody minimum).
- Prevent path traversal and filename ambiguity.
- Verify SHA-256 on write AND on read.
- Provide a manifest for an entire case for audit purposes.

This module is NOT:
- A WORM (write-once-read-many) storage service.
- Signed: the SHA-256 confirms the bytes are intact, not who uploaded them.
- Immutable: a filesystem administrator could still delete or modify files
  on disk. The SHA-256 check on read makes tampering *detectable*, but that
  is not the same as preventing it.

The caller supplies the storage root; this module performs no environment
access. See main.py's `_resolve_evidence_root()` for how EVIDENCE_STORAGE_ROOT
is honoured.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_EXTENSION = re.compile(r"^\.[A-Za-z0-9]{1,8}$")


class EvidenceStoreError(RuntimeError):
    """Base class for evidence-storage errors."""


class EvidenceIntegrityError(EvidenceStoreError):
    """Raised when a stored or supplied file does not match its SHA-256."""


class EvidenceNotFoundError(EvidenceStoreError):
    """Raised when a requested stored file does not exist."""


@dataclass(frozen=True)
class StoredEvidence:
    case_id: str
    sha256: str
    extension: str
    relative_path: str
    absolute_path: Path
    size_bytes: int

    def to_metadata(self) -> dict:
        return {
            "case_id": self.case_id,
            "sha256": self.sha256,
            "extension": self.extension,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
        }


class EvidenceStore:
    """Content-addressed file storage scoped by case_id."""

    def __init__(self, root: str | Path) -> None:
        root_path = Path(root).expanduser()
        if not root_path.is_absolute():
            raise EvidenceStoreError(f"Evidence root must be absolute, got: {root_path}")
        root_path.mkdir(parents=True, exist_ok=True)
        if not root_path.is_dir():
            raise EvidenceStoreError(f"Evidence root is not a directory: {root_path}")
        self.root = root_path.resolve()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_sha256(value: str) -> str:
        lowered = (value or "").strip().lower()
        if not _HEX_SHA256.fullmatch(lowered):
            raise ValueError("SHA-256 must be 64 lowercase hexadecimal characters")
        return lowered

    @staticmethod
    def _validate_case_id(value: str) -> str:
        case_id = (value or "").strip()
        if not case_id or len(case_id) > 128:
            raise ValueError("case_id must be non-empty and <= 128 characters")
        if "/" in case_id or "\\" in case_id or case_id in {".", ".."}:
            raise ValueError("case_id must not contain path separators")
        if any(ord(c) < 32 for c in case_id):
            raise ValueError("case_id must not contain control characters")
        return case_id

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        ext = (extension or "").strip()
        if not ext:
            return ".bin"
        if not ext.startswith("."):
            ext = "." + ext
        if not _SAFE_EXTENSION.fullmatch(ext):
            raise ValueError(f"Unsafe file extension: {extension!r}")
        return ext.lower()

    def _case_dir(self, case_id: str) -> Path:
        safe_case = self._validate_case_id(case_id)
        target = (self.root / safe_case).resolve()
        # Defense-in-depth: confirm the resolved path is still inside root.
        if self.root != target and self.root not in target.parents:
            raise ValueError("Resolved case directory escapes storage root")
        return target

    def _target_path(self, case_id: str, sha256: str, extension: str) -> Path:
        safe_sha = self._validate_sha256(sha256)
        safe_ext = self._normalize_extension(extension)
        return self._case_dir(case_id) / f"{safe_sha}{safe_ext}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def save(self, case_id: str, sha256: str, extension: str, data: bytes) -> StoredEvidence:
        """Atomically persist `data`, verifying that it matches `sha256`."""
        if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
            raise ValueError("Evidence data must be a non-empty bytes object")
        safe_sha = self._validate_sha256(sha256)
        computed = hashlib.sha256(bytes(data)).hexdigest()
        if computed != safe_sha:
            raise EvidenceIntegrityError(
                f"Computed SHA-256 {computed} does not match declared {safe_sha}"
            )
        case_dir = self._case_dir(case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        target = self._target_path(case_id, safe_sha, extension)

        # Write to a temp file in the same directory, fsync, then rename.
        # This is atomic on POSIX and on modern Windows; it cannot leave a
        # partially-written file at `target` under normal failure modes.
        fd, tmp_name = tempfile.mkstemp(prefix=".ev_", dir=case_dir)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(bytes(data))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        except Exception:
            try:
                os.remove(tmp_name)
            except OSError:
                pass
            raise

        return StoredEvidence(
            case_id=case_id,
            sha256=safe_sha,
            extension=self._normalize_extension(extension),
            relative_path=str(target.relative_to(self.root)),
            absolute_path=target,
            size_bytes=len(data),
        )

    def read(
        self,
        case_id: str,
        sha256: str,
        extension: Optional[str] = None,
        verify: bool = True,
    ) -> tuple[bytes, StoredEvidence]:
        """Read a stored file, re-verifying its SHA-256 by default."""
        if extension is None:
            found = self.find(case_id, sha256)
            if found is None:
                raise EvidenceNotFoundError(
                    f"No stored evidence for case {case_id!r} with SHA-256 {sha256!r}"
                )
            extension = found.suffix
        target = self._target_path(case_id, sha256, extension)
        if not target.is_file():
            raise EvidenceNotFoundError(f"No stored evidence at {target.name}")
        data = target.read_bytes()
        if verify:
            computed = hashlib.sha256(data).hexdigest()
            if computed != self._validate_sha256(sha256):
                raise EvidenceIntegrityError(
                    f"Stored file hash mismatch for {target.name}: "
                    f"expected {sha256}, computed {computed}"
                )
        return data, StoredEvidence(
            case_id=case_id,
            sha256=self._validate_sha256(sha256),
            extension=self._normalize_extension(extension),
            relative_path=str(target.relative_to(self.root)),
            absolute_path=target,
            size_bytes=len(data),
        )

    def exists(self, case_id: str, sha256: str, extension: str) -> bool:
        try:
            return self._target_path(case_id, sha256, extension).is_file()
        except ValueError:
            return False

    def find(self, case_id: str, sha256: str) -> Optional[Path]:
        """Return the on-disk file for a given SHA-256 regardless of extension."""
        safe_sha = self._validate_sha256(sha256)
        case_dir = self._case_dir(case_id)
        if not case_dir.is_dir():
            return None
        for candidate in case_dir.iterdir():
            if candidate.is_file() and candidate.name.startswith(safe_sha + "."):
                return candidate
        return None

    def manifest(self, case_id: str) -> list[dict]:
        """List every preserved file for a case, with size and hash."""
        case_dir = self._case_dir(case_id)
        if not case_dir.is_dir():
            return []
        entries: list[dict] = []
        for entry in sorted(case_dir.iterdir()):
            if not entry.is_file() or entry.name.startswith("."):
                continue
            sha_part, dot, ext_part = entry.name.partition(".")
            if not dot or not _HEX_SHA256.fullmatch(sha_part):
                continue
            entries.append({
                "sha256": sha_part,
                "extension": "." + ext_part,
                "relative_path": str(entry.relative_to(self.root)),
                "size_bytes": entry.stat().st_size,
            })
        return entries