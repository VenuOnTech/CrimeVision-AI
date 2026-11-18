"""Best-effort on-disk persistence for the memory fallback.

This is NOT durable storage. It survives a controlled process restart
(close() -> save, init -> load). A disk failure, corruption, or a hard
kill can still lose everything. When Neo4j is connected, this module is
never written to or read from: Neo4j stays the source of truth.

The file format is a single JSON object with a `version` field so future
changes can migrate rather than clobber.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SNAPSHOT_VERSION = 1


class MemorySnapshot:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> Optional[Dict[str, Any]]:
        if not self.exists():
            return None
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            logger.warning("Could not read memory snapshot %s: %s", self.path, exc)
            return None
        if not isinstance(payload, dict):
            logger.warning("Memory snapshot has unexpected top-level type: %s", type(payload).__name__)
            return None
        if int(payload.get("version", 0)) != SNAPSHOT_VERSION:
            logger.warning("Memory snapshot version %r is not supported", payload.get("version"))
            return None
        return payload

    def save(
        self,
        memory_store: Dict[str, List[Dict[str, Any]]],
        cases_metadata: Dict[str, Dict[str, Any]],
        cross_case_links: List[Dict[str, Any]],
    ) -> bool:
        payload = {
            "version": SNAPSHOT_VERSION,
            "memory_store": memory_store,
            "cases_metadata": cases_metadata,
            "cross_case_links": cross_case_links,
        }
        try:
            if self.path.parent and str(self.path.parent) not in {"", "."}:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(self.path.suffix + ".tmp")
            with temp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, default=str, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
            return True
        except Exception as exc:
            logger.warning("Could not write memory snapshot %s: %s", self.path, exc)
            return False