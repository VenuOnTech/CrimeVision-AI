"""Case-scoped Neo4j persistence for CrimeVision AI.

This module stores investigation metadata and analysis records. It does not
calculate MATCH / CONTRADICTION decisions: those are produced by the research
inference engine. If Neo4j is unavailable, writes are kept in a process-local
memory store and explicitly reported as volatile (not durable across restarts).

New VisualEvidence and WitnessStatement nodes are scoped to a case using stable
SHA-256-derived IDs. Analysis artefacts (heatmap matrix, tokens, patches, model
provenance) are persisted on the CORRELATES_TO relationship so a stored record
can be reproduced and audited without re-running inference.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional

from memory_snapshot import MemorySnapshot

try:
    from neo4j import GraphDatabase
except ImportError:  # Permit memory-only unit tests and development installs.
    GraphDatabase = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
_ALLOWED_ASSESSMENTS = {"MATCH", "CONTRADICTION", "INDETERMINATE"}
_RELATION_TYPE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")

# Bounds on persisted analysis artefacts. These are defensive limits, not
# expected values. A CLIP ViT-B/32 analysis on a typical witness statement
# produces ~49 patches x <=76 tokens = <=3,724 cells, well under MAX_HEATMAP_CELLS.
MAX_HEATMAP_ROWS = 1000
MAX_HEATMAP_COLS = 1000
MAX_HEATMAP_CELLS = 250_000
MAX_STRING_LIST_ITEMS = 1000
MAX_PROVENANCE_STRING = 512


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _stable_id(*parts: str, prefix: str = "") -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}{hashlib.sha256(payload).hexdigest()}"


def _json_object(value: Any) -> Dict[str, Any]:
    """Return a JSON-safe dictionary for evidence-quality metadata."""
    if value is None:
        return {}
    if isinstance(value, dict):
        try:
            result = json.loads(json.dumps(value, default=str))
            return result if isinstance(result, dict) else {"value": result}
        except (TypeError, ValueError):
            return {"value": str(value)}
    return {"value": str(value)}


def _record_data(record: Any) -> Dict[str, Any]:
    """Normalize a Neo4j Record or mapping into a plain dictionary."""
    if hasattr(record, "data"):
        return record.data()
    return dict(record)


# ---------------------------------------------------------------------------
# Validation helpers for persisted analysis artefacts
# ---------------------------------------------------------------------------
def _validate_heatmap(value: Any) -> Optional[List[List[float]]]:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError("heatmap_data must be a non-empty list of rows")
    if len(value) > MAX_HEATMAP_ROWS:
        raise ValueError(f"heatmap_data has too many rows (max {MAX_HEATMAP_ROWS})")
    total_cells = 0
    normalized: List[List[float]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list):
            raise ValueError(f"heatmap_data[{row_index}] must be a list")
        if len(row) > MAX_HEATMAP_COLS:
            raise ValueError(f"heatmap_data[{row_index}] has too many columns (max {MAX_HEATMAP_COLS})")
        total_cells += len(row)
        if total_cells > MAX_HEATMAP_CELLS:
            raise ValueError(f"heatmap_data exceeds {MAX_HEATMAP_CELLS} cells")
        normalized_row: List[float] = []
        for cell in row:
            try:
                float_cell = float(cell)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"heatmap_data[{row_index}] contains a non-numeric value"
                ) from exc
            if not math.isfinite(float_cell):
                raise ValueError(f"heatmap_data[{row_index}] contains a non-finite value")
            normalized_row.append(float_cell)
        normalized.append(normalized_row)
    return normalized


def _validate_string_list(value: Any, field_name: str) -> Optional[List[str]]:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings")
    if len(value) > MAX_STRING_LIST_ITEMS:
        raise ValueError(f"{field_name} has too many items (max {MAX_STRING_LIST_ITEMS})")
    return [str(item) for item in value]


def _validate_class_probabilities(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("class_probabilities must be a list with at least two items")
    normalized: List[float] = []
    for item in value:
        try:
            float_item = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError("class_probabilities must contain numeric values") from exc
        if not math.isfinite(float_item) or not 0.0 <= float_item <= 1.0:
            raise ValueError("class_probabilities must be finite floats in [0, 1]")
        normalized.append(float_item)
    total = sum(normalized)
    if abs(total - 1.0) > 1e-3:
        raise ValueError(f"class_probabilities must sum to approximately 1 (got {total})")
    return normalized


def _validate_provenance_string(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) > MAX_PROVENANCE_STRING:
        raise ValueError(f"{field_name} exceeds {MAX_PROVENANCE_STRING} characters")
    return text or None


def _validate_optional_float(value: Any, field_name: str) -> Optional[float]:
    if value is None:
        return None
    try:
        float_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(float_value):
        raise ValueError(f"{field_name} must be a finite number")
    return float_value


def _validate_optional_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    return int_value


def _dumps_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=False, separators=(",", ":"))


def _loads_or_none(raw: Any) -> Any:
    if raw is None or raw == "":
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


class KnowledgeGraphEngine:
    """Manage cases, evidence pairs, and cross-case links."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "",
        *,
        driver: Any = None,
        connect: bool = True,
        ensure_constraints: bool = True,
        snapshot_path: Optional[str] = None,
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = driver
        self.memory_store: Dict[str, List[Dict[str, Any]]] = {}
        self.cases_metadata: Dict[str, Dict[str, Any]] = {}
        self.cross_case_links: List[Dict[str, Any]] = []
        self._neo4j_healthy = False
        self._last_storage_error: Optional[str] = None
        self._constraints_ready = False
        self._last_persistence_status: Dict[str, Any] = {
            "backend": "memory",
            "durable": False,
            "neo4j_connected": False,
            "persisted": False,
            "message": "Neo4j has not been connected.",
            "last_error": None,
        }

        if self.driver is None and connect and GraphDatabase is not None:
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                )
            except Exception as exc:  # pragma: no cover - environment dependent
                self._mark_database_unavailable(exc)

        if self.driver is not None:
            try:
                self.driver.verify_connectivity()
                self._neo4j_healthy = True
                if ensure_constraints:
                    self._ensure_constraints()
                logger.info("Connected to Neo4j Knowledge Graph")
            except Exception as exc:  # pragma: no cover - environment dependent
                self._mark_database_unavailable(exc)
        elif GraphDatabase is None:
            self._last_storage_error = "neo4j Python driver is not installed"

        self._snapshot: Optional[MemorySnapshot] = (
            MemorySnapshot(snapshot_path) if snapshot_path else None
        )
        if self._snapshot is not None and self.driver is None:
            loaded = self._snapshot.load()
            if loaded:
                try:
                    self._restore_from_snapshot(loaded)
                except Exception as exc:
                    logger.warning("Could not restore memory snapshot: %s", exc)

        self._last_persistence_status = self.get_storage_status()

    # ------------------------------------------------------------------
    # Connection, status, and transaction helpers
    # ------------------------------------------------------------------
    def _mark_database_unavailable(self, exc: Exception) -> None:
        self._neo4j_healthy = False
        self._last_storage_error = str(exc)
        logger.warning("Neo4j unavailable; using volatile in-memory storage: %s", exc)
        if self.driver is not None:
            try:
                self.driver.close()
            except Exception:  # pragma: no cover
                pass
        self.driver = None

    def _ensure_constraints(self) -> None:
        if self.driver is None or not self._neo4j_healthy:
            return
        statements = (
            "CREATE CONSTRAINT crimevision_case_id IF NOT EXISTS "
            "FOR (n:Case) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT crimevision_visual_id IF NOT EXISTS "
            "FOR (n:VisualEvidence) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT crimevision_statement_id IF NOT EXISTS "
            "FOR (n:WitnessStatement) REQUIRE n.id IS UNIQUE",
        )
        all_succeeded = True
        try:
            with self.driver.session() as session:
                for statement in statements:
                    try:
                        session.run(statement).consume()
                    except Exception as exc:  # pragma: no cover
                        all_succeeded = False
                        logger.warning("Could not create Neo4j constraint: %s", exc)
            self._constraints_ready = all_succeeded
        except Exception as exc:  # pragma: no cover
            self._constraints_ready = False
            logger.warning("Neo4j constraint setup was not completed: %s", exc)

    def _run(self, query: str, **params: Any) -> List[Dict[str, Any]]:
        if self.driver is None or not self._neo4j_healthy:
            raise RuntimeError("Neo4j is not connected")
        try:
            with self.driver.session() as session:
                result = session.run(query, **params)
                rows = [_record_data(row) for row in result]
            self._last_storage_error = None
            return rows
        except Exception as exc:
            self._mark_database_unavailable(exc)
            raise

    def _write_transaction(self, work: Callable[[Any], Any]) -> Any:
        if self.driver is None or not self._neo4j_healthy:
            raise RuntimeError("Neo4j is not connected")
        try:
            with self.driver.session() as session:
                result = session.execute_write(work)
            self._last_storage_error = None
            return result
        except Exception as exc:
            self._mark_database_unavailable(exc)
            raise

    def get_storage_status(self) -> Dict[str, Any]:
        connected = bool(self.driver is not None and self._neo4j_healthy)
        return {
            "backend": "neo4j" if connected else "memory",
            "durable": connected,
            "neo4j_connected": connected,
            "constraints_ready": self._constraints_ready if connected else False,
            "message": (
                "Neo4j connection is available; successful writes are durable."
                if connected
                else "Stored only in process memory; records may be lost when the API restarts."
            ),
            "last_error": self._last_storage_error,
        }

    def close(self) -> None:
        self._flush_snapshot()
        if self.driver is not None:
            try:
                self.driver.close()
            finally:
                self.driver = None
                self._neo4j_healthy = False

    def _restore_from_snapshot(self, payload: Dict[str, Any]) -> None:
        memory = payload.get("memory_store") or {}
        cases = payload.get("cases_metadata") or {}
        links = payload.get("cross_case_links") or []
        if isinstance(memory, dict):
            for case_id, records in memory.items():
                if isinstance(records, list):
                    self.memory_store[str(case_id)] = [
                        rec for rec in records if isinstance(rec, dict)
                    ]
        if isinstance(cases, dict):
            for case_id, meta in cases.items():
                if isinstance(meta, dict):
                    self.cases_metadata[str(case_id)] = dict(meta)
        if isinstance(links, list):
            self.cross_case_links = [link for link in links if isinstance(link, dict)]
        logger.info(
            "Restored memory snapshot: %d cases, %d evidence records, %d links",
            len(self.cases_metadata),
            sum(len(v) for v in self.memory_store.values()),
            len(self.cross_case_links),
        )

    def _flush_snapshot(self) -> None:
        snapshot = getattr(self, "_snapshot", None)
        if snapshot is None:
            return
        if getattr(self, "_neo4j_healthy", False):
            return
        try:
            snapshot.save(
                getattr(self, "memory_store", {}),
                getattr(self, "cases_metadata", {}),
                getattr(self, "cross_case_links", []),
            )
        except Exception as exc:
            logger.warning("Memory snapshot flush failed: %s", exc)

    def flush_snapshot(self) -> Dict[str, Any]:
        """Explicitly persist the memory store to disk (no-op when Neo4j is healthy)."""
        snapshot = getattr(self, "_snapshot", None)
        if snapshot is None:
            return {"written": False, "reason": "snapshot path not configured"}
        if self._neo4j_healthy:
            return {"written": False, "reason": "Neo4j is connected; snapshot is not needed"}
        written = snapshot.save(self.memory_store, self.cases_metadata, self.cross_case_links)
        return {
            "written": written,
            "path": str(snapshot.path),
            "case_count": len(self.cases_metadata),
            "evidence_count": sum(len(v) for v in self.memory_store.values()),
        }

    # ------------------------------------------------------------------
    # Case metadata
    # ------------------------------------------------------------------
    def _load_case_metadata(self, case_id: str) -> Optional[Dict[str, Any]]:
        cached = self.cases_metadata.get(case_id)
        if cached is not None:
            return cached
        if self.driver is None or not self._neo4j_healthy:
            return None
        try:
            rows = self._run(
                """
                MATCH (c:Case {id: $case_id})
                RETURN c.id AS id, c.name AS name,
                       c.description AS description, c.category AS category,
                       toString(c.created_at) AS created_at
                """,
                case_id=case_id,
            )
            if not rows:
                return None
            row = rows[0]
            record = {
                "id": case_id,
                "name": row.get("name") or f"Case {case_id}",
                "description": row.get("description") or "Active forensic investigation case.",
                "category": row.get("category") or "General Forensics",
                "created_at": row.get("created_at") or _utc_now(),
            }
            self.cases_metadata[case_id] = record
            self.memory_store.setdefault(case_id, [])
            return record
        except Exception:
            return None

    def create_case(
        self,
        case_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        case_id = str(case_id or "").strip()
        if not case_id:
            raise ValueError("case_id must not be empty")
        if len(case_id) > 128:
            raise ValueError("case_id must be 128 characters or fewer")
        if self._load_case_metadata(case_id) is not None:
            raise ValueError(f"Case '{case_id}' already exists")

        created_at = _utc_now()
        record = {
            "id": case_id,
            "name": (name or f"Case {case_id}").strip(),
            "description": (description or "Active forensic investigation case.").strip(),
            "category": (category or "General Forensics").strip(),
            "created_at": created_at,
        }
        if not record["name"]:
            raise ValueError("case name must not be empty")

        persisted = False
        if self.driver is not None and self._neo4j_healthy:
            def create_tx(tx: Any) -> Optional[Dict[str, Any]]:
                result = tx.run(
                    """
                    CREATE (c:Case {
                        id: $case_id, name: $name,
                        description: $description, category: $category,
                        created_at: $created_at
                    })
                    RETURN c.id AS id
                    """,
                    case_id=case_id,
                    name=record["name"],
                    description=record["description"],
                    category=record["category"],
                    created_at=created_at,
                ).single()
                return _record_data(result) if result is not None else None

            try:
                created = self._write_transaction(create_tx)
                persisted = bool(created and created.get("id") == case_id)
                if not persisted:
                    raise RuntimeError("Neo4j did not confirm case creation")
            except Exception as exc:
                self._last_storage_error = str(exc)

        self.cases_metadata[case_id] = record
        self.memory_store.setdefault(case_id, [])
        self._last_persistence_status = {
            **self.get_storage_status(),
            "persisted": persisted,
            "message": "Case persisted to Neo4j." if persisted else "Case exists only in volatile process memory.",
            "error": self._last_storage_error if not persisted else None,
        }
        return dict(record)

    def update_case(
        self,
        case_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        case_id = str(case_id or "").strip()
        if not case_id:
            raise ValueError("case_id must not be empty")
        current = self._load_case_metadata(case_id)
        if current is None:
            raise KeyError(f"Case '{case_id}' does not exist")

        updated = dict(current)
        if name is not None:
            if not name.strip():
                raise ValueError("case name must not be empty")
            updated["name"] = name.strip()
        if description is not None:
            updated["description"] = description.strip()
        if category is not None:
            updated["category"] = category.strip()

        persisted = False
        if self.driver is not None and self._neo4j_healthy:
            def update_tx(tx: Any) -> Optional[Dict[str, Any]]:
                result = tx.run(
                    """
                    MATCH (c:Case {id: $case_id})
                    SET c.name = $name,
                        c.description = $description,
                        c.category = $category
                    RETURN c.id AS id
                    """,
                    case_id=case_id,
                    name=updated["name"],
                    description=updated["description"],
                    category=updated["category"],
                ).single()
                return _record_data(result) if result is not None else None

            try:
                result = self._write_transaction(update_tx)
                if result is None:
                    raise KeyError(f"Case '{case_id}' does not exist in Neo4j")
                persisted = True
            except KeyError:
                raise
            except Exception as exc:
                self._last_storage_error = str(exc)

        self.cases_metadata[case_id] = updated
        self._last_persistence_status = {
            **self.get_storage_status(),
            "persisted": persisted,
            "message": "Case update persisted to Neo4j." if persisted else "Case update exists only in volatile process memory.",
            "error": self._last_storage_error if not persisted else None,
        }
        return dict(updated)

    def get_cases(self) -> List[Dict[str, Any]]:
        cases: Dict[str, Dict[str, Any]] = {
            case_id: dict(metadata)
            for case_id, metadata in self.cases_metadata.items()
        }
        if self.driver is not None and self._neo4j_healthy:
            try:
                rows = self._run(
                    """
                    MATCH (c:Case)
                    OPTIONAL MATCH (c)-[:CONTAINS]->(t:WitnessStatement)-[r:CORRELATES_TO]->(v:VisualEvidence)
                    WHERE r.case_id = c.id
                       OR (r IS NULL)
                       OR (r.case_id IS NULL AND t.case_id = c.id AND v.case_id = c.id)
                    RETURN c.id AS id, c.name AS name,
                           c.description AS description, c.category AS category,
                           toString(c.created_at) AS created_at,
                           count(DISTINCT r) AS evidence_count,
                           sum(CASE WHEN r.assessment_status = 'CONTRADICTION' THEN 1 ELSE 0 END) AS contradiction_count,
                           sum(CASE WHEN r.assessment_status = 'MATCH' THEN 1 ELSE 0 END) AS congruent_count,
                           sum(CASE WHEN r.assessment_status IS NULL OR r.assessment_status = 'INDETERMINATE' THEN 1 ELSE 0 END) AS indeterminate_count
                    """
                )
                for row in rows:
                    case_id = row.get("id")
                    if not case_id:
                        continue
                    cases[case_id] = {
                        "id": case_id,
                        "name": row.get("name") or f"Case {case_id}",
                        "description": row.get("description") or "Active forensic investigation case.",
                        "category": row.get("category") or "General Forensics",
                        "created_at": row.get("created_at") or _utc_now(),
                        "evidence_count": int(row.get("evidence_count") or 0),
                        "contradiction_count": int(row.get("contradiction_count") or 0),
                        "congruent_count": int(row.get("congruent_count") or 0),
                        "indeterminate_count": int(row.get("indeterminate_count") or 0),
                    }
            except Exception:
                logger.exception("Could not load cases from Neo4j")

        for case_id, records in self.memory_store.items():
            if case_id not in cases:
                cases[case_id] = {
                    **self.cases_metadata.get(
                        case_id,
                        {
                            "id": case_id,
                            "name": f"Case {case_id}",
                            "description": "Active forensic investigation case.",
                            "category": "General Forensics",
                            "created_at": _utc_now(),
                        },
                    )
                }
            if not self._neo4j_healthy:
                cases[case_id].update(self._count_records(records))
            else:
                cases[case_id].setdefault("evidence_count", len(records))
                cases[case_id].setdefault("contradiction_count", 0)
                cases[case_id].setdefault("congruent_count", 0)
                cases[case_id].setdefault("indeterminate_count", 0)

        for case in cases.values():
            case.setdefault("evidence_count", 0)
            case.setdefault("contradiction_count", 0)
            case.setdefault("congruent_count", 0)
            case.setdefault("indeterminate_count", 0)
        return list(cases.values())

    @staticmethod
    def _count_records(records: List[Dict[str, Any]]) -> Dict[str, int]:
        return {
            "evidence_count": len(records),
            "contradiction_count": sum(row.get("assessment_status") == "CONTRADICTION" for row in records),
            "congruent_count": sum(row.get("assessment_status") == "MATCH" for row in records),
            "indeterminate_count": sum(row.get("assessment_status", "INDETERMINATE") == "INDETERMINATE" for row in records),
        }

    # ------------------------------------------------------------------
    # Cross-case relationships
    # ------------------------------------------------------------------
    def add_cross_case_relation(
        self,
        source_case_id: str,
        target_case_id: str,
        relation_type: str = "SHARED_SUSPECT",
        description: str = "",
        confidence: float = 0.85,
        shared_entity: Optional[str] = None,
    ) -> Dict[str, Any]:
        source_case_id = str(source_case_id or "").strip()
        target_case_id = str(target_case_id or "").strip()
        if not source_case_id or not target_case_id or source_case_id == target_case_id:
            raise ValueError("Cross-case links require two different non-empty case IDs")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")
        relation_type = str(relation_type or "SHARED_SUSPECT").strip().upper()
        if not _RELATION_TYPE.fullmatch(relation_type):
            raise ValueError("relation_type must contain only uppercase letters, digits, and underscores")
        if self._load_case_metadata(source_case_id) is None:
            raise KeyError(f"Source case '{source_case_id}' does not exist")
        if self._load_case_metadata(target_case_id) is None:
            raise KeyError(f"Target case '{target_case_id}' does not exist")

        link_id = _stable_id(source_case_id, target_case_id, relation_type, prefix="link_")
        record: Dict[str, Any] = {
            "id": link_id,
            "source_case_id": source_case_id,
            "target_case_id": target_case_id,
            "relation_type": relation_type,
            "description": description or f"Correlated {relation_type.replace('_', ' ').title()}",
            "confidence": confidence,
            "confidence_interpretation": (
                "heuristic_candidate_not_calibrated"
                if relation_type == "POTENTIAL_KEYWORD_OVERLAP"
                else "investigator_assigned_not_calibrated"
            ),
            "shared_entity": shared_entity or "",
            "timestamp": _utc_now(),
            "persisted": False,
            "storage_backend": "memory",
        }

        persisted = False
        if self.driver is not None and self._neo4j_healthy:
            def create_link_tx(tx: Any) -> Optional[Dict[str, Any]]:
                result = tx.run(
                    """
                    MATCH (source:Case {id: $source_id}), (target:Case {id: $target_id})
                    MERGE (source)-[r:CROSS_CASE_CORRELATION {id: $link_id}]->(target)
                    SET r.relation_type = $relation_type,
                        r.description = $description,
                        r.confidence = $confidence,
                        r.confidence_interpretation = $confidence_interpretation,
                        r.shared_entity = $shared_entity,
                        r.timestamp = $timestamp
                    RETURN r.id AS id
                    """,
                    source_id=source_case_id,
                    target_id=target_case_id,
                    link_id=link_id,
                    relation_type=relation_type,
                    description=record["description"],
                    confidence=confidence,
                    confidence_interpretation=record["confidence_interpretation"],
                    shared_entity=record["shared_entity"],
                    timestamp=record["timestamp"],
                ).single()
                return _record_data(result) if result is not None else None

            try:
                result = self._write_transaction(create_link_tx)
                if result is None:
                    raise RuntimeError("Neo4j did not find both cases; link was not persisted")
                persisted = True
            except Exception as exc:
                self._last_storage_error = str(exc)

        record["persisted"] = persisted
        record["storage_backend"] = "neo4j" if persisted else "memory"
        record["persistence_message"] = (
            "Cross-case relationship persisted to Neo4j."
            if persisted
            else "Cross-case relationship exists only in volatile process memory."
        )
        self.cross_case_links = [item for item in self.cross_case_links if item["id"] != link_id]
        self.cross_case_links.append(record)
        return dict(record)

    def get_cross_case_relations(self) -> List[Dict[str, Any]]:
        links: Dict[str, Dict[str, Any]] = {
            item["id"]: dict(item) for item in self.cross_case_links
        }
        if self.driver is not None and self._neo4j_healthy:
            try:
                rows = self._run(
                    """
                    MATCH (source:Case)-[r:CROSS_CASE_CORRELATION]->(target:Case)
                    RETURN r.id AS id,
                           source.id AS source_case_id,
                           target.id AS target_case_id,
                           coalesce(r.relation_type, r.type, 'SHARED_SUSPECT') AS relation_type,
                           coalesce(r.description, '') AS description,
                           coalesce(r.confidence, 0.0) AS confidence,
                           coalesce(r.confidence_interpretation, 'legacy_unverified') AS confidence_interpretation,
                           coalesce(r.shared_entity, '') AS shared_entity,
                           toString(r.timestamp) AS timestamp
                    """
                )
                for row in rows:
                    link_id = row.get("id") or _stable_id(
                        row.get("source_case_id", ""),
                        row.get("target_case_id", ""),
                        row.get("relation_type", ""),
                        prefix="link_",
                    )
                    links[link_id] = {
                        **row,
                        "id": link_id,
                        "persisted": True,
                        "storage_backend": "neo4j",
                        "persistence_message": "Cross-case relationship loaded from Neo4j.",
                    }
            except Exception:
                logger.exception("Could not load cross-case relationships from Neo4j")
        return list(links.values())

    # ------------------------------------------------------------------
    # Evidence records
    # ------------------------------------------------------------------
    def _ensure_case_metadata(self, case_id: str) -> Dict[str, Any]:
        existing = self._load_case_metadata(case_id)
        if existing is not None:
            return existing
        return self.create_case(case_id, name=f"Case {case_id}")

    def create_evidential_link(
        self,
        case_id: str,
        evidence_filename: str,
        statement_text: str,
        sinkhorn_cost: float,
        timestamp: Optional[str] = None,
        audit_report: Optional[str] = None,
        assessment_status: str = "INDETERMINATE",
        model_epistemic_uncertainty: Optional[float] = None,
        visual_quality_metrics: Optional[Dict[str, Any]] = None,
        is_blurry: Optional[bool] = None,
        visual_uncertainty: Optional[float] = None,
        quality_label: str = "",
        investigator_advisory: Optional[str] = None,
        discrepancy_info: Optional[str] = None,
        evidence_sha256: Optional[str] = None,
        # ---- NEW: analysis artefacts persisted for reproducibility (#69, #77) ----
        heatmap_data: Optional[List[List[float]]] = None,
        tokens: Optional[List[str]] = None,
        patches: Optional[List[str]] = None,
        ot_diagnostics: Optional[Dict[str, Any]] = None,
        global_clip_similarity: Optional[float] = None,
        standard_entropic_ot_cost: Optional[float] = None,
        class_probabilities: Optional[List[float]] = None,
                # ---- NEW: model provenance persisted for audit (#65, #88) ----
        model_name: Optional[str] = None,
        model_revision: Optional[str] = None,
        checkpoint_sha256: Optional[str] = None,
        calibration_version: Optional[int] = None,
        feature_schema_version: Optional[int] = None,
        # ---- NEW: evidence preservation provenance (#25, #27) ----
        uploader_id: Optional[str] = None,
        acquisition_timestamp: Optional[str] = None,
        ingestion_timestamp: Optional[str] = None,
        storage_path: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        original_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        case_id = str(case_id or "").strip()
        evidence_filename = str(evidence_filename or "uploaded_evidence").strip()
        statement_text = str(statement_text or "").strip()
        if not case_id or not statement_text:
            raise ValueError("case_id and statement_text are required")
        if len(case_id) > 128:
            raise ValueError("case_id must be 128 characters or fewer")

        assessment_status = str(assessment_status or "INDETERMINATE").upper()
        if assessment_status not in _ALLOWED_ASSESSMENTS:
            raise ValueError(f"Unsupported assessment_status: {assessment_status}")
        sinkhorn_cost = float(sinkhorn_cost)
        if not math.isfinite(sinkhorn_cost) or sinkhorn_cost < -1e-8:
            raise ValueError("sinkhorn_cost must be finite and non-negative")
        sinkhorn_cost = max(sinkhorn_cost, 0.0)

        if model_epistemic_uncertainty is not None:
            model_epistemic_uncertainty = float(model_epistemic_uncertainty)
            if not math.isfinite(model_epistemic_uncertainty) or not 0.0 <= model_epistemic_uncertainty <= 1.0:
                raise ValueError("model_epistemic_uncertainty must be finite and between 0 and 1")

        if evidence_sha256 is not None:
            evidence_sha256 = str(evidence_sha256).lower().strip()
            if not re.fullmatch(r"[0-9a-f]{64}", evidence_sha256):
                raise ValueError("evidence_sha256 must be a 64-character hexadecimal SHA-256 digest")

        # Validate analysis artefacts.
        heatmap_data = _validate_heatmap(heatmap_data)
        tokens = _validate_string_list(tokens, "tokens")
        patches = _validate_string_list(patches, "patches")
        class_probabilities = _validate_class_probabilities(class_probabilities)
        global_clip_similarity = _validate_optional_float(global_clip_similarity, "global_clip_similarity")
        standard_entropic_ot_cost = _validate_optional_float(standard_entropic_ot_cost, "standard_entropic_ot_cost")
        model_name = _validate_provenance_string(model_name, "model_name")
        model_revision = _validate_provenance_string(model_revision, "model_revision")
        checkpoint_sha256 = _validate_provenance_string(checkpoint_sha256, "checkpoint_sha256")
        if checkpoint_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256.lower()):
            raise ValueError("checkpoint_sha256 must be a 64-character hexadecimal SHA-256 digest")
        calibration_version = _validate_optional_int(calibration_version, "calibration_version")
        feature_schema_version = _validate_optional_int(feature_schema_version, "feature_schema_version")
        ot_diagnostics = _json_object(ot_diagnostics) if ot_diagnostics is not None else None

        uploader_id = _validate_provenance_string(uploader_id, "uploader_id") or "anonymous"
        acquisition_timestamp = _validate_provenance_string(acquisition_timestamp, "acquisition_timestamp")
        ingestion_timestamp = _validate_provenance_string(ingestion_timestamp, "ingestion_timestamp") or _utc_now()
        storage_path = _validate_provenance_string(storage_path, "storage_path")
        original_filename = _validate_provenance_string(original_filename, "original_filename") or evidence_filename
        file_size_bytes = _validate_optional_int(file_size_bytes, "file_size_bytes")
        if file_size_bytes is not None and file_size_bytes < 0:
            raise ValueError("file_size_bytes must be non-negative")

        case_metadata = self._ensure_case_metadata(case_id)  # noqa: F841 (kept for clarity)
        timestamp = timestamp or _utc_now()
        evidence_identity = evidence_sha256 or _stable_id(evidence_filename)
        visual_id = _stable_id(case_id, evidence_identity, prefix="visual_")
        statement_id = _stable_id(case_id, evidence_identity, statement_text, prefix="statement_")
        relationship_id = _stable_id(case_id, evidence_identity, statement_text, prefix="pair_")
        is_contradiction = (
            True if assessment_status == "CONTRADICTION"
            else False if assessment_status == "MATCH"
            else None
        )

        record_item: Dict[str, Any] = {
            "case_id": case_id,
            "evidence_id": visual_id,
            "evidence_sha256": evidence_sha256,
            "evidence_filename": evidence_filename,
            "statement_id": statement_id,
            "statement_text": statement_text,
            "sinkhorn_cost": sinkhorn_cost,
            "is_contradiction": is_contradiction,
            "assessment_status": assessment_status,
            "model_epistemic_uncertainty": model_epistemic_uncertainty,
            "visual_quality_metrics": _json_object(visual_quality_metrics),
            "is_blurry": is_blurry,
            "visual_uncertainty": visual_uncertainty,
            "quality_label": quality_label or "",
            "investigator_advisory": investigator_advisory or "",
            "discrepancy_info": discrepancy_info or "",
            "audit_report": audit_report or "",
            "timestamp": timestamp,
            # Analysis artefacts (also served to the frontend for heatmap)
            "heatmap_data": heatmap_data,
            "tokens": tokens,
            "patches": patches,
            "ot_diagnostics": ot_diagnostics,
            "global_clip_similarity": global_clip_similarity,
            "standard_entropic_ot_cost": standard_entropic_ot_cost,
            "class_probabilities": class_probabilities,
            # Model provenance
            "model_name": model_name,
            "model_revision": model_revision,
            "checkpoint_sha256": checkpoint_sha256,
            "calibration_version": calibration_version,
            "feature_schema_version": feature_schema_version,
            # Evidence preservation provenance
            "uploader_id": uploader_id,
            "acquisition_timestamp": acquisition_timestamp,
            "ingestion_timestamp": ingestion_timestamp,
            "storage_path": storage_path,
            "original_filename": original_filename,
            "file_size_bytes": file_size_bytes,
        }

        records = self.memory_store.setdefault(case_id, [])
        records[:] = [
            existing for existing in records
            if existing.get("statement_id") != statement_id
        ]
        records.append(record_item)

        persisted = False
        storage_error: Optional[str] = None
        if self.driver is not None and self._neo4j_healthy:
            metrics_json = json.dumps(record_item["visual_quality_metrics"], sort_keys=True)
            heatmap_json = _dumps_or_none(heatmap_data)
            tokens_json = _dumps_or_none(tokens)
            patches_json = _dumps_or_none(patches)
            ot_diagnostics_json = _dumps_or_none(ot_diagnostics)
            class_probabilities_json = _dumps_or_none(class_probabilities)

            def persist_evidence_tx(tx: Any) -> Optional[Dict[str, Any]]:
                result = tx.run(
                    """
                    MATCH (c:Case {id: $case_id})
                    MERGE (v:VisualEvidence {id: $visual_id})
                    SET v.case_id = $case_id,
                        v.filename = $evidence_filename,
                        v.original_filename = $original_filename,
                        v.content_sha256 = $evidence_sha256,
                        v.uploader_id = $uploader_id,
                        v.acquisition_timestamp = $acquisition_timestamp,
                        v.ingestion_timestamp = $ingestion_timestamp,
                        v.storage_path = $storage_path,
                        v.file_size_bytes = $file_size_bytes
                    MERGE (c)-[:CONTAINS]->(v)
                    MERGE (t:WitnessStatement {id: $statement_id})
                    SET t.case_id = $case_id,
                        t.text = $statement_text
                    MERGE (c)-[:CONTAINS]->(t)
                    MERGE (t)-[r:CORRELATES_TO {id: $relationship_id}]->(v)
                    SET r.case_id = $case_id,
                        r.sinkhorn_cost = $sinkhorn_cost,
                        r.flagged_contradiction = $is_contradiction,
                        r.assessment_status = $assessment_status,
                        r.model_epistemic_uncertainty = $model_epistemic_uncertainty,
                        r.visual_quality_metrics_json = $visual_quality_metrics_json,
                        r.is_blurry = $is_blurry,
                        r.visual_uncertainty = $visual_uncertainty,
                        r.quality_label = $quality_label,
                        r.investigator_advisory = $investigator_advisory,
                        r.discrepancy_info = $discrepancy_info,
                        r.audit_report = $audit_report,
                        r.timestamp = $timestamp,
                        r.heatmap_data_json = $heatmap_data_json,
                        r.tokens_json = $tokens_json,
                        r.patches_json = $patches_json,
                        r.ot_diagnostics_json = $ot_diagnostics_json,
                        r.class_probabilities_json = $class_probabilities_json,
                        r.global_clip_similarity = $global_clip_similarity,
                        r.standard_entropic_ot_cost = $standard_entropic_ot_cost,
                        r.model_name = $model_name,
                        r.model_revision = $model_revision,
                        r.checkpoint_sha256 = $checkpoint_sha256,
                        r.calibration_version = $calibration_version,
                        r.feature_schema_version = $feature_schema_version
                    RETURN r.id AS id
                    """,
                    case_id=case_id,
                    visual_id=visual_id,
                    evidence_filename=evidence_filename,
                    evidence_sha256=evidence_sha256 or "",
                    original_filename=original_filename,
                    uploader_id=uploader_id,
                    acquisition_timestamp=acquisition_timestamp,
                    ingestion_timestamp=ingestion_timestamp,
                    storage_path=storage_path,
                    file_size_bytes=file_size_bytes,
                    statement_id=statement_id,
                    statement_text=statement_text,
                    relationship_id=relationship_id,
                    sinkhorn_cost=sinkhorn_cost,
                    is_contradiction=is_contradiction,
                    assessment_status=assessment_status,
                    model_epistemic_uncertainty=model_epistemic_uncertainty,
                    visual_quality_metrics_json=metrics_json,
                    is_blurry=is_blurry,
                    visual_uncertainty=visual_uncertainty,
                    quality_label=quality_label or "",
                    investigator_advisory=investigator_advisory or "",
                    discrepancy_info=discrepancy_info or "",
                    audit_report=audit_report or "",
                    timestamp=timestamp,
                    heatmap_data_json=heatmap_json,
                    tokens_json=tokens_json,
                    patches_json=patches_json,
                    ot_diagnostics_json=ot_diagnostics_json,
                    class_probabilities_json=class_probabilities_json,
                    global_clip_similarity=global_clip_similarity,
                    standard_entropic_ot_cost=standard_entropic_ot_cost,
                    model_name=model_name,
                    model_revision=model_revision,
                    checkpoint_sha256=checkpoint_sha256,
                    calibration_version=calibration_version,
                    feature_schema_version=feature_schema_version,
                ).single()
                return _record_data(result) if result is not None else None

            try:
                result = self._write_transaction(persist_evidence_tx)
                if result is None:
                    raise RuntimeError(f"Case '{case_id}' does not exist in Neo4j")
                persisted = True
            except Exception as exc:
                storage_error = str(exc)

        record_item["persisted"] = persisted
        record_item["storage_backend"] = "neo4j" if persisted else "memory"
        record_item["persistence_message"] = (
            "Persisted to Neo4j."
            if persisted
            else "Saved only in process memory; this record is not durable across API restarts."
        )
        if storage_error:
            record_item["persistence_error"] = storage_error
        self._last_persistence_status = {
            **self.get_storage_status(),
            "persisted": persisted,
            "message": record_item["persistence_message"],
            "error": storage_error or self._last_storage_error,
        }
        return dict(record_item)

    def _read_case_records_from_neo4j(self, case_id: str) -> List[Dict[str, Any]]:
        rows = self._run(
            """
            MATCH (c:Case {id: $case_id})-[:CONTAINS]->(t:WitnessStatement)-[r:CORRELATES_TO]->(v:VisualEvidence)
            WHERE r.case_id = $case_id
               OR (r.case_id IS NULL AND t.case_id = $case_id AND v.case_id = $case_id)
            RETURN v.id AS evidence_id,
                   v.content_sha256 AS evidence_sha256,
                   v.filename AS evidence_filename,
                   v.original_filename AS original_filename,
                   v.uploader_id AS uploader_id,
                   v.acquisition_timestamp AS acquisition_timestamp,
                   v.ingestion_timestamp AS ingestion_timestamp,
                   v.storage_path AS storage_path,
                   v.file_size_bytes AS file_size_bytes,
                   t.id AS statement_id,
                   t.text AS statement_text,
                   r.sinkhorn_cost AS sinkhorn_cost,
                   r.flagged_contradiction AS is_contradiction,
                   r.assessment_status AS assessment_status,
                   r.model_epistemic_uncertainty AS model_epistemic_uncertainty,
                   r.visual_quality_metrics_json AS visual_quality_metrics_json,
                   r.is_blurry AS is_blurry,
                   r.visual_uncertainty AS visual_uncertainty,
                   r.quality_label AS quality_label,
                   r.investigator_advisory AS investigator_advisory,
                   r.discrepancy_info AS discrepancy_info,
                   r.audit_report AS audit_report,
                   toString(r.timestamp) AS timestamp,
                   r.heatmap_data_json AS heatmap_data_json,
                   r.tokens_json AS tokens_json,
                   r.patches_json AS patches_json,
                   r.ot_diagnostics_json AS ot_diagnostics_json,
                   r.class_probabilities_json AS class_probabilities_json,
                   r.global_clip_similarity AS global_clip_similarity,
                   r.standard_entropic_ot_cost AS standard_entropic_ot_cost,
                   r.model_name AS model_name,
                   r.model_revision AS model_revision,
                   r.checkpoint_sha256 AS checkpoint_sha256,
                   r.calibration_version AS calibration_version,
                   r.feature_schema_version AS feature_schema_version
            """,
            case_id=case_id,
        )
        records: List[Dict[str, Any]] = []
        for row in rows:
            try:
                metrics = json.loads(row.get("visual_quality_metrics_json") or "{}")
            except (TypeError, ValueError):
                metrics = {}
            if not isinstance(metrics, dict):
                metrics = {}

            assessment = row.get("assessment_status") or "INDETERMINATE"
            if assessment not in _ALLOWED_ASSESSMENTS:
                assessment = "INDETERMINATE"

            records.append({
                "case_id": case_id,
                "evidence_id": row.get("evidence_id"),
                "evidence_sha256": row.get("evidence_sha256"),
                "evidence_filename": row.get("evidence_filename") or "Unknown evidence",
                "original_filename": row.get("original_filename") or row.get("evidence_filename") or "Unknown evidence",
                "uploader_id": row.get("uploader_id") or "anonymous",
                "acquisition_timestamp": row.get("acquisition_timestamp"),
                "ingestion_timestamp": row.get("ingestion_timestamp"),
                "storage_path": row.get("storage_path"),
                "file_size_bytes": row.get("file_size_bytes"),
                "statement_id": row.get("statement_id"),
                "statement_text": row.get("statement_text") or "",
                "sinkhorn_cost": row.get("sinkhorn_cost"),
                "is_contradiction": row.get("is_contradiction"),
                "assessment_status": assessment,
                "model_epistemic_uncertainty": row.get("model_epistemic_uncertainty"),
                "visual_quality_metrics": metrics,
                "is_blurry": row.get("is_blurry"),
                "visual_uncertainty": row.get("visual_uncertainty"),
                "quality_label": row.get("quality_label") or "",
                "investigator_advisory": row.get("investigator_advisory") or "",
                "discrepancy_info": row.get("discrepancy_info") or "",
                "audit_report": row.get("audit_report") or "",
                "timestamp": row.get("timestamp") or "Recorded",
                # Analysis artefacts — deserialized so frontend can render directly.
                "heatmap_data": _loads_or_none(row.get("heatmap_data_json")),
                "tokens": _loads_or_none(row.get("tokens_json")),
                "patches": _loads_or_none(row.get("patches_json")),
                "ot_diagnostics": _loads_or_none(row.get("ot_diagnostics_json")),
                "class_probabilities": _loads_or_none(row.get("class_probabilities_json")),
                "global_clip_similarity": row.get("global_clip_similarity"),
                "standard_entropic_ot_cost": row.get("standard_entropic_ot_cost"),
                # Model provenance
                "model_name": row.get("model_name"),
                "model_revision": row.get("model_revision"),
                "checkpoint_sha256": row.get("checkpoint_sha256"),
                "calibration_version": row.get("calibration_version"),
                "feature_schema_version": row.get("feature_schema_version"),
                "persisted": True,
                "storage_backend": "neo4j",
                "persistence_message": "Persisted to Neo4j.",
            })
        return records

    def get_all_evidence_records(self) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for case_id in list(self.memory_store):
            for row in self.memory_store.get(case_id, []):
                key = row.get("statement_id") or _stable_id(
                    case_id, row.get("evidence_filename", ""), row.get("statement_text", "")
                )
                merged[str(key)] = dict(row)
        if self.driver is not None and self._neo4j_healthy:
            try:
                case_ids = [row["id"] for row in self.get_cases()]
                for case_id in case_ids:
                    for row in self._read_case_records_from_neo4j(case_id):
                        key = row.get("statement_id") or _stable_id(
                            case_id, row.get("evidence_filename", ""), row.get("statement_text", "")
                        )
                        merged[str(key)] = row
            except Exception:
                logger.exception("Could not load all evidence records from Neo4j")
        return list(merged.values())

    def get_case_context(self, case_id: str) -> str:
        records = self.memory_store.get(case_id, [])
        if self.driver is not None and self._neo4j_healthy:
            try:
                db_records = self._read_case_records_from_neo4j(case_id)
                merged = {
                    (row.get("evidence_id"), row.get("statement_id")): row
                    for row in records
                }
                merged.update({
                    (row.get("evidence_id"), row.get("statement_id")): row
                    for row in db_records
                })
                records = list(merged.values())
            except Exception:
                logger.exception("Could not build Neo4j case context")
        if not records:
            return "No previous evidence history found for this case."
        lines = ["CASE GRAPH HISTORY:"]
        for row in records:
            lines.append(
                f"- Statement: {row.get('statement_text', '')!r} | "
                f"Evidence: {row.get('evidence_filename', '')!r} | "
                f"Sinkhorn divergence diagnostic: {row.get('sinkhorn_cost')} | "
                f"Classifier assessment: {row.get('assessment_status', 'INDETERMINATE')}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Clear evidence versus delete case
    # ------------------------------------------------------------------
    def clear_case_evidence(self, case_id: str) -> Dict[str, Any]:
        case_id = str(case_id or "").strip()
        if not case_id:
            raise ValueError("case_id must not be empty")
        if self._load_case_metadata(case_id) is None and case_id not in self.memory_store:
            raise KeyError(f"Case '{case_id}' does not exist")
        self.memory_store[case_id] = []
        database_cleared = False
        error: Optional[str] = None

        if self.driver is not None and self._neo4j_healthy:
            def clear_tx(tx: Any) -> bool:
                exists = tx.run(
                    "MATCH (c:Case {id: $case_id}) RETURN c.id AS id",
                    case_id=case_id,
                ).single()
                if exists is None:
                    return False
                tx.run(
                    """
                    MATCH (c:Case {id: $case_id})-[ownership:CONTAINS]->()
                    DELETE ownership
                    """,
                    case_id=case_id,
                ).consume()
                tx.run(
                    """
                    MATCH (n)
                    WHERE n.case_id = $case_id
                      AND (n:VisualEvidence OR n:WitnessStatement)
                    DETACH DELETE n
                    """,
                    case_id=case_id,
                ).consume()
                return True

            try:
                database_cleared = bool(self._write_transaction(clear_tx))
                if not database_cleared:
                    error = f"Case '{case_id}' was not found in Neo4j; no database records were cleared."
            except Exception as exc:
                error = str(exc)
        elif case_id not in self.cases_metadata and not self.memory_store.get(case_id):
            error = "Case is absent from the volatile memory store and Neo4j is unavailable."

        return {
            "status": "cleared" if (database_cleared or self.driver is None) else "partially_cleared",
            "case_id": case_id,
            "case_deleted": False,
            "database_cleared": database_cleared,
            "storage_status": self.get_storage_status(),
            "error": error,
        }

    def delete_case(self, case_id: str) -> Dict[str, Any]:
        case_id = str(case_id or "").strip()
        if not case_id:
            raise ValueError("case_id must not be empty")

        if self._load_case_metadata(case_id) is None and case_id not in self.memory_store:
            raise KeyError(f"Case '{case_id}' does not exist")

        self.memory_store.pop(case_id, None)
        self.cases_metadata.pop(case_id, None)
        self.cross_case_links = [
            link for link in self.cross_case_links
            if link.get("source_case_id") != case_id and link.get("target_case_id") != case_id
        ]

        database_deleted = False
        error: Optional[str] = None
        if self.driver is not None and self._neo4j_healthy:
            def delete_tx(tx: Any) -> bool:
                exists = tx.run(
                    "MATCH (c:Case {id: $case_id}) RETURN c.id AS id",
                    case_id=case_id,
                ).single()
                if exists is None:
                    return False
                tx.run(
                    """
                    MATCH (c:Case {id: $case_id})-[ownership:CONTAINS]->()
                    DELETE ownership
                    """,
                    case_id=case_id,
                ).consume()
                tx.run(
                    """
                    MATCH (n)
                    WHERE n.case_id = $case_id
                      AND (n:VisualEvidence OR n:WitnessStatement)
                    DETACH DELETE n
                    """,
                    case_id=case_id,
                ).consume()
                tx.run(
                    "MATCH (c:Case {id: $case_id}) DETACH DELETE c",
                    case_id=case_id,
                ).consume()
                return True

            try:
                database_deleted = bool(self._write_transaction(delete_tx))
                if not database_deleted:
                    error = f"Case '{case_id}' was not found in Neo4j."
            except Exception as exc:
                error = str(exc)
        else:
            error = self._last_storage_error or "Neo4j is unavailable; deletion affected volatile memory only."

        return {
            "status": "deleted_from_memory" if not database_deleted else "deleted",
            "case_id": case_id,
            "case_deleted": True,
            "database_deleted": database_deleted,
            "storage_status": self.get_storage_status(),
            "error": error,
        }

    def clear_case_data(self, case_id: Optional[str] = None) -> Dict[str, Any]:
        if case_id is not None:
            return self.delete_case(case_id)

        self.memory_store.clear()
        self.cases_metadata.clear()
        self.cross_case_links.clear()
        database_cleared = False
        error: Optional[str] = None
        if self.driver is not None and self._neo4j_healthy:
            try:
                self._write_transaction(
                    lambda tx: tx.run("MATCH (n) DETACH DELETE n").consume()
                )
                database_cleared = True
            except Exception as exc:
                error = str(exc)
        else:
            error = self._last_storage_error or "Neo4j is unavailable; all-case clear affected volatile memory only."
        return {
            "status": "cleared_all",
            "case_id": None,
            "case_deleted": True,
            "database_cleared": database_cleared,
            "storage_status": self.get_storage_status(),
            "error": error,
        }

    # ------------------------------------------------------------------
    # Graph-shaped frontend payloads
    # ------------------------------------------------------------------
    @staticmethod
    def _node_safe_id(value: str) -> str:
        return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _normalized_assessment(record: Dict[str, Any]) -> str:
        status = str(record.get("assessment_status") or "INDETERMINATE").upper()
        if status not in _ALLOWED_ASSESSMENTS:
            return "INDETERMINATE"
        return status

    def get_case_graph(self, case_id: str) -> Dict[str, Any]:
        case_id = str(case_id)
        case_info = self._load_case_metadata(case_id)
        if case_info is None:
            case_info = {
                "id": case_id,
                "name": f"Case {case_id}",
                "description": "Active forensic investigation case.",
                "category": "General Forensics",
            }

        records_by_key: Dict[Any, Dict[str, Any]] = {}
        for record in self.memory_store.get(case_id, []):
            key = (record.get("evidence_id"), record.get("statement_id") or record.get("statement_text"))
            records_by_key[key] = dict(record)

        if self.driver is not None and self._neo4j_healthy:
            try:
                for record in self._read_case_records_from_neo4j(case_id):
                    key = (record.get("evidence_id"), record.get("statement_id") or record.get("statement_text"))
                    records_by_key[key] = record
            except Exception:
                logger.exception("Could not fetch case graph from Neo4j")

        records = list(records_by_key.values())
        case_node_id = f"case_{self._node_safe_id(case_id)}"
        nodes: List[Dict[str, Any]] = [{
            "id": case_node_id,
            "position": {"x": 460, "y": 20},
            "data": {
                "label": f"📁 {case_info.get('name', f'Case {case_id}')}",
                "case_id": case_id,
                "type": "case",
            },
            "draggable": True,
            "style": {
                "background": "#0f172a",
                "color": "#38bdf8",
                "border": "2px solid #0284c7",
                "borderRadius": "12px",
                "fontWeight": "bold",
                "padding": "12px 20px",
            },
        }]
        edges: List[Dict[str, Any]] = []
        visual_nodes: Dict[str, str] = {}
        statement_nodes: Dict[str, str] = {}
        visual_counter = 0
        statement_counter = 0
        status_counts = {"MATCH": 0, "CONTRADICTION": 0, "INDETERMINATE": 0}

        for record in records:
            filename = record.get("evidence_filename") or "Unknown evidence"
            evidence_key = str(record.get("evidence_id") or record.get("evidence_filename") or filename)
            if evidence_key not in visual_nodes:
                visual_node_id = f"visual_{self._node_safe_id(case_id + '|' + evidence_key)}"
                visual_nodes[evidence_key] = visual_node_id
                nodes.append({
                    "id": visual_node_id,
                    "position": {"x": 460, "y": 150 + visual_counter * 140},
                    "data": {
                        "label": f"🎥 {filename}",
                        "case_id": case_id,
                        "filename": filename,
                        "content_sha256": record.get("evidence_sha256"),
                        "uploader_id": record.get("uploader_id"),
                        "storage_path": record.get("storage_path"),
                        "type": "visual",
                    },
                    "draggable": True,
                    "style": {
                        "background": "#1e293b",
                        "color": "#f8fafc",
                        "border": "2px solid #3b82f6",
                        "borderRadius": "10px",
                        "padding": "10px 16px",
                    },
                })
                edges.append({
                    "id": f"edge_{case_node_id}_{visual_node_id}",
                    "source": case_node_id,
                    "target": visual_node_id,
                    "style": {"stroke": "#64748b", "strokeWidth": 1.8, "strokeDasharray": "4 4"},
                })
                visual_counter += 1

            statement_key = str(record.get("statement_id") or _stable_id(case_id, evidence_key, record.get("statement_text", "")))
            statement = record.get("statement_text") or ""
            assessment = self._normalized_assessment(record)
            status_counts[assessment] += 1
            if statement_key not in statement_nodes:
                statement_node_id = f"statement_{self._node_safe_id(case_id + '|' + statement_key)}"
                statement_nodes[statement_key] = statement_node_id
                if assessment == "CONTRADICTION":
                    x, border, icon = 820, "#ef4444", "🔴"
                elif assessment == "MATCH":
                    x, border, icon = 70, "#22c55e", "🟢"
                else:
                    x, border, icon = 445, "#64748b", "⚪"
                y = 280 + statement_counter * 120
                statement_counter += 1
                short_statement = statement if len(statement) <= 28 else statement[:25] + "..."
                nodes.append({
                    "id": statement_node_id,
                    "position": {"x": x, "y": y},
                    "data": {
                        "label": f'{icon} "{short_statement}"',
                        "case_id": case_id,
                        "full_text": statement,
                        "assessment_status": assessment,
                        "is_contradiction": True if assessment == "CONTRADICTION" else False if assessment == "MATCH" else None,
                        "sinkhorn_cost": record.get("sinkhorn_cost"),
                        "type": "statement",
                    },
                    "draggable": True,
                    "style": {
                        "background": "#1e293b",
                        "color": "#f8fafc",
                        "border": f"1.5px solid {border}",
                        "borderRadius": "10px",
                        "padding": "10px 14px",
                    },
                })

            visual_node_id = visual_nodes[evidence_key]
            statement_node_id = statement_nodes[statement_key]
            edge_color = (
                "#ef4444" if assessment == "CONTRADICTION"
                else "#22c55e" if assessment == "MATCH"
                else "#64748b"
            )
            cost = record.get("sinkhorn_cost")
            edges.append({
                "id": f"pair_{self._node_safe_id(case_id + '|' + evidence_key + '|' + statement_key)}",
                "source": visual_node_id,
                "target": statement_node_id,
                "label": f"Sinkhorn diagnostic: {cost if cost is not None else 'N/A'} | {assessment}",
                "animated": assessment == "CONTRADICTION",
                "style": {"stroke": edge_color, "strokeWidth": 2.0},
            })
            record["assessment_status"] = assessment

        return {
            "case_id": case_id,
            "case_info": case_info,
            "nodes": nodes,
            "edges": edges,
            "evidence_list": records,
            "storage_status": self.get_storage_status(),
            "assessment_counts": status_counts,
        }

    def get_multi_case_graph(self) -> Dict[str, Any]:
        cases = self.get_cases()
        all_nodes: List[Dict[str, Any]] = []
        all_edges: List[Dict[str, Any]] = []
        all_evidence: List[Dict[str, Any]] = []
        spacing = 1100

        for index, case in enumerate(cases):
            case_id = case["id"]
            graph = self.get_case_graph(case_id)
            offset_x = 50 + index * spacing
            all_evidence.extend(graph.get("evidence_list", []))
            for node in graph.get("nodes", []):
                adjusted = dict(node)
                position = dict(node.get("position", {}))
                position["x"] = position.get("x", 0) + offset_x
                adjusted["position"] = position
                all_nodes.append(adjusted)
            all_edges.extend(graph.get("edges", []))

        cross_links = self.get_cross_case_relations()
        for link in cross_links:
            source_id = f"case_{self._node_safe_id(link['source_case_id'])}"
            target_id = f"case_{self._node_safe_id(link['target_case_id'])}"
            all_edges.append({
                "id": link["id"],
                "source": source_id,
                "target": target_id,
                "label": f"🔗 {link['relation_type'].replace('_', ' ')}",
                "animated": True,
                "style": {"stroke": "#f59e0b", "strokeWidth": 2.8, "strokeDasharray": "5 5"},
                "data": {"persisted": link.get("persisted", False)},
            })

        return {
            "mode": "multi_case",
            "total_cases": len(cases),
            "cases": cases,
            "nodes": all_nodes,
            "edges": all_edges,
            "cross_case_links": cross_links,
            "evidence_list": all_evidence,
            "storage_status": self.get_storage_status(),
        }