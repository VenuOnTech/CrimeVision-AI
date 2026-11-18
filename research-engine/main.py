from __future__ import annotations

import asyncio
import datetime
import hashlib
import io
import logging
import os
import re

import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image
from pydantic import BaseModel, Field

from api_security import (
    assert_loopback_bind,
    get_bool_env,
    get_positive_int,
    parse_allowed_origins,
    read_upload_limited,
    validate_case_id,
    validate_upload_filename,
)
from evidence_store import (
    EvidenceIntegrityError,
    EvidenceNotFoundError,
    EvidenceStore,
)
from logging_config import configure_logging
from neo4j_connector import KnowledgeGraphEngine
from src.inference import EvidentialInferenceEngine

ROOT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ROOT_ENV_FILE, override=False)

configure_logging()
logger = logging.getLogger("crimevision.research_api")

MAX_UPLOAD_BYTES = get_positive_int("MAX_UPLOAD_BYTES", 100 * 1024 * 1024)
MAX_STATEMENT_BYTES = get_positive_int("MAX_STATEMENT_BYTES", 64 * 1024)
MAX_STATEMENT_CHARS = get_positive_int("MAX_STATEMENT_CHARS", 20_000)
MAX_VIDEO_DURATION_SECONDS = get_positive_int("MAX_VIDEO_DURATION_SECONDS", 300)
MAX_VIDEO_FRAME_COUNT = get_positive_int("MAX_VIDEO_FRAME_COUNT", 200_000)
MAX_CONCURRENT_ANALYSIS = get_positive_int("MAX_CONCURRENT_ANALYSIS", 2)
ALLOWED_ORIGINS = parse_allowed_origins(os.getenv("CORS_ALLOWED_ORIGINS"))
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()

Image.MAX_IMAGE_PIXELS = 40_000_000


def _resolve_evidence_root() -> Path:
    raw = os.getenv("EVIDENCE_STORAGE_ROOT", "evidence")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    return path


CONTENT_TYPE_BY_EXTENSION = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".bmp": "image/bmp", ".webp": "image/webp",
    ".tif": "image/tiff", ".tiff": "image/tiff",
    ".mp4": "video/mp4", ".avi": "video/x-msvideo",
    ".mov": "video/quicktime", ".mkv": "video/x-matroska",
}

app = FastAPI(
    title="CrimeVision AI Engine",
    version="1.4",
    docs_url="/docs" if APP_ENV == "development" else None,
    redoc_url=None,
    openapi_url="/openapi.json" if APP_ENV == "development" else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Investigator-Id"],
)

MAX_ANALYSIS_REQUEST_BYTES = MAX_UPLOAD_BYTES + MAX_STATEMENT_BYTES + (2 * 1024 * 1024)
analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSIS)


@app.exception_handler(Exception)
async def structured_unhandled_error(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_code": "internal_error",
            "detail": "An unexpected error occurred. Check the server log for details.",
        },
    )


@app.middleware("http")
async def reject_oversized_analysis_request(request, call_next):
    if request.method == "POST" and request.url.path == "/analyze_evidence/":
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_ANALYSIS_REQUEST_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "status": "error",
                            "error_code": "request_too_large",
                            "detail": "Analysis request exceeds the configured upload limit.",
                        },
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "error_code": "invalid_content_length",
                        "detail": "Invalid Content-Length header.",
                    },
                )
    return await call_next(request)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = (request.headers.get("X-Request-Id") or "").strip() or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "unhandled request error",
            extra={"request_id": request_id, "method": request.method, "path": request.url.path},
        )
        raise
    elapsed_ms = round((time.monotonic() - started) * 1000.0, 2)
    response.headers["X-Request-Id"] = request_id
    logger.info(
        "request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "elapsed_ms": elapsed_ms,
        },
    )
    return response


device = "cuda" if torch.cuda.is_available() else "cpu"
inference_engine = None
neo4j_password = os.getenv("NEO4J_PASSWORD", "").strip()
if not neo4j_password:
    logger.warning(
        "NEO4J_PASSWORD is not configured. No default password will be used; "
        "the connector may operate in volatile memory fallback mode."
    )
_memory_snapshot_path = os.getenv("MEMORY_SNAPSHOT_PATH", "memory_snapshot.json").strip() or None
kg_engine = KnowledgeGraphEngine(
    uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    user=os.getenv("NEO4J_USER", "neo4j"),
    password=neo4j_password,
    snapshot_path=_memory_snapshot_path,
)
evidence_store = EvidenceStore(_resolve_evidence_root())


def get_inference_engine():
    global inference_engine
    if inference_engine is None:
        head_path = os.getenv("EVIDENTIAL_HEAD_PATH") or None
        calibration_path = os.getenv("EVIDENTIAL_CALIBRATION_PATH") or None
        conformal_path = os.getenv("CONFORMAL_ARTIFACT_PATH") or None
        if conformal_path and (head_path or calibration_path):
            raise RuntimeError(
                "Set either EVIDENTIAL_HEAD_PATH+EVIDENTIAL_CALIBRATION_PATH OR "
                "CONFORMAL_ARTIFACT_PATH, not both. They are independent "
                "decision mechanisms."
            )
        model_revision = os.getenv("CLIP_MODEL_REVISION") or None
        if head_path and not model_revision:
            checkpoint_metadata = torch.load(head_path, map_location="cpu", weights_only=True)
            model_revision = checkpoint_metadata.get("model_revision")
            if not model_revision:
                raise RuntimeError("Evidential checkpoint is missing its CLIP model revision")
        inference_engine = EvidentialInferenceEngine(
            model_name=os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"),
            model_revision=model_revision,
            device=os.getenv("DEFAULT_DEVICE", "auto"),
            head_checkpoint_path=head_path,
            calibration_path=calibration_path,
            conformal_artifact_path=conformal_path,
            trust_mode=os.getenv("TRUST_MODE", "heterogeneity"),
        )
    return inference_engine


def _get_uploader_id(request: Request) -> str:
    """Read the investigator header. Assertion-only; not authentication."""
    raw = (request.headers.get("X-Investigator-Id") or "").strip()
    if not raw:
        return "anonymous"
    if len(raw) > 128:
        raw = raw[:128]
    if any(ord(c) < 32 for c in raw):
        raise HTTPException(status_code=422, detail="X-Investigator-Id contains control characters.")
    return raw


def _validate_iso_timestamp(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None or not value.strip():
        return None
    text = value.strip()
    try:
        datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be a valid ISO 8601 timestamp",
        ) from exc
    return text


@app.on_event("startup")
async def load_ai_models():
    allow_non_loopback = get_bool_env("ALLOW_NON_LOOPBACK", False)
    host = os.getenv("RESEARCH_API_HOST", "127.0.0.1")
    assert_loopback_bind(host, allow_non_loopback)

    logger.info("Initializing CrimeVision AI CLIP and Evidential OT engine on %s", device)
    get_inference_engine()
    logger.info("Research engine is ready. Evidence root: %s", evidence_store.root)


@app.on_event("shutdown")
async def shutdown_event():
    if kg_engine:
        kg_engine.close()


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "service": "research-engine"}


@app.get("/ready", tags=["Health"])
async def ready():
    return {
        "status": "ready" if inference_engine is not None else "initializing",
        "inference_engine_loaded": inference_engine is not None,
        "storage": kg_engine.get_storage_status(),
        "evidence_root": str(evidence_store.root),
    }


def extract_sharpest_sampled_frame(
    video_bytes: bytes,
    suffix: str = ".mp4",
) -> tuple[Optional[Image.Image], dict]:
    """Extract a representative frame plus a full sampling report."""
    sample_count = max(5, get_positive_int("VIDEO_SAMPLE_POINTS", 9))
    report: dict = {
        "sample_count": sample_count,
        "frame_count": 0,
        "fps": 0.0,
        "duration_seconds": 0.0,
        "samples": [],
        "chosen_index": None,
        "chosen_frame_index": None,
        "chosen_timestamp_seconds": None,
        "chosen_sharpness": None,
    }
    temp_path = ""
    cap = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix if suffix in {".mp4", ".avi", ".mov", ".mkv"} else ".mp4",
        ) as handle:
            handle.write(video_bytes)
            temp_path = handle.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return None, report

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return None, report

        if total_frames > MAX_VIDEO_FRAME_COUNT:
            raise ValueError(
                f"Video has approximately {total_frames} frames; the configured "
                f"limit is {MAX_VIDEO_FRAME_COUNT}. Trim the clip or raise MAX_VIDEO_FRAME_COUNT."
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps > 0 and total_frames / float(fps) > MAX_VIDEO_DURATION_SECONDS:
            raise ValueError(
                f"Video duration is approximately {total_frames / float(fps):.1f}s; "
                f"the configured limit is {MAX_VIDEO_DURATION_SECONDS}s."
            )

        positions = [
            int((0.10 + 0.80 * (i / max(1, sample_count - 1))) * (total_frames - 1))
            for i in range(sample_count)
        ]

        best_frame = None
        best_sharpness = float("-inf")
        best_index = -1

        for index, frame_index in enumerate(positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(0, frame_index), total_frames - 1))
            success, frame = cap.read()
            timestamp = round(frame_index / fps, 3) if fps > 0 else None
            if not success or frame is None:
                report["samples"].append({
                    "index": index,
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "sharpness": None,
                    "decoded": False,
                })
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
            noise_level = float(cv2.mean(cv2.absdiff(gray, smoothed))[0])
            sobel_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
            edge_energy = float(cv2.mean(cv2.magnitude(sobel_x, sobel_y))[0])
            sharpness = edge_energy / (noise_level + 1.0)

            report["samples"].append({
                "index": index,
                "frame_index": frame_index,
                "timestamp_seconds": timestamp,
                "sharpness": round(sharpness, 6),
                "decoded": True,
            })

            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_index = index
                best_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        report["frame_count"] = total_frames
        report["fps"] = round(fps, 3) if fps > 0 else 0.0
        report["duration_seconds"] = round(total_frames / fps, 3) if fps > 0 else 0.0

        if best_frame is None:
            return None, report

        report["chosen_index"] = best_index
        report["chosen_frame_index"] = positions[best_index]
        report["chosen_timestamp_seconds"] = (
            round(positions[best_index] / fps, 3) if fps > 0 else None
        )
        report["chosen_sharpness"] = round(best_sharpness, 6)
        return Image.fromarray(best_frame), report
    finally:
        if cap is not None:
            cap.release()
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


class CaseCreateSchema(BaseModel):
    case_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    name: Optional[str] = Field(default=None, max_length=200)
    description: Optional[str] = Field(default=None, max_length=4000)
    category: Optional[str] = Field(default="General Forensics", max_length=100)


class CaseUpdateSchema(BaseModel):
    name: Optional[str] = Field(default=None, max_length=200)
    description: Optional[str] = Field(default=None, max_length=4000)
    category: Optional[str] = Field(default=None, max_length=100)


class CrossCaseRelationSchema(BaseModel):
    source_case_id: str = Field(min_length=1, max_length=128)
    target_case_id: str = Field(min_length=1, max_length=128)
    relation_type: Optional[str] = Field(default="SHARED_SUSPECT", max_length=80)
    description: Optional[str] = Field(default="", max_length=2000)
    confidence: Optional[float] = Field(default=0.85, ge=0.0, le=1.0)
    shared_entity: Optional[str] = Field(default=None, max_length=200)


@app.get("/cases")
async def list_cases():
    return await asyncio.to_thread(kg_engine.get_cases)


@app.post("/cases")
async def create_case(payload: CaseCreateSchema):
    try:
        created = await asyncio.to_thread(
            kg_engine.create_case,
            case_id=payload.case_id,
            name=payload.name,
            description=payload.description,
            category=payload.category,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "status": "created",
        "case": created,
        "cases": await asyncio.to_thread(kg_engine.get_cases),
        "storage_status": kg_engine.get_storage_status(),
    }


@app.put("/cases/{case_id}")
async def update_case(case_id: str, payload: CaseUpdateSchema):
    try:
        updated = await asyncio.to_thread(
            kg_engine.update_case,
            case_id=case_id,
            name=payload.name,
            description=payload.description,
            category=payload.category,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "status": "updated",
        "case": updated,
        "cases": await asyncio.to_thread(kg_engine.get_cases),
        "storage_status": kg_engine.get_storage_status(),
    }


@app.delete("/cases/{case_id}")
async def delete_case(case_id: str):
    try:
        result = await asyncio.to_thread(kg_engine.delete_case, case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {**result, "cases": await asyncio.to_thread(kg_engine.get_cases)}


@app.get("/graph/all")
@app.get("/graph/multi/all")
async def get_multi_graph():
    return await asyncio.to_thread(kg_engine.get_multi_case_graph)


@app.post("/cross_case_relation")
async def create_cross_case_relation(payload: CrossCaseRelationSchema):
    try:
        link = await asyncio.to_thread(
            kg_engine.add_cross_case_relation,
            source_case_id=payload.source_case_id,
            target_case_id=payload.target_case_id,
            relation_type=payload.relation_type,
            description=payload.description,
            confidence=payload.confidence,
            shared_entity=payload.shared_entity,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "status": "linked",
        "link": link,
        "multi_graph": await asyncio.to_thread(kg_engine.get_multi_case_graph),
    }


@app.get("/cross_case_relations")
async def get_cross_case_relations():
    return await asyncio.to_thread(kg_engine.get_cross_case_relations)


@app.get("/cross_case_suggestions")
async def get_cross_case_suggestions():
    all_records = await asyncio.to_thread(kg_engine.get_all_evidence_records)
    records_by_case = {}
    for record in all_records:
        case_id = str(record.get("case_id") or "").strip()
        statement = str(record.get("statement_text") or "").strip()
        if case_id and statement:
            records_by_case.setdefault(case_id, []).append(record)

    cases_list = await asyncio.to_thread(kg_engine.get_cases)
    case_names = {str(item["id"]): item.get("name") for item in cases_list}
    case_ids = sorted(records_by_case)
    terms = (
        "red", "blue", "black", "white", "sedan", "car", "vehicle",
        "motorcycle", "jacket", "hoodie", "gun", "knife", "mask",
        "suspect", "alley", "warehouse", "downtown",
    )
    suggestions = []
    seen = set()
    for left_index, left_case_id in enumerate(case_ids):
        for right_case_id in case_ids[left_index + 1:]:
            for left_record in records_by_case[left_case_id]:
                left_text = str(left_record.get("statement_text") or "").lower()
                for right_record in records_by_case[right_case_id]:
                    right_text = str(right_record.get("statement_text") or "").lower()
                    shared_terms = [
                        term for term in terms
                        if re.search(rf"\b{re.escape(term)}\b", left_text)
                        and re.search(rf"\b{re.escape(term)}\b", right_text)
                    ]
                    for term in shared_terms:
                        key = (left_case_id, right_case_id, term)
                        if key in seen:
                            continue
                        seen.add(key)
                        suggestions.append({
                            "id": f"{left_case_id}:{right_case_id}:{term}",
                            "source_case_id": left_case_id,
                            "source_case_name": case_names.get(left_case_id),
                            "target_case_id": right_case_id,
                            "target_case_name": case_names.get(right_case_id),
                            "relation_type": "POTENTIAL_KEYWORD_OVERLAP",
                            "suggestion_type": "heuristic_keyword_overlap",
                            "shared_entity": term,
                            "description": (
                                f"The term '{term}' appears in statements from both cases. "
                                "This is a text overlap only and requires investigator verification."
                            ),
                        })
    return suggestions


@app.get("/graph/{case_id}")
async def get_graph(case_id: str):
    if case_id in {"all", "multi"}:
        return await asyncio.to_thread(kg_engine.get_multi_case_graph)
    return await asyncio.to_thread(kg_engine.get_case_graph, case_id)


@app.post("/clear/{case_id}")
async def clear_case(case_id: str):
    try:
        result = await asyncio.to_thread(kg_engine.clear_case_evidence, case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        **result,
        "nodes": [],
        "edges": [],
        "evidence_list": [],
        "cases": await asyncio.to_thread(kg_engine.get_cases),
    }


_HEAVY_FIELDS = {"heatmap_data", "tokens", "patches", "ot_diagnostics", "transport_plan"}


@app.get("/evidence/{case_id}")
async def list_evidence_metadata(case_id: str):
    if not validate_case_id(case_id):
        raise HTTPException(status_code=422, detail="Invalid case_id.")
    graph = await asyncio.to_thread(kg_engine.get_case_graph, case_id)
    light = [
        {key: value for key, value in record.items() if key not in _HEAVY_FIELDS}
        for record in graph.get("evidence_list", [])
    ]
    return {
        "case_id": case_id,
        "evidence_count": len(light),
        "evidence": light,
        "storage_status": graph.get("storage_status"),
    }


@app.get("/evidence/{case_id}/manifest")
async def get_evidence_manifest(case_id: str):
    if not validate_case_id(case_id):
        raise HTTPException(status_code=422, detail="Invalid case_id.")
    manifest = await asyncio.to_thread(evidence_store.manifest, case_id)
    graph = await asyncio.to_thread(kg_engine.get_case_graph, case_id)
    graph_records = graph.get("evidence_list", [])

    on_disk_hashes = {entry["sha256"] for entry in manifest}
    graph_hashes = {
        record.get("evidence_sha256")
        for record in graph_records
        if record.get("evidence_sha256")
    }
    return {
        "case_id": case_id,
        "evidence_root": str(evidence_store.root),
        "files": manifest,
        "file_count": len(manifest),
        "graph_record_count": len(graph_records),
        "hashes_present_on_disk_only": sorted(on_disk_hashes - graph_hashes),
        "hashes_referenced_in_graph_only": sorted(graph_hashes - on_disk_hashes),
        "note": (
            "Only files preserved after this build are listed. Legacy analyses "
            "created before evidence preservation was enabled will show under "
            "hashes_referenced_in_graph_only and cannot be downloaded."
        ),
    }


@app.get("/evidence/{case_id}/{sha256}/file")
async def download_evidence_file(case_id: str, sha256: str):
    if not validate_case_id(case_id):
        raise HTTPException(status_code=422, detail="Invalid case_id.")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
        raise HTTPException(status_code=422, detail="Invalid SHA-256.")

    try:
        data, stored = await asyncio.to_thread(evidence_store.read, case_id, sha256.lower())
    except EvidenceNotFoundError:
        raise HTTPException(status_code=404, detail="No stored evidence for this case and SHA-256.")
    except EvidenceIntegrityError as exc:
        logger.error("Stored evidence integrity check failed for case_id=%s sha=%s: %s", case_id, sha256, exc)
        raise HTTPException(
            status_code=409,
            detail="Stored file failed integrity verification. The original bytes on disk do not match the recorded SHA-256.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    media_type = CONTENT_TYPE_BY_EXTENSION.get(stored.extension, "application/octet-stream")
    download_name = f"{case_id}_{stored.sha256}{stored.extension}"
    headers = {
        "Content-Disposition": f'attachment; filename="{download_name}"',
        "X-Evidence-SHA256": stored.sha256,
        "X-Evidence-Size-Bytes": str(stored.size_bytes),
    }
    return Response(content=data, media_type=media_type, headers=headers)


@app.post("/evidence/{case_id}/verify")
async def verify_evidence_hash(case_id: str, evidence_file: UploadFile = File(...)):
    if not validate_case_id(case_id):
        raise HTTPException(status_code=422, detail="Invalid case_id.")
    filename = evidence_file.filename or ""
    if not validate_upload_filename(filename):
        raise HTTPException(status_code=400, detail="Evidence filename is missing, unsafe or too long.")
    try:
        file_bytes = await read_upload_limited(evidence_file, MAX_UPLOAD_BYTES)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    computed_sha256 = hashlib.sha256(file_bytes).hexdigest()

    graph = await asyncio.to_thread(kg_engine.get_case_graph, case_id)
    matches = []
    for record in graph.get("evidence_list", []):
        if str(record.get("evidence_sha256") or "").lower() != computed_sha256:
            continue
        on_disk = await asyncio.to_thread(
            evidence_store.exists,
            case_id,
            computed_sha256,
            Path(filename).suffix.lower() or ".bin",
        )
        matches.append({
            "evidence_id": record.get("evidence_id"),
            "evidence_sha256": record.get("evidence_sha256"),
            "evidence_filename": record.get("evidence_filename"),
            "original_filename": record.get("original_filename"),
            "storage_path": record.get("storage_path"),
            "statement_id": record.get("statement_id"),
            "sinkhorn_cost": record.get("sinkhorn_cost"),
            "assessment_status": record.get("assessment_status"),
            "uploader_id": record.get("uploader_id"),
            "acquisition_timestamp": record.get("acquisition_timestamp"),
            "ingestion_timestamp": record.get("ingestion_timestamp"),
            "timestamp": record.get("timestamp"),
            "persisted": record.get("persisted", False),
            "stored_in_backend": on_disk,
        })

    return {
        "status": "match" if matches else "no_match",
        "case_id": case_id,
        "uploaded_filename": filename,
        "computed_sha256": computed_sha256,
        "match_count": len(matches),
        "matches": matches,
        "note": (
            "Byte-for-byte SHA-256 match with a stored analysis record. If "
            "stored_in_backend is true, the original file is also preserved on "
            "disk. Neither fact alone establishes chain-of-custody: there is "
            "no signed acquisition log or trusted timestamp."
            if matches
            else (
                "No evidence record in this case has this SHA-256. The file "
                "may be different, or it may belong to a different case."
            )
        ),
    }


@app.post("/analyze_evidence/")
async def analyze_evidence(
    request: Request,
    case_id: str = Form(...),
    evidence_file: UploadFile = File(...),
    statement_text: Optional[str] = Form(default=None),
    statement_file: Optional[UploadFile] = File(default=None),
    acquisition_timestamp: Optional[str] = Form(default=None),
):
    if not validate_case_id(case_id):
        raise HTTPException(
            status_code=422,
            detail="case_id must be 1-128 characters using letters, digits, dot, underscore or hyphen.",
        )

    uploader_id = _get_uploader_id(request)
    acquisition_timestamp = _validate_iso_timestamp(acquisition_timestamp, "acquisition_timestamp")

    if statement_file is not None and (statement_text or "").strip():
        raise HTTPException(
            status_code=422,
            detail="Provide the witness statement either as text or as a .txt file, not both.",
        )

    if statement_file is not None:
        statement_filename = statement_file.filename or ""
        if not validate_upload_filename(statement_filename) or not statement_filename.lower().endswith(".txt"):
            raise HTTPException(status_code=415, detail="Witness statement files must have a safe .txt filename.")
        try:
            statement_bytes = await read_upload_limited(statement_file, MAX_STATEMENT_BYTES)
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        try:
            final_statement = statement_bytes.decode("utf-8", errors="strict").strip()
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=415, detail="Witness statement must be valid UTF-8.") from exc
    else:
        final_statement = (statement_text or "").strip()
        if len(final_statement.encode("utf-8")) > MAX_STATEMENT_BYTES:
            raise HTTPException(status_code=413, detail="Witness statement exceeds the configured byte limit.")

    if not final_statement:
        raise HTTPException(
            status_code=422,
            detail="A genuine witness statement is required; no synthetic statement is inserted.",
        )
    if len(final_statement) > MAX_STATEMENT_CHARS:
        raise HTTPException(status_code=413, detail=f"Witness statement exceeds {MAX_STATEMENT_CHARS} characters.")

    filename = evidence_file.filename or ""
    if not validate_upload_filename(filename):
        raise HTTPException(status_code=400, detail="Evidence filename is missing, unsafe or too long.")
    try:
        file_bytes = await read_upload_limited(evidence_file, MAX_UPLOAD_BYTES)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded evidence file is empty.")

    extension = os.path.splitext(filename.lower())[1]
    content_type = (evidence_file.content_type or "").lower().split(";", 1)[0].strip()
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    image_formats = {
        ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP",
        ".webp": "WEBP", ".tif": "TIFF", ".tiff": "TIFF",
    }

    video_sampling_report: Optional[dict] = None
    if extension in video_extensions:
        if content_type and not (content_type.startswith("video/") or content_type == "application/octet-stream"):
            raise HTTPException(status_code=415, detail="Declared content type does not match a supported video extension.")
        try:
            image, video_sampling_report = extract_sharpest_sampled_frame(file_bytes, suffix=extension)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if image is None:
            raise HTTPException(status_code=422, detail="Video decoding failed or no valid frame was extracted.")
    elif extension in image_formats:
        if content_type and not (content_type.startswith("image/") or content_type == "application/octet-stream"):
            raise HTTPException(status_code=415, detail="Declared content type does not match a supported image extension.")
        try:
            with Image.open(io.BytesIO(file_bytes)) as decoded_image:
                actual_format = (decoded_image.format or "").upper()
                if actual_format != image_formats[extension]:
                    raise ValueError("File extension does not match the decoded image format.")
                if decoded_image.width * decoded_image.height > Image.MAX_IMAGE_PIXELS:
                    raise ValueError("Image dimensions exceed the configured safety limit.")
                decoded_image.verify()
            with Image.open(io.BytesIO(file_bytes)) as decoded_image:
                image = decoded_image.convert("RGB")
                image.load()
        except Exception:
            logger.exception("Image validation failed for uploaded filename=%s", filename)
            raise HTTPException(status_code=415, detail="The uploaded file is not a valid supported image.") from None
    else:
        raise HTTPException(status_code=415, detail="Only supported image and video evidence is accepted.")

    evidence_sha256 = hashlib.sha256(file_bytes).hexdigest()
    ingestion_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    try:
        stored = await asyncio.to_thread(
            evidence_store.save, case_id, evidence_sha256, extension or ".bin", file_bytes
        )
    except (ValueError, EvidenceIntegrityError) as exc:
        logger.exception("Evidence preservation failed for case_id=%s", case_id)
        raise HTTPException(
            status_code=500,
            detail=f"Evidence preservation failed: {exc}",
        ) from exc

    async with analysis_semaphore:
        try:
            inference_result = await asyncio.to_thread(
                get_inference_engine().analyze_image, image, final_statement
            )
        except (ValueError, RuntimeError) as exc:
            logger.exception("Evidence inference failed for case_id=%s", case_id)
            raise HTTPException(
                status_code=422,
                detail=(
                    "Evidence analysis could not be completed. The original file "
                    f"was preserved at {stored.relative_path} (SHA-256 {stored.sha256}). "
                    "Check the research-engine server log for details."
                ),
            ) from None

        assessment_status = inference_result["assessment_status"]
        sinkhorn_divergence = float(inference_result["sinkhorn_divergence"])
        conformal_payload = inference_result.get("conformal")

        if assessment_status == "INDETERMINATE":
            if conformal_payload:
                abstention_detail = (
                    "Conformal prediction set at target coverage "
                    f"{conformal_payload['coverage_guarantee']:.2f} is "
                    f"{conformal_payload['prediction_set']}; the system abstains because "
                    "the set is not a singleton. Calibrated threshold "
                    f"q_hat = {conformal_payload['q_hat']:.6f}."
                )
            elif inference_result["probabilities"] is None:
                abstention_detail = (
                    "No trained evidential checkpoint with its matching calibration artifact "
                    "and no conformal artifact is loaded. The API therefore provides "
                    "diagnostics only."
                )
            else:
                uncertainty_value = inference_result["model_epistemic_uncertainty"]
                uncertainty_text = (
                    f" Model epistemic uncertainty: {uncertainty_value:.4f}."
                    if uncertainty_value is not None else ""
                )
                abstention_detail = (
                    "The trained classifier abstained and did not issue a "
                    "MATCH/CONTRADICTION decision." + uncertainty_text
                )
            audit_report = (
                "ASSESSMENT: INDETERMINATE. The system computed a debiased Sinkhorn "
                f"divergence diagnostic of {sinkhorn_divergence:.6f}. "
                f"{abstention_detail} The divergence is not a factual finding. "
                "Human examination is required."
            )
        elif conformal_payload:
            audit_report = (
                f"CONFORMAL ASSESSMENT: {assessment_status}. "
                f"Prediction set at target coverage {conformal_payload['coverage_guarantee']:.2f}: "
                f"{conformal_payload['prediction_set']}. Calibrated threshold "
                f"q_hat = {conformal_payload['q_hat']:.6f}. "
                f"Debiased Sinkhorn divergence diagnostic: {sinkhorn_divergence:.6f}. "
                "The coverage guarantee is marginal and distribution-free under "
                "exchangeability; it is not a per-case correctness guarantee and not a "
                "legal conclusion. Human review is required."
            )
        else:
            probability = inference_result["probabilities"][1]
            audit_report = (
                f"MODEL ASSESSMENT: {assessment_status}. The trained evidential classifier "
                f"assigned contradiction probability {probability:.4f}; model epistemic "
                f"uncertainty was {inference_result['model_epistemic_uncertainty']:.4f}. "
                f"The debiased Sinkhorn divergence diagnostic was {sinkhorn_divergence:.6f}. "
                "These are automated research outputs, not independently verified physical "
                "observations or legal conclusions. Human review is required."
            )

        try:
            persistence_record = await asyncio.to_thread(
                kg_engine.create_evidential_link,
                case_id=case_id,
                evidence_filename=filename,
                original_filename=filename,
                evidence_sha256=evidence_sha256,
                statement_text=final_statement,
                sinkhorn_cost=sinkhorn_divergence,
                timestamp=ingestion_timestamp,
                audit_report=audit_report,
                assessment_status=assessment_status,
                model_epistemic_uncertainty=inference_result["model_epistemic_uncertainty"],
                visual_quality_metrics=inference_result["visual_quality_metrics"],
                investigator_advisory=inference_result["indeterminate_reason"] or "Human review remains required.",
                heatmap_data=inference_result["heatmap_data"],
                tokens=inference_result["tokens"],
                patches=inference_result["patches"],
                ot_diagnostics=inference_result["ot_diagnostics"],
                global_clip_similarity=inference_result["global_clip_similarity"],
                standard_entropic_ot_cost=inference_result["standard_entropic_ot_cost"],
                class_probabilities=inference_result["probabilities"],
                model_name=inference_result["model_name"],
                model_revision=inference_result["model_revision"],
                checkpoint_sha256=inference_result["checkpoint_sha256"],
                calibration_version=inference_result["calibration_version"],
                feature_schema_version=inference_result["feature_schema_version"],
                uploader_id=uploader_id,
                acquisition_timestamp=acquisition_timestamp,
                ingestion_timestamp=ingestion_timestamp,
                storage_path=stored.relative_path,
                file_size_bytes=stored.size_bytes,
            )
            graph_status = {
                "backend": persistence_record.get("storage_backend", "memory"),
                "persisted": bool(persistence_record.get("persisted")),
                "message": persistence_record.get("persistence_message", "Persistence status unavailable."),
                "error": persistence_record.get("persistence_error"),
            }
        except Exception:
            logger.exception("Graph persistence failed for case_id=%s", case_id)
            graph_status = {
                "backend": "none",
                "persisted": False,
                "message": (
                    "Analysis completed and evidence preserved on disk, but graph "
                    "persistence failed. Check server logs for details."
                ),
                "error": "Internal graph persistence error.",
            }

        graph_data = await asyncio.to_thread(kg_engine.get_case_graph, case_id)

    return {
        "status": "success",
        "case_id": case_id,
        "evidence_processed": filename,
        "original_filename": filename,
        "storage_path": stored.relative_path,
        "file_size_bytes": stored.size_bytes,
        "uploader_id": uploader_id,
        "acquisition_timestamp": acquisition_timestamp,
        "ingestion_timestamp": ingestion_timestamp,
        "statement_text": final_statement,
        "assessment_status": assessment_status,
        "indeterminate_reason": inference_result["indeterminate_reason"],
        "is_contradiction": inference_result["is_contradiction"],
        "sinkhorn_divergence": sinkhorn_divergence,
        "standard_entropic_ot_cost": inference_result["standard_entropic_ot_cost"],
        "global_clip_similarity": inference_result["global_clip_similarity"],
        "class_probabilities": inference_result["probabilities"],
        "model_epistemic_uncertainty": inference_result["model_epistemic_uncertainty"],
        "visual_quality_metrics": inference_result["visual_quality_metrics"],
        "tokens": inference_result["tokens"],
        "patches": inference_result["patches"],
        "heatmap_data": inference_result["heatmap_data"],
        "transport_plan": inference_result["transport_plan"],
        "ot_diagnostics": inference_result["ot_diagnostics"],
        "model_name": inference_result["model_name"],
        "model_revision": inference_result["model_revision"],
        "checkpoint_sha256": inference_result["checkpoint_sha256"],
        "calibration_version": inference_result["calibration_version"],
        "feature_schema_version": inference_result["feature_schema_version"],
        "conformal": conformal_payload,
        "graph_status": graph_status,
        "storage_status": kg_engine.get_storage_status(),
        "evidence_sha256": evidence_sha256,
        "video_sampling": video_sampling_report,
        "investigator_advisory": inference_result["indeterminate_reason"] or "Human review remains required.",
        "audit_report": audit_report,
        "timestamp": ingestion_timestamp,
        "nodes": graph_data["nodes"],
        "edges": graph_data["edges"],
        "evidence_list": graph_data["evidence_list"],
    }


@app.get("/admin/memory", tags=["Admin"])
async def memory_status():
    storage = kg_engine.get_storage_status()
    snapshot = getattr(kg_engine, "_snapshot", None)
    return {
        "storage": storage,
        "memory_backend_active": not storage.get("neo4j_connected", False),
        "snapshot_path": str(snapshot.path) if snapshot is not None else None,
        "case_count": len(kg_engine.cases_metadata),
        "evidence_count": sum(len(v) for v in kg_engine.memory_store.values()),
        "note": (
            "The memory snapshot survives only a controlled process restart. "
            "Disk failure, corruption, or a hard kill can lose data. Neo4j "
            "remains the source of truth when connected."
        ),
    }


@app.post("/admin/memory/flush", tags=["Admin"])
async def flush_memory_snapshot():
    return await asyncio.to_thread(kg_engine.flush_snapshot)


if __name__ == "__main__":
    import uvicorn

    allow_non_loopback = get_bool_env("ALLOW_NON_LOOPBACK", False)
    host = os.getenv("RESEARCH_API_HOST", "127.0.0.1")
    assert_loopback_bind(host, allow_non_loopback)
    uvicorn.run(app, host=host, port=8001)