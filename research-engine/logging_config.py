"""Structured JSON logging for the CrimeVision research API.

Usage:
    from logging_config import configure_logging
    configure_logging()

Environment variables:
    LOG_LEVEL         — DEBUG / INFO / WARNING / ERROR (default INFO)
    LOG_JSON          — 1 (default) emits one JSON object per line; 0 emits human text
    LOG_FILE          — path to a rotating log file; if unset, stdout only
    LOG_MAX_BYTES     — rotation threshold in bytes (default 10 MiB)
    LOG_BACKUP_COUNT  — number of rotated files to keep (default 5)

The formatter hoists any `extra={...}` kwarg onto the JSON line. The request
middleware in main.py attaches `request_id`, `method`, `path`, `status`, and
`elapsed_ms` to every completed request.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from typing import Any

_RESERVED = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = value
        return json.dumps(payload, default=str, ensure_ascii=False)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def configure_logging() -> None:
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    use_json = _bool_env("LOG_JSON", True)
    log_file = os.getenv("LOG_FILE", "").strip()

    formatter: logging.Formatter = JsonFormatter() if use_json else logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s [%(request_id)s] %(message)s",
        defaults={"request_id": "-"},
    )

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file:
        directory = os.path.dirname(os.path.abspath(log_file))
        if directory:
            os.makedirs(directory, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=int(os.getenv("LOG_MAX_BYTES", 10 * 1024 * 1024)),
            backupCount=int(os.getenv("LOG_BACKUP_COUNT", 5)),
            encoding="utf-8",
        )
        rotating.setFormatter(formatter)
        root.addHandler(rotating)

    root.setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(max(level, logging.WARNING))
    logging.getLogger("neo4j").setLevel(max(level, logging.WARNING))