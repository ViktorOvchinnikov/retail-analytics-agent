from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path


LOG_FIELDS = {
    "attempt",
    "attempts",
    "bytes_processed",
    "conversation_id",
    "count",
    "elapsed_ms",
    "duration_ms",
    "error_category",
    "exception_class",
    "job_id",
    "model",
    "planning_stage",
    "provider_status",
    "repair_count",
    "report_id",
    "request_id",
    "rows",
    "suppressed",
    "user_id",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", "log"),
        }
        payload.update(
            {key: getattr(record, key) for key in LOG_FIELDS if hasattr(record, key)}
        )
        if record.exc_info:
            payload["error_type"] = record.exc_info[0].__name__
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(path: Path) -> None:
    formatter = JsonFormatter()
    package_logger = logging.getLogger("retail_agent")
    package_logger.setLevel(logging.INFO)
    package_logger.propagate = False
    for handler in package_logger.handlers:
        handler.close()
    package_logger.handlers.clear()

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    package_logger.addHandler(stream)

    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    package_logger.addHandler(file_handler)


def pseudonym(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]
