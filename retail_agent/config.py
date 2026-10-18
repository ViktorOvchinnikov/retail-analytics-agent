from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DATASET_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    gemini_model: str
    google_cloud_project: str | None
    bigquery_location: str
    bigquery_dataset: str
    bigquery_max_bytes: int
    database_path: Path
    log_path: Path
    min_group_size: int
    manager_1_brands: tuple[str, ...]
    manager_2_brands: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        dataset = os.getenv(
            "BIGQUERY_DATASET", "bigquery-public-data.thelook_ecommerce"
        )
        if not DATASET_RE.fullmatch(dataset):
            raise ValueError("BIGQUERY_DATASET must have the form project.dataset")
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
            google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            bigquery_location=os.getenv("BIGQUERY_LOCATION", "US"),
            bigquery_dataset=dataset,
            bigquery_max_bytes=int(os.getenv("BIGQUERY_MAX_BYTES", str(1024**3))),
            database_path=Path(os.getenv("RETAIL_AGENT_DB_PATH", ".retail-agent.sqlite3")),
            log_path=Path(os.getenv("RETAIL_AGENT_LOG_PATH", "retail-agent.jsonl")),
            min_group_size=int(os.getenv("RETAIL_AGENT_MIN_GROUP_SIZE", "10")),
            manager_1_brands=_csv_env(
                "MANAGER_1_BRANDS", ("Calvin Klein", "Levi's")
            ),
            manager_2_brands=_csv_env("MANAGER_2_BRANDS", ("Carhartt",)),
        )


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())
