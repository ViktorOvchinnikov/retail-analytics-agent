from __future__ import annotations

from pathlib import Path

import pytest

from retail_agent.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        gemini_api_key="test-key",
        gemini_model="test-model",
        google_cloud_project="test-project",
        bigquery_location="US",
        bigquery_dataset="bigquery-public-data.thelook_ecommerce",
        bigquery_max_bytes=1024**3,
        database_path=tmp_path / "agent.sqlite3",
        log_path=tmp_path / "agent.jsonl",
        min_group_size=10,
        manager_1_brands=("Calvin Klein", "Levi's"),
        manager_2_brands=("Carhartt",),
    )
