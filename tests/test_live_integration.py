from __future__ import annotations

import os
from dataclasses import replace
from datetime import date

import pytest
from google.cloud import bigquery

from retail_agent.agent import RetailAgent
from retail_agent.analytics import BigQueryRunner
from retail_agent.config import Settings


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_INTEGRATION") != "1",
        reason="set RUN_LIVE_INTEGRATION=1 to call Gemini and BigQuery",
    ),
]


class RecordingRunner(BigQueryRunner):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.results = []

    def execute(self, query, request_id):
        result = super().execute(query, request_id)
        self.results.append(result)
        return result


def test_live_manager_aggregate_matches_independent_bigquery(tmp_path) -> None:
    settings = replace(
        Settings.from_env(),
        database_path=tmp_path / "live.sqlite3",
        log_path=tmp_path / "live.jsonl",
    )
    assert settings.gemini_api_key, "GEMINI_API_KEY is required"
    assert settings.google_cloud_project, "GOOGLE_CLOUD_PROJECT is required"
    runner = RecordingRunner(settings)
    agent = RetailAgent(settings, runner=runner)

    output = agent.chat(
        "Show revenue, order count, and AOV for Calvin Klein in calendar year 2024.",
        user_id="manager_1",
        conversation_id="live-integration",
    )

    assert runner.results and "## Evidence:" in output
    actual = runner.results[0].rows[0]
    sql = f"""SELECT
  ROUND(SUM(CAST(oi.sale_price AS NUMERIC)), 2) AS revenue,
  COUNT(DISTINCT oi.order_id) AS order_count,
  ROUND(SAFE_DIVIDE(SUM(CAST(oi.sale_price AS NUMERIC)),
                    COUNT(DISTINCT oi.order_id)), 2) AS aov
FROM `{settings.bigquery_dataset}.order_items` AS oi
JOIN `{settings.bigquery_dataset}.orders` AS o
  ON o.order_id = oi.order_id AND o.user_id = oi.user_id
JOIN `{settings.bigquery_dataset}.products` AS p ON p.id = oi.product_id
WHERE oi.status = 'Complete' AND o.status = 'Complete'
  AND p.brand = @brand
  AND DATE(oi.created_at) >= @start
  AND DATE(oi.created_at) < @end"""
    parameters = [
        bigquery.ScalarQueryParameter("brand", "STRING", "Calvin Klein"),
        bigquery.ScalarQueryParameter("start", "DATE", date(2024, 1, 1)),
        bigquery.ScalarQueryParameter("end", "DATE", date(2025, 1, 1)),
    ]
    config = bigquery.QueryJobConfig(
        use_query_cache=True,
        maximum_bytes_billed=settings.bigquery_max_bytes,
        query_parameters=parameters,
    )
    expected = dict(
        next(
            iter(
                runner.client.query(
                    sql,
                    job_config=config,
                    location=settings.bigquery_location,
                    retry=None,
                    timeout=15,
                ).result(timeout=70)
            )
        ).items()
    )
    agent.close()

    assert actual["revenue"] == expected["revenue"]
    assert actual["order_count"] == expected["order_count"]
    assert actual["aov"] == expected["aov"]
