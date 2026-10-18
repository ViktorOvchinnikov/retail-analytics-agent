from __future__ import annotations

import os

import pytest

from evals.run_evals import print_summary, run_live


pytestmark = [
    pytest.mark.evaluation,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_EVALS") != "1",
        reason="set RUN_LIVE_EVALS=1 to call real Gemini and BigQuery",
    ),
]


def test_live_golden_evaluation_suite() -> None:
    results = run_live()
    print_summary(results, live=True)

    failures = [result for result in results if result.status == "failed"]
    assert not failures, [
        (result.case_id, result.category, result.detail) for result in failures
    ]
