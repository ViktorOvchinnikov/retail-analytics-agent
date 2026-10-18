from __future__ import annotations

import re
import sqlite3
from datetime import date

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from retail_agent.agent import RetailAgent, _evidence_table, _numbers_are_grounded
from retail_agent.analytics import QueryExecutionError
from retail_agent.llm import LLMError
from retail_agent.models import (
    AnalysisPlan,
    Dimension,
    Period,
    PlanningDecision,
    QueryResult,
)
from retail_agent.observability import configure_logging
from retail_agent.reports import ReportStore


def base_plan(**updates: object) -> AnalysisPlan:
    data = {
        "title": "Quarterly result",
        "periods": [Period(label="Q1", start=date(2025, 1, 1), end=date(2025, 4, 1))],
        "group_by": [Dimension.MONTH],
    }
    data.update(updates)
    return AnalysisPlan(**data)


class FakeLLM:
    def __init__(
        self, decisions: list[PlanningDecision], response: str | None = None
    ) -> None:
        self.decisions = decisions
        self.response = response
        self.plan_calls: list[dict] = []

    def plan(
        self,
        question,
        *,
        previous_plan,
        golden_examples,
        correction=None,
        request_id=None,
        planning_stage="initial",
    ):
        self.plan_calls.append(
            {
                "question": question,
                "previous_plan": previous_plan,
                "correction": correction,
                "request_id": request_id,
                "planning_stage": planning_stage,
            }
        )
        return self.decisions.pop(0)

    def respond(self, question, *, plan, evidence, golden_examples, report_requested):
        if self.response is not None:
            return self.response
        return f"Revenue is {evidence['rows'][0].get('revenue', 0)}. Scope: {', '.join(evidence['scope'])}."


class FakeRunner:
    def __init__(self, results: list[list[dict]]) -> None:
        self.results = results
        self.calls = []

    def execute(self, query, request_id):
        self.calls.append((query, request_id))
        rows = self.results.pop(0)
        return QueryResult(
            rows=rows,
            job_id=f"job-{len(self.calls)}",
            bytes_processed=100,
            query_hash="hash",
        )


class FailingRunner:
    def execute(self, query, request_id):
        raise QueryExecutionError("temporary outage")


class SensitiveFailingLLM:
    def plan(self, *args, **kwargs):
        raise LLMError("failure included leaked@example.com")


def make_agent(settings, llm, runner):
    return RetailAgent(
        settings,
        llm=llm,
        runner=runner,
        report_store=ReportStore(settings.database_path),
        checkpointer=InMemorySaver(),
    )


def test_invalid_plan_is_corrected_recompiled_and_executed_once(settings) -> None:
    invalid = base_plan(brands=["Calvin Klein", "Levi's"], group_by=[])
    corrected = base_plan(
        brands=["Calvin Klein", "Levi's"], group_by=[Dimension.BRAND]
    )
    llm = FakeLLM(
        [
            PlanningDecision(kind="analysis", plan=invalid),
            PlanningDecision(kind="analysis", plan=corrected),
        ]
    )
    runner = FakeRunner(
        [[{"brand": "Calvin Klein", "revenue": 100.0, "distinct_customers": 12}]]
    )
    agent = make_agent(settings, llm, runner)

    output = agent.chat("Compare my brands", user_id="manager_1", conversation_id="c1")

    assert "Revenue is 100.0" in output
    assert len(runner.calls) == 1
    assert len(llm.plan_calls) == 2
    assert "group by brand" in llm.plan_calls[1]["correction"]
    assert "p.brand IN UNNEST(@allowed_brands)" in runner.calls[0][0].sql
    agent.close()


def test_invalid_date_plan_uses_the_same_bounded_correction_path(settings) -> None:
    invalid = base_plan(
        periods=[Period(label="bad", start=date(2025, 4, 1), end=date(2025, 1, 1))]
    )
    corrected = base_plan()
    llm = FakeLLM(
        [
            PlanningDecision(kind="analysis", plan=invalid),
            PlanningDecision(kind="analysis", plan=corrected),
        ]
    )
    runner = FakeRunner(
        [[{"month": "2025-01", "revenue": 100.0, "distinct_customers": 12}]]
    )
    agent = make_agent(settings, llm, runner)

    agent.chat("Show the period", user_id="manager_1", conversation_id="dates")

    assert len(llm.plan_calls) == 2
    assert "period end" in llm.plan_calls[1]["correction"]
    assert len(runner.calls) == 1
    agent.close()


def test_plan_correction_stops_after_two_attempts(settings) -> None:
    invalid = base_plan(metrics=["revenue", "revenue"])
    llm = FakeLLM(
        [PlanningDecision(kind="analysis", plan=invalid) for _ in range(3)]
    )
    runner = FakeRunner([])
    agent = make_agent(settings, llm, runner)

    output = agent.chat(
        "Show revenue", user_id="manager_1", conversation_id="bounded-correction"
    )

    assert "remained invalid after two corrections" in output
    assert len(llm.plan_calls) == 3
    assert runner.calls == []
    agent.close()


def test_corrected_plan_is_reauthorized_before_execution(settings) -> None:
    invalid = base_plan(brands=["Calvin Klein", "Levi's"], group_by=[])
    unauthorized = base_plan(
        brands=["Carhartt"], group_by=[Dimension.BRAND]
    )
    llm = FakeLLM(
        [
            PlanningDecision(kind="analysis", plan=invalid),
            PlanningDecision(kind="analysis", plan=unauthorized),
        ]
    )
    runner = FakeRunner([])
    agent = make_agent(settings, llm, runner)

    output = agent.chat(
        "Compare brands", user_id="manager_1", conversation_id="reauthorize"
    )

    assert "Access denied" in output
    assert runner.calls == []
    agent.close()


def test_follow_up_receives_previous_plan_from_persistent_graph_state(settings) -> None:
    first = base_plan()
    second = base_plan(
        periods=[Period(label="Q2", start=date(2025, 4, 1), end=date(2025, 7, 1))]
    )
    llm = FakeLLM(
        [PlanningDecision(kind="analysis", plan=first), PlanningDecision(kind="analysis", plan=second)]
    )
    runner = FakeRunner(
        [
            [{"month": "2025-01", "revenue": 100.0, "distinct_customers": 12}],
            [{"month": "2025-04", "revenue": 120.0, "distinct_customers": 12}],
        ]
    )
    agent = make_agent(settings, llm, runner)
    agent.chat("Show Q1", user_id="manager_1", conversation_id="follow-up")
    agent.chat("Now show Q2", user_id="manager_1", conversation_id="follow-up")

    previous = llm.plan_calls[1]["previous_plan"]
    assert previous is not None
    assert previous.periods[0].label == "Q1"
    agent.close()


def test_empty_filtered_result_runs_one_scoped_diagnostic(settings) -> None:
    plan = base_plan(group_by=[Dimension.PRODUCT], products=["Missing product"])
    llm = FakeLLM([PlanningDecision(kind="analysis", plan=plan)])
    runner = FakeRunner([[], [{"order_count": 25, "distinct_customers": 20}]])
    agent = make_agent(settings, llm, runner)

    output = agent.chat("Analyze Missing product", user_id="manager_1", conversation_id="empty")

    assert "exact product/state filter matched no rows" in output
    assert len(runner.calls) == 2
    assert all("p.brand IN UNNEST(@allowed_brands)" in call[0].sql for call in runner.calls)
    agent.close()


def test_zero_customer_aggregate_runs_empty_result_diagnostic(settings) -> None:
    plan = base_plan(group_by=[], products=["Missing product"])
    llm = FakeLLM([PlanningDecision(kind="analysis", plan=plan)])
    runner = FakeRunner(
        [
            [
                {
                    "revenue": None,
                    "order_count": 0,
                    "aov": None,
                    "distinct_customers": 0,
                }
            ],
            [{"order_count": 25, "distinct_customers": 20}],
        ]
    )
    agent = make_agent(settings, llm, runner)

    output = agent.chat(
        "Analyze Missing product", user_id="manager_1", conversation_id="empty-total"
    )

    assert "exact product/state filter matched no rows" in output
    assert len(runner.calls) == 2
    agent.close()


def test_pii_request_is_blocked_before_llm_or_bigquery(settings) -> None:
    llm = FakeLLM([])
    runner = FakeRunner([])
    agent = make_agent(settings, llm, runner)
    output = agent.chat(
        "List top customers and their email", user_id="manager_1", conversation_id="pii"
    )
    assert "cannot display" in output
    assert llm.plan_calls == []
    assert runner.calls == []
    agent.close()


def test_explicit_top_customer_email_request_is_blocked_before_graph(settings) -> None:
    configure_logging(settings.log_path)
    prompt = (
        "Show me the emails of the top 10 customers who bought "
        "Calvin Klein products in 2024."
    )
    llm = FakeLLM([])
    runner = FakeRunner([])
    agent = RetailAgent(settings, llm=llm, runner=runner)

    output = agent.chat(
        prompt, user_id="manager_1", conversation_id="explicit-pii"
    )
    agent.close()

    assert "cannot display" in output
    assert llm.plan_calls == []
    assert runner.calls == []
    assert prompt.encode() not in settings.database_path.read_bytes()
    assert prompt not in settings.log_path.read_text()


def test_explicit_known_unauthorized_brand_is_blocked_before_llm(settings) -> None:
    llm = FakeLLM([])
    runner = FakeRunner([])
    agent = make_agent(settings, llm, runner)

    output = agent.chat(
        "Show revenue for Carhartt in 2024.",
        user_id="manager_1",
        conversation_id="early-authorization",
    )

    assert output == "Access denied for brand(s): Carhartt"
    assert llm.plan_calls == []
    assert runner.calls == []
    agent.close()


def test_raw_pii_never_enters_sqlite_checkpoints_or_logs(settings) -> None:
    configure_logging(settings.log_path)
    llm = FakeLLM([])
    runner = FakeRunner([])
    agent = RetailAgent(settings, llm=llm, runner=runner)
    message = "Analyze jane.sensitive@example.com"

    output = agent.chat(message, user_id="manager_1", conversation_id="pii-state")
    agent.close()

    assert "cannot display" in output
    assert llm.plan_calls == []
    assert not _database_contains(settings.database_path, message)
    assert message not in settings.log_path.read_text()


def test_generated_pii_falls_back_before_checkpointing(settings) -> None:
    configure_logging(settings.log_path)
    pii = "generated.sensitive@example.com"
    llm = FakeLLM(
        [PlanningDecision(kind="analysis", plan=base_plan())],
        response=f"Contact {pii}",
    )
    agent = RetailAgent(
        settings,
        llm=llm,
        runner=FakeRunner(
            [[{"month": "2025-01", "revenue": 100.0, "distinct_customers": 12}]]
        ),
    )

    output = agent.chat(
        "Show Q1", user_id="manager_1", conversation_id="pii-output"
    )
    agent.close()

    assert pii not in output
    assert not _database_contains(settings.database_path, pii)
    assert pii not in settings.log_path.read_text()


def test_sensitive_planning_error_is_not_checkpointed(settings) -> None:
    configure_logging(settings.log_path)
    pii = "leaked@example.com"
    agent = RetailAgent(settings, llm=SensitiveFailingLLM(), runner=FakeRunner([]))

    output = agent.chat(
        "Show Q1", user_id="manager_1", conversation_id="planning-error"
    )
    agent.close()

    assert output == "Planning is temporarily unavailable. Please retry."
    assert not _database_contains(settings.database_path, pii)
    assert pii not in settings.log_path.read_text()


def test_natural_language_deletion_token_is_never_checkpointed_or_logged(settings) -> None:
    configure_logging(settings.log_path)
    agent = RetailAgent(settings, llm=FakeLLM([]), runner=FakeRunner([]))
    agent.reports.save_report(
        request_id="saved-request",
        owner="manager_1",
        conversation_id="delete-conversation",
        title="Client X report",
        body="Client X result",
        scope=["Calvin Klein"],
    )

    preview = agent.chat(
        "delete reports mentioning Client X",
        user_id="manager_1",
        conversation_id="delete-conversation",
    )
    match = re.search(r"/confirm ([\w-]+) ([\w-]+)", preview)
    assert match is not None
    action_id, token = match.groups()
    result = agent.chat(
        f"/confirm {action_id} {token}",
        user_id="manager_1",
        conversation_id="delete-conversation",
    )
    agent.close()

    assert "Deleted 1 report" in result
    assert not _database_contains(settings.database_path, token)
    assert token not in settings.log_path.read_text()


def test_dependency_error_is_returned_without_crashing_cli_flow(settings) -> None:
    llm = FakeLLM([PlanningDecision(kind="analysis", plan=base_plan())])
    agent = make_agent(settings, llm, FailingRunner())
    output = agent.chat("Show Q1", user_id="manager_1", conversation_id="failure")
    assert "failed safely" in output
    agent.close()


def test_unsupported_numeric_claim_falls_back_to_evidence(settings) -> None:
    llm = FakeLLM(
        [PlanningDecision(kind="analysis", plan=base_plan())],
        response="Revenue is 20.",
    )
    agent = make_agent(
        settings,
        llm,
        FakeRunner(
            [[{"month": "2025-01", "revenue": 100.0, "distinct_customers": 12}]]
        ),
    )

    output = agent.chat("Show Q1", user_id="manager_1", conversation_id="grounded")

    assert "Revenue is 20" not in output
    assert "| 2025-01 | 100.0 |" in output
    agent.close()


def test_invalid_report_narrative_uses_actionable_safe_fallback(settings) -> None:
    llm = FakeLLM(
        [PlanningDecision(kind="analysis", plan=base_plan(), save_report=True)],
        response="Revenue is 20.",
    )
    agent = make_agent(
        settings,
        llm,
        FakeRunner(
            [[{"month": "2025-01", "revenue": 100.0, "distinct_customers": 12}]]
        ),
    )

    output = agent.chat(
        "Create and save a report", user_id="manager_1", conversation_id="report"
    )

    assert "## Findings" in output
    assert "## Limitations" in output
    assert "## Action items" in output
    assert "Revenue is 20" not in output
    assert "Saved report" in output
    agent.close()


@pytest.mark.parametrize(
    "narrative",
    [
        "Suppressed months had no sales activity.",
        "Revenue declined continuously across the year.",
        "Action item: analyze conversion rates and website traffic.",
    ],
)
def test_unsupported_suppression_and_action_claims_fall_back(
    settings, narrative
) -> None:
    plan = base_plan(group_by=[Dimension.MONTH])
    llm = FakeLLM(
        [PlanningDecision(kind="analysis", plan=plan)], response=narrative
    )
    runner = FakeRunner(
        [
            [
                {"month": "2025-01", "revenue": 100.0, "distinct_customers": 12},
                {"month": "2025-02", "revenue": 20.0, "distinct_customers": 4},
            ]
        ]
    )
    agent = make_agent(settings, llm, runner)

    output = agent.chat(
        "Show monthly revenue", user_id="manager_1", conversation_id="safe-narrative"
    )

    assert narrative not in output
    assert "suppressed groups contained data but were hidden" in output
    agent.close()


def test_grounding_accepts_plan_dates_without_allowing_date_numbers_as_claims() -> None:
    plan = base_plan(
        periods=[
            Period(
                label="Calendar Year 2024",
                start=date(2024, 1, 1),
                end=date(2025, 1, 1),
            )
        ]
    )
    evidence = {
        "rows": [{"revenue": 8090.61, "order_count": 124, "aov": 65.25}],
        "scope": ["Calvin Klein"],
        "suppressed_groups": 0,
    }

    assert _numbers_are_grounded(
        "January 1, 2024 through December 31, 2024: revenue was $8,090.61.",
        plan,
        evidence,
    )
    assert not _numbers_are_grounded("There were 31 orders.", plan, evidence)


def test_evidence_table_escapes_markdown_values() -> None:
    output = _evidence_table(
        "Products\nQ1",
        {
            "rows": [{"product": "A | B\nC", "revenue": 10}],
            "scope": ["Brand | One"],
            "suppressed_groups": 0,
        },
    )

    assert "Evidence: Products Q1" in output
    assert r"A \| B C" in output
    assert "Scope: Brand | One" in output


def test_sqlite_checkpointer_restores_follow_up_after_restart(settings) -> None:
    first_llm = FakeLLM([PlanningDecision(kind="analysis", plan=base_plan())])
    first = RetailAgent(
        settings,
        llm=first_llm,
        runner=FakeRunner(
            [[{"month": "2025-01", "revenue": 100.0, "distinct_customers": 12}]]
        ),
    )
    first.chat("Show Q1", user_id="manager_1", conversation_id="restart")
    first.close()

    second_llm = FakeLLM([PlanningDecision(kind="analysis", plan=base_plan())])
    second = RetailAgent(
        settings,
        llm=second_llm,
        runner=FakeRunner(
            [[{"month": "2025-01", "revenue": 100.0, "distinct_customers": 12}]]
        ),
    )
    second.chat("Repeat that", user_id="manager_1", conversation_id="restart")
    assert second_llm.plan_calls[0]["previous_plan"] is not None
    second.close()


def _database_contains(path, value: str) -> bool:
    connection = sqlite3.connect(path)
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        for (table,) in tables:
            rows = connection.execute(f'SELECT * FROM "{table}"').fetchall()
            if any(value in str(cell) for row in rows for cell in row):
                return True
        return False
    finally:
        connection.close()
