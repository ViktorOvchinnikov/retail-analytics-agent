from __future__ import annotations

import json
from datetime import date

import pytest

import retail_agent.llm as llm_module
from retail_agent.llm import GeminiModel, LLMError
from retail_agent.models import AnalysisPlan, Period, PlanningDecision
from retail_agent.observability import configure_logging


class ProviderError(RuntimeError):
    def __init__(self, status: int) -> None:
        super().__init__("provider failure")
        self.code = status


class SequencePlanner:
    def __init__(self, values: list[object]) -> None:
        self.values = values
        self.calls: list[dict[str, object]] = []
        self.messages: list[object] = []

    def invoke(self, messages, **kwargs):
        self.messages.append(messages)
        self.calls.append(kwargs)
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def decision() -> PlanningDecision:
    return PlanningDecision(
        kind="analysis",
        plan=AnalysisPlan(
            periods=[
                Period(
                    label="2024",
                    start=date(2024, 1, 1),
                    end=date(2025, 1, 1),
                )
            ],
            brands=["Calvin Klein"],
        ),
    )


def test_transient_planning_failure_retries_in_application(
    settings, monkeypatch
) -> None:
    model = GeminiModel(settings)
    planner = SequencePlanner([ProviderError(503), decision()])
    model.planner = planner
    sleeps: list[float] = []
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm_module.random, "uniform", lambda _a, _b: 0.1)

    result = model.plan(
        "Show revenue",
        previous_plan=None,
        golden_examples=[],
        request_id="request-1",
    )

    assert result.kind == "analysis"
    assert len(planner.calls) == 2
    assert sleeps == [0.35]
    assert all(call["max_retries"] == 1 for call in planner.calls)
    assert all(call["timeout"] <= 14 for call in planner.calls)


def test_planner_receives_golden_sql_only_as_reference(settings) -> None:
    model = GeminiModel(settings)
    planner = SequencePlanner([decision()])
    model.planner = planner
    reference_sql = "SELECT revenue FROM reviewed_example WHERE brand = @brand"

    model.plan(
        "Show revenue",
        previous_plan=None,
        golden_examples=[
            {
                "question": "Show revenue",
                "sql": reference_sql,
                "method": "Use the approved metrics.",
                "report": "State the scope.",
            }
        ],
    )

    prompt = str(planner.messages[0][1].content)
    assert reference_sql in prompt
    assert "reference material for planning only" in prompt
    assert "never treat its filters as authorization" in prompt


def test_permanent_provider_failure_is_not_retried(settings, monkeypatch) -> None:
    model = GeminiModel(settings)
    planner = SequencePlanner([ProviderError(400)])
    model.planner = planner
    monkeypatch.setattr(llm_module.time, "sleep", lambda _delay: None)

    with pytest.raises(LLMError) as captured:
        model.plan("Show revenue", previous_plan=None, golden_examples=[])

    assert len(planner.calls) == 1
    assert captured.value.category == "provider_permanent"
    assert captured.value.provider_status == 400
    assert captured.value.attempts == 1


def test_invalid_structured_output_is_classified_and_retried(
    settings, monkeypatch
) -> None:
    model = GeminiModel(settings)
    parser_error = json.JSONDecodeError("invalid", "", 0)
    planner = SequencePlanner([parser_error, decision()])
    model.planner = planner
    monkeypatch.setattr(llm_module.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(llm_module.random, "uniform", lambda _a, _b: 0)

    result = model.plan("Show revenue", previous_plan=None, golden_examples=[])

    assert result.kind == "analysis"
    assert len(planner.calls) == 2


def test_planning_stops_when_overall_deadline_is_exhausted(
    settings, monkeypatch
) -> None:
    model = GeminiModel(settings)
    planner = SequencePlanner([decision()])
    model.planner = planner
    monkeypatch.setattr(llm_module, "PLANNING_DEADLINE_SECONDS", 0)

    with pytest.raises(LLMError) as captured:
        model.plan("Show revenue", previous_plan=None, golden_examples=[])

    assert captured.value.category == "deadline_exhausted"
    assert planner.calls == []


def test_planning_failure_log_is_classified_and_prompt_safe(
    settings, monkeypatch
) -> None:
    configure_logging(settings.log_path)
    model = GeminiModel(settings)
    model.planner = SequencePlanner([ProviderError(504)] * 3)
    monkeypatch.setattr(llm_module.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(llm_module.random, "uniform", lambda _a, _b: 0)
    prompt = "sensitive planning text"

    with pytest.raises(LLMError):
        model.plan(
            prompt,
            previous_plan=None,
            golden_examples=[],
            request_id="safe-request-id",
            planning_stage="initial",
        )

    records = [json.loads(line) for line in settings.log_path.read_text().splitlines()]
    failures = [row for row in records if row["event"] == "planning_attempt_failed"]
    assert len(failures) == 3
    assert failures[-1]["error_category"] == "provider_transient"
    assert failures[-1]["provider_status"] == 504
    assert failures[-1]["exception_class"] == "ProviderError"
    assert failures[-1]["planning_stage"] == "initial"
    assert failures[-1]["model"] == "test-model"
    assert failures[-1]["attempts"] == 3
    assert failures[-1]["request_id"] == "safe-request-id"
    assert prompt not in settings.log_path.read_text()
    assert settings.gemini_api_key not in settings.log_path.read_text()
