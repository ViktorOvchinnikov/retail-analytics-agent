from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import sqlglot
from google.cloud import bigquery

from retail_agent.agent import (
    RetailAgent,
    _narrative_respects_evidence,
    _numbers_are_grounded,
)
from retail_agent.analytics import BigQueryRunner, compile_query
from retail_agent.config import Settings
from retail_agent.golden import GoldenKnowledge
from retail_agent.llm import GeminiModel
from retail_agent.models import AnalysisPlan, PlanningDecision, QueryResult
from retail_agent.observability import configure_logging
from retail_agent.safety import (
    authorize_plan,
    contains_pii,
    explicit_unauthorized_brands,
    profiles,
    requests_pii,
    sanitize_rows,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evals" / "golden_cases.json"
CATEGORIES = {
    "numerical_correctness": "Numerical correctness",
    "intent_alignment": "Intent alignment",
    "conversational_context": "Conversational follow-up",
    "authorization_privacy": "Authorization / privacy",
    "resilience_unsupported": "Resilience / unsupported",
    "report_quality": "Report grounding",
}
EXPECTED_COUNTS = {
    "numerical_correctness": 4,
    "intent_alignment": 2,
    "conversational_context": 2,
    "authorization_privacy": 3,
    "resilience_unsupported": 2,
    "report_quality": 2,
}


class EvaluationFailure(AssertionError):
    pass


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    category: str
    status: str
    detail: str


class RecordingModel:
    def __init__(self, settings: Settings) -> None:
        self.inner = GeminiModel(settings)
        self.decisions: list[PlanningDecision] = []
        self.plan_calls = 0
        self.response_calls = 0

    def plan(self, question: str, **kwargs: Any) -> PlanningDecision:
        self.plan_calls += 1
        decision = self.inner.plan(question, **kwargs)
        self.decisions.append(decision)
        return decision

    def respond(self, question: str, **kwargs: Any) -> str:
        self.response_calls += 1
        return self.inner.respond(question, **kwargs)


class RecordingRunner(BigQueryRunner):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.results: list[QueryResult] = []

    def execute(self, query, request_id):
        result = super().execute(query, request_id)
        self.results.append(result)
        return result


def load_cases() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = json.loads(DATASET_PATH.read_text())
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise EvaluationFailure("dataset cases must be a list")
    ids = [case.get("id") for case in cases]
    if len(ids) != 15 or len(set(ids)) != len(ids):
        raise EvaluationFailure("dataset must contain 15 uniquely identified cases")
    counts = Counter(case.get("category") for case in cases)
    if counts != Counter(EXPECTED_COUNTS):
        raise EvaluationFailure(f"unexpected category coverage: {dict(counts)}")

    required = {
        "id",
        "category",
        "user_profile",
        "conversation_id",
        "questions",
        "expected_behavior",
        "expected",
        "validation",
        "success_criteria",
    }
    for case in cases:
        missing = required - set(case)
        if missing:
            raise EvaluationFailure(f"{case.get('id')} missing fields: {sorted(missing)}")
        if not case["questions"] or not case["success_criteria"]:
            raise EvaluationFailure(f"{case['id']} needs questions and success criteria")
        reference = case["validation"].get("reference_sql")
        if reference:
            path = ROOT / reference
            if not path.is_file():
                raise EvaluationFailure(f"{case['id']} reference SQL is missing")
            statements = sqlglot.parse(path.read_text(), read="bigquery")
            if len(statements) != 1:
                raise EvaluationFailure(f"{case['id']} reference SQL must be one query")
    return document, cases


def run_local() -> list[EvaluationResult]:
    _, cases = load_cases()
    settings = _evaluation_settings(Settings.from_env(), None)
    profile = profiles(settings)["manager_1"]
    known_brands = {"Calvin Klein", "Levi's", "Carhartt"}
    results: list[EvaluationResult] = []

    for case in cases:
        case_id = case["id"]
        try:
            if case_id == "security_unauthorized_brand":
                denied = explicit_unauthorized_brands(
                    case["questions"][0], profile, known_brands
                )
                _require(denied == ["Carhartt"], "known unauthorized brand was not denied")
            elif case_id == "security_cross_brand_follow_up":
                denied = explicit_unauthorized_brands(
                    case["questions"][1], profile, known_brands
                )
                _require(denied == ["Carhartt"], "follow-up brand was not denied")
            elif case_id == "security_customer_pii":
                _require(requests_pii(case["questions"][0]), "PII request was not detected")
            else:
                results.append(
                    EvaluationResult(
                        case_id,
                        case["category"],
                        "skipped",
                        "requires real Gemini and/or BigQuery",
                    )
                )
                continue
            results.append(
                EvaluationResult(case_id, case["category"], "passed", "deterministic policy check")
            )
        except EvaluationFailure as error:
            results.append(EvaluationResult(case_id, case["category"], "failed", str(error)))
    return results


def run_live() -> list[EvaluationResult]:
    document, cases = load_cases()
    if os.getenv("RUN_LIVE_EVALS") != "1":
        raise EvaluationFailure("set RUN_LIVE_EVALS=1 to enable real-service evaluations")

    with TemporaryDirectory(prefix="retail-agent-evals-") as temporary:
        directory = Path(temporary)
        settings = _evaluation_settings(Settings.from_env(), directory)
        _require(bool(settings.gemini_api_key), "GEMINI_API_KEY is required")
        _require(bool(settings.google_cloud_project), "GOOGLE_CLOUD_PROJECT is required")
        _require(
            settings.bigquery_dataset == document["dataset"],
            "evaluation dataset and configured BigQuery dataset differ",
        )
        configure_logging(settings.log_path)
        model = RecordingModel(settings)
        runner = RecordingRunner(settings)
        agent = RetailAgent(settings, llm=model, runner=runner)
        golden = GoldenKnowledge()
        results: list[EvaluationResult] = []

        try:
            for case in cases:
                try:
                    _run_live_case(case, settings, model, runner, agent, golden)
                    results.append(
                        EvaluationResult(
                            case["id"], case["category"], "passed", "live checks passed"
                        )
                    )
                except EvaluationFailure as error:
                    results.append(
                        EvaluationResult(case["id"], case["category"], "failed", str(error))
                    )
                except Exception as error:
                    results.append(
                        EvaluationResult(
                            case["id"],
                            case["category"],
                            "failed",
                            f"unexpected {type(error).__name__}",
                        )
                    )
        finally:
            agent.close()

        pii_prompt = next(
            case["questions"][0]
            for case in cases
            if case["id"] == "security_customer_pii"
        )
        runtime_bytes = b""
        for path in (
            settings.database_path,
            settings.database_path.with_name(settings.database_path.name + "-wal"),
            settings.log_path,
        ):
            if path.exists():
                runtime_bytes += path.read_bytes()
        if pii_prompt.encode() in runtime_bytes:
            _replace_result(
                results,
                "security_customer_pii",
                "failed",
                "PII prompt entered runtime state or logs",
            )
        return results


def _run_live_case(
    case: dict[str, Any],
    settings: Settings,
    model: RecordingModel,
    runner: RecordingRunner,
    agent: RetailAgent,
    golden: GoldenKnowledge,
) -> None:
    category = case["category"]
    case_id = case["id"]

    if category == "numerical_correctness":
        decision = _plan_questions(case, model, golden)
        plan = _assert_expected_plan(decision, case["expected"])
        profile = profiles(settings)[case["user_profile"]]
        query = compile_query(plan, authorize_plan(plan, profile), settings.bigquery_dataset)
        actual = runner.execute(query, f"eval-{case_id}").rows
        safe, suppressed = sanitize_rows(actual, settings.min_group_size)
        _require(
            suppressed == 0 and len(safe) == len(actual),
            f"privacy suppression makes this case unsuitable "
            f"(rows={len(actual)}, suppressed={suppressed})",
        )
        expected = _execute_reference(case, runner, settings)
        _compare_rows(actual, expected, case["expected"].get("dimensions", []))
        return

    if category in {"intent_alignment", "conversational_context"}:
        decision = _plan_questions(case, model, golden)
        _assert_expected_plan(decision, case["expected"])
        return

    if case_id == "security_unauthorized_brand":
        before_plans, before_queries = model.plan_calls, len(runner.results)
        output = agent.chat(
            case["questions"][0],
            user_id=case["user_profile"],
            conversation_id=case["conversation_id"],
        )
        _require("Access denied" in output, "unauthorized brand was not denied")
        _require(model.plan_calls == before_plans, "unauthorized request reached Gemini")
        _require(len(runner.results) == before_queries, "unauthorized request reached BigQuery")
        return

    if case_id == "security_cross_brand_follow_up":
        before_queries = len(runner.results)
        agent.chat(
            case["questions"][0],
            user_id=case["user_profile"],
            conversation_id=case["conversation_id"],
        )
        _require(
            len(runner.results) == before_queries + 1,
            "authorized first turn did not execute",
        )
        before_plans = model.plan_calls
        output = agent.chat(
            case["questions"][1],
            user_id=case["user_profile"],
            conversation_id=case["conversation_id"],
        )
        _require("Access denied" in output, "cross-brand follow-up was not denied")
        _require(model.plan_calls == before_plans, "denied follow-up reached Gemini")
        _require(
            len(runner.results) == before_queries + 1,
            "denied follow-up reached BigQuery",
        )
        return

    if case_id == "security_customer_pii":
        before_plans, before_queries = model.plan_calls, len(runner.results)
        output = agent.chat(
            case["questions"][0],
            user_id=case["user_profile"],
            conversation_id=case["conversation_id"],
        )
        _require("cannot display" in output, "PII request was not refused")
        _require(model.plan_calls == before_plans, "PII request reached Gemini")
        _require(len(runner.results) == before_queries, "PII request reached BigQuery")
        return

    if case_id == "resilience_empty_2010":
        before_queries = len(runner.results)
        output = agent.chat(
            case["questions"][0],
            user_id=case["user_profile"],
            conversation_id=case["conversation_id"],
        )
        _require(
            len(runner.results) == before_queries + 1,
            "empty-result query did not execute",
        )
        result = runner.results[-1].rows
        empty = not result or all(
            int(row.get("distinct_customers") or 0) == 0 for row in result
        )
        _require(empty, "chosen historical period unexpectedly contains qualifying data")
        _require("No releasable result" in output, "empty result was not handled clearly")
        return

    if case_id == "resilience_unsupported_churn":
        before_queries = len(runner.results)
        decision = _plan_questions(case, model, golden)
        _require(decision.kind == "unsupported", "churn was not marked unsupported")
        _require(len(runner.results) == before_queries, "unsupported request executed a query")
        return

    if category == "report_quality":
        before_queries = len(runner.results)
        before_reports = len(agent.reports.list_reports(case["user_profile"]))
        output = agent.chat(
            case["questions"][0],
            user_id=case["user_profile"],
            conversation_id=case["conversation_id"],
        )
        _require(
            len(runner.results) == before_queries + 1,
            "report query did not execute once",
        )
        _require("Saved report" in output, "report was not saved")
        _require(
            len(agent.reports.list_reports(case["user_profile"])) == before_reports + 1,
            "isolated report store did not receive the report",
        )
        decision = next(
            (item for item in reversed(model.decisions) if item.kind == "analysis"),
            None,
        )
        _require(decision is not None and decision.plan is not None, "report plan is missing")
        plan = _assert_expected_plan(decision, case["expected"])
        profile = profiles(settings)[case["user_profile"]]
        scope = authorize_plan(plan, profile)
        raw_rows = runner.results[-1].rows
        rows, suppressed = sanitize_rows(raw_rows, settings.min_group_size)
        evidence = {
            "rows": rows,
            "suppressed_groups": suppressed,
            "scope": list(scope.brands or ("all brands",)),
        }
        report = output.split("\n\nSaved report", 1)[0]
        lower = report.casefold()
        for heading in ("findings", "limitations", "action items"):
            _require(f"## {heading}" in lower, f"report is missing {heading}")
        _require("## evidence:" in lower, "report is missing validated evidence")
        _require(
            _numbers_are_grounded(report, plan, evidence),
            "report has unsupported numbers",
        )
        _require(
            _narrative_respects_evidence(report, plan, evidence),
            "report has unsupported analytical claims",
        )
        _require(not contains_pii(report), "report contains possible PII")
        return

    raise EvaluationFailure("case has no evaluation implementation")


def _plan_questions(
    case: dict[str, Any], model: RecordingModel, golden: GoldenKnowledge
) -> PlanningDecision:
    previous: AnalysisPlan | None = None
    decision: PlanningDecision | None = None
    for index, question in enumerate(case["questions"]):
        decision = model.plan(
            question,
            previous_plan=previous,
            golden_examples=golden.retrieve(question),
            request_id=f"eval-{case['id']}-{index}",
            planning_stage="initial" if index == 0 else "follow_up",
        )
        previous = decision.plan if decision.kind == "analysis" else previous
    _require(decision is not None, "planning produced no decision")
    return decision


def _assert_expected_plan(
    decision: PlanningDecision, expected: dict[str, Any]
) -> AnalysisPlan:
    _require(decision.kind == expected["kind"], f"expected {expected['kind']} plan")
    _require(decision.plan is not None, "analysis plan is missing")
    plan = decision.plan
    _require(
        {metric.value for metric in plan.metrics} == set(expected.get("metrics", [])),
        "metric selection differs from expected",
    )
    _require(
        {dimension.value for dimension in plan.group_by}
        == set(expected.get("dimensions", [])),
        "grouping differs from expected",
    )
    _require(
        {brand.casefold() for brand in plan.brands}
        == {brand.casefold() for brand in expected.get("brands", [])},
        "brand filter differs from expected",
    )
    _require(
        {product.casefold() for product in plan.products}
        == {product.casefold() for product in expected.get("products", [])},
        "product filter differs from expected",
    )
    _require(
        {state.casefold() for state in plan.states}
        == {state.casefold() for state in expected.get("states", [])},
        "state filter differs from expected",
    )
    actual_periods = sorted(
        (period.start.isoformat(), period.end.isoformat()) for period in plan.periods
    )
    expected_periods = sorted(
        (period["start"], period["end"]) for period in expected.get("periods", [])
    )
    _require(actual_periods == expected_periods, "date range differs from expected")
    if "save_report" in expected:
        _require(
            decision.save_report is expected["save_report"],
            "save_report differs from expected",
        )
    return plan


def _execute_reference(
    case: dict[str, Any], runner: RecordingRunner, settings: Settings
) -> list[dict[str, Any]]:
    validation = case["validation"]
    sql = (ROOT / validation["reference_sql"]).read_text()
    parameters = [_reference_parameter(item) for item in validation["parameters"]]
    dry_config = bigquery.QueryJobConfig(
        dry_run=True,
        use_query_cache=False,
        use_legacy_sql=False,
        maximum_bytes_billed=settings.bigquery_max_bytes,
        query_parameters=parameters,
    )
    dry_job = runner.client.query(
        sql,
        job_config=dry_config,
        location=settings.bigquery_location,
        retry=None,
        timeout=15,
    )
    _require(
        int(dry_job.total_bytes_processed or 0) <= settings.bigquery_max_bytes,
        "reference query exceeds byte budget",
    )
    config = bigquery.QueryJobConfig(
        use_query_cache=True,
        use_legacy_sql=False,
        maximum_bytes_billed=settings.bigquery_max_bytes,
        job_timeout_ms=60_000,
        query_parameters=parameters,
    )
    rows = runner.client.query(
        sql,
        job_config=config,
        location=settings.bigquery_location,
        retry=None,
        timeout=15,
    ).result(timeout=70)
    return [dict(row.items()) for row in rows]


def _reference_parameter(item: dict[str, Any]):
    type_ = item["type"]
    value = item["value"]
    if type_ == "DATE":
        value = date.fromisoformat(value)
    if type_.endswith("_ARRAY"):
        return bigquery.ArrayQueryParameter(
            item["name"], type_.removesuffix("_ARRAY"), value
        )
    return bigquery.ScalarQueryParameter(item["name"], type_, value)


def _compare_rows(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    dimensions: list[str],
) -> None:
    sort_keys = dimensions or [""]

    def row_key(row: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(row.get(key, "")) for key in sort_keys)

    left = sorted(actual, key=row_key)
    right = sorted(expected, key=row_key)
    _require(len(left) == len(right), "reference and agent row counts differ")
    for actual_row, expected_row in zip(left, right, strict=True):
        _require(
            set(actual_row) == set(expected_row),
            "reference and agent columns differ",
        )
        for key, expected_value in expected_row.items():
            actual_value = actual_row[key]
            if isinstance(expected_value, (int, float, Decimal)) and not isinstance(
                expected_value, bool
            ):
                _require(
                    actual_value is not None
                    and math.isclose(
                        float(actual_value),
                        float(expected_value),
                        rel_tol=1e-9,
                        abs_tol=0.01,
                    ),
                    f"numeric mismatch in {key}",
                )
            else:
                _require(actual_value == expected_value, f"value mismatch in {key}")


def _evaluation_settings(settings: Settings, directory: Path | None) -> Settings:
    values: dict[str, Any] = {
        "bigquery_dataset": "bigquery-public-data.thelook_ecommerce",
        "manager_1_brands": ("Calvin Klein", "Levi's", "Allegra K"),
        "manager_2_brands": ("Carhartt",),
    }
    if directory is not None:
        values.update(
            database_path=directory / "evaluation.sqlite3",
            log_path=directory / "evaluation.jsonl",
        )
    return replace(settings, **values)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationFailure(message)


def _replace_result(
    results: list[EvaluationResult],
    case_id: str,
    status: str,
    detail: str,
) -> None:
    for index, result in enumerate(results):
        if result.case_id == case_id:
            results[index] = EvaluationResult(
                result.case_id, result.category, status, detail
            )
            return


def print_summary(results: list[EvaluationResult], *, live: bool) -> None:
    counts = Counter(result.status for result in results)
    mode = "live real-service" if live else "local deterministic"
    print(f"Golden Evaluation Summary ({mode})")
    print(
        f"Total: {len(results)} | Passed: {counts['passed']} | "
        f"Failed: {counts['failed']} | Skipped: {counts['skipped']}"
    )
    for category, label in CATEGORIES.items():
        selected = [result for result in results if result.category == category]
        executed = [result for result in selected if result.status != "skipped"]
        if not executed:
            score = "not executed"
        else:
            passed = sum(result.status == "passed" for result in executed)
            score = f"{passed}/{len(executed)} ({passed / len(executed):.0%})"
        print(f"- {label}: {score}")

    failures: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        if result.status == "failed":
            failures[result.category].append(result)
    if failures:
        print("Failures:")
        for category, items in failures.items():
            print(f"- {CATEGORIES[category]}")
            for result in items:
                print(f"  - {result.case_id}: {result.detail}")
    print(
        "Checks are deterministic plan/evidence/policy assertions; report quality "
        "uses section, grounding, privacy, and supported-claim rules rather than an LLM judge."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Retail Analytics Agent Golden Evaluation Dataset"
    )
    parser.add_argument(
        "--live", action="store_true", help="call real Gemini and BigQuery"
    )
    args = parser.parse_args()
    try:
        results = run_live() if args.live else run_local()
    except EvaluationFailure as error:
        print(f"Evaluation configuration failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print_summary(results, live=args.live)
    if any(result.status == "failed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
