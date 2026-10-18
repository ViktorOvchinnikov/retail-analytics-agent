from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import ValidationError

from .analytics import (
    BigQueryRunner,
    PlanCorrectionError,
    QueryExecutionError,
    QueryValidationError,
    compile_query,
    plan_without_narrow_filters,
)
from .config import Settings
from .golden import GoldenKnowledge
from .llm import GeminiModel, LLMError, LanguageModel
from .models import AnalysisPlan
from .observability import pseudonym
from .reports import ReportError, ReportStore
from .safety import (
    AuthorizationError,
    authorize_plan,
    contains_pii,
    explicit_unauthorized_brands,
    profiles,
    requests_pii,
    sanitize_rows,
)


logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    decision: dict[str, object]
    current_plan: dict[str, object]
    last_plan: dict[str, object]
    last_response: str
    evidence: dict[str, object]
    correction: str
    repair_count: int
    next_step: str
    output: str
    error: str


@dataclass(frozen=True)
class AgentContext:
    message: str
    user_id: str
    conversation_id: str
    request_id: str


class RetailAgent:
    def __init__(
        self,
        settings: Settings,
        *,
        llm: LanguageModel | None = None,
        runner: BigQueryRunner | None = None,
        report_store: ReportStore | None = None,
        checkpointer: object | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm or GeminiModel(settings)
        self.runner = runner or BigQueryRunner(settings)
        self.reports = report_store or ReportStore(settings.database_path)
        self.golden = GoldenKnowledge()
        self.profiles = profiles(settings)
        self._checkpoint_connection: sqlite3.Connection | None = None
        if checkpointer is None:
            self._checkpoint_connection = sqlite3.connect(
                settings.database_path, check_same_thread=False
            )
            checkpointer = SqliteSaver(self._checkpoint_connection)
        self.graph = self._build_graph(checkpointer)

    def close(self) -> None:
        self.reports.close()
        if self._checkpoint_connection is not None:
            self._checkpoint_connection.close()

    def chat(self, message: str, *, user_id: str, conversation_id: str) -> str:
        if user_id not in self.profiles:
            raise ValueError(f"unknown user: {user_id}")
        context = AgentContext(
            message=message,
            user_id=user_id,
            conversation_id=conversation_id,
            request_id=str(uuid.uuid4()),
        )
        config = {"configurable": {"thread_id": f"{user_id}:{conversation_id}"}}
        started = time.monotonic()

        command = parse_report_command(message)
        if command:
            prior_state = dict(self.graph.get_state(config).values)
            output = self._report(command, prior_state, context)
            _event(
                "turn_completed",
                context,
                elapsed_ms=round((time.monotonic() - started) * 1000, 1),
            )
            return output

        if requests_pii(message) or contains_pii(message):
            _event("input_pii_blocked", context)
            output = (
                "I cannot display customer-level identifiers or PII. "
                "I can provide aggregate brand, product, state, or monthly analysis."
            )
            _event(
                "turn_completed",
                context,
                elapsed_ms=round((time.monotonic() - started) * 1000, 1),
            )
            return output

        profile = self.profiles[user_id]
        known_brands = {
            brand
            for candidate in self.profiles.values()
            for brand in candidate.allowed_brands
            if brand != "*"
        }
        denied = explicit_unauthorized_brands(message, profile, known_brands)
        if denied:
            _event("authorization_denied", context)
            output = "Access denied for brand(s): " + ", ".join(denied)
            _event(
                "turn_completed",
                context,
                elapsed_ms=round((time.monotonic() - started) * 1000, 1),
            )
            return output

        state = self.graph.invoke(
            {
                "decision": {},
                "current_plan": {},
                "evidence": {},
                "correction": "",
                "repair_count": 0,
                "next_step": "",
                "output": "",
                "error": "",
            },
            config=config,
            context=context,
        )
        _event(
            "turn_completed",
            context,
            elapsed_ms=round((time.monotonic() - started) * 1000, 1),
        )
        return state["output"]

    def _build_graph(self, checkpointer: object) -> object:
        graph = StateGraph(AgentState, context_schema=AgentContext)
        graph.add_node("plan", self._plan)
        graph.add_node("execute", self._execute)
        graph.add_node("repair", self._repair)
        graph.add_node("respond", self._respond)
        graph.add_node("schema", self._schema)
        graph.add_node("unsupported", self._unsupported)
        graph.add_node("blocked", self._blocked)

        graph.add_edge(START, "plan")
        graph.add_conditional_edges(
            "plan",
            lambda state: str(state["decision"].get("kind", "unsupported")),
            {"analysis": "execute", "schema": "schema", "unsupported": "unsupported"},
        )
        graph.add_conditional_edges(
            "execute",
            lambda state: state["next_step"],
            {"repair": "repair", "respond": "respond", "error": "blocked"},
        )
        graph.add_conditional_edges(
            "repair",
            lambda state: state["next_step"],
            {"execute": "execute", "error": "blocked"},
        )
        for node in ("respond", "schema", "unsupported", "blocked"):
            graph.add_edge(node, END)
        return graph.compile(checkpointer=checkpointer)

    def _plan(
        self, state: AgentState, runtime: Runtime[AgentContext]
    ) -> dict[str, object]:
        context = runtime.context
        previous = _load_plan(state.get("last_plan"))
        examples = self.golden.retrieve(context.message)
        try:
            decision = self.llm.plan(
                context.message,
                previous_plan=previous,
                golden_examples=examples,
                request_id=context.request_id,
                planning_stage="initial",
            )
        except (LLMError, ValidationError) as error:
            _event(
                "planning_failed",
                context,
                **_planning_error_fields(error, "initial"),
            )
            return {
                "decision": {
                    "kind": "unsupported",
                    "message": "Planning is temporarily unavailable. Please retry.",
                },
                "error": "Planning failed safely.",
            }
        if contains_pii(decision.model_dump_json()):
            _event("plan_pii_blocked", context)
            return {
                "decision": {
                    "kind": "unsupported",
                    "message": "The proposed analysis contained sensitive data and was blocked.",
                },
                "current_plan": {},
            }
        return {
            "decision": decision.model_dump(mode="json"),
            "current_plan": decision.plan.model_dump(mode="json") if decision.plan else {},
        }

    def _repair(
        self, state: AgentState, runtime: Runtime[AgentContext]
    ) -> dict[str, object]:
        context = runtime.context
        invalid_plan = _load_plan(state.get("current_plan"))
        try:
            decision = self.llm.plan(
                context.message,
                previous_plan=invalid_plan,
                golden_examples=self.golden.retrieve(context.message),
                correction=state["correction"],
                request_id=context.request_id,
                planning_stage="correction",
            )
            if decision.kind != "analysis" or decision.plan is None:
                raise LLMError("plan correction did not return an analysis")
        except (LLMError, ValidationError) as error:
            _event(
                "plan_correction_failed",
                context,
                **_planning_error_fields(error, "correction"),
            )
            return {
                "current_plan": {},
                "next_step": "error",
                "error": "Plan correction failed safely.",
            }
        if contains_pii(decision.model_dump_json()):
            _event("plan_pii_blocked", context)
            return {
                "current_plan": {},
                "next_step": "error",
                "error": "The corrected analysis contained sensitive data and was blocked.",
            }
        return {
            "decision": decision.model_dump(mode="json"),
            "current_plan": decision.plan.model_dump(mode="json"),
            "repair_count": state.get("repair_count", 0) + 1,
            "correction": "",
            "next_step": "execute",
        }

    def _execute(
        self, state: AgentState, runtime: Runtime[AgentContext]
    ) -> dict[str, object]:
        context = runtime.context
        request_id = context.request_id
        try:
            plan = AnalysisPlan.model_validate(state["current_plan"])
            profile = self.profiles[context.user_id]
            scope = authorize_plan(plan, profile)
            compiled = compile_query(plan, scope, self.settings.bigquery_dataset)
            result = self.runner.execute(compiled, request_id)
            rows = [_json_safe_row(row) for row in result.rows]
            if not any(int(row.get("distinct_customers") or 0) > 0 for row in rows):
                rows = []

            empty_diagnostic: str | None = None
            if not rows and (plan.products or plan.states):
                diagnostic_plan = plan_without_narrow_filters(plan)
                diagnostic_query = compile_query(
                    diagnostic_plan, scope, self.settings.bigquery_dataset
                )
                diagnostic = self.runner.execute(
                    diagnostic_query, request_id + "_availability"
                )
                available = any(
                    int(row.get("order_count") or 0) > 0 for row in diagnostic.rows
                )
                empty_diagnostic = (
                    "Accessible data exists for the period, but the exact product/state "
                    "filter matched no rows. Check the spelling or choose another filter."
                    if available
                    else "No accessible completed orders exist in the requested period."
                )

            safe_rows, suppressed = sanitize_rows(rows, self.settings.min_group_size)
            evidence = {
                "rows": safe_rows,
                "suppressed_groups": suppressed,
                "empty_diagnostic": empty_diagnostic,
                "scope": list(scope.brands) if scope.brands is not None else ["all brands"],
                "job_id": result.job_id,
                "query_hash": result.query_hash,
                "bytes_processed": result.bytes_processed,
            }
            _event(
                "query_succeeded",
                context,
                job_id=result.job_id,
                rows=len(safe_rows),
                suppressed=suppressed,
                bytes_processed=result.bytes_processed,
            )
            return {"evidence": evidence, "next_step": "respond"}
        except PlanCorrectionError as error:
            repairs = state.get("repair_count", 0)
            if repairs < 2:
                _event("plan_correction", context, repair_count=repairs + 1)
                return {
                    "correction": str(error),
                    "next_step": "repair",
                }
            return {"error": "The analysis plan remained invalid after two corrections.", "next_step": "error"}
        except AuthorizationError as error:
            _event("authorization_denied", context)
            return {"error": str(error), "next_step": "error"}
        except QueryValidationError:
            logger.exception(
                "compiler_validation_failed",
                extra=_log_extra("compiler_validation_failed", context),
            )
            return {
                "error": "The query compiler produced an invalid query. No query was executed.",
                "next_step": "error",
            }
        except (QueryExecutionError, ValueError) as error:
            logger.exception(
                "query_failed", extra=_log_extra("query_failed", context)
            )
            return {"error": f"Analysis failed safely: {error}", "next_step": "error"}
        except Exception:
            logger.exception(
                "dependency_failed", extra=_log_extra("dependency_failed", context)
            )
            return {
                "error": "An external dependency failed. No result was released; please retry.",
                "next_step": "error",
            }

    def _respond(
        self, state: AgentState, runtime: Runtime[AgentContext]
    ) -> dict[str, object]:
        context = runtime.context
        plan = AnalysisPlan.model_validate(state["current_plan"])
        evidence = state["evidence"]
        if not evidence.get("rows"):
            detail = evidence.get("empty_diagnostic") or (
                "No groups met the minimum privacy threshold."
                if evidence.get("suppressed_groups")
                else "No data matched the requested scope."
            )
            output = f"No releasable result. {detail}"
        else:
            evidence_table = _evidence_table(plan.title, evidence)
            report_requested = bool(state["decision"].get("save_report"))
            safe_fallback = (
                _safe_report_fallback(evidence_table) if report_requested else evidence_table
            )
            try:
                narrative = self.llm.respond(
                    context.message,
                    plan=plan,
                    evidence=evidence,
                    golden_examples=self.golden.retrieve(context.message),
                    report_requested=report_requested,
                )
            except LLMError:
                logger.exception(
                    "response_generation_failed",
                    extra=_log_extra("response_generation_failed", context),
                )
                output = safe_fallback
            else:
                if contains_pii(narrative) or not _numbers_are_grounded(
                    narrative, plan, evidence
                ) or not _narrative_respects_evidence(narrative, plan, evidence):
                    _event("output_validation_blocked", context)
                    output = safe_fallback
                else:
                    output = f"{narrative.rstrip()}\n\n{evidence_table}"
            if contains_pii(output):
                _event("output_pii_blocked", context)
                output = evidence_table

        result: dict[str, object] = {
            "output": output,
            "last_response": output,
            "last_plan": plan.model_dump(mode="json"),
        }
        if state["decision"].get("save_report") and evidence.get("rows"):
            scope = evidence.get("scope")
            report_id = self.reports.save_report(
                request_id=context.request_id,
                owner=context.user_id,
                conversation_id=context.conversation_id,
                title=plan.title,
                body=output,
                scope=scope if isinstance(scope, list) else None,
            )
            result["output"] = f"{output}\n\nSaved report `{report_id}`."
            _event("report_saved", context, report_id=report_id)
        return result

    def _schema(self, state: AgentState) -> dict[str, str]:
        return {"output": SCHEMA_RESPONSE}

    def _unsupported(self, state: AgentState) -> dict[str, str]:
        message = state.get("decision", {}).get("message")
        if message and contains_pii(str(message)):
            message = "The generated response contained sensitive data and was blocked."
        return {
            "output": str(message or "That analysis is outside the prototype's supported operations.")
        }

    def _blocked(self, state: AgentState) -> dict[str, str]:
        return {"output": state.get("error") or "The request was blocked safely."}

    def _report(
        self,
        command: dict[str, str],
        state: AgentState,
        context: AgentContext,
    ) -> str:
        action = command["action"]
        try:
            if action == "list":
                reports = self.reports.list_reports(context.user_id)
                if not reports:
                    return "No saved reports."
                lines = [f"- `{report.report_id}` — {report.title}" for report in reports]
                return "Saved reports:\n" + "\n".join(lines)
            if action == "save":
                body = state.get("last_response")
                plan = _load_plan(state.get("last_plan"))
                if not body or plan is None:
                    return "There is no completed analysis to save."
                report_id = self.reports.save_report(
                    request_id=context.request_id,
                    owner=context.user_id,
                    conversation_id=context.conversation_id,
                    title=plan.title,
                    body=body,
                    scope=plan.brands or list(self.profiles[context.user_id].allowed_brands),
                )
                return f"Saved report `{report_id}`."
            if action == "prepare":
                pending = self.reports.prepare_delete(
                    owner=context.user_id,
                    conversation_id=context.conversation_id,
                    contains=command.get("contains"),
                    current_conversation_only=command.get("scope") == "conversation",
                )
                if pending is None:
                    return "No owned reports matched; nothing was deleted."
                lines = [f"- `{item.report_id}` — {item.title}" for item in pending.reports]
                return (
                    "Deletion preview (nothing deleted yet):\n"
                    + "\n".join(lines)
                    + f"\n\nConfirm with `/confirm {pending.action_id} {pending.token}` "
                    f"before {pending.expires_at}, or cancel with `/cancel {pending.action_id}`."
                )
            if action == "confirm":
                receipt = self.reports.confirm_delete(
                    owner=context.user_id,
                    conversation_id=context.conversation_id,
                    action_id=command["action_id"],
                    token=command["token"],
                )
                _event("reports_deleted", context, count=receipt.deleted_count)
                return (
                    f"Deleted {receipt.deleted_count} report(s). "
                    f"Receipt: `{receipt.action_id}`."
                )
            if action == "cancel":
                cancelled = self.reports.cancel_delete(
                    owner=context.user_id,
                    conversation_id=context.conversation_id,
                    action_id=command["action_id"],
                )
                return "Deletion cancelled." if cancelled else "No pending deletion matched."
        except ReportError as error:
            _event("report_operation_failed", context)
            return f"Report operation failed safely: {error}"
        return "Unknown report command."


def parse_report_command(message: str) -> dict[str, str] | None:
    text = message.strip()
    lower = text.casefold()
    if lower in {"/reports", "list reports", "list my reports", "show my reports"}:
        return {"action": "list"}
    if lower in {"/save", "save this report", "save this analysis"}:
        return {"action": "save"}
    confirm = re.fullmatch(r"/?confirm\s+([\w-]+)\s+([\w-]+)", text, re.I)
    if confirm:
        return {"action": "confirm", "action_id": confirm.group(1), "token": confirm.group(2)}
    cancel = re.fullmatch(r"/?cancel\s+([\w-]+)", text, re.I)
    if cancel:
        return {"action": "cancel", "action_id": cancel.group(1)}
    if re.fullmatch(
        r"delete\s+(?:all\s+)?(?:the\s+)?reports(?:\s+we\s+made)?\s+(?:in|from)\s+(?:this|the current)\s+conversation",
        text,
        re.I,
    ):
        return {"action": "prepare", "scope": "conversation"}
    match = re.fullmatch(
        r"delete\s+(?:all\s+)?(?:the\s+)?reports\s+(?:mentioning|containing)\s+(.+)",
        text,
        re.I,
    )
    if match:
        return {"action": "prepare", "contains": match.group(1).strip()}
    return None


def _load_plan(value: object) -> AnalysisPlan | None:
    if not value:
        return None
    try:
        return AnalysisPlan.model_validate(value)
    except ValidationError:
        return None


def _json_safe_row(row: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, Decimal):
            result[key] = float(value)
        elif isinstance(value, (date, datetime)):
            result[key] = value.isoformat()
        else:
            result[key] = value
    return result


def _evidence_table(title: str, evidence: dict[str, object]) -> str:
    rows = evidence.get("rows")
    if not isinstance(rows, list) or not rows:
        return "No releasable result."
    columns = list(rows[0])
    lines = [
        f"## Evidence: {_markdown_text(title)}",
        "",
        "| " + " | ".join(_markdown_cell(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(row.get(column, "")) for column in columns
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Scope: "
            + ", ".join(_markdown_text(value) for value in evidence.get("scope", [])),
            f"Suppressed groups: {evidence.get('suppressed_groups', 0)}",
        ]
    )
    if evidence.get("suppressed_groups"):
        lines.append(
            "Privacy note: suppressed groups contained data but were hidden; "
            "their values and trends cannot be inferred."
        )
    return "\n".join(lines)


def _safe_report_fallback(evidence_table: str) -> str:
    return f"""## Findings

The validated evidence is shown below for the requested scope and period.

## Limitations

- Results include completed orders only.
- Groups below the privacy threshold are excluded.

## Action items

- Compare the validated metrics with another period using the same definitions.
- Review monthly, product, or state breakdowns where the privacy threshold permits.

{evidence_table}"""


NUMBER_RE = re.compile(r"(?<![\w.-])-?\d[\d,]*(?:\.\d+)?%?")


def _numbers_are_grounded(
    output: str, plan: AnalysisPlan, evidence: dict[str, object]
) -> bool:
    periods = [period.model_dump(mode="json") for period in plan.periods]
    source = json.dumps(
        {
            "rows": evidence.get("rows", []),
            "scope": evidence.get("scope", []),
            "suppressed_groups": evidence.get("suppressed_groups", 0),
            "periods": periods,
        },
        default=str,
    )
    allowed = {_normalized_number(value) for value in NUMBER_RE.findall(source)}
    checked_output = output
    for period in plan.periods:
        for value in (period.start, period.end, period.end - timedelta(days=1)):
            for rendered in (
                value.isoformat(),
                f"{value.strftime('%B')} {value.day}, {value.year}",
                f"{value.strftime('%b')} {value.day}, {value.year}",
            ):
                checked_output = re.sub(
                    re.escape(rendered), "", checked_output, flags=re.IGNORECASE
                )
    return all(
        _normalized_number(value) in allowed
        for value in NUMBER_RE.findall(checked_output)
    )


def _normalized_number(value: str) -> tuple[Decimal, bool]:
    return Decimal(value.rstrip("%").replace(",", "")), value.endswith("%")


UNSUPPORTED_ANALYSIS_RE = re.compile(
    r"\b(conversion rates?|website traffic|site traffic|marketing spend|ad spend|"
    r"inventory levels?|customer acquisition|a/?b test(?:ing)?|product pages?)\b",
    re.I,
)
SUPPRESSION_AS_NO_DATA_RE = re.compile(
    r"\b(suppressed|hidden|omitted)\s+(months?|groups?)\b.{0,60}"
    r"\b(no|zero|without|lack of)\s+(sales|orders|activity|data)\b",
    re.I | re.S,
)
CONTINUOUS_TREND_RE = re.compile(
    r"\b(continuously|continuous|uninterrupted|every month|month[- ]over[- ]month)\b",
    re.I,
)


def _narrative_respects_evidence(
    output: str, plan: AnalysisPlan, evidence: dict[str, object]
) -> bool:
    if UNSUPPORTED_ANALYSIS_RE.search(output):
        return False
    if evidence.get("suppressed_groups"):
        if SUPPRESSION_AS_NO_DATA_RE.search(output):
            return False
        if any(dimension.value == "month" for dimension in plan.group_by):
            if CONTINUOUS_TREND_RE.search(output):
                return False
    return True


def _planning_error_fields(
    error: LLMError | ValidationError, stage: str
) -> dict[str, object]:
    if isinstance(error, LLMError):
        return {
            "error_category": error.category,
            "provider_status": error.provider_status,
            "exception_class": error.exception_class or type(error).__name__,
            "planning_stage": error.planning_stage or stage,
            "model": error.model,
            "duration_ms": error.duration_ms,
            "attempts": error.attempts,
        }
    return {
        "error_category": "structured_output",
        "exception_class": type(error).__name__,
        "planning_stage": stage,
        "attempts": 1,
    }


def _markdown_text(value: object) -> str:
    return " ".join(str(value).replace("\r", "").split("\n"))


def _markdown_cell(value: object) -> str:
    return _markdown_text(value).replace("|", r"\|")


def _event(event: str, context: AgentContext, **fields: object) -> None:
    logger.info(event, extra=_log_extra(event, context, **fields))


def _log_extra(event: str, context: AgentContext, **fields: object) -> dict[str, object]:
    return {
        "event": event,
        "request_id": context.request_id,
        "conversation_id": pseudonym(context.conversation_id),
        "user_id": pseudonym(context.user_id),
        **fields,
    }


SCHEMA_RESPONSE = """The prototype analyzes completed orders from `orders`,
`order_items`, `products`, and `users`. It supports revenue, distinct order
count, and AOV, grouped by brand, product, state, or month, with up to two
periods or selected products for comparison. It does not expose customer rows
or direct identifiers."""
