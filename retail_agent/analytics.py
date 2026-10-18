from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import sqlglot
from google.api_core import exceptions as google_exceptions
from google.cloud import bigquery
from sqlglot import exp

from .config import Settings
from .models import AnalysisPlan, Dimension, Metric, QueryResult
from .safety import AuthorizedScope


logger = logging.getLogger(__name__)


class QueryValidationError(RuntimeError):
    pass


class QueryExecutionError(RuntimeError):
    pass


class PlanCorrectionError(ValueError):
    pass


@dataclass(frozen=True)
class Parameter:
    name: str
    type_: str
    value: object


@dataclass(frozen=True)
class CompiledQuery:
    sql: str
    parameters: tuple[Parameter, ...]


METRIC_SQL = {
    Metric.REVENUE: "ROUND(SUM(sale_price), 2) AS revenue",
    Metric.ORDER_COUNT: "COUNT(DISTINCT order_id) AS order_count",
    Metric.AOV: (
        "ROUND(SAFE_DIVIDE(SUM(sale_price), COUNT(DISTINCT order_id)), 2) AS aov"
    ),
}
DIMENSION_SQL = {
    Dimension.BRAND: ("brand", "brand"),
    Dimension.PRODUCT: ("product", "product"),
    Dimension.STATE: ("state", "state"),
    Dimension.MONTH: ("FORMAT_DATE('%Y-%m', event_date)", "month"),
}
TABLES = ("orders", "order_items", "products", "users")


def compile_query(
    plan: AnalysisPlan, scope: AuthorizedScope, dataset: str
) -> CompiledQuery:
    validate_plan_for_execution(plan)
    parameters: list[Parameter] = []
    predicates = ["oi.status = 'Complete'", "o.status = 'Complete'"]

    if scope.brands is not None:
        predicates.append("p.brand IN UNNEST(@allowed_brands)")
        parameters.append(Parameter("allowed_brands", "STRING_ARRAY", list(scope.brands)))

    period_predicates: list[str] = []
    period_cases: list[str] = []
    for index, period in enumerate(plan.periods):
        start_name = f"period_{index}_start"
        end_name = f"period_{index}_end"
        label_name = f"period_{index}_label"
        clause = (
            f"DATE(oi.created_at) >= @{start_name} "
            f"AND DATE(oi.created_at) < @{end_name}"
        )
        period_predicates.append(f"({clause})")
        period_cases.append(f"WHEN {clause} THEN @{label_name}")
        parameters.extend(
            (
                Parameter(start_name, "DATE", period.start),
                Parameter(end_name, "DATE", period.end),
            )
        )
        if len(plan.periods) == 2:
            parameters.append(Parameter(label_name, "STRING", period.label))
    predicates.append("(" + " OR ".join(period_predicates) + ")")

    if plan.products:
        predicates.append("LOWER(p.name) IN UNNEST(@products)")
        parameters.append(
            Parameter("products", "STRING_ARRAY", [value.casefold() for value in plan.products])
        )
    if plan.states:
        predicates.append("LOWER(u.state) IN UNNEST(@states)")
        parameters.append(
            Parameter("states", "STRING_ARRAY", [value.casefold() for value in plan.states])
        )

    source_columns = [
        "oi.order_id",
        "oi.user_id",
        "CAST(oi.sale_price AS NUMERIC) AS sale_price",
        "DATE(oi.created_at) AS event_date",
        "p.brand",
        "p.name AS product",
        "u.state",
    ]
    if len(plan.periods) == 2:
        source_columns.append(
            "CASE " + " ".join(period_cases) + " END AS comparison_period"
        )

    groups: list[tuple[str, str]] = []
    if len(plan.periods) == 2:
        groups.append(("comparison_period", "period"))
    groups.extend(DIMENSION_SQL[dimension] for dimension in plan.group_by)

    select_columns = [f"{expression} AS {alias}" for expression, alias in groups]
    select_columns.extend(METRIC_SQL[metric] for metric in plan.metrics)
    select_columns.append("COUNT(DISTINCT user_id) AS distinct_customers")
    group_sql = ""
    if groups:
        group_sql = "\nGROUP BY " + ", ".join(str(index) for index in range(1, len(groups) + 1))

    order_sql = (
        "ORDER BY month ASC"
        if Dimension.MONTH in plan.group_by
        else f"ORDER BY {plan.metrics[0].value} DESC"
    )

    sql = f"""WITH scoped_items AS (
  SELECT
    {', '.join(source_columns)}
  FROM `{dataset}.order_items` AS oi
  JOIN `{dataset}.orders` AS o
    ON o.order_id = oi.order_id AND o.user_id = oi.user_id
  JOIN `{dataset}.products` AS p ON p.id = oi.product_id
  JOIN `{dataset}.users` AS u ON u.id = oi.user_id
  WHERE {' AND '.join(predicates)}
)
SELECT
  {', '.join(select_columns)}
FROM scoped_items{group_sql}
{order_sql}
LIMIT {plan.limit}"""
    validate_sql(sql, dataset)
    return CompiledQuery(sql=sql, parameters=tuple(parameters))


def validate_plan_for_execution(plan: AnalysisPlan) -> None:
    if len(set(plan.metrics)) != len(plan.metrics):
        raise PlanCorrectionError("metrics must be unique")
    if len(set(plan.group_by)) != len(plan.group_by):
        raise PlanCorrectionError("grouping dimensions must be unique")

    for period in plan.periods:
        if period.end <= period.start:
            raise PlanCorrectionError(
                "period end must be after start (end is exclusive)"
            )
    if len(plan.periods) == 2:
        first, second = sorted(plan.periods, key=lambda period: period.start)
        if first.end > second.start:
            raise PlanCorrectionError("comparison periods must not overlap")

    comparisons = (
        (plan.brands, Dimension.BRAND, "brand"),
        (plan.products, Dimension.PRODUCT, "product"),
        (plan.states, Dimension.STATE, "state"),
    )
    for values, dimension, label in comparisons:
        if len(values) > 1 and dimension not in plan.group_by:
            raise PlanCorrectionError(
                f"a {label} comparison must group by {dimension.value}"
            )
    if len(plan.periods) == 2 and plan.periods[0].label.casefold() == plan.periods[1].label.casefold():
        raise PlanCorrectionError("comparison periods need distinct labels")


def validate_sql(sql: str, dataset: str) -> None:
    try:
        statements = sqlglot.parse(sql, read="bigquery")
    except sqlglot.errors.ParseError as error:
        raise QueryValidationError("compiler produced invalid GoogleSQL") from error
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise QueryValidationError("exactly one SELECT query is required")

    statement = statements[0]
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Command)
    if any(statement.find(node) is not None for node in forbidden):
        raise QueryValidationError("mutating or scripted SQL is forbidden")
    if statement.find(exp.Star) is not None:
        raise QueryValidationError("wildcard projections are forbidden")
    if statement.find(exp.Anonymous) is not None:
        raise QueryValidationError("unapproved functions are forbidden")

    allowed = {f"{dataset}.{table}".casefold() for table in TABLES}
    allowed.add("scoped_items")
    for table in statement.find_all(exp.Table):
        name = ".".join(
            part
            for part in (table.catalog, table.db, table.name)
            if part
        ).casefold()
        if name not in allowed:
            raise QueryValidationError(f"table is not allowlisted: {name}")


class BigQueryRunner:
    def __init__(
        self,
        settings: Settings,
        client: bigquery.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._client = client
        self._sleep = sleeper

    @property
    def client(self) -> bigquery.Client:
        if self._client is None:
            self._client = bigquery.Client(
                project=self.settings.google_cloud_project,
                location=self.settings.bigquery_location,
            )
        return self._client

    def list_brands(self, limit: int = 50) -> list[str]:
        sql = f"""SELECT DISTINCT brand
FROM `{self.settings.bigquery_dataset}.products`
WHERE brand IS NOT NULL
ORDER BY brand
LIMIT {max(1, min(limit, 200))}"""
        rows = self.client.query(sql, retry=None, timeout=20).result(timeout=30)
        return [str(row["brand"]) for row in rows]

    def execute(self, query: CompiledQuery, request_id: str) -> QueryResult:
        query_hash = _query_hash(query)
        parameters = _bigquery_parameters(query.parameters)
        dry_config = bigquery.QueryJobConfig(
            dry_run=True,
            use_query_cache=False,
            use_legacy_sql=False,
            query_parameters=parameters,
            maximum_bytes_billed=self.settings.bigquery_max_bytes,
        )
        try:
            dry_job = self._call_with_transient_retries(
                lambda: self.client.query(
                    query.sql,
                    job_config=dry_config,
                    location=self.settings.bigquery_location,
                    retry=None,
                    timeout=15,
                )
            )
        except google_exceptions.BadRequest as error:
            raise QueryValidationError("BigQuery rejected the compiled query") from error
        except google_exceptions.GoogleAPICallError as error:
            raise QueryExecutionError("BigQuery request failed") from error
        estimated = int(dry_job.total_bytes_processed or 0)
        if estimated > self.settings.bigquery_max_bytes:
            raise QueryExecutionError(
                f"query estimate {estimated} exceeds byte budget"
            )

        job_config = bigquery.QueryJobConfig(
            use_legacy_sql=False,
            use_query_cache=True,
            query_parameters=parameters,
            maximum_bytes_billed=self.settings.bigquery_max_bytes,
            job_timeout_ms=60_000,
        )
        job_id = "retail_agent_" + hashlib.sha256(
            f"{request_id}:{query_hash}".encode()
        ).hexdigest()[:24]
        try:
            job = self._submit_or_reconcile(query.sql, job_config, job_id)
        except google_exceptions.BadRequest as error:
            raise QueryValidationError("BigQuery rejected the compiled query") from error
        except google_exceptions.GoogleAPICallError as error:
            raise QueryExecutionError("BigQuery submission failed") from error
        try:
            row_iterator = self._call_with_transient_retries(
                lambda: job.result(timeout=70)
            )
        except TimeoutError as error:
            self.client.cancel_job(
                job_id, location=self.settings.bigquery_location, retry=None
            )
            raise QueryExecutionError("BigQuery timed out; cancellation requested") from error
        except google_exceptions.BadRequest as error:
            raise QueryValidationError("BigQuery rejected the compiled query") from error
        except google_exceptions.GoogleAPICallError as error:
            raise QueryExecutionError("BigQuery execution failed") from error
        rows = [dict(row.items()) for row in row_iterator]
        return QueryResult(
            rows=rows,
            job_id=job.job_id,
            bytes_processed=int(job.total_bytes_processed or estimated),
            query_hash=query_hash,
        )

    def _submit_or_reconcile(
        self, sql: str, config: bigquery.QueryJobConfig, job_id: str
    ) -> object:
        for attempt in range(3):
            try:
                return self.client.query(
                    sql,
                    job_config=config,
                    job_id=job_id,
                    location=self.settings.bigquery_location,
                    retry=None,
                    job_retry=None,
                    timeout=15,
                )
            except google_exceptions.Conflict:
                return self.client.get_job(
                    job_id, location=self.settings.bigquery_location, retry=None
                )
            except TRANSIENT_ERRORS as error:
                logger.info(
                    "bigquery_submission_retry",
                    extra={"event": "bigquery_submission_retry", "attempt": attempt + 1},
                )
                try:
                    return self.client.get_job(
                        job_id, location=self.settings.bigquery_location, retry=None
                    )
                except google_exceptions.NotFound:
                    if attempt == 2:
                        raise QueryExecutionError("BigQuery submission failed") from error
                    self._sleep(0.25 * (2**attempt))
        raise QueryExecutionError("BigQuery submission failed")

    def _call_with_transient_retries(self, operation: Callable[[], object]) -> object:
        for attempt in range(3):
            try:
                return operation()
            except TRANSIENT_ERRORS as error:
                if attempt == 2:
                    raise QueryExecutionError("BigQuery is temporarily unavailable") from error
                logger.info(
                    "bigquery_retry",
                    extra={"event": "bigquery_retry", "attempt": attempt + 1},
                )
                self._sleep(0.25 * (2**attempt))
        raise QueryExecutionError("BigQuery is temporarily unavailable")


TRANSIENT_ERRORS = (
    google_exceptions.TooManyRequests,
    google_exceptions.ServiceUnavailable,
    google_exceptions.InternalServerError,
    google_exceptions.BadGateway,
    google_exceptions.GatewayTimeout,
)


def _bigquery_parameters(parameters: Iterable[Parameter]) -> list[object]:
    result: list[object] = []
    for parameter in parameters:
        if parameter.type_ == "STRING_ARRAY":
            result.append(
                bigquery.ArrayQueryParameter(parameter.name, "STRING", parameter.value)
            )
        else:
            result.append(
                bigquery.ScalarQueryParameter(
                    parameter.name, parameter.type_, parameter.value
                )
            )
    return result


def _query_hash(query: CompiledQuery) -> str:
    payload = {
        "sql": query.sql,
        "parameters": [
            {"name": item.name, "type": item.type_, "value": item.value}
            for item in query.parameters
        ],
    }
    canonical = json.dumps(
        payload, default=str, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def plan_without_narrow_filters(plan: AnalysisPlan) -> AnalysisPlan:
    return plan.model_copy(
        update={
            "metrics": [Metric.ORDER_COUNT],
            "group_by": [],
            "products": [],
            "states": [],
            "limit": 1,
        }
    )


