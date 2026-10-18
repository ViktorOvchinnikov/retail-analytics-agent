from datetime import date

import pytest

from retail_agent.analytics import (
    PlanCorrectionError,
    QueryValidationError,
    compile_query,
    validate_sql,
)
from retail_agent.models import AnalysisPlan, Dimension, Metric, Period, UserProfile
from retail_agent.safety import AuthorizationError, authorize_plan


DATASET = "bigquery-public-data.thelook_ecommerce"


def plan(**updates: object) -> AnalysisPlan:
    values = {
        "periods": [Period(label="Q1", start=date(2025, 1, 1), end=date(2025, 4, 1))],
        "group_by": [Dimension.MONTH],
    }
    values.update(updates)
    return AnalysisPlan(**values)


def test_manager_scope_is_parameterized_and_applied_before_aggregation() -> None:
    profile = UserProfile(
        user_id="manager", role="manager", allowed_brands=("Calvin Klein", "Levi's")
    )
    query = compile_query(plan(), authorize_plan(plan(), profile), DATASET)

    assert "p.brand IN UNNEST(@allowed_brands)" in query.sql
    assert "Calvin Klein" not in query.sql
    allowed = next(parameter for parameter in query.parameters if parameter.name == "allowed_brands")
    assert allowed.value == ["Calvin Klein", "Levi's"]
    assert all(f"`{DATASET}.{table}`" in query.sql for table in ("orders", "order_items", "products", "users"))


def test_manager_cannot_request_unassigned_brand() -> None:
    profile = UserProfile(
        user_id="manager", role="manager", allowed_brands=("Calvin Klein",)
    )
    with pytest.raises(AuthorizationError, match="Access denied"):
        authorize_plan(plan(brands=["Carhartt"]), profile)


def test_comparison_requires_its_grouping_dimension() -> None:
    profile = UserProfile(
        user_id="manager", role="manager", allowed_brands=("Calvin Klein", "Levi's")
    )
    invalid = plan(brands=["Calvin Klein", "Levi's"])
    with pytest.raises(PlanCorrectionError, match="group by brand"):
        compile_query(invalid, authorize_plan(invalid, profile), DATASET)


@pytest.mark.parametrize("metric", list(Metric))
@pytest.mark.parametrize("group_by", [[], [Dimension.BRAND]])
def test_each_supported_metric_has_a_valid_ordering(metric, group_by) -> None:
    profile = UserProfile(
        user_id="manager", role="manager", allowed_brands=("Calvin Klein",)
    )
    analysis = plan(metrics=[metric], group_by=group_by)
    query = compile_query(analysis, authorize_plan(analysis, profile), DATASET)

    assert f"ORDER BY {metric.value} DESC" in query.sql


def test_recoverable_date_error_reaches_execution_validation() -> None:
    profile = UserProfile(
        user_id="manager", role="manager", allowed_brands=("Calvin Klein",)
    )
    invalid = plan(
        periods=[Period(label="bad", start=date(2025, 4, 1), end=date(2025, 1, 1))]
    )

    with pytest.raises(PlanCorrectionError, match="period end"):
        compile_query(invalid, authorize_plan(invalid, profile), DATASET)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM `bigquery-public-data.thelook_ecommerce.orders` WHERE TRUE",
        "SELECT * FROM `other.dataset.orders`",
        "SELECT 1; SELECT 2",
        "SELECT suspicious_remote_function(1)",
    ],
)
def test_sql_validator_rejects_unsafe_queries(sql: str) -> None:
    with pytest.raises(QueryValidationError):
        validate_sql(sql, DATASET)
