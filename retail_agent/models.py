from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Metric(str, Enum):
    REVENUE = "revenue"
    ORDER_COUNT = "order_count"
    AOV = "aov"


class Dimension(str, Enum):
    BRAND = "brand"
    PRODUCT = "product"
    STATE = "state"
    MONTH = "month"


class Period(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=40)
    start: date
    end: date


class AnalysisPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="Retail analysis", min_length=1, max_length=100)
    metrics: list[Metric] = Field(
        default_factory=lambda: [Metric.REVENUE, Metric.ORDER_COUNT, Metric.AOV],
        min_length=1,
        max_length=3,
    )
    group_by: list[Dimension] = Field(default_factory=list, max_length=2)
    periods: list[Period] = Field(min_length=1, max_length=2)
    brands: list[str] = Field(default_factory=list, max_length=10)
    products: list[str] = Field(default_factory=list, max_length=10)
    states: list[str] = Field(default_factory=list, max_length=10)
    limit: int = Field(default=20, ge=1, le=50)


class PlanningDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["analysis", "schema", "unsupported"]
    plan: AnalysisPlan | None = None
    message: str | None = Field(default=None, max_length=300)
    save_report: bool = False

    @model_validator(mode="after")
    def plan_matches_kind(self) -> "PlanningDecision":
        if self.kind == "analysis" and self.plan is None:
            raise ValueError("analysis decisions require a plan")
        if self.kind != "analysis" and self.plan is not None:
            raise ValueError("non-analysis decisions cannot include a plan")
        return self


class UserProfile(BaseModel):
    user_id: str
    role: Literal["manager", "ceo"]
    allowed_brands: tuple[str, ...]


class QueryResult(BaseModel):
    rows: list[dict[str, object]]
    job_id: str
    bytes_processed: int
    query_hash: str


class ReportSummary(BaseModel):
    report_id: str
    title: str
    created_at: str


class PendingDeletion(BaseModel):
    action_id: str
    token: str
    reports: list[ReportSummary]
    expires_at: str


class DeletionReceipt(BaseModel):
    action_id: str
    deleted_count: int
    report_ids: list[str]
    committed_at: str
