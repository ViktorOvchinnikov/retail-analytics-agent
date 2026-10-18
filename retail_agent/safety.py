from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .config import Settings
from .models import AnalysisPlan, UserProfile


class AuthorizationError(ValueError):
    pass


PII_REQUEST_PATTERNS = (
    re.compile(r"\b(e-?mails?|street addresses?|postal codes?|zip codes?|ip addresses?)\b", re.I),
    re.compile(r"\b(customer|user)\s+(name|id|identifier|email|address)\b", re.I),
    re.compile(r"\b(top|list|show)\s+(customers|users)\b", re.I),
    re.compile(r"\btop\s+\d+\s+(customers|users)\b", re.I),
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
FORBIDDEN_RESULT_FIELDS = {
    "id",
    "user_id",
    "customer_id",
    "first_name",
    "last_name",
    "email",
    "street_address",
    "postal_code",
    "ip_address",
    "latitude",
    "longitude",
}


@dataclass(frozen=True)
class AuthorizedScope:
    brands: tuple[str, ...] | None


def profiles(settings: Settings) -> dict[str, UserProfile]:
    return {
        "manager_1": UserProfile(
            user_id="manager_1",
            role="manager",
            allowed_brands=settings.manager_1_brands,
        ),
        "manager_2": UserProfile(
            user_id="manager_2",
            role="manager",
            allowed_brands=settings.manager_2_brands,
        ),
        "ceo": UserProfile(user_id="ceo", role="ceo", allowed_brands=("*",)),
    }


def authorize_plan(plan: AnalysisPlan, profile: UserProfile) -> AuthorizedScope:
    if profile.role == "ceo":
        return AuthorizedScope(tuple(plan.brands) or None)
    if not profile.allowed_brands:
        raise AuthorizationError("This manager has no assigned brands.")

    canonical = {brand.casefold(): brand for brand in profile.allowed_brands}
    if not plan.brands:
        return AuthorizedScope(profile.allowed_brands)

    denied = [brand for brand in plan.brands if brand.casefold() not in canonical]
    if denied:
        raise AuthorizationError(
            "Access denied for brand(s): " + ", ".join(sorted(denied))
        )
    return AuthorizedScope(tuple(canonical[brand.casefold()] for brand in plan.brands))


def requests_pii(text: str) -> bool:
    return any(pattern.search(text) for pattern in PII_REQUEST_PATTERNS)


def contains_pii(text: str) -> bool:
    return bool(EMAIL_RE.search(text) or IP_RE.search(text))


def explicit_unauthorized_brands(
    text: str, profile: UserProfile, known_brands: Iterable[str]
) -> list[str]:
    if profile.role == "ceo":
        return []
    allowed = {brand.casefold() for brand in profile.allowed_brands}
    return sorted(
        brand
        for brand in set(known_brands)
        if brand.casefold() not in allowed
        and re.search(rf"(?<!\w){re.escape(brand)}(?!\w)", text, re.I)
    )


def sanitize_rows(
    rows: Iterable[dict[str, object]], min_group_size: int
) -> tuple[list[dict[str, object]], int]:
    safe: list[dict[str, object]] = []
    suppressed = 0
    for row in rows:
        forbidden = {key.casefold() for key in row} & FORBIDDEN_RESULT_FIELDS
        if forbidden:
            raise ValueError(f"query returned forbidden field(s): {sorted(forbidden)}")
        customers = row.get("distinct_customers")
        if not isinstance(customers, int) or customers < min_group_size:
            suppressed += 1
            continue
        clean = {
            key: value
            for key, value in row.items()
            if key != "distinct_customers"
        }
        if contains_pii(" ".join(str(value) for value in clean.values())):
            raise ValueError("query result contains possible PII")
        safe.append(clean)
    return safe, suppressed
