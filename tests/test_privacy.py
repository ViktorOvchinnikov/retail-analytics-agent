import pytest

from retail_agent.safety import contains_pii, requests_pii, sanitize_rows


def test_pii_requests_and_values_are_detected() -> None:
    assert requests_pii("Show the top customers and their email addresses")
    assert contains_pii("Contact jane@example.com")
    assert contains_pii("Source address 192.168.1.2")


def test_small_groups_are_suppressed_and_internal_count_removed() -> None:
    rows = [
        {"brand": "A", "revenue": 100.0, "distinct_customers": 9},
        {"brand": "B", "revenue": 500.0, "distinct_customers": 12},
    ]
    safe, suppressed = sanitize_rows(rows, 10)
    assert safe == [{"brand": "B", "revenue": 500.0}]
    assert suppressed == 1


def test_forbidden_result_column_fails_closed() -> None:
    with pytest.raises(ValueError, match="forbidden field"):
        sanitize_rows([{"email": "secret@example.com", "distinct_customers": 20}], 10)
