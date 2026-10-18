import json
import logging

from retail_agent.observability import configure_logging


def test_logging_is_package_scoped_and_only_keeps_approved_fields(tmp_path) -> None:
    path = tmp_path / "agent.jsonl"
    configure_logging(path)

    logging.getLogger("third_party").warning(
        "external", extra={"prompt": "customer@example.com", "api_token": "secret"}
    )
    logging.getLogger("retail_agent.test").info(
        "query_succeeded",
        extra={
            "event": "query_succeeded",
            "request_id": "request-1",
            "rows": 2,
            "prompt": "customer@example.com",
            "api_token": "secret",
        },
    )
    logging.getLogger("retail_agent.test").warning(
        "unstructured customer@example.com"
    )

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0] == {
        "timestamp": records[0]["timestamp"],
        "level": "INFO",
        "event": "query_succeeded",
        "request_id": "request-1",
        "rows": 2,
    }
    assert records[1] == {
        "timestamp": records[1]["timestamp"],
        "level": "WARNING",
        "event": "log",
    }
    assert "customer@example.com" not in path.read_text()
