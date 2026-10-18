from __future__ import annotations

import sqlglot

from retail_agent.golden import GoldenKnowledge


def test_golden_examples_include_reviewed_reference_sql() -> None:
    knowledge = GoldenKnowledge()

    assert len(knowledge.examples) == 3
    for example in knowledge.examples:
        assert set(example) == {"question", "sql", "method", "report"}
        parsed = sqlglot.parse_one(example["sql"], read="bigquery")
        sql = example["sql"]
        assert "@brand" in sql
        assert "o.status = 'Complete'" in sql
        assert "oi.status = 'Complete'" in sql
        assert ">= @" in sql and "< @" in sql
        columns = {
            column.name.casefold()
            for column in parsed.find_all(sqlglot.exp.Column)
        }
        assert not columns & {
            "email",
            "first_name",
            "last_name",
            "street_address",
            "ip_address",
            "latitude",
            "longitude",
        }


def test_retrieval_keeps_sql_as_reference_context() -> None:
    examples = GoldenKnowledge().retrieve("monthly revenue by brand")

    assert examples
    assert all("sql" in example for example in examples)
