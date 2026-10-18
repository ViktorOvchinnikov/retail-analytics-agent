from google.api_core.exceptions import ServiceUnavailable

from retail_agent.analytics import (
    BigQueryRunner,
    CompiledQuery,
    Parameter,
    _query_hash,
)


def test_transient_bigquery_error_is_bounded_and_recovers(settings) -> None:
    sleeps: list[float] = []
    runner = BigQueryRunner(settings, client=object(), sleeper=sleeps.append)
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ServiceUnavailable("try again")
        return "ok"

    assert runner._call_with_transient_retries(operation) == "ok"
    assert attempts == 3
    assert sleeps == [0.25, 0.5]


def test_idempotent_bigquery_job_disables_library_job_retry(settings) -> None:
    calls: list[dict[str, object]] = []

    class Client:
        def query(self, sql: str, **kwargs: object) -> object:
            calls.append(kwargs)
            return object()

    runner = BigQueryRunner(settings, client=Client())
    runner._submit_or_reconcile("SELECT 1", object(), "job-id")

    assert calls[0]["job_retry"] is None


def test_query_hash_includes_bound_parameters() -> None:
    sql = "SELECT @brand"
    calvin_klein = CompiledQuery(
        sql=sql,
        parameters=(Parameter("brand", "STRING", "Calvin Klein"),),
    )
    allegra_k = CompiledQuery(
        sql=sql,
        parameters=(Parameter("brand", "STRING", "Allegra K"),),
    )

    assert _query_hash(calvin_klein) == _query_hash(calvin_klein)
    assert _query_hash(calvin_klein) != _query_hash(allegra_k)
