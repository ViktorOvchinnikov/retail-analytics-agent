from evals.run_evals import EXPECTED_COUNTS, load_cases, run_local


def test_golden_evaluation_dataset_has_required_coverage() -> None:
    document, cases = load_cases()

    assert document["dataset"] == "bigquery-public-data.thelook_ecommerce"
    assert len(cases) == 15
    assert {
        category: sum(case["category"] == category for case in cases)
        for category in EXPECTED_COUNTS
    } == EXPECTED_COUNTS
    assert all(
        "reference_sql" not in case["validation"]
        or case["validation"]["reference_sql"].startswith("evals/reference_sql/")
        for case in cases
    )


def test_local_golden_evaluations_are_credential_free() -> None:
    results = run_local()

    assert len(results) == 15
    assert sum(result.status == "passed" for result in results) == 3
    assert sum(result.status == "skipped" for result in results) == 12
    assert not [result for result in results if result.status == "failed"]
