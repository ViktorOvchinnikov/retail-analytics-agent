# Retail Analytics Agent

Minimal CLI prototype for the Retail Analytics Agent take-home assignment. The production design is in [architecture.md](architecture.md); this implementation follows its Section 11 scope.

LangGraph was chosen for explicit conditional flow, bounded correction, and persisted conversation state while keeping one orchestrator. The author has prior hands-on experience with LangChain/LangGraph and production LLM systems; this prototype uses only the graph and checkpoint features needed for the assignment.

## What it implements

- One LangGraph conversation graph backed by SQLite checkpoints; raw prompts use run-scoped context and are not checkpointed.
- Gemini structured planning and evidence-grounded report writing.
- Deterministic, parameterized BigQuery SQL for revenue, distinct order count, and AOV.
- Grouping by brand, product, state, and month; up to two periods or selected products for comparison.
- Server-owned manager/CEO brand scopes, read-only SQL validation, query dry runs, byte/time limits, aggregate suppression, and final PII checks.
- SQLite saved reports with exact deletion previews, expiring confirmation tokens, ownership checks, idempotent confirmation, and audit metadata.
- App-owned Gemini planning retries (three attempts, 14 seconds each, 40 seconds overall), bounded semantic plan correction, transient BigQuery retries, empty-result diagnosis, deterministic response fallback, and privacy-safe JSON logs.
- A small local Golden Knowledge set retrieved lexically before planning/reporting.

Customer-level output, churn/cohort logic, arbitrary SQL, charts, email, web search, production SSO, DLP, hybrid vector retrieval, and cloud observability remain production work.

## Quick start with Docker Compose

Docker Compose is optional, but it provides the shortest path to the interactive CLI. You still need a valid Gemini API key, a billing-enabled Google Cloud project with BigQuery access, and Google Application Default Credentials (ADC).

Create the local configuration and set `GEMINI_API_KEY`, `GOOGLE_CLOUD_PROJECT`, and the mock manager brands:

```bash
cp .env.example .env
# Edit .env before continuing.
```

Authenticate on the host and set the ADC quota project to the same project used in `.env`:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_GCP_PROJECT_ID
```

Compose mounts the host's default ADC file read-only; credentials are never copied into the image. Build the image, then start a manager or CEO session:

```bash
docker compose build
docker compose run --rm cli --user manager_1 --conversation demo
docker compose run --rm cli --user ceo --conversation ceo-demo
```

SQLite checkpoints, saved reports, and JSON logs are stored in the named `retail-agent-state` volume. To verify persistence, create or save a report, exit, start another manager session, and enter `/reports`:

```bash
docker compose run --rm cli --user manager_1 --conversation demo
```

The bind mount expects ADC at `$HOME/.config/gcloud/application_default_credentials.json`, the default location created by `gcloud auth application-default login` on macOS and Linux. The image targets `linux/amd64`; Docker Desktop uses emulation on Apple Silicon. The non-root container user defaults to UID 1000. On a Linux host with another UID, build with `APP_UID=$(id -u) docker compose build` so it can read a mode-600 ADC file. The container does not include the Google Cloud CLI. The existing Python workflow remains fully supported; use the **Setup** and **Run** sections below when running without Docker.

## Setup

Requires Python 3.11+, a Gemini API key, a billing-enabled Google Cloud project, and BigQuery Application Default Credentials.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
gcloud auth application-default login
```

Set `GEMINI_API_KEY` and `GOOGLE_CLOUD_PROJECT` in `.env`. The mock managers must use brands present in the public dataset. List them with:

```bash
retail-agent --list-brands
```

Then set `MANAGER_1_BRANDS` and `MANAGER_2_BRANDS` as comma-separated values. The defaults are common TheLook brands but should be verified because the synthetic public dataset can change.

## Run

```bash
retail-agent --user manager_1 --conversation demo
```

Example prompts:

```text
Show monthly revenue, order count, and AOV for Q1 2026.
Compare that with Q4 2025.
Create and save a report comparing two products last quarter, with action items.
What data can you analyze?
delete all reports from this conversation
```

Deletion only occurs after the CLI displays exact report IDs and the user enters the generated `/confirm ACTION_ID TOKEN` command. A plain “yes” cannot delete reports.

## Test

```bash
pip install -e ".[dev]"
pytest
```

Tests mock Gemini and BigQuery. They exercise compiler allowlists and parameterized brand scope, manager isolation, PII suppression, bounded plan correction, empty-result diagnosis, persistent follow-ups, and deletion confirmation. Live Gemini/BigQuery behavior requires the credentials above.

Run the opt-in real-service smoke test with:

```bash
RUN_LIVE_INTEGRATION=1 pytest -m integration
```

## Golden evaluations

The human-readable [Golden Evaluation Dataset](evals/golden_cases.json) contains
15 held-out cases for numerical correctness, intent alignment, follow-ups,
authorization/privacy, empty and unsupported requests, and report grounding.
It is separate from `retail_agent/data/golden.json` and is never retrieved by
the agent.

Four numerical cases execute independent SQL from
[`evals/reference_sql/`](evals/reference_sql/) at evaluation time. These queries
implement the documented metric definitions independently of the agent's SQL
compiler. The runner compares structured plans and validated BigQuery rows,
using a 0.01 absolute tolerance for numeric values; it does not parse free-form
answers or use an LLM judge.

Run credential-free dataset and deterministic security checks:

```bash
python -m evals.run_evals
pytest tests/test_evaluation.py
```

Run all 15 cases against real Gemini and BigQuery:

```bash
RUN_LIVE_EVALS=1 python -m evals.run_evals --live
RUN_LIVE_EVALS=1 pytest -m evaluation -s
```

Live evaluations use temporary SQLite state, the existing query byte limits,
and the fixed public TheLook dataset. The summary reports pass/fail/skip counts
and category rates. Local results intentionally mark service-dependent cases as
skipped. Report-quality scores are deterministic structural, numerical
grounding, privacy, and supported-claim checks; they are not a subjective
editorial assessment. Public data and Gemini behavior can change, so rerun the
suite after model, prompt, compiler, or dataset changes.

## Prototype notes

- The CLI-selected user is a mock identity. Production uses corporate SSO and centrally managed grants.
- The public tables cannot reproduce production authorized views; the prototype service account should have read-only access.
- The compiler supports a deliberately small analytical vocabulary. Unsupported requests receive a clear response; the LLM never gets raw-SQL fallback access.
- Structured logs go to stdout and `retail-agent.jsonl`. Planning failures include stage, model, attempt, duration, sanitized exception class/category, and provider status. Logs contain pseudonymous user/conversation IDs, request/job IDs, timings, retry/error categories, and query bytes—not prompts, result rows, exception payloads, or credentials.
- Revenue uses completed item `sale_price`; order count is distinct completed orders in the same scope; AOV is their ratio. Dates are half-open UTC ranges. These definitions must be agreed with the client before production use.
