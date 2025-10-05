# AI Eval + RAG Prototype

This repository ships two lightweight deliverables:
- A prompt-reliability harness for tracking regressions against a curated test set.
- A FastAPI retrieval prototype with basic guardrails and TF-IDF indexing.

The codebase is organised for small, fast iterations: tests are JSON-controlled, the harness can run entirely against a mock client, and the API surfaces simple diagnostics to plug into monitoring.

## Repository Layout
- `eval/` - Harness entry point, OpenAI client shim, test suite, and JSON reports.
- `rag_api/` - FastAPI application, TF-IDF corpus/index, and guardrail helpers.
- `scripts/` - Runner wrappers (`run_eval.sh`, `smoke_api.sh`).
- `Dockerfile`, `docker-compose.yml` - Containerisation for the API service.
- `Makefile` - Convenience targets for local workflows.

## Prerequisites & Environment
- Python 3.10+ (tested on 3.11).
- Optional: Docker / Docker Compose for container runs.
- Environment variables (loaded from `.env`, `../.env`, or `../../.env` by default):
  - `OPENAI_API_KEY` - Required for live calls (skip if using the mock client).
  - `OPENAI_BASE_URL`, `OPENAI_MODEL` - Optional overrides for the OpenAI SDK wrapper.
  - `OPENAI_MOCK=true` - Forces the deterministic mock client regardless of CLI flags.

### Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Or use `make install` once the virtual environment is active.

## Make & Script Shortcuts
- `make eval` - Run the harness with the mock client (`--mock`).
- `make run-api` - Launch FastAPI locally with auto-reload (`uvicorn`).
- `make smoke` - Hit `/health`, `/answer`, a guardrail block, and `/metrics` on localhost via `scripts/smoke_api.sh` (requires `jq` in PATH).
- `make docker-build` / `make docker-up` - Container workflows.

`scripts/run_eval.sh` forwards all CLI arguments to `eval/eval_harness.py`; `scripts/smoke_api.sh` targets `http://127.0.0.1:8000` by default; edit the script if you need another host.

## Prompt Reliability Harness (`eval/eval_harness.py`)
### Test Suite
- Definitions live in `eval/tests.json` (10 cases: 5 synthetic, 5 anonymised real).
- Supports invariance (e.g. `syn_02` vs `syn_01`) and perturbation pairs (`syn_06` vs `syn_05`).
- Evaluation methods: `regex`, `contains`, `exact` (validated on load).
- Currency normalisation / conversion variants are covered by the mock client for deterministic runs.
- Responses are normalised (Unicode + whitespace) before consistency checks to reduce false diffs.

### Running the Harness
```bash
# Deterministic mock (default in Makefile)
python eval/eval_harness.py --tests eval/tests.json --mock

# Live API usage with custom model & report directory
python eval/eval_harness.py \
  --tests eval/tests.json \
  --model gpt-4o-mini \
  --temperature 0.1 \
  --output-dir eval/reports \
  --invariance-mode identical
```
Flags of note:
- `--mock` - Use the canned `MockEchoClient`. Disable to call OpenAI via `OPENAI_API_KEY`.
- `--dry-run` - Skip model calls and echo `[dry-run] <test_id>` outputs.
- `--invariance-mode` - `both-pass` (default) counts variants as consistent if both pass; `identical` also requires identical normalised responses.
- `--output-dir` - Saves timestamped JSON reports (`eval/reports/eval_YYYYMMDD-HHMMSS.json`).

Running the harness prints per-case pass/fail lines, followed by a summary containing accuracy, invariance rate, and perturbation divergence rate.

## OpenAI Client Shim (`eval/openai_client.py`)
- `build_client()` returns either the real Responses API wrapper or the deterministic mock.
- The real client honours `OPENAI_MODEL`, `OPENAI_BASE_URL`, and forwards `temperature`/`max_output_tokens` overrides.
- The mock recognises the library FAQ prompts plus numeric normalisation cases used in the test suite.

## Retrieval API (`rag_api/app.py`)
- Base URL: `POST /answer`, `GET /metrics`, and `GET /health`.
- Request body: `query: str`, optional `k` (1-10) and `similarity` (`cosine` | `dot`).
- Response includes the composed answer, scored snippets (`doc_id`, `title`, `text`, `score`), guardrail metadata, and the `index_config` echo.
- Guardrails (`rag_api/guardrails.py`) apply a denylist for credential / PII / prompt-injection phrases and a simple word-budget check before reaching the retriever.
- Retrieval stack (`rag_api/corpus.py`) loads 12 curated FAQ snippets, builds a TF-IDF matrix, and supports cosine or raw dot-product search. The top snippet is summarised with provenance titles by `compose_answer()`.
- Index comparison: I expose TF-IDF with cosine and dot similarity. On this short, mixed-length corpus, cosine is more stable (length-normalized) and usually ranks the intended policy note higher. I keep `k=3` by default to minimize noise on small corpora, with `k` configurable per request.
- Monitoring: In-memory counters track request volume, guardrail blocks, p95 latency (ms), and a >0.2 TF-IDF hit-rate proxy; the aggregates surface at `GET /metrics`.

### Running & Testing the API
```bash
# Local dev server
uvicorn rag_api.app:app --host 0.0.0.0 --port 8000 --reload
# or
make run-api

# Smoke test once it is running
bash scripts/smoke_api.sh
```
Example `POST /answer` payload:
```json
{
  "query": "How do I get a library card?",
  "k": 3,
  "similarity": "cosine"
}
```
Inspect live metrics while the API is running:
```bash
curl -s localhost:8000/metrics | jq
```

## Docker Support
```bash
# Build image
make docker-build

# Bring up stack (exposes FastAPI on localhost:8000)
make docker-up
```

## Extending & Next Steps
- Add semantic grading (embeddings, vector search) once baseline scores stabilise.
- Wire the harness into CI/nightly jobs and archive JSON reports (`eval/reports/`).
- Push the `/metrics` counters into your telemetry stack; the in-app stats are in-memory only.
- Expand guardrails or replace denylist matching with a policy engine as production risks evolve.
