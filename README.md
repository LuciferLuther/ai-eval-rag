# AI Eval + RAG Prototype

This repo contains two lean deliverables for measuring prompt reliability and prototyping a retrieval-augmented answering service.

## Environment Setup

1. Create a virtual environment (`python -m venv .venv`) and activate it.
2. Install dependencies: `pip install -r requirements.txt`.
3. populate `.env` with  `OPENAI_API_KEY` if you plan to hit the live API.

Common helper targets:

```bash
make install      # install dependencies
make eval         # run harness with the mock client
make run-api      # start FastAPI with reload
make smoke        # smoke test against a running API instance
```

## Question 1 - Prompt Reliability Mini-Harness

- **Test set** lives in `eval/tests.json` with 10 cases (5 synthetic + 5 anonymised real-world scenarios). Synthetic prompts cover everyday library help-desk flows (card signup, renewals, printing, events). The real prompts anonymise prior client engagements (financial cross-validation, fire safety automation, agency tooling, healthcare booking) while preserving their key facts.
- **Invariance test:** `syn_02_get_library_card_invariance` paraphrases `syn_01` with slang and typos to ensure the signup policy answer stays consistent.
- **Perturbation test:** `syn_05_printing_perturbation` adds an unrelated Wi-Fi clause while expecting the pricing policy to remain precise.
- **Harness:** `eval/eval_harness.py` loads the suite, calls into `eval/openai_client.py`, and scores responses with regex/exact/substring checks. It reports accuracy plus invariance and perturbation consistency rates. Use `--mock` for deterministic output or supply a real API key.
- **Automation:** `scripts/run_eval.sh` wraps the harness for CI or local runs, and `make eval` defaults to the mock client.
- **Ship vs. later:**
  - Ship now: keep curating the test mix, land the harness + reporting, wire it into CI/nightly runs.
  - Later: add semantic grading (e.g. embeddings), persist trend dashboards, and auto-diff regression explanations for failing variants.

## Question 2 - Retrieval + Guardrails FastAPI Prototype

- **Corpus:** `rag_api/corpus.py` defines 12 purpose-written snippets framed as a community library FAQ (cards, renewals, printing, events, volunteering, donations). `CorpusIndex` builds a TF-IDF matrix with support for cosine or raw dot-product similarity.
- **Endpoint:** `POST /answer` (implemented in `rag_api/app.py`) accepts `query`, optional `k`, and `similarity`. It returns the naive answer plus the top-k snippets.
- **Guardrail:** `rag_api/guardrails.py` enforces a denylist for credential/PII-seeking queries, short-circuiting the answer. This keeps risky requests blocked without adding latency and can be expanded with analytics.
- **Index comparison:** Switch `similarity` between `cosine` and `dot`. Cosine normalises vectors for balanced snippets, while dot-product is exposed for experiments when raw term frequency matters (see response `index_config`).
- **Monitoring metrics:**
  1. **P95 end-to-end latency** - capture via FastAPI middleware logging into tooling such as OpenTelemetry.
  2. **Retrieval hit-rate** - percentage of queries whose top score exceeds a relevance threshold; log alongside query metadata to surface drift.
- **Smoke test:** once the API is running (`make run-api`), execute `scripts/smoke_api.sh` to hit `/health` and `/answer`.

## Docker & Compose

- Build the image: `docker build -t ai-eval-rag .`
- Run via compose: `docker compose up --build` to expose the service on `localhost:8000`.

## Next Steps

- Fill in production telemetry (latency + hit-rate collectors) and push eval reports to your preferred observability stack.
- Extend the harness with semantic grading and alerting once you have baseline stability.
- Capture decision notes and metrics snapshots in the companion Google Doc for submission.