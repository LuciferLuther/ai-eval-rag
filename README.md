# AI Eval + RAG Prototype

This repo contains two lean deliverables for measuring prompt reliability and prototyping a retrieval-augmented answering service.

## Environment Setup

1. Create a virtual environment (`python -m venv .venv`) and activate it.
2. Install dependencies: `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and populate `OPENAI_API_KEY` if you plan to hit the live API.

Common helper targets:

```bash
make install      # install dependencies
make eval         # run harness with the mock client
make run-api      # start FastAPI with reload
make smoke        # smoke test against a running API instance
```

## Question 1 – Prompt Reliability Mini-Harness

- **Test set** lives in `eval/tests.json` with 10 cases (5 synthetic + 5 redacted real-world edge cases). Synthetic items include explicit generation rules in `generation_rule`, covering account recovery, compliance export, usage guardrails, and billing scenarios.
- **Invariance test:** `syn_02_reset_password_mobile_invariance` paraphrases `syn_01` with typos to ensure the response remains stable.
- **Perturbation test:** `syn_05_billing_perturbation` injects a distractor clause while expecting the budget guardrail answer to stay precise.
- **Harness:** `eval/eval_harness.py` loads the suite, calls into `eval/openai_client.py`, and scores responses with regex/exact/substring checks. It reports accuracy plus invariance and perturbation consistency rates. Use `--mock` for deterministic output or supply a real API key.
- **Automation:** `scripts/run_eval.sh` wraps the harness for CI or local runs, and `make eval` defaults to the mock client.
- **Ship vs. later:**
  - Ship now: curate the mixed test set, land the harness + reporting, wire it into CI/nightly runs.
  - Later: add semantic scoring (e.g. embeddings), persist trend dashboards, and auto-diff regression explanations for failing variants.

## Question 2 – Retrieval + Guardrails FastAPI Prototype

- **Corpus:** `rag_api/corpus.py` defines 12 purpose-written snippets spanning security, billing, compliance, and integrations. `CorpusIndex` builds a TF-IDF matrix with support for cosine or raw dot-product similarity.
- **Endpoint:** `POST /answer` (implemented in `rag_api/app.py`) accepts `query`, optional `k`, and `similarity`. It returns the naive answer plus top-k snippets.
- **Guardrail:** `rag_api/guardrails.py` enforces a denylist for credential/PII-seeking queries, short-circuiting the answer. This is practical because it blocks high-risk requests without adding latency and can be extended with analytics.
- **Index comparison:** Switch `similarity` between `cosine` and `dot`. Cosine is the default; it normalises vectors and produced tighter rankings on mixed-length snippets, while dot-product is exposed for experimentation when raw term frequency matters (see response `index_config`).
- **Monitoring metrics:**
  1. **P95 end-to-end latency** – capture via FastAPI middleware logging into e.g. OpenTelemetry.
  2. **Retrieval hit-rate** – percentage of queries whose top score exceeds a relevance threshold; log alongside query metadata to surface drift.
- **Smoke test:** once the API is running (`make run-api`), execute `scripts/smoke_api.sh` to hit `/health` and `/answer`.

## Docker & Compose

- Build the image: `docker build -t ai-eval-rag .`
- Run via compose: `docker compose up --build` to expose the service on `localhost:8000`.

## Next Steps

- Fill in production telemetry (latency + hit-rate collectors) and push eval reports to your preferred observability stack.
- Extend the harness with semantic grading and alerting once you have baseline stability.
- Capture decision notes and metrics snapshots in the companion Google Doc for submission.