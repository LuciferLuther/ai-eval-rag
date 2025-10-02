.PHONY: install eval run-api smoke docker-build docker-up

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

eval:
	python eval/eval_harness.py --tests eval/tests.json --mock

run-api:
	uvicorn rag_api.app:app --host 0.0.0.0 --port 8000 --reload

smoke:
	bash scripts/smoke_api.sh

docker-build:
	docker build -t ai-eval-rag .

docker-up:
	docker compose up --build