.PHONY: help setup fetch test experiment analyze serve docker-build up down pull-model clean

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv and install dependencies
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

fetch:  ## Download + cache the Wikipedia corpus
	$(PY) -m rag_eval.corpus

test:  ## Run the unit test suite
	$(PY) -m pytest tests/ -q

experiment:  ## Run the full RAG + judge experiment -> outputs/results.csv
	$(PY) -m evaluation.run_experiment

analyze:  ## Fit regression + render outputs/analysis.png
	$(PY) -m evaluation.analyze

serve:  ## Run the API locally (requires Ollama running on localhost)
	$(VENV)/bin/uvicorn app.main:app --reload

docker-build:  ## Build the API image
	docker compose build

up:  ## Start API + Ollama containers
	docker compose up -d

down:  ## Stop containers
	docker compose down

pull-model:  ## Pull the LLM into the running Ollama container
	docker compose exec ollama ollama pull llama3.2:3b

clean:  ## Remove caches and generated artifacts
	rm -rf cache/* outputs/*.csv outputs/*.png
	find . -type d -name __pycache__ -exec rm -rf {} +
