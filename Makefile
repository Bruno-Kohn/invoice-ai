.PHONY: help install train evaluate demo preprocess lint test clean all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -e ".[dev]"

preprocess: ## Run preprocessing pipeline on dataset
	python -m src.pipeline --stage preprocess --config configs/preprocessing.yaml

train: ## Train quality assessment CNN
	python -m src.quality.train --config configs/pipeline.yaml

evaluate: ## Run evaluation metrics on all experiments
	python -m src.evaluation.metrics --config configs/experiments.yaml --output results/metrics/

demo: ## Launch Streamlit demo app
	streamlit run app.py

lint: ## Run linter (ruff)
	ruff check src/ tests/
	ruff format --check src/ tests/

format: ## Auto-format code
	ruff format src/ tests/

test: ## Run tests
	pytest tests/ -v

clean: ## Remove generated files
	rm -rf results/metrics/* results/figures/* models/quality_cnn/*.pth
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: preprocess train evaluate ## Run full pipeline (preprocess → train → evaluate)
