.PHONY: install test benchmark docs lint clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode with dev dependencies
	pip install -e ".[dev,app]"

test: ## Run all tests
	python -m pytest tests/ -v --tb=short

test-quick: ## Run fast tests only (no long training)
	python -m pytest tests/ -v --tb=short -k "not trained_model and not trained_heat and not trained_wave and not trained_inverse and not trained_hnn"

benchmark: ## Run quick benchmarks
	python -c "from src.benchmarks import BenchmarkRunner; r = BenchmarkRunner('quick'); r.run_all(verbose=True); print(r.generate_markdown_table())"

benchmark-full: ## Run full benchmarks (slow)
	python -c "from src.benchmarks import BenchmarkRunner; r = BenchmarkRunner('full'); r.run_all(verbose=True); print(r.generate_markdown_table())"

app: ## Launch Streamlit web app
	streamlit run src/app.py

lint: ## Run linter (ruff)
	python -m ruff check src/ tests/ examples/

format: ## Auto-format code (ruff)
	python -m ruff format src/ tests/ examples/

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
