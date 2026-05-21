.PHONY: run dev test lint clean

run:
	uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

dev:
	pip install -e ".[dev]"
	ruff check src/ tests/

test:
	pytest -v

lint:
	ruff check src/ tests/
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache dist *.egg-info
