.PHONY: dev lint clean chat

dev:
	pip install -e ".[dev]"
	ruff check src/

lint:
	ruff check src/
	mypy src/

chat:
	python -m src.agent.cli chat $(if $(d),-d) $(if $(s),-s)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache dist *.egg-info
