.PHONY: run dev lint clean ask chat

run:
	uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

dev:
	pip install -e ".[dev]"
	ruff check src/

lint:
	ruff check src/
	mypy src/

ask:
	python -m src.agent.cli ask "$(q)" $(if $(v),-v) $(if $(s),-s)

chat:
	python -m src.agent.cli chat $(if $(v),-v) $(if $(s),-s)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache dist *.egg-info
