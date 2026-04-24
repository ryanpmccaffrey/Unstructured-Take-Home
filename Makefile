install:
	uv sync

format:
	uv run ruff format

ingest:
	uv run python -m scripts.ingest

eval:
	uv run python -m scripts.run_eval
