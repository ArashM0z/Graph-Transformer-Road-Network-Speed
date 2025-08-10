.PHONY: install dev lint test smoke train docker clean

install: ; pip install -e .
dev: ; pip install -e ".[dev]"
lint: ; ruff check src tests
test: ; pytest --cov=graph_tt --cov-report=term-missing
smoke: ; WANDB_MODE=disabled gtt-train --config configs/smoke.yaml --demo
train: ; gtt-train --config configs/default.yaml
docker: ; docker build -t graph-tt:latest .
clean: ; rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache mlruns wandb
