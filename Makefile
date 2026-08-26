PYTHON ?= python3.12

.PHONY: install api test lint demo-data prepare-data train-models frontend-install frontend-dev

install:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install -e ".[dev,ml]"

api:
	.venv/bin/uvicorn backend.app.main:app --reload --port 8000

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check backend ml scripts tests

demo-data:
	.venv/bin/python scripts/generate_demo_data.py

prepare-data:
	.venv/bin/python scripts/prepare_data.py --input data/raw/paysim.csv

train-models:
	.venv/bin/python scripts/train_models.py

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev
