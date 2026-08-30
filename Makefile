PYTHON ?= python3.12
DEMO_ROOT ?= artifacts/demo

.PHONY: install install-llm api normal test lint demo demo-data demo-cases prepare-data train-models reference-profile behavior-history relationship-history frontend-install frontend-dev

install:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install -e ".[dev,ml]"

install-llm:
	.venv/bin/python -m pip install -e ".[dev,ml,llm]"

api:
	.venv/bin/uvicorn backend.app.main:app --reload --port 8000

normal:
	env -u FRAUDETECT_CASE_DATABASE \
		-u FRAUDETECT_BEHAVIORAL_HISTORY_DB \
		-u FRAUDETECT_RELATIONSHIP_HISTORY_DB \
		.venv/bin/uvicorn backend.app.main:app --reload --port 8000

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check backend ml scripts tests

demo-data:
	.venv/bin/python scripts/generate_demo_data.py

demo-cases:
	.venv/bin/python scripts/seed_demo_cases.py --output-dir "$(DEMO_ROOT)"

demo:
	@test -f "$(DEMO_ROOT)/cases.sqlite" \
		-a -f "$(DEMO_ROOT)/behavior.sqlite" \
		-a -f "$(DEMO_ROOT)/relationship.sqlite" \
		|| { echo "Showcase databases are unavailable; run 'make demo-cases' first."; exit 1; }
	FRAUDETECT_CASE_DATABASE="$(DEMO_ROOT)/cases.sqlite" \
	FRAUDETECT_BEHAVIORAL_HISTORY_DB="$(DEMO_ROOT)/behavior.sqlite" \
	FRAUDETECT_RELATIONSHIP_HISTORY_DB="$(DEMO_ROOT)/relationship.sqlite" \
		.venv/bin/uvicorn backend.app.main:app --reload --port 8000

prepare-data:
	.venv/bin/python scripts/prepare_data.py --input data/raw/paysim.csv

train-models:
	.venv/bin/python scripts/train_models.py

reference-profile:
	.venv/bin/python scripts/build_reference_profile.py

behavior-history:
	.venv/bin/python scripts/build_behavior_history.py

relationship-history:
	.venv/bin/python scripts/build_relationship_history.py

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev
