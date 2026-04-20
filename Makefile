PYTHON ?= python3.11
PIP := $(PYTHON) -m pip
CONFIG ?= configs/gemma2_conservative.yaml

.PHONY: install test smoke stage_a stage_b stage_c eval format

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

smoke:
	PYTHON_BIN=$(PYTHON) ./scripts/smoke_test.sh

stage_a:
	./scripts/run_stage_a.sh $(CONFIG)

stage_b:
	./scripts/run_stage_b.sh $(CONFIG)

stage_c:
	./scripts/run_stage_c.sh $(CONFIG)

eval:
	./scripts/run_eval_all.sh $(CONFIG)

format:
	$(PYTHON) -m compileall src tests
