PYTHON ?= py -3.12
PIP := $(PYTHON) -m pip
CONFIG ?= configs/adaptive_bridge/gemma2_first_milestone.yaml
DEBUG_CONFIG ?= configs/adaptive_bridge/debug_tiny.yaml

.PHONY: install test adaptive_smoke adaptive_train adaptive_eval format

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

adaptive_smoke:
	powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_debug_smoke.ps1 -Config "$(DEBUG_CONFIG)"

adaptive_train:
	powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_train.ps1 -Config "$(CONFIG)"

adaptive_eval:
	powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_eval.ps1 -Config "$(CONFIG)"

format:
	$(PYTHON) -m compileall src tests
