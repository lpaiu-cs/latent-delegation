PYTHON ?= py -3.12
PIP := $(PYTHON) -m pip
CONFIG ?= configs/adaptive_bridge/gemma2_first_milestone.yaml
DEBUG_CONFIG ?= configs/adaptive_bridge/debug_tiny.yaml
REAL_TRAIN_DIR ?= outputs/adaptive_bridge/real_seed42_warm_start/train
REAL_EVAL_DIR ?= outputs/adaptive_bridge/real_seed42_warm_start/eval
THREE_SEED_CONFIG ?= configs/adaptive_bridge/gemma2_three_seed_replication.yaml
THREE_SEED_TRAIN_DIR ?= outputs/adaptive_bridge/real_seed42_43_44_warm_start/train
THREE_SEED_EVAL_DIR ?= outputs/adaptive_bridge/real_seed42_43_44_warm_start/eval
HARDENED_EVAL_JSON ?= outputs/adaptive_bridge/real_seed42_43_44_warm_start/eval/paired_uncertainty.json
ROUTE_ABLATION_DIR ?= outputs/adaptive_bridge/route_ablation
ROUTE_ABLATION_JSON ?= outputs/adaptive_bridge/route_ablation/results.json

.PHONY: install test adaptive_smoke adaptive_train adaptive_real_seed42 adaptive_eval adaptive_real_eval adaptive_three_seed_train adaptive_three_seed_eval adaptive_eval_hardening adaptive_route_ablation format

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

adaptive_smoke:
	powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_debug_smoke.ps1 -Config "$(DEBUG_CONFIG)"

adaptive_train:
	powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_train.ps1 -Config "$(CONFIG)"

adaptive_real_seed42:
	powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_real_warm_start_seed42.ps1 -Config "$(CONFIG)"

adaptive_eval:
	powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_eval.ps1 -Config "$(CONFIG)"

adaptive_real_eval:
	powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_eval.ps1 -Config "$(CONFIG)" -TrainDir "$(REAL_TRAIN_DIR)" -OutputDir "$(REAL_EVAL_DIR)"

adaptive_three_seed_train:
	powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_real_three_seed_train.ps1 -Config "$(THREE_SEED_CONFIG)" -OutputDir "$(THREE_SEED_TRAIN_DIR)"

adaptive_three_seed_eval:
	powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_real_three_seed_eval.ps1 -Config "$(THREE_SEED_CONFIG)" -TrainDir "$(THREE_SEED_TRAIN_DIR)" -OutputDir "$(THREE_SEED_EVAL_DIR)"

adaptive_eval_hardening:
	powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_eval_hardening.ps1 -Config "$(THREE_SEED_CONFIG)" -TrainDir "$(THREE_SEED_TRAIN_DIR)" -EvalDir "$(THREE_SEED_EVAL_DIR)" -OutputPath "$(HARDENED_EVAL_JSON)"

adaptive_route_ablation:
	powershell -ExecutionPolicy Bypass -File .\scripts\adaptive_bridge\run_route_ablation.ps1 -Config "$(THREE_SEED_CONFIG)" -TrainDir "$(THREE_SEED_TRAIN_DIR)" -OutputDir "$(ROUTE_ABLATION_DIR)" -ResultsPath "$(ROUTE_ABLATION_JSON)"

format:
	$(PYTHON) -m compileall src tests
