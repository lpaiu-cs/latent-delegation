"""I/O and config helpers."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Model-loading settings."""

    family: str
    large_model_name: str
    small_model_name: str
    tokenizer_name: str
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: str
    torch_dtype: str
    freeze_backbones: bool
    gradient_checkpointing: bool
    trust_remote_code: bool
    allow_qwen_fallback: bool
    debug_random_init: bool
    debug_large_hidden_size: int | None = None
    debug_large_intermediate_size: int | None = None
    debug_large_num_attention_heads: int | None = None
    debug_large_num_key_value_heads: int | None = None
    debug_small_hidden_size: int | None = None
    debug_small_intermediate_size: int | None = None
    debug_small_num_attention_heads: int | None = None
    debug_small_num_key_value_heads: int | None = None
    debug_vocab_size: int | None = None
    debug_max_position_embeddings: int | None = None


@dataclass
class SplitConfig:
    """Layer boundary settings."""

    large_prefix_end: int
    large_removed_start: int
    large_removed_end: int
    large_suffix_start: int
    small_entry_target_layer: int
    small_delegate_start: int
    small_delegate_end: int


@dataclass
class AdapterConfig:
    """Adapter hyperparameters."""

    entry_projector: str
    return_adapter_rank: int
    bridge_rank: int
    gate_init: float
    use_rmsnorm_after_entry: bool
    rms_norm_eps: float


@dataclass
class StageTrainConfig:
    """Per-stage optimization settings."""

    max_steps: int
    save_every: int
    kl_weight: float | None = None
    ce_weight: float | None = None
    delta_reg_weight: float | None = None
    train_entry_projector: bool = False
    entry_lr: float | None = None
    return_lr: float | None = None
    gate_lr: float | None = None


@dataclass
class TrainingConfig:
    """Shared training settings."""

    seed: int
    seq_len: int
    micro_batch_size: int
    grad_accum_steps: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    log_every: int
    stage_a: StageTrainConfig
    stage_b: StageTrainConfig
    stage_c: StageTrainConfig


@dataclass
class DataConfig:
    """Data settings."""

    use_synthetic_data: bool
    cache_dir: str
    train_wikitext_examples: int
    train_gsm8k_examples: int
    val_wikitext_examples: int
    gsm8k_eval_examples: int
    strategyqa_eval_examples: int
    synthetic_text_repeats: int


@dataclass
class EvalConfig:
    """Evaluation settings."""

    max_new_tokens: int
    gsm8k_examples: int
    strategyqa_examples: int
    speed_prompt_tokens: int
    speed_decode_tokens: int


@dataclass
class ExperimentMeta:
    """Top-level experiment metadata."""

    name: str
    output_root: str


@dataclass
class ExperimentConfig:
    """Repository-wide configuration."""

    experiment: ExperimentMeta
    model: ModelConfig
    split: SplitConfig
    adapters: AdapterConfig
    training: TrainingConfig
    data: DataConfig
    eval: EvalConfig
    raw: dict[str, Any]


def _stage_config_from_dict(values: dict[str, Any]) -> StageTrainConfig:
    return StageTrainConfig(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML config into typed dataclasses."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    training = raw["training"]
    training_cfg = TrainingConfig(
        seed=training["seed"],
        seq_len=training["seq_len"],
        micro_batch_size=training["micro_batch_size"],
        grad_accum_steps=training["grad_accum_steps"],
        num_workers=training["num_workers"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        max_grad_norm=training["max_grad_norm"],
        log_every=training["log_every"],
        stage_a=_stage_config_from_dict(training["stage_a"]),
        stage_b=_stage_config_from_dict(training["stage_b"]),
        stage_c=_stage_config_from_dict(training["stage_c"]),
    )

    config = ExperimentConfig(
        experiment=ExperimentMeta(**raw["experiment"]),
        model=ModelConfig(**raw["model"]),
        split=SplitConfig(**raw["split"]),
        adapters=AdapterConfig(**raw["adapters"]),
        training=training_cfg,
        data=DataConfig(**raw["data"]),
        eval=EvalConfig(**raw["eval"]),
        raw=raw,
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    """Validate split boundaries and high-level constraints."""

    split = config.split
    assert split.large_prefix_end + 1 == split.large_removed_start
    assert split.large_removed_end + 1 == split.large_suffix_start
    assert split.large_prefix_end >= 0
    assert split.large_removed_start <= split.large_removed_end
    assert split.small_entry_target_layer + 1 == split.small_delegate_start
    assert split.small_delegate_start <= split.small_delegate_end
    if config.model.family != "gemma2":
        raise ValueError(f"Unsupported model family: {config.model.family}")
    if config.model.allow_qwen_fallback:
        raise ValueError("Qwen fallback is not implemented in v1 and must remain disabled by default.")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""

    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def timestamp() -> str:
    """Return a filesystem-friendly UTC timestamp."""

    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def create_run_dir(config: ExperimentConfig, stage_name: str) -> Path:
    """Create and return a run directory for a stage."""

    root = ensure_dir(Path(config.experiment.output_root) / config.experiment.name / stage_name / timestamp())
    return root


def save_json(path: str | Path, payload: Any) -> None:
    """Write JSON to disk."""

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_text(path: str | Path, content: str) -> None:
    """Write a UTF-8 text file."""

    Path(path).write_text(content, encoding="utf-8")


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to CSV."""

    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_config_snapshot(path: str | Path, config: ExperimentConfig) -> None:
    """Write the original YAML config payload."""

    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.raw, handle, sort_keys=False)


def git_commit_hash() -> str | None:
    """Return the git commit hash if the workspace is inside a repository."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def export_run_metadata(path: str | Path, config: ExperimentConfig, extra: dict[str, Any] | None = None) -> None:
    """Save run metadata including git hash when available."""

    payload = {
        "experiment_name": config.experiment.name,
        "git_commit_hash": git_commit_hash(),
        "raw_config": asdict(config.experiment),
    }
    if extra:
        payload.update(extra)
    save_json(path, payload)
