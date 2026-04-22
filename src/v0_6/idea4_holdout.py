"""Holdout-slice helpers for Idea 4 output probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
from torch.utils.data import DataLoader

from src.data.build_corpus import TokenSequenceDataset, _sample_indices, build_corpus_bundle
from src.data.collators import CausalLMCollator
from src.utils.io import ExperimentConfig


@dataclass
class ProbeHoldoutSlice:
    """One concrete output-probe slice plus its definition."""

    dataloader: DataLoader
    sample_metadata: list[dict[str, Any]]
    heldout_policy: str
    slice_definition: dict[str, Any]


def _fresh_holdout_settings(config: ExperimentConfig) -> dict[str, Any]:
    idea4 = config.raw.get("idea4", {})
    fresh = dict(idea4.get("fresh_holdout", {}))
    return {
        "dataset_name": fresh.get("dataset_name", "wikitext"),
        "dataset_config_name": fresh.get("dataset_config_name", "wikitext-103-v1"),
        "split": fresh.get("split", "test"),
        "sample_count": int(fresh.get("sample_count", config.data.val_wikitext_examples)),
        "sampling_seed": int(fresh.get("sampling_seed", 7606)),
        "nonempty_only": bool(fresh.get("nonempty_only", True)),
        "untouched_justification": str(
            fresh.get(
                "untouched_justification",
                "Prior v0.5.x and v0_6 output probes used seed-matched wikitext validation slices for model selection; this confirmation slice is sampled from the untouched wikitext test split.",
            )
        ),
    }


def build_probe_holdout_slice(
    config: ExperimentConfig,
    tokenizer: Any,
    *,
    holdout_policy: str,
    seed: int,
) -> ProbeHoldoutSlice:
    """Build the requested output-probe slice."""

    if holdout_policy == "main_validation":
        corpus = build_corpus_bundle(config, tokenizer, stage_name="stage_b", split_name="validation")
        dataloader = DataLoader(
            corpus.dataset,
            batch_size=config.training.micro_batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            collate_fn=CausalLMCollator(),
        )
        return ProbeHoldoutSlice(
            dataloader=dataloader,
            sample_metadata=corpus.sample_metadata,
            heldout_policy="stage_b validation split reused for the matching seed",
            slice_definition={
                "policy": "main_validation",
                "dataset_name": "wikitext",
                "dataset_config_name": "wikitext-103-v1",
                "split": "validation",
                "sample_count": len(corpus.sample_metadata),
                "sampling_seed": config.training.seed + 101,
            },
        )

    if holdout_policy != "fresh_untouched":
        raise ValueError(f"Unsupported holdout policy: {holdout_policy}")

    settings = _fresh_holdout_settings(config)
    dataset = load_dataset(
        settings["dataset_name"],
        settings["dataset_config_name"],
        split=settings["split"],
    )
    if settings["nonempty_only"]:
        valid_text_indices = [idx for idx, row in enumerate(dataset) if str(row["text"]).strip()]
    else:
        valid_text_indices = list(range(len(dataset)))
    selected = _sample_indices(len(valid_text_indices), settings["sample_count"], settings["sampling_seed"])
    sample_metadata = [
        {
            "dataset": f"{settings['dataset_name']}_{settings['split']}",
            "id": valid_text_indices[index],
        }
        for index in selected
    ]
    texts = [str(dataset[valid_text_indices[index]]["text"]).strip() for index in selected]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.training.seq_len,
    )
    examples = [
        {
            "input_ids": encoded["input_ids"][index],
            "attention_mask": encoded["attention_mask"][index],
        }
        for index in range(encoded["input_ids"].shape[0])
    ]
    dataloader = DataLoader(
        TokenSequenceDataset(examples),
        batch_size=config.training.micro_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=CausalLMCollator(),
    )
    return ProbeHoldoutSlice(
        dataloader=dataloader,
        sample_metadata=sample_metadata,
        heldout_policy="fresh untouched wikitext test slice shared across seeds",
        slice_definition={
            "policy": "fresh_untouched",
            "dataset_name": settings["dataset_name"],
            "dataset_config_name": settings["dataset_config_name"],
            "split": settings["split"],
            "sample_count": len(sample_metadata),
            "sampling_seed": settings["sampling_seed"],
            "nonempty_only": settings["nonempty_only"],
            "untouched_justification": settings["untouched_justification"],
        },
    )
