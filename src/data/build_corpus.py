"""Minimal dataset building and tokenization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from src.utils.io import ExperimentConfig, ensure_dir


SYNTHETIC_TEXTS = [
    "The capital of France is Paris and the Seine river crosses the city center.",
    "Question: If Alice has 3 apples and buys 2 more, how many apples does she have? Answer: 5.",
    "Question: Is the sky blue during a clear day? Answer: yes.",
    "A transformer residual stream can be projected into another latent space with a learned affine map.",
    "Question: What is 12 plus 7? Answer: 19.",
    "Question: Can penguins usually fly? Answer: no.",
]


@dataclass
class CorpusBundle:
    """Tokenized tensors and sampling metadata."""

    dataset: Dataset
    sample_metadata: list[dict[str, Any]]


class TokenSequenceDataset(Dataset):
    """Simple in-memory token sequence dataset."""

    def __init__(self, examples: list[dict[str, torch.Tensor]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        return {
            "input_ids": example["input_ids"].clone(),
            "attention_mask": example["attention_mask"].clone(),
        }


def _sample_indices(total: int, sample_count: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total, generator=generator).tolist()
    return permutation[: min(total, sample_count)]


def _format_gsm8k_record(record: dict[str, Any]) -> str:
    return f"Question: {record['question'].strip()}\nAnswer: {record['answer'].strip()}"


def _format_strategyqa_record(record: dict[str, Any]) -> str:
    answer_value = record.get("answer", record.get("label"))
    answer = "yes" if bool(answer_value) else "no"
    return f"Question: {record['question'].strip()}\nAnswer: {answer}"


def _load_training_texts(config: ExperimentConfig, stage_name: str) -> tuple[list[str], list[dict[str, Any]]]:
    if config.data.use_synthetic_data:
        repeats = max(1, config.data.synthetic_text_repeats)
        texts = SYNTHETIC_TEXTS * repeats
        metadata = [{"dataset": "synthetic", "id": index} for index in range(len(texts))]
        return texts, metadata

    texts: list[str] = []
    metadata: list[dict[str, Any]] = []

    wikitext = load_dataset("wikitext", "wikitext-103-v1", split="train")
    valid_text_indices = [idx for idx, row in enumerate(wikitext) if row["text"].strip()]
    selected_wiki = _sample_indices(len(valid_text_indices), config.data.train_wikitext_examples, config.training.seed)
    for offset in selected_wiki:
        actual_idx = valid_text_indices[offset]
        texts.append(wikitext[actual_idx]["text"].strip())
        metadata.append({"dataset": "wikitext", "id": actual_idx})

    gsm8k = load_dataset("gsm8k", "main", split="train")
    selected_gsm = _sample_indices(len(gsm8k), config.data.train_gsm8k_examples, config.training.seed + 17)
    for idx in selected_gsm:
        texts.append(_format_gsm8k_record(gsm8k[idx]))
        metadata.append({"dataset": "gsm8k_train", "id": idx})

    if stage_name == "stage_a":
        cutoff = max(1, math.ceil(len(texts) * 0.75))
        texts = texts[:cutoff]
        metadata = metadata[:cutoff]
    return texts, metadata


def _load_eval_texts(config: ExperimentConfig) -> tuple[list[str], list[dict[str, Any]]]:
    if config.data.use_synthetic_data:
        texts = SYNTHETIC_TEXTS[: config.data.val_wikitext_examples]
        metadata = [{"dataset": "synthetic_eval", "id": index} for index in range(len(texts))]
        return texts, metadata

    wikitext = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    valid_text_indices = [idx for idx, row in enumerate(wikitext) if row["text"].strip()]
    selected = _sample_indices(len(valid_text_indices), config.data.val_wikitext_examples, config.training.seed + 101)
    texts = [wikitext[valid_text_indices[idx]]["text"].strip() for idx in selected]
    metadata = [{"dataset": "wikitext_validation", "id": valid_text_indices[idx]} for idx in selected]
    return texts, metadata


def build_eval_examples(task_name: str, config: ExperimentConfig) -> list[dict[str, Any]]:
    """Build raw evaluation examples for GSM8K or StrategyQA."""

    if config.data.use_synthetic_data:
        if task_name == "gsm8k":
            return [
                {"id": 0, "question": "What is 2 + 3?", "answer": "5"},
                {"id": 1, "question": "What is 7 - 4?", "answer": "3"},
            ][: config.eval.gsm8k_examples]
        if task_name == "strategyqa":
            return [
                {"id": 0, "question": "Is water wet?", "answer": True},
                {"id": 1, "question": "Can humans breathe underwater without equipment?", "answer": False},
            ][: config.eval.strategyqa_examples]
        raise ValueError(f"Unsupported task: {task_name}")

    if task_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
        selected = _sample_indices(len(dataset), config.eval.gsm8k_examples, config.training.seed + 201)
        return [{"id": idx, "question": dataset[idx]["question"], "answer": dataset[idx]["answer"]} for idx in selected]
    if task_name == "strategyqa":
        candidates: list[tuple[str, str | None, str]] = [
            ("tasksource/strategy-qa", None, "validation"),
            ("SelfCorrect/strategyqa", None, "train"),
            ("njf/StrategyQA", None, "train"),
        ]
        last_error: Exception | None = None
        for dataset_name, config_name, split_name in candidates:
            try:
                dataset = load_dataset(dataset_name, config_name, split=split_name)
                selected = _sample_indices(len(dataset), config.eval.strategyqa_examples, config.training.seed + 301)
                examples: list[dict[str, Any]] = []
                for idx in selected:
                    answer_value = dataset[idx].get("answer", dataset[idx].get("label"))
                    examples.append(
                        {
                            "id": idx,
                            "question": dataset[idx]["question"],
                            "answer": answer_value,
                            "dataset_name": dataset_name,
                            "source_split": split_name,
                        }
                    )
                return examples
            except Exception as exc:
                last_error = exc
        raise RuntimeError("Unable to load a StrategyQA dataset candidate.") from last_error
    raise ValueError(f"Unsupported task: {task_name}")


def _cache_path(config: ExperimentConfig, split_name: str, stage_name: str) -> Path:
    root = ensure_dir(Path(config.data.cache_dir))
    return root / f"{config.experiment.name}_{stage_name}_{split_name}_len{config.training.seq_len}_seed{config.training.seed}.pt"


def _tokenize_texts(tokenizer: Any, texts: list[str], seq_len: int) -> list[dict[str, torch.Tensor]]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    return [
        {
            "input_ids": encoded["input_ids"][index],
            "attention_mask": encoded["attention_mask"][index],
        }
        for index in range(encoded["input_ids"].shape[0])
    ]


def build_corpus_bundle(
    config: ExperimentConfig,
    tokenizer: Any,
    stage_name: str,
    split_name: str,
) -> CorpusBundle:
    """Build or load a tokenized corpus bundle."""

    cache_path = _cache_path(config, split_name, stage_name)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        dataset = TokenSequenceDataset(payload["examples"])
        return CorpusBundle(dataset=dataset, sample_metadata=payload["sample_metadata"])

    if split_name == "train":
        texts, sample_metadata = _load_training_texts(config, stage_name)
    elif split_name == "validation":
        texts, sample_metadata = _load_eval_texts(config)
    else:
        raise ValueError(f"Unsupported split: {split_name}")

    examples = _tokenize_texts(tokenizer, texts, config.training.seq_len)
    payload = {"examples": examples, "sample_metadata": sample_metadata}
    torch.save(payload, cache_path)
    dataset = TokenSequenceDataset(examples)
    return CorpusBundle(dataset=dataset, sample_metadata=sample_metadata)
