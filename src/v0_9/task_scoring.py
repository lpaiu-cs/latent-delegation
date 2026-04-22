"""Task formatting and log-likelihood scoring helpers for v0.9."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset

from src.v0_9.common import LMTaskSpec, MultichoiceTaskSpec


WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TaskExample:
    """One formatted multiple-choice example."""

    task_name: str
    example_id: str
    prompt: str
    choices: list[str]
    label_index: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LMExample:
    """One formatted LM evaluation example."""

    task_name: str
    example_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ChoiceBatch:
    """Batched option tensors for one multichoice example."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    continuation_mask: torch.Tensor
    continuation_token_counts: torch.Tensor
    truncated_flags: list[bool]


def sample_indices(total: int, sample_count: int, seed: int) -> list[int]:
    """Sample deterministic dataset indices."""

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total, generator=generator).tolist()
    return permutation[: min(total, sample_count)]


def _clean_text(text: str) -> str:
    text = text.replace("[title]", " ")
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" ?", "?")
    text = text.replace(" !", "!")
    text = text.replace(" n't", "n't")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _choice_with_leading_space(text: str) -> str:
    text = _clean_text(text)
    if not text:
        return text
    return text if text[0] in {".", ",", ";", ":", "?", "!"} else f" {text}"


def _hellaswag_example(record: dict[str, Any], index: int) -> TaskExample:
    prompt = _clean_text(record["ctx"])
    choices = [_choice_with_leading_space(ending) for ending in record["endings"]]
    return TaskExample(
        task_name="hellaswag",
        example_id=str(record.get("ind", index)),
        prompt=prompt,
        choices=choices,
        label_index=int(record["label"]),
        metadata={
            "dataset_index": index,
            "source_id": record.get("source_id"),
            "activity_label": record.get("activity_label"),
        },
    )


def _piqa_example(record: dict[str, Any], index: int) -> TaskExample:
    prompt = f"Goal: {_clean_text(record['goal'])}\nSolution:"
    choices = [_choice_with_leading_space(record["sol1"]), _choice_with_leading_space(record["sol2"])]
    return TaskExample(
        task_name="piqa",
        example_id=str(index),
        prompt=prompt,
        choices=choices,
        label_index=int(record["label"]),
        metadata={"dataset_index": index},
    )


def _winogrande_example(record: dict[str, Any], index: int) -> TaskExample:
    sentence = str(record["sentence"])
    if "_" not in sentence:
        raise ValueError("WinoGrande sentence is missing the blank marker.")
    prefix, suffix = sentence.split("_", maxsplit=1)
    prompt = prefix
    choices = [f"{record['option1']}{suffix}", f"{record['option2']}{suffix}"]
    return TaskExample(
        task_name="winogrande",
        example_id=str(index),
        prompt=prompt,
        choices=choices,
        label_index=int(record["answer"]) - 1,
        metadata={"dataset_index": index, "sentence": sentence},
    )


def _arc_example(task_name: str, record: dict[str, Any], index: int) -> TaskExample:
    choice_labels = list(record["choices"]["label"])
    choice_texts = [_choice_with_leading_space(text) for text in record["choices"]["text"]]
    answer_key = str(record["answerKey"])
    if answer_key not in choice_labels:
        raise ValueError(f"ARC answer key {answer_key} not found in choices for index {index}.")
    prompt = f"Question: {_clean_text(record['question'])}\nAnswer:"
    return TaskExample(
        task_name=task_name,
        example_id=str(record.get("id", index)),
        prompt=prompt,
        choices=choice_texts,
        label_index=choice_labels.index(answer_key),
        metadata={"dataset_index": index, "choice_labels": choice_labels},
    )


def _ptb_example(record: dict[str, Any], index: int) -> LMExample:
    return LMExample(
        task_name="ptb_test",
        example_id=str(index),
        text=_clean_text(record["sentence"]),
        metadata={"dataset_index": index},
    )


def build_multichoice_examples(spec: MultichoiceTaskSpec) -> tuple[list[TaskExample], dict[str, Any], list[dict[str, Any]]]:
    """Load and sample one multiple-choice task."""

    dataset = load_dataset(spec.dataset_name, spec.dataset_config_name, split=spec.split)
    selected = sample_indices(len(dataset), spec.sample_count, spec.sampling_seed)
    examples: list[TaskExample] = []
    sample_metadata: list[dict[str, Any]] = []
    skipped = 0
    for dataset_index in selected:
        record = dataset[dataset_index]
        try:
            if spec.name == "hellaswag":
                example = _hellaswag_example(record, dataset_index)
            elif spec.name == "piqa":
                example = _piqa_example(record, dataset_index)
            elif spec.name == "winogrande":
                example = _winogrande_example(record, dataset_index)
            elif spec.name == "arc_easy":
                example = _arc_example(spec.name, record, dataset_index)
            elif spec.name == "arc_challenge":
                example = _arc_example(spec.name, record, dataset_index)
            else:
                raise ValueError(f"Unsupported multichoice task: {spec.name}")
        except Exception:
            skipped += 1
            continue
        examples.append(example)
        sample_metadata.append(
            {
                "task_name": spec.name,
                "dataset_name": spec.dataset_name,
                "dataset_config_name": spec.dataset_config_name,
                "split": spec.split,
                "dataset_index": dataset_index,
                "example_id": example.example_id,
            }
        )
    slice_definition = {
        "task_name": spec.name,
        "dataset_name": spec.dataset_name,
        "dataset_config_name": spec.dataset_config_name,
        "split": spec.split,
        "sample_count_requested": spec.sample_count,
        "sample_count_actual": len(examples),
        "sampling_seed": spec.sampling_seed,
        "skipped_examples": skipped,
    }
    return examples, slice_definition, sample_metadata


def build_lm_examples(spec: LMTaskSpec) -> tuple[list[LMExample], dict[str, Any], list[dict[str, Any]]]:
    """Load and sample one LM-style benchmark slice."""

    dataset = load_dataset(spec.dataset_name, spec.dataset_config_name, split=spec.split)
    selected = sample_indices(len(dataset), spec.sample_count, spec.sampling_seed)
    examples: list[LMExample] = []
    sample_metadata: list[dict[str, Any]] = []
    for dataset_index in selected:
        record = dataset[dataset_index]
        if spec.name == "lambada_openai":
            text = _clean_text(record["text"])
        else:
            raise ValueError(f"Unsupported LM task: {spec.name}")
        if not text:
            continue
        example = LMExample(
            task_name=spec.name,
            example_id=str(dataset_index),
            text=text,
            metadata={"dataset_index": dataset_index},
        )
        examples.append(example)
        sample_metadata.append(
            {
                "task_name": spec.name,
                "dataset_name": spec.dataset_name,
                "dataset_config_name": spec.dataset_config_name,
                "split": spec.split,
                "dataset_index": dataset_index,
                "example_id": example.example_id,
            }
        )
    slice_definition = {
        "task_name": spec.name,
        "dataset_name": spec.dataset_name,
        "dataset_config_name": spec.dataset_config_name,
        "split": spec.split,
        "sample_count_requested": spec.sample_count,
        "sample_count_actual": len(examples),
        "sampling_seed": spec.sampling_seed,
    }
    return examples, slice_definition, sample_metadata


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )
    return encoded["input_ids"][0].tolist()


def build_choice_batch(
    tokenizer: Any,
    prompt: str,
    choices: list[str],
    *,
    max_seq_len: int,
    device: torch.device,
) -> ChoiceBatch:
    """Build one batched multiple-choice scoring input."""

    bos = [tokenizer.bos_token_id] if getattr(tokenizer, "bos_token_id", None) is not None else []
    prompt_ids = _encode_text(tokenizer, prompt)

    sequences: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    continuation_masks: list[torch.Tensor] = []
    token_counts: list[int] = []
    truncated_flags: list[bool] = []
    pad_token_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else 0
    for choice in choices:
        choice_ids = _encode_text(tokenizer, choice)
        if not choice_ids:
            raise ValueError("Multiple-choice continuation may not be empty.")
        body_budget = max_seq_len - len(bos)
        truncated = False
        if len(choice_ids) > body_budget:
            choice_ids = choice_ids[:body_budget]
            truncated = True
        prompt_budget = max(0, body_budget - len(choice_ids))
        if len(prompt_ids) > prompt_budget:
            prompt_ids_trimmed = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
            truncated = True
        else:
            prompt_ids_trimmed = list(prompt_ids)
        input_ids = bos + prompt_ids_trimmed + choice_ids
        continuation_start = len(bos) + len(prompt_ids_trimmed)
        continuation_mask = [0] * continuation_start + [1] * len(choice_ids)
        sequences.append(torch.tensor(input_ids, dtype=torch.long))
        attention_masks.append(torch.ones(len(input_ids), dtype=torch.long))
        continuation_masks.append(torch.tensor(continuation_mask, dtype=torch.long))
        token_counts.append(len(choice_ids))
        truncated_flags.append(truncated)

    max_length = max(sequence.shape[0] for sequence in sequences)
    padded_ids = []
    padded_attention = []
    padded_continuation = []
    for input_ids, attention_mask, continuation_mask in zip(sequences, attention_masks, continuation_masks, strict=True):
        pad = max_length - input_ids.shape[0]
        if pad > 0:
            input_ids = torch.cat([input_ids, torch.full((pad,), pad_token_id, dtype=torch.long)], dim=0)
            attention_mask = torch.cat([attention_mask, torch.zeros(pad, dtype=torch.long)], dim=0)
            continuation_mask = torch.cat([continuation_mask, torch.zeros(pad, dtype=torch.long)], dim=0)
        padded_ids.append(input_ids)
        padded_attention.append(attention_mask)
        padded_continuation.append(continuation_mask)

    return ChoiceBatch(
        input_ids=torch.stack(padded_ids, dim=0).to(device),
        attention_mask=torch.stack(padded_attention, dim=0).to(device),
        continuation_mask=torch.stack(padded_continuation, dim=0).to(device),
        continuation_token_counts=torch.tensor(token_counts, dtype=torch.long, device=device),
        truncated_flags=truncated_flags,
    )


def continuation_logprob_summaries(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    continuation_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return continuation token log-prob sums and means for a batched choice input."""

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    target_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    continuation_target_mask = continuation_mask[:, 1:].to(target_log_probs.dtype)
    token_counts = continuation_target_mask.sum(dim=1).clamp_min(1.0)
    summed = (target_log_probs * continuation_target_mask).sum(dim=1)
    averaged = summed / token_counts
    return {
        "sum_logprob": summed,
        "avg_logprob": averaged,
        "token_count": token_counts,
    }


def score_multichoice_example(
    model: torch.nn.Module,
    tokenizer: Any,
    example: TaskExample,
    *,
    max_seq_len: int,
    device: torch.device,
    length_normalize: bool,
) -> dict[str, Any]:
    """Score one multiple-choice example by conditional log-likelihood."""

    batch = build_choice_batch(
        tokenizer,
        example.prompt,
        example.choices,
        max_seq_len=max_seq_len,
        device=device,
    )
    outputs = model(batch.input_ids, attention_mask=batch.attention_mask)
    choice_scores = continuation_logprob_summaries(outputs.logits, batch.input_ids, batch.continuation_mask)
    ranking_scores = choice_scores["avg_logprob"] if length_normalize else choice_scores["sum_logprob"]
    winner = int(ranking_scores.argmax(dim=0).item())
    sorted_scores = torch.sort(ranking_scores.detach().float(), descending=True).values
    runner_up = float(sorted_scores[1].cpu()) if sorted_scores.numel() > 1 else float(sorted_scores[0].cpu())
    margin = float(sorted_scores[0].cpu()) - runner_up
    return {
        "predicted_index": winner,
        "correct": winner == example.label_index,
        "score_margin": margin,
        "ranking_score_name": "avg_logprob" if length_normalize else "sum_logprob",
        "choice_sum_logprob": [float(value) for value in choice_scores["sum_logprob"].detach().cpu().tolist()],
        "choice_avg_logprob": [float(value) for value in choice_scores["avg_logprob"].detach().cpu().tolist()],
        "choice_token_count": [int(value) for value in choice_scores["token_count"].detach().cpu().tolist()],
        "truncated_flags": list(batch.truncated_flags),
    }
