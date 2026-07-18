from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


def prepare_chunk(
    chunk: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Convert one token chunk into model-ready tensors.
    """

    if chunk.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional chunk, got {chunk.shape}"
        )

    input_ids = chunk.unsqueeze(0).to(device)

    attention_mask = torch.ones_like(
        input_ids,
        dtype=torch.long,
    )

    labels = input_ids.clone()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }