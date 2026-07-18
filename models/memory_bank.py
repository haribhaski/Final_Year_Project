"""
models/memory_bank.py

Persistent latent memory bank for a GPT-2 memory-augmented language model.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

@dataclass
class MemoryState:
    slots: Tensor
    age: Tensor
    write_count: Tensor
    read_count: Tensor
    confidence: Tensor

    def detach(self) -> "MemoryState":
        return MemoryState(
            slots=self.slots.detach(),
            age=self.age.detach(),
            write_count=self.write_count.detach(),
            read_count=self.read_count.detach(),
            confidence=self.confidence.detach(),
        )

    def to(self, device: torch.device | str) -> "MemoryState":
        return MemoryState(
            slots=self.slots.to(device),
            age=self.age.to(device),
            write_count=self.write_count.to(device),
            read_count=self.read_count.to(device),
            confidence=self.confidence.to(device),
        )


class MemoryBank(nn.Module):
    """Persistent latent memory bank with stable gated updates."""

    VALID_NORMALIZATIONS = {"layernorm", "rms", "l2", "none"}

    def __init__(
        self,
        num_slots: int,
        d_model: int,
        init_std: float = 0.02,
        normalization: str = "layernorm",
        max_slot_norm: Optional[float] = None,
        trainable_initial_memory: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if num_slots <= 0:
            raise ValueError("num_slots must be greater than zero.")
        if d_model <= 0:
            raise ValueError("d_model must be greater than zero.")
        if normalization not in self.VALID_NORMALIZATIONS:
            raise ValueError(f"Unsupported normalization: {normalization}")
        if max_slot_norm is not None and max_slot_norm <= 0:
            raise ValueError("max_slot_norm must be positive.")

        self.num_slots = num_slots
        self.d_model = d_model
        self.normalization = normalization
        self.max_slot_norm = max_slot_norm
        self.eps = eps

        initial = torch.empty(num_slots, d_model)
        nn.init.normal_(initial, mean=0.0, std=init_std)
        if trainable_initial_memory:
            self.initial_slots = nn.Parameter(initial)
        else:
            self.register_buffer("initial_slots", initial)

        self.layer_norm = nn.LayerNorm(d_model) if normalization == "layernorm" else nn.Identity()

    def initialize(
        self,
        batch_size: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> MemoryState:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        slots = self.initial_slots.unsqueeze(0).expand(batch_size, -1, -1).clone()
        slots = slots.to(
            device=device if device is not None else slots.device,
            dtype=dtype if dtype is not None else slots.dtype,
        )
        slots = self.normalize_slots(slots)

        age = torch.zeros(batch_size, self.num_slots, device=slots.device, dtype=torch.long)
        write_count = torch.zeros_like(age)
        read_count = torch.zeros_like(age)
        confidence = torch.zeros(batch_size, self.num_slots, device=slots.device, dtype=slots.dtype)

        return MemoryState(slots, age, write_count, read_count, confidence)

    def forward(
        self,
        state: MemoryState,
        candidate: Tensor,
        write_gate: Tensor,
        erase_gate: Optional[Tensor] = None,
        write_mask: Optional[Tensor] = None,
        confidence: Optional[Tensor] = None,
    ) -> MemoryState:
        self._validate_state(state)
        self._validate_update_tensor(candidate, "candidate", self.d_model)
        self._validate_update_tensor(write_gate, "write_gate", 1)

        if erase_gate is not None:
            self._validate_update_tensor(erase_gate, "erase_gate", 1)
        if write_mask is not None:
            self._validate_update_tensor(write_mask, "write_mask", 1)

        if candidate.shape[:2] != (state.slots.size(0), self.num_slots):
            raise ValueError("candidate has an invalid batch or slot dimension.")
        if write_gate.shape[:2] != (state.slots.size(0), self.num_slots):
            raise ValueError("write_gate has an invalid batch or slot dimension.")

        write_gate = write_gate.clamp(0.0, 1.0)
        if write_mask is not None:
            write_gate = write_gate * write_mask.to(write_gate.dtype).clamp(0.0, 1.0)

        if erase_gate is None:
            new_slots = (1.0 - write_gate) * state.slots + write_gate * candidate
        else:
            erase_gate = erase_gate.clamp(0.0, 1.0)
            if write_mask is not None:
                erase_gate = erase_gate * write_mask.to(erase_gate.dtype).clamp(0.0, 1.0)
            new_slots = (1.0 - erase_gate) * state.slots + write_gate * candidate

        new_slots = self.normalize_slots(new_slots)
        wrote = write_gate.squeeze(-1) > 1e-4
        new_age = torch.where(wrote, torch.zeros_like(state.age), state.age + 1)
        new_write_count = state.write_count + wrote.long()

        if confidence is None:
            new_confidence = state.confidence
        else:
            if confidence.dim() == 3 and confidence.size(-1) == 1:
                confidence = confidence.squeeze(-1)
            if confidence.shape != state.confidence.shape:
                raise ValueError("confidence must have shape [B, N] or [B, N, 1].")
            new_confidence = torch.where(wrote, confidence.to(state.confidence.dtype), state.confidence)

        return MemoryState(
            slots=new_slots,
            age=new_age,
            write_count=new_write_count,
            read_count=state.read_count,
            confidence=new_confidence,
        )

    def record_reads(self, state: MemoryState, read_weights: Tensor, threshold: float = 1e-4) -> MemoryState:
        self._validate_state(state)
        if read_weights.dim() == 3 and read_weights.size(-1) == 1:
            read_weights = read_weights.squeeze(-1)
        if read_weights.shape != state.read_count.shape:
            raise ValueError("read_weights must have shape [B, N] or [B, N, 1].")

        read_events = (read_weights > threshold).long()
        return MemoryState(
            slots=state.slots,
            age=state.age,
            write_count=state.write_count,
            read_count=state.read_count + read_events,
            confidence=state.confidence,
        )

    def normalize_slots(self, slots: Tensor) -> Tensor:
        if slots.dim() != 3:
            raise ValueError("slots must have shape [B, N, D].")

        if self.normalization == "layernorm":
            slots = self.layer_norm(slots)
        elif self.normalization == "rms":
            rms = slots.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
            slots = slots / rms
        elif self.normalization == "l2":
            slots = F.normalize(slots, p=2, dim=-1, eps=self.eps)

        if self.max_slot_norm is not None:
            norms = slots.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)
            slots = slots * (self.max_slot_norm / norms).clamp(max=1.0)

        return slots

    @torch.no_grad()
    def collapse_metrics(self, state: MemoryState) -> Dict[str, Tensor]:
        self._validate_state(state)
        slots = state.slots.float()
        normalized = F.normalize(slots, p=2, dim=-1, eps=self.eps)
        gram = torch.bmm(normalized, normalized.transpose(1, 2))
        eye = torch.eye(self.num_slots, device=slots.device, dtype=slots.dtype).unsqueeze(0)
        denominator = max(slots.size(0) * self.num_slots * (self.num_slots - 1), 1)
        pairwise_cosine = (gram * (1.0 - eye)).sum() / denominator

        effective_ranks = []
        stable_ranks = []
        for sample in slots:
            s = torch.linalg.svdvals(sample)
            p = s / s.sum().clamp_min(self.eps)
            effective_ranks.append((-(p * p.clamp_min(self.eps).log()).sum()).exp())
            stable_ranks.append(s.pow(2).sum() / s.max().pow(2).clamp_min(self.eps))

        norms = slots.norm(p=2, dim=-1)
        return {
            "pairwise_cosine": pairwise_cosine,
            "effective_rank": torch.stack(effective_ranks).mean(),
            "stable_rank": torch.stack(stable_ranks).mean(),
            "slot_norm_mean": norms.mean(),
            "slot_norm_std": norms.std(unbiased=False),
            "unused_slot_fraction": (state.write_count == 0).float().mean(),
        }

    def _validate_state(self, state: MemoryState) -> None:
        if not isinstance(state, MemoryState):
            raise TypeError("state must be a MemoryState.")
        if state.slots.dim() != 3 or state.slots.shape[1:] != (self.num_slots, self.d_model):
            raise ValueError(f"state.slots must have shape [B, {self.num_slots}, {self.d_model}].")
        expected_meta = (state.slots.size(0), self.num_slots)
        for name in ("age", "write_count", "read_count", "confidence"):
            if getattr(state, name).shape != expected_meta:
                raise ValueError(f"state.{name} must have shape {expected_meta}.")

    @staticmethod
    def _validate_update_tensor(tensor: Tensor, name: str, final_dim: int) -> None:
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a tensor.")
        if tensor.dim() != 3 or tensor.size(-1) != final_dim:
            raise ValueError(f"{name} must have shape [B, N, {final_dim}].")


def _smoke_test() -> None:
    torch.manual_seed(42)
    bank = MemoryBank(num_slots=8, d_model=64, normalization="layernorm", max_slot_norm=12.0)
    state = bank.initialize(batch_size=4)
    candidate = torch.randn(4, 8, 64)
    write_gate = torch.sigmoid(torch.randn(4, 8, 1))
    erase_gate = torch.sigmoid(torch.randn(4, 8, 1)) * 0.2
    state = bank(state, candidate, write_gate, erase_gate, confidence=write_gate.squeeze(-1))
    state = bank.record_reads(state, torch.softmax(torch.randn(4, 8), dim=-1))
    metrics = bank.collapse_metrics(state)
    print("Memory shape:", tuple(state.slots.shape))
    print({k: round(float(v.item()), 4) for k, v in metrics.items()})


if __name__ == "__main__":
    _smoke_test()