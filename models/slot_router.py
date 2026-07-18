from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RoutingOutput:
    """
    Structured output of the slot router.

    Attributes
    ----------
    weights:
        Routing weights, shape [B, N].
    mask:
        Binary active-slot mask, shape [B, N].
    logits:
        Raw routing logits, shape [B, N].
    selected_indices:
        Top-k selected slot indices, shape [B, K] when top-k is enabled.
        Otherwise None.
    """

    weights: Tensor
    mask: Tensor
    logits: Tensor
    selected_indices: Optional[Tensor]


class SlotRouter(nn.Module):
    """
    Select relevant memory slots for each input representation.

    Parameters
    ----------
    d_model:
        Hidden dimension D.
    num_slots:
        Number of memory slots N.
    hidden_dim:
        Hidden size of the MLP router.
    mode:
        One of {"softmax", "sigmoid", "gumbel_softmax", "cosine"}.
    top_k:
        Number of active slots. None means all slots remain active.
    temperature:
        Routing temperature.
    dropout:
        Dropout inside the MLP router.
    straight_through:
        Whether gumbel-softmax should return hard one-hot assignments in the
        forward pass while preserving gradients through the soft distribution.
    use_layer_norm:
        Apply LayerNorm to query inputs.
    learnable_slot_embeddings:
        For cosine mode, use trainable slot embeddings.
    eps:
        Numerical stability constant.
    """

    VALID_MODES = {"softmax", "sigmoid", "gumbel_softmax", "cosine"}

    def __init__(
        self,
        d_model: int,
        num_slots: int,
        hidden_dim: Optional[int] = None,
        mode: str = "softmax",
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        dropout: float = 0.0,
        straight_through: bool = False,
        use_layer_norm: bool = True,
        learnable_slot_embeddings: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be greater than zero.")
        if num_slots <= 0:
            raise ValueError("num_slots must be greater than zero.")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero.")
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}, got {mode!r}."
            )
        if top_k is not None and not 1 <= top_k <= num_slots:
            raise ValueError(
                f"top_k must be between 1 and num_slots={num_slots}."
            )
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.d_model = d_model
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim or max(d_model // 2, 1)
        self.mode = mode
        self.top_k = top_k
        self.temperature = float(temperature)
        self.straight_through = straight_through
        self.eps = eps

        self.query_norm = (
            nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        )

        self.router = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, num_slots),
        )

        slot_embeddings = torch.empty(num_slots, d_model)
        nn.init.normal_(slot_embeddings, mean=0.0, std=0.02)

        if learnable_slot_embeddings:
            self.slot_embeddings = nn.Parameter(slot_embeddings)
        else:
            self.register_buffer("slot_embeddings", slot_embeddings)

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.slot_projection = nn.Linear(d_model, d_model, bias=False)

    def set_temperature(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        self.temperature = float(temperature)

    def forward(
        self,
        query: Tensor,
        memory_slots: Optional[Tensor] = None,
        slot_mask: Optional[Tensor] = None,
    ) -> RoutingOutput:
        """
        Route each sample to one or more memory slots.

        Parameters
        ----------
        query:
            Sequence summary or hidden representation, shape [B, D].
        memory_slots:
            Optional current memory, shape [B, N, D].

            In cosine mode:
            - when supplied, routing compares the query against current memory;
            - when omitted, learned slot embeddings are used.
        slot_mask:
            Optional availability mask, shape [B, N] or [B, N, 1].

        Returns
        -------
        RoutingOutput
        """
        self._validate_query(query)

        batch_size = query.size(0)
        prepared_mask = self._prepare_slot_mask(
            slot_mask,
            batch_size=batch_size,
            device=query.device,
        )

        if self.mode == "cosine":
            logits = self._cosine_logits(query, memory_slots)
        else:
            logits = self.router(self.query_norm(query))

        masked_logits = self._mask_logits(logits, prepared_mask)
        dense_weights = self._activate(masked_logits)
        weights, active_mask, selected_indices = self._sparsify(
            dense_weights=dense_weights,
            slot_mask=prepared_mask,
        )

        return RoutingOutput(
            weights=weights,
            mask=active_mask,
            logits=logits,
            selected_indices=selected_indices,
        )

    def _cosine_logits(
        self,
        query: Tensor,
        memory_slots: Optional[Tensor],
    ) -> Tensor:
        projected_query = self.query_projection(
            self.query_norm(query)
        )  # [B, D]
        projected_query = F.normalize(
            projected_query,
            p=2,
            dim=-1,
            eps=self.eps,
        )

        if memory_slots is None:
            slots = self.slot_embeddings.unsqueeze(0).expand(
                query.size(0),
                -1,
                -1,
            )
        else:
            self._validate_memory(memory_slots, query.size(0))
            slots = memory_slots

        projected_slots = self.slot_projection(slots)
        projected_slots = F.normalize(
            projected_slots,
            p=2,
            dim=-1,
            eps=self.eps,
        )

        logits = torch.einsum(
            "bd,bnd->bn",
            projected_query,
            projected_slots,
        )

        return logits

    def _activate(self, logits: Tensor) -> Tensor:
        if self.mode in {"softmax", "cosine"}:
            return torch.softmax(
                logits / self.temperature,
                dim=-1,
            )

        if self.mode == "sigmoid":
            return torch.sigmoid(
                logits / self.temperature
            )

        if self.mode == "gumbel_softmax":
            return F.gumbel_softmax(
                logits,
                tau=self.temperature,
                hard=self.straight_through,
                dim=-1,
            )

        raise RuntimeError(f"Unsupported routing mode: {self.mode}")

    def _sparsify(
        self,
        dense_weights: Tensor,
        slot_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        if self.top_k is None or self.top_k == self.num_slots:
            if slot_mask is None:
                active_mask = torch.ones_like(
                    dense_weights,
                    dtype=torch.bool,
                )
            else:
                active_mask = slot_mask.bool()

            weights = dense_weights * active_mask.to(dense_weights.dtype)

            if self.mode in {"softmax", "gumbel_softmax", "cosine"}:
                weights = weights / weights.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(self.eps)

            return weights, active_mask, None

        _, selected_indices = torch.topk(
            dense_weights,
            k=self.top_k,
            dim=-1,
        )

        active_mask = torch.zeros_like(
            dense_weights,
            dtype=torch.bool,
        )
        active_mask.scatter_(1, selected_indices, True)

        if slot_mask is not None:
            active_mask = active_mask & slot_mask.bool()

        weights = dense_weights * active_mask.to(dense_weights.dtype)

        if self.mode in {"softmax", "gumbel_softmax", "cosine"}:
            weights = weights / weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(self.eps)

        return weights, active_mask, selected_indices

    @torch.no_grad()
    def diagnostics(
        self,
        routing: RoutingOutput | Tensor,
        active_threshold: float = 1e-6,
    ) -> Dict[str, Tensor]:
        """
        Compute routing and slot-utilization statistics.
        """
        if isinstance(routing, RoutingOutput):
            weights = routing.weights
        else:
            weights = routing

        self._validate_weights(weights)

        normalized = weights / weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps)

        entropy = -(
            normalized * normalized.clamp_min(self.eps).log()
        ).sum(dim=-1)

        max_entropy = torch.log(
            torch.tensor(
                float(self.num_slots),
                device=weights.device,
                dtype=weights.dtype,
            )
        ).clamp_min(self.eps)

        mean_usage = weights.mean(dim=0)
        usage_distribution = mean_usage / mean_usage.sum().clamp_min(self.eps)

        return {
            "routing_entropy": entropy.mean(),
            "normalized_routing_entropy": (
                entropy / max_entropy
            ).mean(),
            "active_slots_per_sample": (
                weights > active_threshold
            ).float().sum(dim=-1).mean(),
            "slot_usage_variance": mean_usage.var(unbiased=False),
            "unused_slot_fraction": (
                mean_usage <= active_threshold
            ).float().mean(),
            "maximum_slot_share": usage_distribution.max(),
            "minimum_slot_share": usage_distribution.min(),
            "mean_max_route_weight": weights.max(dim=-1).values.mean(),
        }

    def load_balance_loss(self, weights: Tensor) -> Tensor:
        """
        Encourage equal average utilization across slots.

        The loss operates across the batch, not within each individual sample.
        Therefore individual samples may still specialize to a subset of slots.
        """
        self._validate_weights(weights)

        average_usage = weights.mean(dim=0)
        normalized_usage = average_usage / average_usage.sum().clamp_min(
            self.eps
        )
        target = torch.full_like(
            normalized_usage,
            1.0 / self.num_slots,
        )

        return F.mse_loss(normalized_usage, target)

    def route_diversity_loss(self, weights: Tensor) -> Tensor:
        """
        Penalize identical routing patterns between different batch samples.
        """
        self._validate_weights(weights)

        batch_size = weights.size(0)
        if batch_size <= 1:
            return weights.new_zeros(())

        normalized = F.normalize(
            weights,
            p=2,
            dim=-1,
            eps=self.eps,
        )
        similarities = normalized @ normalized.transpose(0, 1)

        identity = torch.eye(
            batch_size,
            device=weights.device,
            dtype=weights.dtype,
        )

        return (
            similarities.mul(1.0 - identity).pow(2).sum()
            / (batch_size * (batch_size - 1))
        )

    def commitment_loss(
        self,
        query: Tensor,
        routing: RoutingOutput,
        memory_slots: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encourage the query to stay close to the selected slot representation.

        This is especially useful for cosine routing and slot specialization.
        """
        self._validate_query(query)
        self._validate_weights(routing.weights)

        if memory_slots is None:
            slots = self.slot_embeddings.unsqueeze(0).expand(
                query.size(0),
                -1,
                -1,
            )
        else:
            self._validate_memory(memory_slots, query.size(0))
            slots = memory_slots

        selected_representation = torch.einsum(
            "bn,bnd->bd",
            routing.weights,
            slots,
        )

        query_normalized = F.normalize(
            query,
            p=2,
            dim=-1,
            eps=self.eps,
        )
        selected_normalized = F.normalize(
            selected_representation,
            p=2,
            dim=-1,
            eps=self.eps,
        )

        return (
            1.0
            - F.cosine_similarity(
                query_normalized,
                selected_normalized,
                dim=-1,
            )
        ).mean()

    @staticmethod
    def _mask_logits(
        logits: Tensor,
        slot_mask: Optional[Tensor],
    ) -> Tensor:
        if slot_mask is None:
            return logits

        minimum = torch.finfo(logits.dtype).min
        return logits.masked_fill(~slot_mask.bool(), minimum)

    def _prepare_slot_mask(
        self,
        slot_mask: Optional[Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        if slot_mask is None:
            return None

        if not torch.is_tensor(slot_mask):
            raise TypeError("slot_mask must be a torch.Tensor.")

        if slot_mask.dim() == 3 and slot_mask.size(-1) == 1:
            slot_mask = slot_mask.squeeze(-1)

        expected_shape = (batch_size, self.num_slots)

        if tuple(slot_mask.shape) != expected_shape:
            raise ValueError(
                f"slot_mask must have shape {expected_shape}, "
                f"got {tuple(slot_mask.shape)}."
            )

        prepared = slot_mask.to(device=device).bool()

        if (~prepared).all(dim=-1).any():
            raise ValueError(
                "Every sample must have at least one available memory slot."
            )

        return prepared

    def _validate_query(self, query: Tensor) -> None:
        if not torch.is_tensor(query):
            raise TypeError("query must be a torch.Tensor.")
        if query.dim() != 2:
            raise ValueError(
                f"query must have shape [B, D], got {tuple(query.shape)}."
            )
        if query.size(-1) != self.d_model:
            raise ValueError(
                f"query final dimension must be {self.d_model}, "
                f"got {query.size(-1)}."
            )

    def _validate_memory(
        self,
        memory_slots: Tensor,
        batch_size: int,
    ) -> None:
        expected_shape = (
            batch_size,
            self.num_slots,
            self.d_model,
        )
        if tuple(memory_slots.shape) != expected_shape:
            raise ValueError(
                f"memory_slots must have shape {expected_shape}, "
                f"got {tuple(memory_slots.shape)}."
            )

    def _validate_weights(self, weights: Tensor) -> None:
        if not torch.is_tensor(weights):
            raise TypeError("weights must be a torch.Tensor.")
        expected_tail = (self.num_slots,)
        if weights.dim() != 2 or weights.shape[1:] != expected_tail:
            raise ValueError(
                f"weights must have shape [B, {self.num_slots}], "
                f"got {tuple(weights.shape)}."
            )


def _smoke_test() -> None:
    """Run with: python models/slot_router.py"""
    torch.manual_seed(42)

    batch_size = 4
    num_slots = 8
    d_model = 64

    query = torch.randn(batch_size, d_model)
    memory = torch.randn(batch_size, num_slots, d_model)

    print("=== MLP softmax top-k router ===")
    router = SlotRouter(
        d_model=d_model,
        num_slots=num_slots,
        hidden_dim=32,
        mode="softmax",
        top_k=2,
        temperature=0.7,
    )

    output = router(query)
    metrics = router.diagnostics(output)

    assert output.weights.shape == (batch_size, num_slots)
    assert output.mask.shape == (batch_size, num_slots)
    assert output.logits.shape == (batch_size, num_slots)
    assert output.selected_indices is not None
    assert output.selected_indices.shape == (batch_size, 2)
    assert torch.allclose(
        output.weights.sum(dim=-1),
        torch.ones(batch_size),
        atol=1e-5,
    )

    print("Weights shape:", tuple(output.weights.shape))
    print("Selected slots:", output.selected_indices.tolist())
    print(
        "Diagnostics:",
        {k: round(float(v.item()), 6) for k, v in metrics.items()},
    )

    print("\n=== Cosine memory-content router ===")
    cosine_router = SlotRouter(
        d_model=d_model,
        num_slots=num_slots,
        mode="cosine",
        top_k=3,
    )

    cosine_output = cosine_router(
        query=query,
        memory_slots=memory,
    )

    assert cosine_output.weights.shape == (batch_size, num_slots)
    assert torch.allclose(
        cosine_output.weights.sum(dim=-1),
        torch.ones(batch_size),
        atol=1e-5,
    )

    print("Selected slots:", cosine_output.selected_indices.tolist())
    print(
        "Commitment loss:",
        round(
            float(
                cosine_router.commitment_loss(
                    query,
                    cosine_output,
                    memory,
                ).item()
            ),
            6,
        ),
    )


if __name__ == "__main__":
    _smoke_test()
