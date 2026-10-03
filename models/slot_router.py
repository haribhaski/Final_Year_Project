from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RoutingOutput:
    weights: Tensor
    mask: Tensor
    logits: Tensor
    selected_indices: Optional[Tensor]


class SlotRouter(nn.Module):
    """
    Slot router with occupancy-aware allocation.

    Modes
    -----
    softmax:
        Original learned router.

    sigmoid:
        Original independent sigmoid routing.

    gumbel_softmax:
        Original Gumbel routing.

    cosine:
        Content similarity routing.

    occupancy:
        NEW mode.

        Routing score combines:
            1. query/slot compatibility
            2. penalty for slots that have already been written

        score_i = content_score_i - lambda * occupancy_i

        This encourages new facts to use free slots instead of repeatedly
        overwriting the same memory addresses.
    """

    VALID_MODES = {
        "softmax",
        "sigmoid",
        "gumbel_softmax",
        "cosine",
        "occupancy",
    }

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

        # NEW
        occupancy_penalty: float = 2.0,
        free_slot_bonus: float = 1.0,
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
            raise ValueError("dropout must be in [0,1).")

        self.d_model = d_model
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim or max(d_model // 2, 1)

        self.mode = mode
        self.top_k = top_k
        self.temperature = float(temperature)
        self.straight_through = straight_through
        self.eps = eps

        # NEW
        self.occupancy_penalty = float(occupancy_penalty)
        self.free_slot_bonus = float(free_slot_bonus)

        self.query_norm = (
            nn.LayerNorm(d_model)
            if use_layer_norm
            else nn.Identity()
        )

        # Original MLP router
        self.router = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, num_slots),
        )

        # Learned slot addresses
        slot_embeddings = torch.empty(num_slots, d_model)
        nn.init.normal_(slot_embeddings, mean=0.0, std=0.02)

        if learnable_slot_embeddings:
            self.slot_embeddings = nn.Parameter(slot_embeddings)
        else:
            self.register_buffer(
                "slot_embeddings",
                slot_embeddings,
            )

        # Content-based addressing
        self.query_projection = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.slot_projection = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

    # ---------------------------------------------------------
    # Main forward
    # ---------------------------------------------------------

    def forward(
        self,
        query: Tensor,
        memory_slots: Optional[Tensor] = None,
        slot_mask: Optional[Tensor] = None,

        # NEW
        write_count: Optional[Tensor] = None,
    ) -> RoutingOutput:

        self._validate_query(query)

        batch_size = query.size(0)

        prepared_mask = self._prepare_slot_mask(
            slot_mask,
            batch_size=batch_size,
            device=query.device,
        )

        if self.mode == "cosine":
            logits = self._cosine_logits(
                query,
                memory_slots,
            )

        elif self.mode == "occupancy":
            logits = self._occupancy_logits(
                query=query,
                memory_slots=memory_slots,
                write_count=write_count,
            )

        else:
            logits = self.router(
                self.query_norm(query)
            )

        masked_logits = self._mask_logits(
            logits,
            prepared_mask,
        )

        dense_weights = self._activate(
            masked_logits
        )

        weights, active_mask, selected_indices = (
            self._sparsify(
                dense_weights=dense_weights,
                slot_mask=prepared_mask,
            )
        )

        return RoutingOutput(
            weights=weights,
            mask=active_mask,
            logits=logits,
            selected_indices=selected_indices,
        )

    # ---------------------------------------------------------
    # NEW OCCUPANCY ROUTER
    # ---------------------------------------------------------

    def _occupancy_logits(
        self,
        query: Tensor,
        memory_slots: Optional[Tensor],
        write_count: Optional[Tensor],
    ) -> Tensor:

        batch_size = query.size(0)

        # ---------------------------------------
        # Query representation
        # ---------------------------------------

        q = self.query_projection(
            self.query_norm(query)
        )

        q = F.normalize(
            q,
            p=2,
            dim=-1,
            eps=self.eps,
        )

        # ---------------------------------------
        # Slot/address representation
        # ---------------------------------------

        #
        # IMPORTANT:
        # use learned addresses here rather than
        # only memory contents.
        #
        # Otherwise an empty slot has no useful
        # address representation.
        #

        addresses = self.slot_embeddings.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

        projected_addresses = self.slot_projection(
            addresses
        )

        projected_addresses = F.normalize(
            projected_addresses,
            p=2,
            dim=-1,
            eps=self.eps,
        )

        # ---------------------------------------
        # Query ↔ address compatibility
        # ---------------------------------------

        content_score = torch.einsum(
            "bd,bnd->bn",
            q,
            projected_addresses,
        )

        # ---------------------------------------
        # No occupancy information
        # ---------------------------------------

        if write_count is None:
            return content_score

        if write_count.shape != (
            batch_size,
            self.num_slots,
        ):
            raise ValueError(
                "write_count must have shape "
                f"[B,{self.num_slots}], "
                f"got {tuple(write_count.shape)}"
            )

        # 0 = free
        # 1 = occupied

        occupied = (
            write_count > 0
        ).to(content_score.dtype)

        free = 1.0 - occupied

        # ---------------------------------------
        # Final score
        #
        # compatible slots score higher
        # occupied slots score lower
        # free slots receive bonus
        # ---------------------------------------

        logits = (
            content_score
            - self.occupancy_penalty * occupied
            + self.free_slot_bonus * free
        )

        return logits

    # ---------------------------------------------------------
    # Original cosine router
    # ---------------------------------------------------------

    def _cosine_logits(
        self,
        query: Tensor,
        memory_slots: Optional[Tensor],
    ) -> Tensor:

        projected_query = self.query_projection(
            self.query_norm(query)
        )

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

            self._validate_memory(
                memory_slots,
                query.size(0),
            )

            slots = memory_slots

        projected_slots = self.slot_projection(
            slots
        )

        projected_slots = F.normalize(
            projected_slots,
            p=2,
            dim=-1,
            eps=self.eps,
        )

        return torch.einsum(
            "bd,bnd->bn",
            projected_query,
            projected_slots,
        )

    # ---------------------------------------------------------
    # Activation
    # ---------------------------------------------------------

    def _activate(
        self,
        logits: Tensor,
    ) -> Tensor:

        if self.mode in {
            "softmax",
            "cosine",
            "occupancy",
        }:

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

        raise RuntimeError(
            f"Unsupported routing mode: {self.mode}"
        )

    # ---------------------------------------------------------
    # Top-k
    # ---------------------------------------------------------

    def _sparsify(
        self,
        dense_weights: Tensor,
        slot_mask: Optional[Tensor],
    ) -> Tuple[
        Tensor,
        Tensor,
        Optional[Tensor],
    ]:

        if (
            self.top_k is None
            or self.top_k == self.num_slots
        ):

            if slot_mask is None:

                active_mask = torch.ones_like(
                    dense_weights,
                    dtype=torch.bool,
                )

            else:

                active_mask = slot_mask.bool()

            weights = (
                dense_weights
                * active_mask.to(dense_weights.dtype)
            )

            if self.mode in {
                "softmax",
                "gumbel_softmax",
                "cosine",
                "occupancy",
            }:

                weights = (
                    weights
                    / weights.sum(
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(self.eps)
                )

            return (
                weights,
                active_mask,
                None,
            )

        _, selected_indices = torch.topk(
            dense_weights,
            k=self.top_k,
            dim=-1,
        )

        active_mask = torch.zeros_like(
            dense_weights,
            dtype=torch.bool,
        )

        active_mask.scatter_(
            1,
            selected_indices,
            True,
        )

        if slot_mask is not None:

            active_mask = (
                active_mask
                & slot_mask.bool()
            )

        weights = (
            dense_weights
            * active_mask.to(dense_weights.dtype)
        )

        if self.mode in {
            "softmax",
            "gumbel_softmax",
            "cosine",
            "occupancy",
        }:

            weights = (
                weights
                / weights.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(self.eps)
            )

        return (
            weights,
            active_mask,
            selected_indices,
        )

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    @torch.no_grad()
    def diagnostics(
        self,
        routing: RoutingOutput | Tensor,
        active_threshold: float = 1e-6,
    ) -> Dict[str, Tensor]:

        if isinstance(
            routing,
            RoutingOutput,
        ):
            weights = routing.weights
        else:
            weights = routing

        self._validate_weights(weights)

        normalized = (
            weights
            / weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(self.eps)
        )

        entropy = -(
            normalized
            * normalized.clamp_min(
                self.eps
            ).log()
        ).sum(dim=-1)

        max_entropy = torch.log(
            torch.tensor(
                float(self.num_slots),
                device=weights.device,
                dtype=weights.dtype,
            )
        ).clamp_min(self.eps)

        mean_usage = weights.mean(dim=0)

        usage_distribution = (
            mean_usage
            / mean_usage.sum().clamp_min(
                self.eps
            )
        )

        return {
            "routing_entropy":
                entropy.mean(),

            "normalized_routing_entropy":
                (entropy / max_entropy).mean(),

            "active_slots_per_sample":
                (
                    weights > active_threshold
                ).float().sum(
                    dim=-1
                ).mean(),

            "slot_usage_variance":
                mean_usage.var(
                    unbiased=False
                ),

            "unused_slot_fraction":
                (
                    mean_usage
                    <= active_threshold
                ).float().mean(),

            "maximum_slot_share":
                usage_distribution.max(),

            "minimum_slot_share":
                usage_distribution.min(),

            "mean_max_route_weight":
                weights.max(
                    dim=-1
                ).values.mean(),
        }

    # ---------------------------------------------------------
    # Existing auxiliary losses
    # ---------------------------------------------------------

    def load_balance_loss(
        self,
        weights: Tensor,
    ) -> Tensor:

        self._validate_weights(weights)

        average_usage = weights.mean(dim=0)

        normalized_usage = (
            average_usage
            / average_usage.sum().clamp_min(
                self.eps
            )
        )

        target = torch.full_like(
            normalized_usage,
            1.0 / self.num_slots,
        )

        return F.mse_loss(
            normalized_usage,
            target,
        )

    def route_diversity_loss(
        self,
        weights: Tensor,
    ) -> Tensor:

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

        similarities = (
            normalized
            @ normalized.transpose(0, 1)
        )

        identity = torch.eye(
            batch_size,
            device=weights.device,
            dtype=weights.dtype,
        )

        return (
            similarities
            .mul(1.0 - identity)
            .pow(2)
            .sum()
            / (
                batch_size
                * (batch_size - 1)
            )
        )

    def commitment_loss(
        self,
        query: Tensor,
        routing: RoutingOutput,
        memory_slots: Optional[Tensor] = None,
    ) -> Tensor:

        self._validate_query(query)
        self._validate_weights(
            routing.weights
        )

        if memory_slots is None:

            slots = (
                self.slot_embeddings
                .unsqueeze(0)
                .expand(
                    query.size(0),
                    -1,
                    -1,
                )
            )

        else:

            self._validate_memory(
                memory_slots,
                query.size(0),
            )

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

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _mask_logits(
        logits: Tensor,
        slot_mask: Optional[Tensor],
    ) -> Tensor:

        if slot_mask is None:
            return logits

        minimum = torch.finfo(
            logits.dtype
        ).min

        return logits.masked_fill(
            ~slot_mask.bool(),
            minimum,
        )

    def _prepare_slot_mask(
        self,
        slot_mask: Optional[Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[Tensor]:

        if slot_mask is None:
            return None

        if not torch.is_tensor(
            slot_mask
        ):
            raise TypeError(
                "slot_mask must be a Tensor."
            )

        if (
            slot_mask.dim() == 3
            and slot_mask.size(-1) == 1
        ):
            slot_mask = slot_mask.squeeze(-1)

        expected_shape = (
            batch_size,
            self.num_slots,
        )

        if tuple(
            slot_mask.shape
        ) != expected_shape:

            raise ValueError(
                "slot_mask must have shape "
                f"{expected_shape}."
            )

        prepared = (
            slot_mask
            .to(device=device)
            .bool()
        )

        if (
            (~prepared)
            .all(dim=-1)
            .any()
        ):
            raise ValueError(
                "Every sample must have "
                "at least one available slot."
            )

        return prepared

    def _validate_query(
        self,
        query: Tensor,
    ) -> None:

        if (
            query.dim() != 2
            or query.size(-1)
            != self.d_model
        ):
            raise ValueError(
                "query must have shape "
                f"[B,{self.d_model}]."
            )

    def _validate_memory(
        self,
        memory_slots: Tensor,
        batch_size: int,
    ) -> None:

        expected = (
            batch_size,
            self.num_slots,
            self.d_model,
        )

        if tuple(
            memory_slots.shape
        ) != expected:

            raise ValueError(
                "memory_slots must have shape "
                f"{expected}."
            )

    def _validate_weights(
        self,
        weights: Tensor,
    ) -> None:

        expected = (
            self.num_slots,
        )

        if (
            weights.dim() != 2
            or weights.shape[1:]
            != expected
        ):
            raise ValueError(
                "weights must have shape "
                f"[B,{self.num_slots}]."
            )


def _smoke_test():

    torch.manual_seed(42)

    router = SlotRouter(
        d_model=64,
        num_slots=8,
        mode="occupancy",
        top_k=1,
        temperature=0.7,
    )

    q1 = torch.randn(1, 64)
    q2 = torch.randn(1, 64)

    memory = torch.randn(
        1,
        8,
        64,
    )

    # No slots occupied
    counts = torch.zeros(
        1,
        8,
        dtype=torch.long,
    )

    r1 = router(
        q1,
        memory,
        write_count=counts,
    )

    first_slot = int(
        r1.weights.argmax(-1).item()
    )

    # Mark first slot occupied
    counts[
        0,
        first_slot
    ] = 1

    r2 = router(
        q2,
        memory,
        write_count=counts,
    )

    second_slot = int(
        r2.weights.argmax(-1).item()
    )

    print("Fact A slot:", first_slot)
    print("Fact B slot:", second_slot)
    print("A route:", r1.weights)
    print("B route:", r2.weights)

    assert first_slot != second_slot

    print(
        "Occupancy router smoke test PASSED"
    )


if __name__ == "__main__":
    _smoke_test()