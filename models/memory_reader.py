from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MemoryReadOutput:
    """
    Structured memory read output.

    Attributes
    ----------
    context:
        Memory-derived context vectors, shape [B, T, D].
    fused_hidden:
        Hidden states after optional memory fusion, shape [B, T, D].
    attention_weights:
        Per-head read weights, shape [B, H, T, N], or None.
    slot_usage:
        Average attention assigned to each slot, shape [B, N].
    read_confidence:
        Maximum or aggregated read confidence, shape [B, T, 1].
    selected_indices:
        Top-k memory slot indices, shape [B, T, K], or None.
    """

    context: Tensor
    fused_hidden: Tensor
    attention_weights: Optional[Tensor]
    slot_usage: Tensor
    read_confidence: Tensor
    selected_indices: Optional[Tensor]


class MemoryReader(nn.Module):
    """
    Read from a latent memory bank using multi-head content-based attention.

    Parameters
    ----------
    d_model:
        Hidden and memory dimensionality.
    num_slots:
        Number of memory slots.
    num_heads:
        Number of attention heads.
    mode:
        Query construction mode:
        - "token": every token independently queries memory.
        - "summary": one sequence summary queries memory and is broadcast.
        - "learned": a learned global read query is combined with token states.
        - "hybrid": token query + sequence summary + learned query.
    fusion:
        How memory context is fused with hidden states:
        - "residual": h + scale * context
        - "gated": h + sigmoid(g(h,c)) * context
        - "concat": projection([h;c])
        - "none": return hidden states unchanged as fused_hidden
    top_k:
        Optional sparse top-k slot retrieval.
    temperature:
        Softmax temperature.
    dropout:
        Dropout used in projections and attention.
    use_memory_norm:
        Apply LayerNorm to memory slots.
    use_query_norm:
        Apply LayerNorm to hidden/query states.
    learnable_residual_scale:
        Make residual scale trainable.
    residual_scale:
        Initial residual scale.
    return_attention_by_default:
        Whether forward returns full per-head attention unless overridden.
    eps:
        Numerical stability value.
    """

    VALID_MODES = {"token", "summary", "learned", "hybrid"}
    VALID_FUSIONS = {"residual", "gated", "concat", "none"}

    def __init__(
        self,
        d_model: int,
        num_slots: int,
        num_heads: int = 8,
        mode: str = "token",
        fusion: str = "gated",
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        dropout: float = 0.0,
        use_memory_norm: bool = True,
        use_query_norm: bool = True,
        learnable_residual_scale: bool = True,
        residual_scale: float = 0.1,
        return_attention_by_default: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be greater than zero.")
        if num_slots <= 0:
            raise ValueError("num_slots must be greater than zero.")
        if num_heads <= 0:
            raise ValueError("num_heads must be greater than zero.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}, got {mode!r}."
            )
        if fusion not in self.VALID_FUSIONS:
            raise ValueError(
                f"fusion must be one of {self.VALID_FUSIONS}, got {fusion!r}."
            )
        if top_k is not None and not 1 <= top_k <= num_slots:
            raise ValueError(
                f"top_k must be between 1 and num_slots={num_slots}."
            )
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if residual_scale < 0:
            raise ValueError("residual_scale cannot be negative.")

        self.d_model = d_model
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.mode = mode
        self.fusion = fusion
        self.top_k = top_k
        self.temperature = float(temperature)
        self.return_attention_by_default = return_attention_by_default
        self.eps = eps

        self.query_norm = (
            nn.LayerNorm(d_model) if use_query_norm else nn.Identity()
        )
        self.memory_norm = (
            nn.LayerNorm(d_model) if use_memory_norm else nn.Identity()
        )
        self.output_norm = nn.LayerNorm(d_model)

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)

        self.summary_projection = nn.Linear(d_model, d_model)
        self.learned_query = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.learned_query, mean=0.0, std=0.02)

        self.query_combiner = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)

        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.concat_fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        residual_tensor = torch.tensor(float(residual_scale))
        if learnable_residual_scale:
            self.residual_scale = nn.Parameter(residual_tensor)
        else:
            self.register_buffer("residual_scale", residual_tensor)

        self.confidence_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2 if d_model >= 2 else 1),
            nn.GELU(),
            nn.Linear(d_model // 2 if d_model >= 2 else 1, 1),
            nn.Sigmoid(),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for layer in (
            self.query_projection,
            self.key_projection,
            self.value_projection,
            self.output_projection,
        ):
            nn.init.xavier_uniform_(layer.weight)

        nn.init.normal_(
            self.output_projection.weight,
            mean=0.0,
            std=0.01,
        )

        last_gate_layer = self.fusion_gate[-2]
        if isinstance(last_gate_layer, nn.Linear):
            nn.init.zeros_(last_gate_layer.weight)
            nn.init.constant_(last_gate_layer.bias, -2.0)

    def set_temperature(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        self.temperature = float(temperature)

    def forward(
        self,
        hidden_states: Tensor,
        memory_slots: Tensor,
        attention_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        routing_prior: Optional[Tensor] = None,
        memory_confidence: Optional[Tensor] = None,
        return_attention: Optional[bool] = None,
    ) -> MemoryReadOutput:
        """
        Read memory and optionally fuse it with transformer hidden states.

        Parameters
        ----------
        hidden_states:
            Transformer states, shape [B, T, D].
        memory_slots:
            Memory bank, shape [B, N, D].
        attention_mask:
            Token validity mask, shape [B, T].
            Used for summary pooling and diagnostics.
        memory_mask:
            Available-slot mask, shape [B, N] or [B, N, 1].
        routing_prior:
            Optional multiplicative/additive prior over memory reads.
            Shape [B, N] or [B, T, N].
        memory_confidence:
            Confidence associated with each memory slot.
            Shape [B, N] or [B, N, 1].
        return_attention:
            Override default attention-return setting.

        Returns
        -------
        MemoryReadOutput
        """
        self._validate_hidden(hidden_states)
        self._validate_memory(memory_slots, hidden_states.size(0))

        batch_size, sequence_length, _ = hidden_states.shape

        token_mask = self._prepare_attention_mask(
            attention_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=hidden_states.device,
        )

        slot_mask = self._prepare_memory_mask(
            memory_mask,
            batch_size=batch_size,
            device=hidden_states.device,
        )

        prior = self._prepare_routing_prior(
            routing_prior,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        confidence = self._prepare_memory_confidence(
            memory_confidence,
            batch_size=batch_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        queries = self._build_queries(
            hidden_states=hidden_states,
            token_mask=token_mask,
        )

        normalized_memory = self.memory_norm(memory_slots)

        q = self._split_heads(self.query_projection(queries))
        k = self._split_heads(self.key_projection(normalized_memory))
        v = self._split_heads(self.value_projection(normalized_memory))

        scores = torch.einsum("bhtd,bhnd->bhtn", q, k)
        scores = scores / (self.head_dim ** 0.5)
        scores = scores / self.temperature

        if prior is not None:
            scores = scores + prior.clamp_min(self.eps).log().unsqueeze(1)

        if confidence is not None:
            scores = scores + confidence.clamp_min(self.eps).log().unsqueeze(1).unsqueeze(2)

        if slot_mask is not None:
            minimum = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(
                ~slot_mask[:, None, None, :],
                minimum,
            )

        selected_indices = None
        if self.top_k is not None and self.top_k < self.num_slots:
            scores, selected_indices = self._apply_top_k(scores)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        context_heads = torch.einsum(
            "bhtn,bhnd->bhtd",
            weights,
            v,
        )
        context = self._merge_heads(context_heads)
        context = self.output_projection(context)
        context = self.output_norm(context)

        if token_mask is not None:
            context = context * token_mask.unsqueeze(-1).to(context.dtype)

        read_confidence = self._compute_read_confidence(
            context=context,
            weights=weights,
        )

        fused_hidden = self._fuse(
            hidden_states=hidden_states,
            context=context,
            read_confidence=read_confidence,
        )

        slot_usage = weights.mean(dim=1).mean(dim=1)

        should_return_attention = (
            self.return_attention_by_default
            if return_attention is None
            else return_attention
        )

        return MemoryReadOutput(
            context=context,
            fused_hidden=fused_hidden,
            attention_weights=weights if should_return_attention else None,
            slot_usage=slot_usage,
            read_confidence=read_confidence,
            selected_indices=selected_indices,
        )

    def _build_queries(
        self,
        hidden_states: Tensor,
        token_mask: Optional[Tensor],
    ) -> Tensor:
        normalized = self.query_norm(hidden_states)
        batch_size, sequence_length, _ = normalized.shape

        if self.mode == "token":
            return normalized

        summary = self._masked_mean(normalized, token_mask)
        projected_summary = self.summary_projection(summary)
        summary_expanded = projected_summary.unsqueeze(1).expand(
            -1,
            sequence_length,
            -1,
        )

        learned = self.learned_query.expand(
            batch_size,
            sequence_length,
            -1,
        )

        if self.mode == "summary":
            return summary_expanded

        if self.mode == "learned":
            return normalized + learned

        if self.mode == "hybrid":
            combined = torch.cat(
                [normalized, summary_expanded, learned],
                dim=-1,
            )
            return self.query_combiner(combined)

        raise RuntimeError(f"Unsupported query mode: {self.mode}")

    def _apply_top_k(
        self,
        scores: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        top_values, top_indices = torch.topk(
            scores,
            k=self.top_k,
            dim=-1,
        )

        sparse_scores = torch.full_like(
            scores,
            torch.finfo(scores.dtype).min,
        )
        sparse_scores.scatter_(
            dim=-1,
            index=top_indices,
            src=top_values,
        )

        # Compress selected indices from [B,H,T,K] to [B,T,K] by using
        # the most common head-wise slot selection.
        batch_size, _, sequence_length, _ = top_indices.shape
        one_hot = F.one_hot(
            top_indices,
            num_classes=self.num_slots,
        ).sum(dim=1).sum(dim=-2)

        compressed = torch.topk(
            one_hot,
            k=self.top_k,
            dim=-1,
        ).indices

        assert compressed.shape == (
            batch_size,
            sequence_length,
            self.top_k,
        )

        return sparse_scores, compressed

    def _compute_read_confidence(
        self,
        context: Tensor,
        weights: Tensor,
    ) -> Tensor:
        learned_confidence = self.confidence_projection(context)

        max_attention = weights.max(dim=-1).values.mean(
            dim=1,
            keepdim=False,
        ).unsqueeze(-1)

        return 0.5 * learned_confidence + 0.5 * max_attention

    def _fuse(
        self,
        hidden_states: Tensor,
        context: Tensor,
        read_confidence: Tensor,
    ) -> Tensor:
        if self.fusion == "none":
            return hidden_states

        scale = self.residual_scale.abs()

        if self.fusion == "residual":
            return hidden_states + scale * context

        if self.fusion == "gated":
            gate_input = torch.cat(
                [hidden_states, context],
                dim=-1,
            )
            gate = self.fusion_gate(gate_input)
            gate = gate * read_confidence
            return hidden_states + scale * gate * context

        if self.fusion == "concat":
            combined = torch.cat(
                [hidden_states, context],
                dim=-1,
            )
            fused = self.concat_fusion(combined)
            return hidden_states + scale * fused

        raise RuntimeError(f"Unsupported fusion mode: {self.fusion}")

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch_size, length, _ = tensor.shape
        tensor = tensor.view(
            batch_size,
            length,
            self.num_heads,
            self.head_dim,
        )
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor: Tensor) -> Tensor:
        batch_size, _, length, _ = tensor.shape
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        return tensor.view(batch_size, length, self.d_model)

    def _masked_mean(
        self,
        hidden_states: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        if mask is None:
            return hidden_states.mean(dim=1)

        weights = mask.unsqueeze(-1).to(hidden_states.dtype)
        numerator = (hidden_states * weights).sum(dim=1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return numerator / denominator

    @torch.no_grad()
    def diagnostics(
        self,
        output: MemoryReadOutput,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute paper-ready memory-read diagnostics.
        """
        result: Dict[str, Tensor] = {
            "context_norm_mean": output.context.float().norm(dim=-1).mean(),
            "context_norm_std": output.context.float().norm(
                dim=-1
            ).std(unbiased=False),
            "read_confidence_mean": output.read_confidence.float().mean(),
            "read_confidence_std": output.read_confidence.float().std(
                unbiased=False
            ),
            "slot_usage_variance": output.slot_usage.float().var(
                dim=-1,
                unbiased=False,
            ).mean(),
            "unused_slot_fraction": (
                output.slot_usage <= 1e-6
            ).float().mean(),
            "maximum_slot_usage": output.slot_usage.float().max(
                dim=-1
            ).values.mean(),
        }

        if output.attention_weights is not None:
            weights = output.attention_weights.float()
            normalized = weights / weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(self.eps)

            entropy = -(
                normalized
                * normalized.clamp_min(self.eps).log()
            ).sum(dim=-1)

            if attention_mask is not None:
                token_mask = attention_mask[:, None, :].to(entropy.dtype)
                entropy_mean = (
                    entropy * token_mask
                ).sum() / token_mask.sum().clamp_min(1.0)
            else:
                entropy_mean = entropy.mean()

            maximum_entropy = torch.log(
                torch.tensor(
                    float(self.num_slots),
                    device=entropy.device,
                    dtype=entropy.dtype,
                )
            ).clamp_min(self.eps)

            result.update(
                {
                    "attention_entropy": entropy_mean,
                    "normalized_attention_entropy": (
                        entropy_mean / maximum_entropy
                    ),
                    "mean_max_attention": weights.max(
                        dim=-1
                    ).values.mean(),
                    "active_slots_per_token": (
                        weights > 1e-6
                    ).float().sum(dim=-1).mean(),
                }
            )

        return result

    def slot_balance_loss(
        self,
        slot_usage: Tensor,
    ) -> Tensor:
        """
        Encourage balanced slot usage across the batch.

        Parameters
        ----------
        slot_usage:
            Shape [B, N].
        """
        self._validate_slot_usage(slot_usage)

        average_usage = slot_usage.mean(dim=0)
        average_usage = average_usage / average_usage.sum().clamp_min(
            self.eps
        )

        target = torch.full_like(
            average_usage,
            1.0 / self.num_slots,
        )

        return F.mse_loss(average_usage, target)

    def attention_entropy_loss(
        self,
        attention_weights: Tensor,
        target_entropy: Optional[float] = None,
    ) -> Tensor:
        """
        Control read sharpness.

        When target_entropy is None, the loss minimizes entropy and promotes
        sharper retrieval. When a target is supplied, it penalizes deviation
        from that entropy level.
        """
        self._validate_attention_weights(attention_weights)

        normalized = attention_weights / attention_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps)

        entropy = -(
            normalized
            * normalized.clamp_min(self.eps).log()
        ).sum(dim=-1)

        if target_entropy is None:
            return entropy.mean()

        target = entropy.new_tensor(float(target_entropy))
        return (entropy - target).pow(2).mean()

    def head_diversity_loss(
        self,
        attention_weights: Tensor,
    ) -> Tensor:
        """
        Encourage attention heads to retrieve different memory slots.
        """
        self._validate_attention_weights(attention_weights)

        head_patterns = attention_weights.mean(dim=2)
        head_patterns = F.normalize(
            head_patterns,
            p=2,
            dim=-1,
            eps=self.eps,
        )

        similarities = torch.bmm(
            head_patterns,
            head_patterns.transpose(1, 2),
        )

        identity = torch.eye(
            self.num_heads,
            device=similarities.device,
            dtype=similarities.dtype,
        ).unsqueeze(0)

        return (similarities - identity).pow(2).mean()

    def read_consistency_loss(
        self,
        attention_a: Tensor,
        attention_b: Tensor,
    ) -> Tensor:
        """
        Encourage stable retrieval for two augmented views of the same input.
        """
        self._validate_attention_weights(attention_a)
        self._validate_attention_weights(attention_b)

        if attention_a.shape != attention_b.shape:
            raise ValueError(
                "attention_a and attention_b must have identical shapes."
            )

        distribution_a = attention_a.mean(dim=1)
        distribution_b = attention_b.mean(dim=1)

        distribution_a = distribution_a / distribution_a.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps)
        distribution_b = distribution_b / distribution_b.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps)

        return F.mse_loss(distribution_a, distribution_b)

    def _prepare_attention_mask(
        self,
        mask: Optional[Tensor],
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        if mask is None:
            return None

        expected_shape = (batch_size, sequence_length)
        if tuple(mask.shape) != expected_shape:
            raise ValueError(
                f"attention_mask must have shape {expected_shape}, "
                f"got {tuple(mask.shape)}."
            )

        return mask.to(device=device).bool()

    def _prepare_memory_mask(
        self,
        mask: Optional[Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        if mask is None:
            return None

        if mask.dim() == 3 and mask.size(-1) == 1:
            mask = mask.squeeze(-1)

        expected_shape = (batch_size, self.num_slots)
        if tuple(mask.shape) != expected_shape:
            raise ValueError(
                f"memory_mask must have shape {expected_shape}, "
                f"got {tuple(mask.shape)}."
            )

        prepared = mask.to(device=device).bool()

        if (~prepared).all(dim=-1).any():
            raise ValueError(
                "Each batch sample must expose at least one memory slot."
            )

        return prepared

    def _prepare_routing_prior(
        self,
        prior: Optional[Tensor],
        batch_size: int,
        sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if prior is None:
            return None

        if prior.dim() == 2:
            expected = (batch_size, self.num_slots)
            if tuple(prior.shape) != expected:
                raise ValueError(
                    f"routing_prior must have shape {expected} or "
                    f"[B, T, N], got {tuple(prior.shape)}."
                )
            prior = prior.unsqueeze(1).expand(
                -1,
                sequence_length,
                -1,
            )

        elif prior.dim() == 3:
            expected = (
                batch_size,
                sequence_length,
                self.num_slots,
            )
            if tuple(prior.shape) != expected:
                raise ValueError(
                    f"routing_prior must have shape {expected}, "
                    f"got {tuple(prior.shape)}."
                )
        else:
            raise ValueError(
                "routing_prior must have shape [B, N] or [B, T, N]."
            )

        return prior.to(
            device=device,
            dtype=dtype,
        )

    def _prepare_memory_confidence(
        self,
        confidence: Optional[Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if confidence is None:
            return None

        if confidence.dim() == 3 and confidence.size(-1) == 1:
            confidence = confidence.squeeze(-1)

        expected_shape = (batch_size, self.num_slots)

        if tuple(confidence.shape) != expected_shape:
            raise ValueError(
                f"memory_confidence must have shape {expected_shape}, "
                f"got {tuple(confidence.shape)}."
            )

        return confidence.to(
            device=device,
            dtype=dtype,
        ).clamp(0.0, 1.0)

    def _validate_hidden(self, hidden_states: Tensor) -> None:
        if not torch.is_tensor(hidden_states):
            raise TypeError("hidden_states must be a torch.Tensor.")
        if hidden_states.dim() != 3:
            raise ValueError(
                "hidden_states must have shape [B, T, D]."
            )
        if hidden_states.size(-1) != self.d_model:
            raise ValueError(
                f"hidden_states final dimension must be {self.d_model}, "
                f"got {hidden_states.size(-1)}."
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

    def _validate_slot_usage(self, slot_usage: Tensor) -> None:
        if slot_usage.dim() != 2:
            raise ValueError("slot_usage must have shape [B, N].")
        if slot_usage.size(-1) != self.num_slots:
            raise ValueError(
                f"slot_usage final dimension must be {self.num_slots}."
            )

    def _validate_attention_weights(
        self,
        attention_weights: Tensor,
    ) -> None:
        if attention_weights.dim() != 4:
            raise ValueError(
                "attention_weights must have shape [B, H, T, N]."
            )
        if attention_weights.size(1) != self.num_heads:
            raise ValueError(
                f"Expected {self.num_heads} attention heads."
            )
        if attention_weights.size(-1) != self.num_slots:
            raise ValueError(
                f"Expected {self.num_slots} memory slots."
            )


def _smoke_test() -> None:
    """Run with: python models/memory_reader.py"""
    torch.manual_seed(42)

    batch_size = 4
    sequence_length = 16
    num_slots = 8
    d_model = 64

    hidden = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )
    memory = torch.randn(
        batch_size,
        num_slots,
        d_model,
    )

    attention_mask = torch.ones(
        batch_size,
        sequence_length,
        dtype=torch.long,
    )
    attention_mask[0, -3:] = 0

    memory_mask = torch.ones(
        batch_size,
        num_slots,
        dtype=torch.bool,
    )
    memory_mask[1, -2:] = False

    confidence = torch.rand(
        batch_size,
        num_slots,
    )

    for mode in ("token", "summary", "learned", "hybrid"):
        print(f"\n=== {mode} reader ===")

        reader = MemoryReader(
            d_model=d_model,
            num_slots=num_slots,
            num_heads=8,
            mode=mode,
            fusion="gated",
            top_k=3,
            temperature=0.8,
            return_attention_by_default=True,
        )

        output = reader(
            hidden_states=hidden,
            memory_slots=memory,
            attention_mask=attention_mask,
            memory_mask=memory_mask,
            memory_confidence=confidence,
        )

        assert output.context.shape == hidden.shape
        assert output.fused_hidden.shape == hidden.shape
        assert output.slot_usage.shape == (
            batch_size,
            num_slots,
        )
        assert output.read_confidence.shape == (
            batch_size,
            sequence_length,
            1,
        )
        assert output.attention_weights is not None
        assert output.attention_weights.shape == (
            batch_size,
            8,
            sequence_length,
            num_slots,
        )
        assert output.selected_indices is not None
        assert output.selected_indices.shape == (
            batch_size,
            sequence_length,
            3,
        )

        metrics = reader.diagnostics(
            output,
            attention_mask=attention_mask,
        )

        print("Context shape:", tuple(output.context.shape))
        print(
            "Metrics:",
            {
                key: round(float(value.item()), 6)
                for key, value in metrics.items()
            },
        )


if __name__ == "__main__":
    _smoke_test()
