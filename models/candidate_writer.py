from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CandidateWriterOutput:
    """
    Structured writer output.

    Attributes
    ----------
    candidates:
        Final slot-specific candidate vectors, shape [B, N, D].
    deltas:
        Candidate minus previous memory, shape [B, N, D].
    attended_context:
        Token-derived slot context in attention mode, otherwise None.
    """

    candidates: Tensor
    deltas: Tensor
    attended_context: Optional[Tensor]


class CandidateWriter(nn.Module):
    """
    Generate distinct candidate updates for each memory slot.

    Parameters
    ----------
    d_model:
        Hidden and memory dimension D.
    num_slots:
        Number of memory slots N.
    hidden_dim:
        Hidden size of internal feed-forward networks.
    mode:
        One of {"mlp", "gru", "residual", "attention"}.
    dropout:
        Dropout probability.
    use_slot_embeddings:
        Include learned slot-identity embeddings.
    use_memory_content:
        Condition candidates on the previous memory content.
    candidate_activation:
        One of {"tanh", "gelu", "none"}.
    residual_scale:
        Initial multiplier used in residual mode.
    num_attention_heads:
        Number of heads used in attention mode.
    normalize_output:
        Apply LayerNorm to final candidates.
    eps:
        Numerical stability constant.
    """

    VALID_MODES = {"mlp", "gru", "residual", "attention"}
    VALID_ACTIVATIONS = {"tanh", "gelu", "none"}

    def __init__(
        self,
        d_model: int,
        num_slots: int,
        hidden_dim: Optional[int] = None,
        mode: str = "mlp",
        dropout: float = 0.0,
        use_slot_embeddings: bool = True,
        use_memory_content: bool = True,
        candidate_activation: str = "tanh",
        residual_scale: float = 0.1,
        num_attention_heads: int = 8,
        normalize_output: bool = True,
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
        if candidate_activation not in self.VALID_ACTIVATIONS:
            raise ValueError(
                "candidate_activation must be one of "
                f"{self.VALID_ACTIVATIONS}, got {candidate_activation!r}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if residual_scale < 0:
            raise ValueError("residual_scale cannot be negative.")
        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be greater than zero.")
        if d_model % num_attention_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_attention_heads."
            )

        self.d_model = d_model
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim or max(d_model * 2, 1)
        self.mode = mode
        self.use_slot_embeddings = use_slot_embeddings
        self.use_memory_content = use_memory_content
        self.candidate_activation = candidate_activation
        self.eps = eps

        slot_embeddings = torch.empty(num_slots, d_model)
        nn.init.normal_(slot_embeddings, mean=0.0, std=0.02)

        if use_slot_embeddings:
            self.slot_embeddings = nn.Parameter(slot_embeddings)
        else:
            self.register_buffer("slot_embeddings", torch.zeros_like(slot_embeddings))

        self.summary_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.output_norm = (
            nn.LayerNorm(d_model) if normalize_output else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

        input_parts = 1
        if use_memory_content:
            input_parts += 1
        if use_slot_embeddings:
            input_parts += 1

        fusion_input_dim = input_parts * d_model

        self.mlp_writer = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, d_model),
        )

        gru_input_dim = d_model
        if use_slot_embeddings:
            gru_input_dim += d_model

        self.gru_input_projection = nn.Linear(gru_input_dim, d_model)
        self.gru_cell = nn.GRUCell(d_model, d_model)

        self.residual_writer = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, d_model),
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale))
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_fusion = nn.Sequential(
            nn.Linear(3 * d_model, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, d_model),
        )

        self.query_projection = nn.Linear(d_model, d_model)
        self.slot_projection = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in (
            self.mlp_writer[-1],
            self.residual_writer[-1],
            self.attention_fusion[-1],
        ):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        summary: Tensor,
        memory_slots: Tensor,
        token_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        routing_weights: Optional[Tensor] = None,
    ) -> CandidateWriterOutput:
        """
        Generate one candidate vector per memory slot.

        Parameters
        ----------
        summary:
            Sequence-level representation, shape [B, D].
        memory_slots:
            Current memory, shape [B, N, D].
        token_states:
            Token-level hidden states, shape [B, T, D].
            Required only in attention mode.
        attention_mask:
            Token validity mask, shape [B, T], with 1/True for valid tokens.
        routing_weights:
            Optional router weights, shape [B, N] or [B, N, 1].
            These weights modulate candidate deltas but do not replace the
            separate memory gate.

        Returns
        -------
        CandidateWriterOutput
        """
        self._validate_summary(summary)
        self._validate_memory(memory_slots, summary.size(0))

        route = self._prepare_routing_weights(
            routing_weights,
            batch_size=summary.size(0),
            device=summary.device,
            dtype=summary.dtype,
        )

        if self.mode == "mlp":
            raw_candidates = self._mlp_candidates(summary, memory_slots)
            attended_context = None

        elif self.mode == "gru":
            raw_candidates = self._gru_candidates(summary, memory_slots)
            attended_context = None

        elif self.mode == "residual":
            raw_candidates = self._residual_candidates(summary, memory_slots)
            attended_context = None

        elif self.mode == "attention":
            raw_candidates, attended_context = self._attention_candidates(
                summary=summary,
                memory_slots=memory_slots,
                token_states=token_states,
                attention_mask=attention_mask,
            )

        else:
            raise RuntimeError(f"Unsupported writer mode: {self.mode}")

        raw_candidates = self._activate(raw_candidates)

        if self.mode == "residual":
            candidates = raw_candidates
        else:
            candidates = self.output_norm(raw_candidates)

        deltas = candidates - memory_slots

        if route is not None:
            deltas = deltas * route
            candidates = memory_slots + deltas
            candidates = self.output_norm(candidates)

        return CandidateWriterOutput(
            candidates=candidates,
            deltas=candidates - memory_slots,
            attended_context=attended_context,
        )

    def _mlp_candidates(
        self,
        summary: Tensor,
        memory_slots: Tensor,
    ) -> Tensor:
        batch_size = summary.size(0)

        summary_expanded = self.summary_norm(summary).unsqueeze(1).expand(
            -1,
            self.num_slots,
            -1,
        )

        parts = [summary_expanded]

        if self.use_memory_content:
            parts.append(self.memory_norm(memory_slots))

        if self.use_slot_embeddings:
            slot_ids = self.slot_embeddings.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
            parts.append(slot_ids)

        fused = torch.cat(parts, dim=-1)
        return self.mlp_writer(fused)

    def _gru_candidates(
        self,
        summary: Tensor,
        memory_slots: Tensor,
    ) -> Tensor:
        batch_size = summary.size(0)
        summary_expanded = self.summary_norm(summary).unsqueeze(1).expand(
            -1,
            self.num_slots,
            -1,
        )

        gru_parts = [summary_expanded]

        if self.use_slot_embeddings:
            gru_parts.append(
                self.slot_embeddings.unsqueeze(0).expand(
                    batch_size,
                    -1,
                    -1,
                )
            )

        gru_input = torch.cat(gru_parts, dim=-1)
        gru_input = self.gru_input_projection(gru_input)

        flat_input = gru_input.reshape(
            batch_size * self.num_slots,
            self.d_model,
        )
        flat_memory = self.memory_norm(memory_slots).reshape(
            batch_size * self.num_slots,
            self.d_model,
        )

        updated = self.gru_cell(flat_input, flat_memory)
        return updated.reshape(
            batch_size,
            self.num_slots,
            self.d_model,
        )

    def _residual_candidates(
        self,
        summary: Tensor,
        memory_slots: Tensor,
    ) -> Tensor:
        batch_size = summary.size(0)

        summary_expanded = self.summary_norm(summary).unsqueeze(1).expand(
            -1,
            self.num_slots,
            -1,
        )

        parts = [summary_expanded]

        if self.use_memory_content:
            parts.append(self.memory_norm(memory_slots))

        if self.use_slot_embeddings:
            parts.append(
                self.slot_embeddings.unsqueeze(0).expand(
                    batch_size,
                    -1,
                    -1,
                )
            )

        fused = torch.cat(parts, dim=-1)
        delta = self.residual_writer(fused)
        delta = self._activate(delta)

        scale = self.residual_scale.abs()
        return self.output_norm(memory_slots + scale * delta)

    def _attention_candidates(
        self,
        summary: Tensor,
        memory_slots: Tensor,
        token_states: Optional[Tensor],
        attention_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        if token_states is None:
            raise ValueError(
                "token_states is required when mode='attention'."
            )

        self._validate_token_states(
            token_states,
            batch_size=summary.size(0),
        )

        batch_size = summary.size(0)

        summary_query = self.query_projection(
            self.summary_norm(summary)
        ).unsqueeze(1)

        slot_identity = self.slot_embeddings.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

        slot_queries = self.slot_projection(
            self.memory_norm(memory_slots) + slot_identity
        )
        queries = slot_queries + summary_query

        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != token_states.shape[:2]:
                raise ValueError(
                    "attention_mask must have shape "
                    f"{tuple(token_states.shape[:2])}, got "
                    f"{tuple(attention_mask.shape)}."
                )
            key_padding_mask = ~attention_mask.to(
                device=token_states.device
            ).bool()

        attended_context, _ = self.cross_attention(
            query=queries,
            key=token_states,
            value=token_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        fused = torch.cat(
            [
                attended_context,
                self.memory_norm(memory_slots),
                slot_identity,
            ],
            dim=-1,
        )

        candidates = self.attention_fusion(fused)
        return candidates, attended_context

    def _activate(self, tensor: Tensor) -> Tensor:
        if self.candidate_activation == "tanh":
            return torch.tanh(tensor)
        if self.candidate_activation == "gelu":
            return F.gelu(tensor)
        if self.candidate_activation == "none":
            return tensor
        raise RuntimeError(
            f"Unsupported activation: {self.candidate_activation}"
        )

    @torch.no_grad()
    def diagnostics(
        self,
        output: CandidateWriterOutput | Tensor,
    ) -> Dict[str, Tensor]:
        """
        Measure candidate diversity and update geometry.
        """
        if isinstance(output, CandidateWriterOutput):
            candidates = output.candidates
            deltas = output.deltas
        else:
            candidates = output
            self._validate_candidate_tensor(candidates)
            deltas = candidates

        normalized_candidates = F.normalize(
            candidates.float(),
            p=2,
            dim=-1,
            eps=self.eps,
        )

        cosine_matrix = torch.bmm(
            normalized_candidates,
            normalized_candidates.transpose(1, 2),
        )

        identity = torch.eye(
            self.num_slots,
            device=candidates.device,
            dtype=cosine_matrix.dtype,
        ).unsqueeze(0)

        off_diagonal_count = max(
            candidates.size(0)
            * self.num_slots
            * (self.num_slots - 1),
            1,
        )

        mean_pairwise_cosine = (
            cosine_matrix * (1.0 - identity)
        ).sum() / off_diagonal_count

        delta_norms = deltas.float().norm(dim=-1)
        candidate_norms = candidates.float().norm(dim=-1)

        effective_ranks = []
        for sample in candidates.float():
            singular_values = torch.linalg.svdvals(sample)
            probabilities = singular_values / singular_values.sum().clamp_min(
                self.eps
            )
            entropy = -(
                probabilities
                * probabilities.clamp_min(self.eps).log()
            ).sum()
            effective_ranks.append(entropy.exp())

        return {
            "candidate_pairwise_cosine": mean_pairwise_cosine,
            "candidate_effective_rank": torch.stack(
                effective_ranks
            ).mean(),
            "candidate_norm_mean": candidate_norms.mean(),
            "candidate_norm_std": candidate_norms.std(unbiased=False),
            "delta_norm_mean": delta_norms.mean(),
            "delta_norm_std": delta_norms.std(unbiased=False),
            "near_zero_delta_fraction": (
                delta_norms < 1e-4
            ).float().mean(),
        }

    def candidate_diversity_loss(
        self,
        candidates: Tensor,
        margin: float = 0.2,
    ) -> Tensor:
        """
        Penalize overly similar candidate vectors within each sample.

        The loss is zero for pairwise cosine similarity below `margin`.
        """
        self._validate_candidate_tensor(candidates)

        normalized = F.normalize(
            candidates,
            p=2,
            dim=-1,
            eps=self.eps,
        )

        similarities = torch.bmm(
            normalized,
            normalized.transpose(1, 2),
        )

        identity = torch.eye(
            self.num_slots,
            device=candidates.device,
            dtype=candidates.dtype,
        ).unsqueeze(0)

        penalties = F.relu(similarities - margin)
        penalties = penalties * (1.0 - identity)

        denominator = max(
            candidates.size(0)
            * self.num_slots
            * (self.num_slots - 1),
            1,
        )

        return penalties.pow(2).sum() / denominator

    def delta_orthogonality_loss(
        self,
        deltas: Tensor,
    ) -> Tensor:
        """
        Encourage slot-specific update directions to be orthogonal.
        """
        self._validate_candidate_tensor(deltas)

        normalized = F.normalize(
            deltas,
            p=2,
            dim=-1,
            eps=self.eps,
        )
        gram = torch.bmm(
            normalized,
            normalized.transpose(1, 2),
        )

        identity = torch.eye(
            self.num_slots,
            device=deltas.device,
            dtype=deltas.dtype,
        ).unsqueeze(0)

        return (gram - identity).pow(2).mean()

    def _prepare_routing_weights(
        self,
        routing_weights: Optional[Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if routing_weights is None:
            return None

        if routing_weights.dim() == 2:
            routing_weights = routing_weights.unsqueeze(-1)

        expected_shape = (batch_size, self.num_slots, 1)

        if tuple(routing_weights.shape) != expected_shape:
            raise ValueError(
                f"routing_weights must have shape {expected_shape}, "
                f"got {tuple(routing_weights.shape)}."
            )

        return routing_weights.to(
            device=device,
            dtype=dtype,
        ).clamp_min(0.0)

    def _validate_summary(self, summary: Tensor) -> None:
        if not torch.is_tensor(summary):
            raise TypeError("summary must be a torch.Tensor.")
        if summary.dim() != 2:
            raise ValueError(
                f"summary must have shape [B, D], got {tuple(summary.shape)}."
            )
        if summary.size(-1) != self.d_model:
            raise ValueError(
                f"summary final dimension must be {self.d_model}, "
                f"got {summary.size(-1)}."
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

    def _validate_token_states(
        self,
        token_states: Tensor,
        batch_size: int,
    ) -> None:
        if token_states.dim() != 3:
            raise ValueError(
                "token_states must have shape [B, T, D]."
            )
        if token_states.size(0) != batch_size:
            raise ValueError(
                "token_states batch size must match summary batch size."
            )
        if token_states.size(-1) != self.d_model:
            raise ValueError(
                f"token_states final dimension must be {self.d_model}, "
                f"got {token_states.size(-1)}."
            )

    def _validate_candidate_tensor(
        self,
        candidates: Tensor,
    ) -> None:
        if not torch.is_tensor(candidates):
            raise TypeError("candidates must be a torch.Tensor.")

        if candidates.dim() != 3:
            raise ValueError(
                "candidates must have shape [B, N, D]."
            )

        if candidates.size(1) != self.num_slots:
            raise ValueError(
                f"Expected {self.num_slots} slots, "
                f"got {candidates.size(1)}."
            )

        if candidates.size(2) != self.d_model:
            raise ValueError(
                f"Expected final dimension {self.d_model}, "
                f"got {candidates.size(2)}."
            )


def _smoke_test() -> None:
    """Run with: python models/candidate_writer.py"""
    torch.manual_seed(42)

    batch_size = 4
    num_slots = 8
    d_model = 64
    sequence_length = 12

    summary = torch.randn(batch_size, d_model)
    memory = torch.randn(batch_size, num_slots, d_model)
    token_states = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )
    attention_mask = torch.ones(
        batch_size,
        sequence_length,
        dtype=torch.long,
    )
    routing = torch.softmax(
        torch.randn(batch_size, num_slots),
        dim=-1,
    )

    for mode in ("mlp", "gru", "residual", "attention"):
        print(f"\n=== {mode} writer ===")

        writer = CandidateWriter(
            d_model=d_model,
            num_slots=num_slots,
            hidden_dim=128,
            mode=mode,
            num_attention_heads=8,
        )

        output = writer(
            summary=summary,
            memory_slots=memory,
            token_states=token_states if mode == "attention" else None,
            attention_mask=attention_mask if mode == "attention" else None,
            routing_weights=routing,
        )

        metrics = writer.diagnostics(output)

        assert output.candidates.shape == (
            batch_size,
            num_slots,
            d_model,
        )
        assert output.deltas.shape == (
            batch_size,
            num_slots,
            d_model,
        )

        print("Candidate shape:", tuple(output.candidates.shape))
        print(
            "Metrics:",
            {
                key: round(float(value.item()), 6)
                for key, value in metrics.items()
            },
        )


if __name__ == "__main__":
    _smoke_test()
