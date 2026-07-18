from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from transformers import GPT2LMHeadModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gpt2_memory.py requires Hugging Face Transformers. "
        "Install it with: pip install transformers"
    ) from exc

try:
    from .memory_bank import MemoryBank, MemoryState
    from .scalar_gate import ScalarGate
    from .vector_gate import VectorGate
    from .slot_router import RoutingOutput, SlotRouter
    from .candidate_writer import CandidateWriter, CandidateWriterOutput
    from .orthogonal_update import OrthogonalUpdate, OrthogonalUpdateOutput
    from .memory_reader import MemoryReader, MemoryReadOutput
except ImportError:
    # Enables direct execution with:
    # python models/gpt2_memory.py
    from memory_bank import MemoryBank, MemoryState
    from scalar_gate import ScalarGate
    from vector_gate import VectorGate
    from slot_router import RoutingOutput, SlotRouter
    from candidate_writer import CandidateWriter, CandidateWriterOutput
    from orthogonal_update import OrthogonalUpdate, OrthogonalUpdateOutput
    from memory_reader import MemoryReader, MemoryReadOutput


@dataclass
class MemoryGPT2Config:
    """
    Configuration for the external latent-memory components.

    GPT-2's own architecture is loaded from the Hugging Face model/config.
    """

    num_slots: int = 8

    gate_type: str = "vector"
    gate_hidden_dim: Optional[int] = None
    gate_mode: str = "sigmoid"
    gate_temperature: float = 1.0
    gate_top_k: Optional[int] = None
    gate_init_bias: float = -2.0

    router_enabled: bool = True
    router_mode: str = "softmax"
    router_hidden_dim: Optional[int] = None
    router_top_k: Optional[int] = 2
    router_temperature: float = 0.7

    writer_mode: str = "attention"
    writer_hidden_dim: Optional[int] = None
    writer_activation: str = "tanh"
    writer_attention_heads: int = 8
    writer_residual_scale: float = 0.1

    orthogonal_mode: str = "other_slots"
    orthogonal_strength: float = 0.5
    preserve_update_norm: bool = True
    learned_basis_rank: int = 4

    reader_mode: str = "hybrid"
    reader_fusion: str = "gated"
    reader_heads: int = 8
    reader_top_k: Optional[int] = 3
    reader_temperature: float = 0.8
    reader_residual_scale: float = 0.1

    memory_normalization: str = "layernorm"
    memory_max_slot_norm: Optional[float] = None
    trainable_initial_memory: bool = True

    summary_mode: str = "masked_mean"
    read_before_write: bool = True
    detach_memory_between_steps: bool = False
    enable_memory: bool = True
    dropout: float = 0.0

    candidate_diversity_weight: float = 0.0
    update_orthogonality_weight: float = 0.0
    router_balance_weight: float = 0.0
    reader_balance_weight: float = 0.0
    head_diversity_weight: float = 0.0
    memory_collapse_weight: float = 0.0
    gate_sparsity_weight: float = 0.0

    def validate(self, d_model: Optional[int] = None) -> None:
        if self.num_slots <= 0:
            raise ValueError("num_slots must be greater than zero.")
        if self.gate_type not in {"scalar", "vector"}:
            raise ValueError("gate_type must be 'scalar' or 'vector'.")
        if self.summary_mode not in {
            "masked_mean",
            "last_token",
            "first_token",
        }:
            raise ValueError(
                "summary_mode must be masked_mean, last_token, or first_token."
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        nonnegative_fields = {
            "candidate_diversity_weight": self.candidate_diversity_weight,
            "update_orthogonality_weight": self.update_orthogonality_weight,
            "router_balance_weight": self.router_balance_weight,
            "reader_balance_weight": self.reader_balance_weight,
            "head_diversity_weight": self.head_diversity_weight,
            "memory_collapse_weight": self.memory_collapse_weight,
            "gate_sparsity_weight": self.gate_sparsity_weight,
        }
        for name, value in nonnegative_fields.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative.")

        if d_model is not None:
            for name, heads in (
                ("writer_attention_heads", self.writer_attention_heads),
                ("reader_heads", self.reader_heads),
            ):
                if heads <= 0 or d_model % heads != 0:
                    raise ValueError(
                        f"d_model={d_model} must be divisible by "
                        f"{name}={heads}."
                    )


@dataclass
class MemoryGPT2Output:
    """
    Output returned by MemoryAugmentedGPT2LMHeadModel.
    """

    loss: Optional[Tensor]
    lm_loss: Optional[Tensor]
    auxiliary_loss: Tensor
    logits: Tensor
    memory_state: MemoryState
    hidden_states: Tensor
    memory_context: Optional[Tensor]
    read_output: Optional[MemoryReadOutput]
    routing_output: Optional[RoutingOutput]
    writer_output: Optional[CandidateWriterOutput]
    orthogonal_output: Optional[OrthogonalUpdateOutput]
    write_gate: Optional[Tensor]
    diagnostics: Dict[str, Tensor] = field(default_factory=dict)
    auxiliary_losses: Dict[str, Tensor] = field(default_factory=dict)
    past_key_values: Optional[Any] = None


class MemoryAugmentedGPT2LMHeadModel(nn.Module):
    """
    GPT-2 language model with an explicit persistent latent memory state.

    Notes
    -----
    The memory reader is applied to GPT-2's final hidden states before the
    language-model head. Therefore memory directly affects token predictions.

    The write path uses the current sequence representation to update memory
    for future chunks. With `read_before_write=True`, the current chunk reads
    the previous state and writes only after producing its contextualized
    hidden representation, which avoids trivial same-step information leakage.
    """

    def __init__(
        self,
        backbone: GPT2LMHeadModel,
        memory_config: Optional[MemoryGPT2Config] = None,
    ) -> None:
        super().__init__()

        if not isinstance(backbone, GPT2LMHeadModel):
            raise TypeError("backbone must be a GPT2LMHeadModel.")

        self.backbone = backbone
        self.memory_config = memory_config or MemoryGPT2Config()

        d_model = int(backbone.config.n_embd)
        self.memory_config.validate(d_model=d_model)

        self.d_model = d_model
        self.vocab_size = int(backbone.config.vocab_size)
        self.num_slots = self.memory_config.num_slots

        self.memory_bank = MemoryBank(
            num_slots=self.num_slots,
            d_model=d_model,
            normalization=self.memory_config.memory_normalization,
            max_slot_norm=self.memory_config.memory_max_slot_norm,
            trainable_initial_memory=(
                self.memory_config.trainable_initial_memory
            ),
        )

        if self.memory_config.gate_type == "scalar":
            self.write_gate_module: nn.Module = ScalarGate(
                d_model=d_model,
                num_slots=self.num_slots,
                hidden_dim=self.memory_config.gate_hidden_dim,
                dropout=self.memory_config.dropout,
                init_bias=self.memory_config.gate_init_bias,
            )
        else:
            self.write_gate_module = VectorGate(
                d_model=d_model,
                num_slots=self.num_slots,
                hidden_dim=self.memory_config.gate_hidden_dim,
                mode=self.memory_config.gate_mode,
                temperature=self.memory_config.gate_temperature,
                dropout=self.memory_config.dropout,
                init_bias=self.memory_config.gate_init_bias,
                top_k=self.memory_config.gate_top_k,
            )

        self.router: Optional[SlotRouter]
        if self.memory_config.router_enabled:
            self.router = SlotRouter(
                d_model=d_model,
                num_slots=self.num_slots,
                hidden_dim=self.memory_config.router_hidden_dim,
                mode=self.memory_config.router_mode,
                top_k=self.memory_config.router_top_k,
                temperature=self.memory_config.router_temperature,
                dropout=self.memory_config.dropout,
            )
        else:
            self.router = None

        self.writer = CandidateWriter(
            d_model=d_model,
            num_slots=self.num_slots,
            hidden_dim=self.memory_config.writer_hidden_dim,
            mode=self.memory_config.writer_mode,
            dropout=self.memory_config.dropout,
            candidate_activation=self.memory_config.writer_activation,
            residual_scale=self.memory_config.writer_residual_scale,
            num_attention_heads=self.memory_config.writer_attention_heads,
        )

        self.orthogonalizer = OrthogonalUpdate(
            d_model=d_model,
            num_slots=self.num_slots,
            mode=self.memory_config.orthogonal_mode,
            learned_rank=self.memory_config.learned_basis_rank,
            preserve_norm=self.memory_config.preserve_update_norm,
            strength=self.memory_config.orthogonal_strength,
        )

        self.reader = MemoryReader(
            d_model=d_model,
            num_slots=self.num_slots,
            num_heads=self.memory_config.reader_heads,
            mode=self.memory_config.reader_mode,
            fusion=self.memory_config.reader_fusion,
            top_k=self.memory_config.reader_top_k,
            temperature=self.memory_config.reader_temperature,
            dropout=self.memory_config.dropout,
            residual_scale=self.memory_config.reader_residual_scale,
            return_attention_by_default=True,
        )

        # A confidence estimate for each newly written slot.
        self.write_confidence_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "gpt2",
        memory_config: Optional[MemoryGPT2Config] = None,
        **kwargs: Any,
    ) -> "MemoryAugmentedGPT2LMHeadModel":
        """
        Load a pretrained GPT-2 language model and attach memory modules.
        """
        backbone = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
        return cls(backbone=backbone, memory_config=memory_config)

    @classmethod
    def from_gpt2_config(
        cls,
        gpt2_config: Any,
        memory_config: Optional[MemoryGPT2Config] = None,
    ) -> "MemoryAugmentedGPT2LMHeadModel":
        """
        Create a randomly initialized GPT-2 model from a transformers GPT2Config.
        """
        backbone = GPT2LMHeadModel(gpt2_config)
        return cls(backbone=backbone, memory_config=memory_config)

    def initialize_memory(
        self,
        batch_size: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> MemoryState:
        return self.memory_bank.initialize(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def reset_memory(
        self,
        batch_size: int,
        reference: Optional[Tensor] = None,
    ) -> MemoryState:
        device = reference.device if reference is not None else None
        dtype = reference.dtype if reference is not None else None
        return self.initialize_memory(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def detach_memory(self, memory_state: MemoryState) -> MemoryState:
        return memory_state.detach()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        memory_state: Optional[MemoryState] = None,
        memory_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        return_diagnostics: bool = True,
        update_memory: bool = True,
    ) -> MemoryGPT2Output:
        """
        Run GPT-2, read memory, compute logits, and update memory.

        Memory boundaries
        -----------------
        Pass `memory_state=None` to start a fresh sequence/document.
        Pass the returned state into the next chunk to retain long-term context.
        Do not carry state across unrelated examples unless that is intentional.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Provide input_ids or inputs_embeds.")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "Provide only one of input_ids and inputs_embeds."
            )

        reference = (
            inputs_embeds
            if inputs_embeds is not None
            else self.backbone.transformer.wte(input_ids)
        )
        batch_size = reference.size(0)

        if memory_state is None:
            memory_state = self.initialize_memory(
                batch_size=batch_size,
                device=reference.device,
                dtype=reference.dtype,
            )
        else:
            self._validate_memory_state(
                memory_state,
                batch_size=batch_size,
            )
            memory_state = memory_state.to(reference.device)

        transformer_outputs = self.backbone.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )

        base_hidden = transformer_outputs.last_hidden_state
        summary = self._pool_hidden(
            hidden_states=base_hidden,
            attention_mask=attention_mask,
        )

        read_output: Optional[MemoryReadOutput] = None
        routing_output: Optional[RoutingOutput] = None
        writer_output: Optional[CandidateWriterOutput] = None
        orthogonal_output: Optional[OrthogonalUpdateOutput] = None
        write_gate: Optional[Tensor] = None
        context: Optional[Tensor] = None

        working_state = memory_state

        if not self.memory_config.enable_memory:
            fused_hidden = base_hidden
            new_state = working_state
        else:
            if self.memory_config.read_before_write:
                read_output = self._read_memory(
                    hidden_states=base_hidden,
                    memory_state=working_state,
                    attention_mask=attention_mask,
                    memory_mask=memory_mask,
                )
                fused_hidden = read_output.fused_hidden
                context = read_output.context

                if update_memory:
                    (
                        new_state,
                        routing_output,
                        writer_output,
                        orthogonal_output,
                        write_gate,
                    ) = self._write_memory(
                        summary=summary,
                        token_states=base_hidden,
                        attention_mask=attention_mask,
                        memory_state=working_state,
                        memory_mask=memory_mask,
                    )
                else:
                    new_state = working_state

            else:
                if update_memory:
                    (
                        written_state,
                        routing_output,
                        writer_output,
                        orthogonal_output,
                        write_gate,
                    ) = self._write_memory(
                        summary=summary,
                        token_states=base_hidden,
                        attention_mask=attention_mask,
                        memory_state=working_state,
                        memory_mask=memory_mask,
                    )
                else:
                    written_state = working_state

                read_output = self._read_memory(
                    hidden_states=base_hidden,
                    memory_state=written_state,
                    attention_mask=attention_mask,
                    memory_mask=memory_mask,
                )
                fused_hidden = read_output.fused_hidden
                context = read_output.context
                new_state = written_state

            if read_output is not None:
                new_state = self.memory_bank.record_reads(
                    state=new_state,
                    read_weights=read_output.slot_usage,
                )

        if (
            self.memory_config.detach_memory_between_steps
            and update_memory
        ):
            new_state = new_state.detach()

        logits = self.backbone.lm_head(fused_hidden)

        lm_loss = None
        if labels is not None:
            lm_loss = self._causal_language_modeling_loss(
                logits=logits,
                labels=labels,
            )

        auxiliary_losses = self._compute_auxiliary_losses(
            new_state=new_state,
            routing_output=routing_output,
            writer_output=writer_output,
            orthogonal_output=orthogonal_output,
            read_output=read_output,
            write_gate=write_gate,
        )

        if auxiliary_losses:
            auxiliary_loss = torch.stack(
                [value for value in auxiliary_losses.values()]
            ).sum()
        else:
            auxiliary_loss = logits.new_zeros(())

        loss = None
        if lm_loss is not None:
            loss = lm_loss + auxiliary_loss

        diagnostics: Dict[str, Tensor] = {}
        if return_diagnostics:
            diagnostics = self._collect_diagnostics(
                memory_state=new_state,
                routing_output=routing_output,
                writer_output=writer_output,
                orthogonal_output=orthogonal_output,
                read_output=read_output,
                write_gate=write_gate,
                attention_mask=attention_mask,
            )

        return MemoryGPT2Output(
            loss=loss,
            lm_loss=lm_loss,
            auxiliary_loss=auxiliary_loss,
            logits=logits,
            memory_state=new_state,
            hidden_states=fused_hidden,
            memory_context=context,
            read_output=read_output,
            routing_output=routing_output,
            writer_output=writer_output,
            orthogonal_output=orthogonal_output,
            write_gate=write_gate,
            diagnostics=diagnostics,
            auxiliary_losses=auxiliary_losses,
            past_key_values=transformer_outputs.past_key_values,
        )

    def _read_memory(
        self,
        hidden_states: Tensor,
        memory_state: MemoryState,
        attention_mask: Optional[Tensor],
        memory_mask: Optional[Tensor],
    ) -> MemoryReadOutput:
        confidence = memory_state.confidence

        # New memory begins with zero confidence. A small floor prevents all
        # slot logits from becoming effectively masked at initialization.
        confidence_prior = confidence.clamp_min(0.05)

        return self.reader(
            hidden_states=hidden_states,
            memory_slots=memory_state.slots,
            attention_mask=attention_mask,
            memory_mask=memory_mask,
            memory_confidence=confidence_prior,
            return_attention=True,
        )

    def _write_memory(
        self,
        summary: Tensor,
        token_states: Tensor,
        attention_mask: Optional[Tensor],
        memory_state: MemoryState,
        memory_mask: Optional[Tensor],
    ) -> Tuple[
        MemoryState,
        Optional[RoutingOutput],
        CandidateWriterOutput,
        OrthogonalUpdateOutput,
        Tensor,
    ]:
        routing_output = None
        routing_weights = None
        write_mask = memory_mask

        if self.router is not None:
            routing_output = self.router(
                query=summary,
                memory_slots=memory_state.slots,
                slot_mask=memory_mask,
            )
            routing_weights = routing_output.weights
            write_mask = routing_output.mask.unsqueeze(-1)

        writer_output = self.writer(
            summary=summary,
            memory_slots=memory_state.slots,
            token_states=(
                token_states
                if self.memory_config.writer_mode == "attention"
                else None
            ),
            attention_mask=(
                attention_mask
                if self.memory_config.writer_mode == "attention"
                else None
            ),
            routing_weights=routing_weights,
        )

        orthogonal_output = self.orthogonalizer(
            updates=writer_output.deltas,
            memory_slots=memory_state.slots,
        )

        projected_candidate = (
            memory_state.slots + orthogonal_output.updates
        )

        write_gate = self.write_gate_module(
            summary,
            slot_mask=memory_mask,
        )

        if routing_weights is not None:
            write_gate = write_gate * routing_weights.unsqueeze(-1)

        confidence = self.write_confidence_head(
            projected_candidate
        ).squeeze(-1)

        new_state = self.memory_bank(
            state=memory_state,
            candidate=projected_candidate,
            write_gate=write_gate,
            write_mask=write_mask,
            confidence=confidence,
        )

        return (
            new_state,
            routing_output,
            writer_output,
            orthogonal_output,
            write_gate,
        )

    def _pool_hidden(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        if self.memory_config.summary_mode == "first_token":
            return hidden_states[:, 0]

        if self.memory_config.summary_mode == "last_token":
            if attention_mask is None:
                return hidden_states[:, -1]

            indices = attention_mask.long().sum(dim=-1).sub(1).clamp_min(0)
            batch_indices = torch.arange(
                hidden_states.size(0),
                device=hidden_states.device,
            )
            return hidden_states[batch_indices, indices]

        if attention_mask is None:
            return hidden_states.mean(dim=1)

        weights = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return (
            hidden_states * weights
        ).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _causal_language_modeling_loss(
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        if labels.shape != logits.shape[:2]:
            raise ValueError(
                "labels must have shape [B, T] matching logits."
            )

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def _compute_auxiliary_losses(
        self,
        new_state: MemoryState,
        routing_output: Optional[RoutingOutput],
        writer_output: Optional[CandidateWriterOutput],
        orthogonal_output: Optional[OrthogonalUpdateOutput],
        read_output: Optional[MemoryReadOutput],
        write_gate: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}
        cfg = self.memory_config

        if (
            writer_output is not None
            and cfg.candidate_diversity_weight > 0
        ):
            losses["candidate_diversity"] = (
                cfg.candidate_diversity_weight
                * self.writer.candidate_diversity_loss(
                    writer_output.candidates
                )
            )

        if (
            orthogonal_output is not None
            and cfg.update_orthogonality_weight > 0
        ):
            losses["update_orthogonality"] = (
                cfg.update_orthogonality_weight
                * self.orthogonalizer.orthogonality_loss(
                    orthogonal_output.updates
                )
            )

        if (
            routing_output is not None
            and self.router is not None
            and cfg.router_balance_weight > 0
        ):
            losses["router_balance"] = (
                cfg.router_balance_weight
                * self.router.load_balance_loss(
                    routing_output.weights
                )
            )

        if (
            read_output is not None
            and cfg.reader_balance_weight > 0
        ):
            losses["reader_balance"] = (
                cfg.reader_balance_weight
                * self.reader.slot_balance_loss(
                    read_output.slot_usage
                )
            )

        if (
            read_output is not None
            and read_output.attention_weights is not None
            and cfg.head_diversity_weight > 0
        ):
            losses["head_diversity"] = (
                cfg.head_diversity_weight
                * self.reader.head_diversity_loss(
                    read_output.attention_weights
                )
            )

        if cfg.memory_collapse_weight > 0:
            normalized = F.normalize(
                new_state.slots,
                p=2,
                dim=-1,
            )
            gram = torch.bmm(
                normalized,
                normalized.transpose(1, 2),
            )
            identity = torch.eye(
                self.num_slots,
                device=gram.device,
                dtype=gram.dtype,
            ).unsqueeze(0)
            collapse_loss = (
                gram - identity
            ).pow(2).mean()

            losses["memory_collapse"] = (
                cfg.memory_collapse_weight * collapse_loss
            )

        if (
            write_gate is not None
            and cfg.gate_sparsity_weight > 0
        ):
            losses["gate_sparsity"] = (
                cfg.gate_sparsity_weight
                * write_gate.abs().mean()
            )

        return losses

    @torch.no_grad()
    def _collect_diagnostics(
        self,
        memory_state: MemoryState,
        routing_output: Optional[RoutingOutput],
        writer_output: Optional[CandidateWriterOutput],
        orthogonal_output: Optional[OrthogonalUpdateOutput],
        read_output: Optional[MemoryReadOutput],
        write_gate: Optional[Tensor],
        attention_mask: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        diagnostics: Dict[str, Tensor] = {}

        self._merge_metrics(
            diagnostics,
            "memory",
            self.memory_bank.collapse_metrics(memory_state),
        )

        if routing_output is not None and self.router is not None:
            self._merge_metrics(
                diagnostics,
                "router",
                self.router.diagnostics(routing_output),
            )

        if writer_output is not None:
            self._merge_metrics(
                diagnostics,
                "writer",
                self.writer.diagnostics(writer_output),
            )

        if orthogonal_output is not None:
            self._merge_metrics(
                diagnostics,
                "orthogonal",
                self.orthogonalizer.diagnostics(
                    orthogonal_output,
                    memory_slots=memory_state.slots,
                ),
            )

        if read_output is not None:
            self._merge_metrics(
                diagnostics,
                "reader",
                self.reader.diagnostics(
                    read_output,
                    attention_mask=attention_mask,
                ),
            )

        if write_gate is not None:
            diagnostics.update(
                {
                    "gate/mean": write_gate.float().mean(),
                    "gate/std": write_gate.float().std(
                        unbiased=False
                    ),
                    "gate/max": write_gate.float().max(),
                    "gate/min": write_gate.float().min(),
                    "gate/active_fraction": (
                        write_gate > 1e-4
                    ).float().mean(),
                    "gate/within_sample_slot_variance": (
                        write_gate.squeeze(-1)
                        .float()
                        .var(dim=-1, unbiased=False)
                        .mean()
                    ),
                }
            )

        diagnostics.update(
            {
                "memory/mean_age": memory_state.age.float().mean(),
                "memory/mean_write_count": (
                    memory_state.write_count.float().mean()
                ),
                "memory/mean_read_count": (
                    memory_state.read_count.float().mean()
                ),
                "memory/mean_confidence": (
                    memory_state.confidence.float().mean()
                ),
            }
        )

        return diagnostics

    @staticmethod
    def _merge_metrics(
        target: Dict[str, Tensor],
        prefix: str,
        metrics: Mapping[str, Tensor],
    ) -> None:
        for name, value in metrics.items():
            target[f"{prefix}/{name}"] = value

    def _validate_memory_state(
        self,
        state: MemoryState,
        batch_size: int,
    ) -> None:
        if not isinstance(state, MemoryState):
            raise TypeError("memory_state must be a MemoryState.")

        expected_slots = (
            batch_size,
            self.num_slots,
            self.d_model,
        )
        if tuple(state.slots.shape) != expected_slots:
            raise ValueError(
                f"memory_state.slots must have shape {expected_slots}, "
                f"got {tuple(state.slots.shape)}."
            )

    def freeze_backbone(self) -> None:
        """
        Freeze all pretrained GPT-2 parameters.
        """
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all GPT-2 parameters.
        """
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def set_trainable_backbone_layers(
        self,
        num_unfrozen_final_blocks: int,
    ) -> None:
        """
        Freeze GPT-2 except its final `num_unfrozen_final_blocks` blocks.

        The LM head remains trainable. Since GPT-2 ties the LM head to token
        embeddings, the shared embedding weight is also trainable.
        """
        if num_unfrozen_final_blocks < 0:
            raise ValueError(
                "num_unfrozen_final_blocks cannot be negative."
            )

        blocks = self.backbone.transformer.h
        if num_unfrozen_final_blocks > len(blocks):
            raise ValueError(
                "num_unfrozen_final_blocks exceeds GPT-2 block count."
            )

        self.freeze_backbone()

        if num_unfrozen_final_blocks > 0:
            for block in blocks[-num_unfrozen_final_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

        for parameter in self.backbone.transformer.ln_f.parameters():
            parameter.requires_grad = True
        for parameter in self.backbone.lm_head.parameters():
            parameter.requires_grad = True

    def memory_parameters(self):
        """
        Iterate over external-memory parameters only.
        """
        backbone_parameter_ids = {
            id(parameter)
            for parameter in self.backbone.parameters()
        }
        for parameter in self.parameters():
            if id(parameter) not in backbone_parameter_ids:
                yield parameter


def build_recommended_model(
    model_name: str = "gpt2",
) -> MemoryAugmentedGPT2LMHeadModel:
    """
    Build the recommended full vector-gated architecture.
    """
    config = MemoryGPT2Config(
        num_slots=8,
        gate_type="vector",
        gate_mode="sigmoid",
        gate_init_bias=-2.0,
        router_enabled=True,
        router_mode="softmax",
        router_top_k=2,
        router_temperature=0.7,
        writer_mode="attention",
        writer_attention_heads=8,
        orthogonal_mode="other_slots",
        orthogonal_strength=0.5,
        reader_mode="hybrid",
        reader_fusion="gated",
        reader_heads=8,
        reader_top_k=3,
        reader_temperature=0.8,
        candidate_diversity_weight=0.01,
        update_orthogonality_weight=0.01,
        router_balance_weight=0.01,
        reader_balance_weight=0.01,
        head_diversity_weight=0.001,
        memory_collapse_weight=0.01,
        gate_sparsity_weight=0.001,
    )
    return MemoryAugmentedGPT2LMHeadModel.from_pretrained(
        model_name,
        memory_config=config,
    )


def _smoke_test() -> None:
    """
    Lightweight test using a tiny randomly initialized GPT-2 configuration.

    Run:
        python models/gpt2_memory.py
    """
    try:
        from transformers import GPT2Config
    except ImportError:
        return

    torch.manual_seed(42)

    gpt2_config = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_ctx=64,
        n_embd=64,
        n_layer=2,
        n_head=8,
        bos_token_id=0,
        eos_token_id=1,
        use_cache=False,
    )

    memory_config = MemoryGPT2Config(
        num_slots=8,
        gate_type="vector",
        gate_mode="sigmoid",
        router_enabled=True,
        router_mode="softmax",
        router_top_k=2,
        writer_mode="attention",
        writer_attention_heads=8,
        orthogonal_mode="other_slots",
        orthogonal_strength=0.5,
        reader_mode="hybrid",
        reader_heads=8,
        reader_top_k=3,
        candidate_diversity_weight=0.01,
        update_orthogonality_weight=0.01,
        router_balance_weight=0.01,
        reader_balance_weight=0.01,
        memory_collapse_weight=0.01,
    )

    model = MemoryAugmentedGPT2LMHeadModel.from_gpt2_config(
        gpt2_config=gpt2_config,
        memory_config=memory_config,
    )

    batch_size = 2
    sequence_length = 16

    input_ids = torch.randint(
        0,
        gpt2_config.vocab_size,
        (batch_size, sequence_length),
    )
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    first = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    assert first.logits.shape == (
        batch_size,
        sequence_length,
        gpt2_config.vocab_size,
    )
    assert first.memory_state.slots.shape == (
        batch_size,
        memory_config.num_slots,
        gpt2_config.n_embd,
    )
    assert first.loss is not None

    second_ids = torch.randint(
        0,
        gpt2_config.vocab_size,
        (batch_size, sequence_length),
    )

    second = model(
        input_ids=second_ids,
        attention_mask=attention_mask,
        labels=second_ids,
        memory_state=first.memory_state,
    )

    assert (
        second.memory_state.write_count
        >= first.memory_state.write_count
    ).all()

    print("Logits:", tuple(first.logits.shape))
    print("Memory:", tuple(first.memory_state.slots.shape))
    print("LM loss:", round(float(first.lm_loss.item()), 6))
    print(
        "Auxiliary loss:",
        round(float(first.auxiliary_loss.item()), 6),
    )
    print(
        "Memory effective rank:",
        round(
            float(
                first.diagnostics[
                    "memory/effective_rank"
                ].item()
            ),
            6,
        ),
    )


if __name__ == "__main__":
    _smoke_test()

