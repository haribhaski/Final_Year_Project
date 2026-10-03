from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VectorGate(nn.Module):
    """
    Produce a separate gate value for each memory slot.

    Parameters
    ----------
    d_model:
        Hidden-state dimension D.
    num_slots:
        Number of memory slots N.
    hidden_dim:
        Hidden size of the gate MLP. Defaults to d_model // 2.
    mode:
        One of {"sigmoid", "softmax", "gumbel_softmax"}.
    temperature:
        Temperature used by softmax or gumbel-softmax.
    dropout:
        Dropout probability in the gate network.
    init_bias:
        Initial bias of the final projection.

        For sigmoid mode, a negative value such as -2.0 starts with conservative
        writes. For softmax modes, zero is usually suitable.
    top_k:
        Optional number of active slots retained per sample.
        None means that all slots remain active.
    straight_through:
        In gumbel-softmax mode, whether to use hard straight-through routing.
    normalize_topk:
        When top-k routing is enabled, renormalize surviving values:
        - softmax/gumbel modes: surviving gates sum to one;
        - sigmoid mode: surviving gates are scaled by their maximum only when
          normalize_topk=True.
    use_layer_norm:
        Apply LayerNorm to the input hidden representation.
    eps:
        Numerical stability constant.
    """

    VALID_MODES = {"sigmoid", "softmax", "gumbel_softmax"}

    def __init__(
        self,
        d_model: int,
        num_slots: int,
        hidden_dim: Optional[int] = None,
        mode: str = "sigmoid",
        temperature: float = 1.0,
        dropout: float = 0.0,
        init_bias: Optional[float] = None,
        top_k: Optional[int] = None,
        straight_through: bool = False,
        normalize_topk: bool = True,
        use_layer_norm: bool = True,
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
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if top_k is not None and not 1 <= top_k <= num_slots:
            raise ValueError(
                f"top_k must be between 1 and num_slots={num_slots}."
            )

        self.d_model = d_model
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim or max(d_model // 2, 1)
        self.mode = mode
        self.temperature = float(temperature)
        self.top_k = top_k
        self.straight_through = straight_through
        self.normalize_topk = normalize_topk
        self.eps = eps

        self.input_norm = (
            nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        )

        self.network = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, num_slots),
        )

        if init_bias is None:
            init_bias = -2.0 if mode == "sigmoid" else 0.0

        self._reset_parameters(float(init_bias))

    def _reset_parameters(self, init_bias: float) -> None:
        final_layer = self.network[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("The final vector-gate layer must be nn.Linear.")

        nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        nn.init.constant_(final_layer.bias, init_bias)

    def set_temperature(self, temperature: float) -> None:
        """Update routing temperature during training."""
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        self.temperature = float(temperature)

    def logits(self, hidden: Tensor) -> Tensor:
        """
        Return unnormalized slot logits with shape [B, N].
        """
        self._validate_hidden(hidden)
        return self.network(self.input_norm(hidden))

    def forward(
        self,
        hidden: Tensor,
        slot_mask: Optional[Tensor] = None,
        return_logits: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Compute slot-wise gate values.

        Parameters
        ----------
        hidden:
            Input representation with shape [B, D].
        slot_mask:
            Optional mask with shape [B, N] or [B, N, 1].
            Masked slots receive zero gate values.
        return_logits:
            When True, return `(gates, logits)`.

        Returns
        -------
        Tensor or Tuple[Tensor, Tensor]
            gates with shape [B, N, 1].
        """
        raw_logits = self.logits(hidden)

        prepared_mask = None
        if slot_mask is not None:
            prepared_mask = self._prepare_slot_mask(
                slot_mask=slot_mask,
                batch_size=hidden.size(0),
                device=hidden.device,
            )  # [B, N]

        gate_values = self._activate(raw_logits, prepared_mask)
        gate_values = self._apply_top_k(gate_values, prepared_mask)

        if prepared_mask is not None:
            gate_values = gate_values * prepared_mask.to(gate_values.dtype)

        gates = gate_values.unsqueeze(-1)

        if return_logits:
            return gates, raw_logits
        return gates

    def _activate(
        self,
        logits: Tensor,
        slot_mask: Optional[Tensor],
    ) -> Tensor:
        if self.mode == "sigmoid":
            values = torch.sigmoid(logits / self.temperature)

        elif self.mode == "softmax":
            masked_logits = self._mask_logits(logits, slot_mask)
            values = torch.softmax(
                masked_logits / self.temperature,
                dim=-1,
            )

        elif self.mode == "gumbel_softmax":
            masked_logits = self._mask_logits(logits, slot_mask)
            values = F.gumbel_softmax(
                masked_logits,
                tau=self.temperature,
                hard=self.straight_through,
                dim=-1,
            )

        else:
            raise RuntimeError(f"Unsupported gate mode: {self.mode}")

        return values

    def _apply_top_k(
        self,
        values: Tensor,
        slot_mask: Optional[Tensor],
    ) -> Tensor:
        if self.top_k is None or self.top_k == self.num_slots:
            return values

        _, top_indices = torch.topk(
            values,
            k=self.top_k,
            dim=-1,
        )

        sparse_mask = torch.zeros_like(values)
        sparse_mask.scatter_(1, top_indices, 1.0)

        if slot_mask is not None:
            sparse_mask = sparse_mask * slot_mask.to(sparse_mask.dtype)

        sparse_values = values * sparse_mask

        if self.normalize_topk:
            if self.mode in {"softmax", "gumbel_softmax"}:
                sparse_values = sparse_values / sparse_values.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(self.eps)

            elif self.mode == "sigmoid":
                # Preserve [0,1] interpretation while retaining relative strength.
                maximum = sparse_values.max(dim=-1, keepdim=True).values
                sparse_values = torch.where(
                    maximum > self.eps,
                    sparse_values / maximum.clamp_min(self.eps),
                    sparse_values,
                )

        return sparse_values

    @torch.no_grad()
    def diagnostics(
        self,
        gates: Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, Tensor]:
        """
        Compute paper-ready gate diagnostics.

        Metrics
        -------
        gate_mean:
            Mean gate opening over batch and slots.

        gate_std:
            Standard deviation over all gate values.

        within_sample_slot_variance:
            Measures whether different slots receive different gate values.
            This is expected to be approximately zero for a scalar gate and
            non-zero for a functioning vector gate.

        routing_entropy:
            Entropy across slots after normalizing each sample's gates.

        normalized_routing_entropy:
            Entropy divided by log(N), producing a value in approximately [0,1].

        active_slot_fraction:
            Fraction of gates greater than `threshold`.

        unused_slot_fraction:
            Fraction of slots whose mean gate across the batch is negligible.

        load_imbalance:
            Variance of average slot utilization across the batch.

        maximum_slot_share:
            Largest average normalized routing share assigned to any slot.
        """
        self._validate_gates(gates)

        values = gates.squeeze(-1).float()  # [B, N]

        normalized = values / values.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps)

        entropy_per_sample = -(
            normalized * normalized.clamp_min(self.eps).log()
        ).sum(dim=-1)

        max_entropy = torch.log(
            torch.tensor(
                float(self.num_slots),
                device=values.device,
                dtype=values.dtype,
            )
        ).clamp_min(self.eps)

        mean_slot_usage = values.mean(dim=0)
        normalized_mean_usage = mean_slot_usage / mean_slot_usage.sum().clamp_min(
            self.eps
        )

        return {
            "gate_mean": values.mean(),
            "gate_std": values.std(unbiased=False),
            "within_sample_slot_variance": values.var(
                dim=1,
                unbiased=False,
            ).mean(),
            "routing_entropy": entropy_per_sample.mean(),
            "normalized_routing_entropy": (
                entropy_per_sample / max_entropy
            ).mean(),
            "active_slot_fraction": (values > threshold).float().mean(),
            "unused_slot_fraction": (
                mean_slot_usage <= self.eps
            ).float().mean(),
            "load_imbalance": mean_slot_usage.var(unbiased=False),
            "maximum_slot_share": normalized_mean_usage.max(),
            "minimum_gate": values.min(),
            "maximum_gate": values.max(),
        }

    def balance_loss(self, gates: Tensor) -> Tensor:
        """
        Encourage balanced average slot usage across a batch.

        This does not force each individual input to use all slots equally.
        It only discourages the entire batch from repeatedly selecting the
        same small subset of slots.
        """
        self._validate_gates(gates)

        mean_usage = gates.squeeze(-1).mean(dim=0)
        target = mean_usage.mean().detach()

        return (mean_usage - target).pow(2).mean()

    def entropy_loss(
        self,
        gates: Tensor,
        maximize_entropy: bool = False,
    ) -> Tensor:
        """
        Compute routing entropy loss.

        Parameters
        ----------
        maximize_entropy:
            False:
                Minimize entropy, encouraging sparse/specialized routing.
            True:
                Return negative entropy, so minimizing the result encourages
                broader routing.

        Use carefully:
        - low entropy can promote specialization,
        - extremely low entropy can cause slot starvation,
        - very high entropy can make all slots behave similarly.
        """
        self._validate_gates(gates)

        values = gates.squeeze(-1)
        normalized = values / values.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps)

        entropy = -(
            normalized * normalized.clamp_min(self.eps).log()
        ).sum(dim=-1).mean()

        return -entropy if maximize_entropy else entropy

    def specialization_loss(self, gates: Tensor) -> Tensor:
        """
        Encourage different samples to use distinguishable slot allocations.

        The loss penalizes cosine similarity between routing vectors belonging
        to different samples in the same batch.

        This should be used with a small coefficient because semantically
        similar samples may legitimately use similar slots.
        """
        self._validate_gates(gates)

        routing = gates.squeeze(-1)
        routing = F.normalize(routing, p=2, dim=-1, eps=self.eps)

        gram = routing @ routing.transpose(0, 1)
        batch_size = routing.size(0)

        if batch_size <= 1:
            return routing.new_zeros(())

        identity = torch.eye(
            batch_size,
            device=routing.device,
            dtype=routing.dtype,
        )

        return (
            (gram * (1.0 - identity)).pow(2).sum()
            / (batch_size * (batch_size - 1))
        )

    @staticmethod
    def _mask_logits(
        logits: Tensor,
        slot_mask: Optional[Tensor],
    ) -> Tensor:
        if slot_mask is None:
            return logits

        minimum = torch.finfo(logits.dtype).min
        return logits.masked_fill(~slot_mask.bool(), minimum)

    def _validate_hidden(self, hidden: Tensor) -> None:
        if not torch.is_tensor(hidden):
            raise TypeError("hidden must be a torch.Tensor.")
        if hidden.dim() != 2:
            raise ValueError(
                f"hidden must have shape [B, D], got {tuple(hidden.shape)}."
            )
        if hidden.size(-1) != self.d_model:
            raise ValueError(
                f"hidden final dimension must be {self.d_model}, "
                f"got {hidden.size(-1)}."
            )

    def _validate_gates(self, gates: Tensor) -> None:
        if not torch.is_tensor(gates):
            raise TypeError("gates must be a torch.Tensor.")
        if gates.dim() != 3:
            raise ValueError(
                f"gates must have shape [B, N, 1], got {tuple(gates.shape)}."
            )
        if gates.size(1) != self.num_slots or gates.size(2) != 1:
            raise ValueError(
                f"gates must have shape [B, {self.num_slots}, 1], "
                f"got {tuple(gates.shape)}."
            )

    def _prepare_slot_mask(
        self,
        slot_mask: Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
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

        if self.mode in {"softmax", "gumbel_softmax"}:
            if (~prepared).all(dim=-1).any():
                raise ValueError(
                    "Every sample must have at least one unmasked slot in "
                    "softmax or gumbel-softmax mode."
                )

        return prepared


def _smoke_test() -> None:
    """Run with: python models/vector_gate.py"""
    torch.manual_seed(42)

    batch_size = 4
    d_model = 64
    num_slots = 8
    hidden = torch.randn(batch_size, d_model)

    print("=== Sigmoid vector gate ===")
    sigmoid_gate = VectorGate(
        d_model=d_model,
        num_slots=num_slots,
        hidden_dim=32,
        mode="sigmoid",
        init_bias=-2.0,
    )
    gates = sigmoid_gate(hidden)
    metrics = sigmoid_gate.diagnostics(gates)

    assert gates.shape == (batch_size, num_slots, 1)
    assert torch.all(gates >= 0.0)
    assert torch.all(gates <= 1.0)

    print("Shape:", tuple(gates.shape))
    print("First sample:", [round(v, 4) for v in gates[0, :, 0].tolist()])
    print(
        "Diagnostics:",
        {k: round(float(v.item()), 6) for k, v in metrics.items()},
    )

    print("\n=== Softmax top-k vector gate ===")
    sparse_gate = VectorGate(
        d_model=d_model,
        num_slots=num_slots,
        hidden_dim=32,
        mode="softmax",
        temperature=0.7,
        top_k=2,
        normalize_topk=True,
    )
    sparse_gates = sparse_gate(hidden)
    sparse_metrics = sparse_gate.diagnostics(
        sparse_gates,
        threshold=1e-6,
    )

    assert sparse_gates.shape == (batch_size, num_slots, 1)
    assert torch.allclose(
        sparse_gates.squeeze(-1).sum(dim=-1),
        torch.ones(batch_size),
        atol=1e-5,
    )
    assert (
        sparse_gates.squeeze(-1).gt(0).sum(dim=-1) <= 2
    ).all()

    print("Shape:", tuple(sparse_gates.shape))
    print(
        "First sample:",
        [round(v, 4) for v in sparse_gates[0, :, 0].tolist()],
    )
    print(
        "Diagnostics:",
        {
            k: round(float(v.item()), 6)
            for k, v in sparse_metrics.items()
        },
    )


if __name__ == "__main__":
    _smoke_test()
