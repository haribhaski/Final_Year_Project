from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor


class ScalarGate(nn.Module):
    """
    Produce one sigmoid write-gate value per sample and broadcast it to all slots.

    Parameters
    ----------
    d_model:
        Dimension of the input hidden representation.
    num_slots:
        Number of memory slots N.
    hidden_dim:
        Hidden size of the gate MLP. Defaults to d_model // 2.
    dropout:
        Dropout used inside the gate network.
    init_bias:
        Initial bias of the final gate projection.

        A negative value is usually preferred so that the model initially
        writes conservatively. For example:

            sigmoid(-2.0) ~= 0.119
    use_layer_norm:
        Whether to normalize the input before gate prediction.
    """

    def __init__(
        self,
        d_model: int,
        num_slots: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        init_bias: float = -2.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be greater than zero.")
        if num_slots <= 0:
            raise ValueError("num_slots must be greater than zero.")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.d_model = d_model
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim or max(d_model // 2, 1)

        self.input_norm = (
            nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        )

        self.network = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

        self._reset_parameters(init_bias)

    def _reset_parameters(self, init_bias: float) -> None:
        """
        Initialize the final projection conservatively.

        The last layer starts with a small weight scale and a negative bias,
        reducing aggressive writes at the start of training.
        """
        final_layer = self.network[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("The final scalar-gate layer must be nn.Linear.")

        nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        nn.init.constant_(final_layer.bias, init_bias)

    def forward(
        self,
        hidden: Tensor,
        slot_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute scalar gate values and broadcast them over memory slots.

        Parameters
        ----------
        hidden:
            Input representation with shape [B, D].

            This will usually be:
            - a masked mean of fact-token hidden states,
            - the final valid token representation, or
            - another pooled sequence summary.
        slot_mask:
            Optional slot mask with shape [B, N] or [B, N, 1].
            Masked slots receive a gate value of zero.

        Returns
        -------
        Tensor
            Broadcast gate values with shape [B, N, 1].
        """
        self._validate_hidden(hidden)

        scalar_gate = torch.sigmoid(
            self.network(self.input_norm(hidden))
        )  # [B, 1]

        gates = scalar_gate.unsqueeze(1).expand(
            -1,
            self.num_slots,
            -1,
        )  # [B, N, 1]

        if slot_mask is not None:
            slot_mask = self._prepare_slot_mask(
                slot_mask=slot_mask,
                batch_size=hidden.size(0),
                device=hidden.device,
                dtype=gates.dtype,
            )
            gates = gates * slot_mask

        return gates

    @torch.no_grad()
    def diagnostics(self, gates: Tensor) -> Dict[str, Tensor]:
        """
        Return basic scalar-gate diagnostics.

        For an unmasked scalar gate, within-sample slot variance should be zero
        because every slot receives exactly the same value.
        """
        self._validate_gates(gates)

        per_sample_mean = gates.mean(dim=1)
        per_sample_variance = gates.var(dim=1, unbiased=False)

        return {
            "gate_mean": gates.mean(),
            "gate_std_across_batch": per_sample_mean.std(unbiased=False),
            "within_sample_slot_variance": per_sample_variance.mean(),
            "minimum_gate": gates.min(),
            "maximum_gate": gates.max(),
            "open_fraction_0_5": (gates > 0.5).float().mean(),
        }

    def raw_scalar(self, hidden: Tensor) -> Tensor:
        """
        Return the unbroadcast scalar gate with shape [B, 1].

        This is useful for analysis and plotting.
        """
        self._validate_hidden(hidden)
        return torch.sigmoid(self.network(self.input_norm(hidden)))

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
        dtype: torch.dtype,
    ) -> Tensor:
        if not torch.is_tensor(slot_mask):
            raise TypeError("slot_mask must be a torch.Tensor.")

        if slot_mask.dim() == 2:
            slot_mask = slot_mask.unsqueeze(-1)

        expected_shape = (batch_size, self.num_slots, 1)

        if tuple(slot_mask.shape) != expected_shape:
            raise ValueError(
                f"slot_mask must have shape {expected_shape}, "
                f"got {tuple(slot_mask.shape)}."
            )

        return slot_mask.to(device=device, dtype=dtype).clamp(0.0, 1.0)


def _smoke_test() -> None:
    """Run with: python models/scalar_gate.py"""
    torch.manual_seed(42)

    gate = ScalarGate(
        d_model=64,
        num_slots=8,
        hidden_dim=32,
        dropout=0.0,
        init_bias=-2.0,
    )

    hidden = torch.randn(4, 64)
    gates = gate(hidden)
    diagnostics = gate.diagnostics(gates)

    assert gates.shape == (4, 8, 1)
    assert torch.allclose(gates[:, 0], gates[:, 7])
    assert torch.all(gates >= 0.0)
    assert torch.all(gates <= 1.0)

    print("Gate shape:", tuple(gates.shape))
    print("First sample gates:", gates[0, :, 0].tolist())
    print(
        "Diagnostics:",
        {
            name: round(float(value.item()), 6)
            for name, value in diagnostics.items()
        },
    )


if __name__ == "__main__":
    _smoke_test()
