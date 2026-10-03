from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class OrthogonalUpdateOutput:
    """
    Structured orthogonalization result.

    Attributes
    ----------
    updates:
        Final projected updates, shape [B, N, D].
    removed_component:
        Component removed by projection, shape [B, N, D].
    projection_ratio:
        Removed norm divided by original norm, shape [B, N].
    """

    updates: Tensor
    removed_component: Tensor
    projection_ratio: Tensor


class OrthogonalUpdate(nn.Module):
    """
    Project memory updates away from interfering directions.

    Parameters
    ----------
    d_model:
        Memory dimension D.
    num_slots:
        Number of memory slots N.
    mode:
        One of:
        - "slot": remove the component parallel to each slot itself.
        - "other_slots": remove components lying in the span of other slots.
        - "all_slots": remove components lying in the full memory subspace.
        - "pairwise": sequentially orthogonalize slot updates against earlier
          update vectors.
        - "learned_basis": project away from a trainable K-dimensional basis.
        - "none": return updates unchanged.
    learned_rank:
        Number of trainable basis vectors for "learned_basis".
    preserve_norm:
        Rescale projected updates to retain their original L2 norm.
    strength:
        Interpolation factor between original and fully projected update.
        0 means no projection, 1 means full projection.
    detach_basis:
        Detach memory-derived bases from the gradient graph.
    eps:
        Numerical stability constant.
    """

    VALID_MODES = {
        "slot",
        "other_slots",
        "all_slots",
        "pairwise",
        "learned_basis",
        "none",
    }

    def __init__(
        self,
        d_model: int,
        num_slots: int,
        mode: str = "other_slots",
        learned_rank: int = 4,
        preserve_norm: bool = True,
        strength: float = 1.0,
        detach_basis: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be greater than zero.")
        if num_slots <= 0:
            raise ValueError("num_slots must be greater than zero.")
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}, got {mode!r}."
            )
        if learned_rank <= 0:
            raise ValueError("learned_rank must be greater than zero.")
        if learned_rank > d_model:
            raise ValueError("learned_rank cannot exceed d_model.")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1].")

        self.d_model = d_model
        self.num_slots = num_slots
        self.mode = mode
        self.learned_rank = learned_rank
        self.preserve_norm = preserve_norm
        self.strength = float(strength)
        self.detach_basis = detach_basis
        self.eps = eps

        learned_basis = torch.empty(learned_rank, d_model)
        nn.init.orthogonal_(learned_basis)
        self.learned_basis = nn.Parameter(learned_basis)

    def forward(
        self,
        updates: Tensor,
        memory_slots: Tensor,
        external_basis: Optional[Tensor] = None,
    ) -> OrthogonalUpdateOutput:
        """
        Orthogonalize update vectors.

        Parameters
        ----------
        updates:
            Proposed slot updates, shape [B, N, D].
        memory_slots:
            Current memory content, shape [B, N, D].
        external_basis:
            Optional basis with shape [K, D] or [B, K, D].
            When supplied, it replaces the basis implied by the selected mode,
            except for "pairwise" and "slot".

        Returns
        -------
        OrthogonalUpdateOutput
        """
        self._validate_tensor(updates, "updates")
        self._validate_tensor(memory_slots, "memory_slots")

        if updates.shape != memory_slots.shape:
            raise ValueError(
                "updates and memory_slots must have identical shapes."
            )

        original = updates

        if self.mode == "none":
            projected = updates

        elif self.mode == "slot":
            projected = self._project_from_own_slot(
                updates,
                memory_slots,
            )

        elif self.mode == "pairwise":
            projected = self._pairwise_gram_schmidt(updates)

        else:
            if external_basis is not None:
                basis = self._prepare_external_basis(
                    external_basis,
                    batch_size=updates.size(0),
                )
            elif self.mode == "all_slots":
                basis = memory_slots
            elif self.mode == "other_slots":
                projected = self._project_from_other_slots(
                    updates,
                    memory_slots,
                )
                basis = None
            elif self.mode == "learned_basis":
                basis = self.learned_basis.unsqueeze(0).expand(
                    updates.size(0),
                    -1,
                    -1,
                )
            else:
                raise RuntimeError(f"Unsupported mode: {self.mode}")

            if basis is not None:
                if self.detach_basis:
                    basis = basis.detach()
                projected = self._project_from_basis(updates, basis)

        mixed = original + self.strength * (projected - original)

        if self.preserve_norm:
            mixed = self._restore_norm(mixed, original)

        removed = original - mixed
        original_norm = original.norm(dim=-1).clamp_min(self.eps)
        removed_norm = removed.norm(dim=-1)
        ratio = removed_norm / original_norm

        return OrthogonalUpdateOutput(
            updates=mixed,
            removed_component=removed,
            projection_ratio=ratio,
        )

    def _project_from_own_slot(
        self,
        updates: Tensor,
        memory_slots: Tensor,
    ) -> Tensor:
        """
        Remove the component of each update parallel to its own memory slot.
        """
        slot_direction = F.normalize(
            memory_slots,
            p=2,
            dim=-1,
            eps=self.eps,
        )
        parallel = (
            updates * slot_direction
        ).sum(dim=-1, keepdim=True) * slot_direction

        return updates - parallel

    def _project_from_other_slots(
        self,
        updates: Tensor,
        memory_slots: Tensor,
    ) -> Tensor:
        """
        For every slot i, project update_i away from the span of all slots j != i.
        """
        batch_size, num_slots, _ = updates.shape
        results = []

        for slot_index in range(num_slots):
            if num_slots == 1:
                results.append(updates[:, slot_index])
                continue

            before = memory_slots[:, :slot_index]
            after = memory_slots[:, slot_index + 1 :]
            basis = torch.cat([before, after], dim=1)

            if self.detach_basis:
                basis = basis.detach()

            projected = self._project_vectors_from_basis(
                updates[:, slot_index],
                basis,
            )
            results.append(projected)

        return torch.stack(results, dim=1)

    def _project_from_basis(
        self,
        updates: Tensor,
        basis: Tensor,
    ) -> Tensor:
        """
        Project all slot updates away from a shared per-batch basis.
        """
        batch_size, num_slots, _ = updates.shape
        flat = updates.reshape(
            batch_size * num_slots,
            self.d_model,
        )

        expanded_basis = basis.unsqueeze(1).expand(
            -1,
            num_slots,
            -1,
            -1,
        ).reshape(
            batch_size * num_slots,
            basis.size(1),
            self.d_model,
        )

        projected = self._project_vectors_from_basis(
            flat,
            expanded_basis,
        )

        return projected.reshape(
            batch_size,
            num_slots,
            self.d_model,
        )

    def _project_vectors_from_basis(
        self,
        vectors: Tensor,
        basis: Tensor,
    ) -> Tensor:
        """
        Project vectors away from the row-space of a basis.

        vectors: [B, D]
        basis:   [B, K, D]
        """
        if basis.size(1) == 0:
            return vectors

        # QR is applied to basis^T, giving orthonormal columns spanning the
        # basis row-space.
        basis_t = basis.transpose(1, 2)  # [B, D, K]
        q, _ = torch.linalg.qr(basis_t, mode="reduced")  # [B, D, R]

        coefficients = torch.bmm(
            q.transpose(1, 2),
            vectors.unsqueeze(-1),
        )
        projection = torch.bmm(q, coefficients).squeeze(-1)

        return vectors - projection

    def _pairwise_gram_schmidt(self, updates: Tensor) -> Tensor:
        """
        Sequentially orthogonalize slot updates within each sample.

        Slot order matters in this mode. It is therefore best used as an
        ablation rather than the default final architecture.
        """
        orthogonal_vectors = []

        for slot_index in range(self.num_slots):
            vector = updates[:, slot_index]

            for previous in orthogonal_vectors:
                direction = F.normalize(
                    previous,
                    p=2,
                    dim=-1,
                    eps=self.eps,
                )
                parallel = (
                    vector * direction
                ).sum(dim=-1, keepdim=True) * direction
                vector = vector - parallel

            orthogonal_vectors.append(vector)

        return torch.stack(orthogonal_vectors, dim=1)

    def _restore_norm(
        self,
        projected: Tensor,
        original: Tensor,
    ) -> Tensor:
        original_norm = original.norm(
            dim=-1,
            keepdim=True,
        )
        projected_norm = projected.norm(
            dim=-1,
            keepdim=True,
        )

        safe_scale = original_norm / projected_norm.clamp_min(self.eps)

        restored = projected * safe_scale

        # If a vector becomes numerically zero, keep it zero instead of
        # amplifying noise.
        valid = projected_norm > self.eps
        return torch.where(valid, restored, projected)

    @torch.no_grad()
    def diagnostics(
        self,
        output: OrthogonalUpdateOutput | Tensor,
        memory_slots: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute update-geometry diagnostics.
        """
        if isinstance(output, OrthogonalUpdateOutput):
            updates = output.updates
            projection_ratio = output.projection_ratio
        else:
            updates = output
            self._validate_tensor(updates, "updates")
            projection_ratio = updates.new_zeros(
                updates.shape[:2]
            )

        normalized = F.normalize(
            updates.float(),
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
            device=updates.device,
            dtype=gram.dtype,
        ).unsqueeze(0)

        denominator = max(
            updates.size(0)
            * self.num_slots
            * (self.num_slots - 1),
            1,
        )

        pairwise_cosine = (
            gram * (1.0 - identity)
        ).sum() / denominator

        result = {
            "update_pairwise_cosine": pairwise_cosine,
            "update_norm_mean": updates.float().norm(dim=-1).mean(),
            "update_norm_std": updates.float().norm(
                dim=-1
            ).std(unbiased=False),
            "projection_ratio_mean": projection_ratio.float().mean(),
            "projection_ratio_max": projection_ratio.float().max(),
        }

        if memory_slots is not None:
            self._validate_tensor(memory_slots, "memory_slots")
            update_norm = F.normalize(
                updates.float(),
                p=2,
                dim=-1,
                eps=self.eps,
            )
            memory_norm = F.normalize(
                memory_slots.float(),
                p=2,
                dim=-1,
                eps=self.eps,
            )
            alignment = (
                update_norm * memory_norm
            ).sum(dim=-1).abs().mean()
            result["absolute_slot_alignment"] = alignment

        return result

    def orthogonality_loss(self, updates: Tensor) -> Tensor:
        """
        Penalize cosine similarity between update directions.

        The target Gram matrix is the identity.
        """
        self._validate_tensor(updates, "updates")

        normalized = F.normalize(
            updates,
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
            device=updates.device,
            dtype=updates.dtype,
        ).unsqueeze(0)

        return (gram - identity).pow(2).mean()

    def subspace_overlap_loss(
        self,
        updates: Tensor,
        memory_slots: Tensor,
    ) -> Tensor:
        """
        Penalize overlap between update directions and the current memory span.
        """
        self._validate_tensor(updates, "updates")
        self._validate_tensor(memory_slots, "memory_slots")

        q, _ = torch.linalg.qr(
            memory_slots.transpose(1, 2),
            mode="reduced",
        )

        coefficients = torch.bmm(
            updates,
            q,
        )

        return coefficients.pow(2).mean()

    def learned_basis_orthogonality_loss(self) -> Tensor:
        """
        Keep learned interference-basis vectors mutually orthogonal.
        """
        normalized = F.normalize(
            self.learned_basis,
            p=2,
            dim=-1,
            eps=self.eps,
        )

        gram = normalized @ normalized.transpose(0, 1)
        identity = torch.eye(
            self.learned_rank,
            device=gram.device,
            dtype=gram.dtype,
        )

        return (gram - identity).pow(2).mean()

    def _prepare_external_basis(
        self,
        basis: Tensor,
        batch_size: int,
    ) -> Tensor:
        if not torch.is_tensor(basis):
            raise TypeError("external_basis must be a torch.Tensor.")

        if basis.dim() == 2:
            if basis.size(-1) != self.d_model:
                raise ValueError(
                    f"external_basis final dimension must be {self.d_model}."
                )
            basis = basis.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )

        elif basis.dim() == 3:
            if basis.size(0) != batch_size:
                raise ValueError(
                    "external_basis batch size must match updates."
                )
            if basis.size(-1) != self.d_model:
                raise ValueError(
                    f"external_basis final dimension must be {self.d_model}."
                )

        else:
            raise ValueError(
                "external_basis must have shape [K, D] or [B, K, D]."
            )

        return basis

    def _validate_tensor(
        self,
        tensor: Tensor,
        name: str,
    ) -> None:
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")

        expected_tail = (self.num_slots, self.d_model)

        if tensor.dim() != 3 or tensor.shape[1:] != expected_tail:
            raise ValueError(
                f"{name} must have shape [B, {self.num_slots}, "
                f"{self.d_model}], got {tuple(tensor.shape)}."
            )


def _smoke_test() -> None:
    """Run with: python models/orthogonal_update.py"""
    torch.manual_seed(42)

    batch_size = 4
    num_slots = 8
    d_model = 64

    memory = torch.randn(
        batch_size,
        num_slots,
        d_model,
    )

    # Deliberately correlated updates.
    shared = torch.randn(batch_size, 1, d_model)
    updates = shared.expand(
        -1,
        num_slots,
        -1,
    ).clone()
    updates = updates + 0.01 * torch.randn_like(updates)

    for mode in (
        "slot",
        "other_slots",
        "all_slots",
        "pairwise",
        "learned_basis",
    ):
        print(f"\n=== {mode} projection ===")

        module = OrthogonalUpdate(
            d_model=d_model,
            num_slots=num_slots,
            mode=mode,
            preserve_norm=True,
            strength=1.0,
        )

        output = module(
            updates=updates,
            memory_slots=memory,
        )
        metrics = module.diagnostics(
            output,
            memory_slots=memory,
        )

        assert output.updates.shape == updates.shape
        assert output.removed_component.shape == updates.shape
        assert output.projection_ratio.shape == (
            batch_size,
            num_slots,
        )

        print("Output shape:", tuple(output.updates.shape))
        print(
            "Metrics:",
            {
                key: round(float(value.item()), 6)
                for key, value in metrics.items()
            },
        )


if __name__ == "__main__":
    _smoke_test()
