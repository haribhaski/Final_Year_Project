from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.dataset import (
    WikiText103DocumentDataset,
    document_collate_fn,
)
from data.preprocessing import prepare_chunk
from models.gpt2_memory import (
    MemoryAugmentedGPT2LMHeadModel,
    MemoryGPT2Config,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def scalar_value(value: Any) -> float:
    if value is None:
        return 0.0

    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())

    return float(value)


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")

    return sum(values) / len(values)


def get_autocast_context(
    device: torch.device,
    enabled: bool,
):
    if device.type == "cuda" and enabled:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    from contextlib import nullcontext

    return nullcontext()


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate memory-augmented GPT-2 checkpoints "
            "on WikiText-103."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint_best.pt or another checkpoint.",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/wikitext-103",
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=("validation", "test"),
        default="validation",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--min-document-tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help=(
            "Number of documents to evaluate. "
            "Use -1 for the complete split."
        ),
    )

    # -------------------------------------------------------------
    # Architecture arguments
    # -------------------------------------------------------------

    parser.add_argument(
        "--num-slots",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--gate-type",
        type=str,
        choices=("scalar", "vector"),
        default="vector",
    )

    parser.add_argument(
        "--gate-mode",
        type=str,
        choices=("sigmoid", "softmax", "gumbel_softmax"),
        default="sigmoid",
    )

    parser.add_argument(
        "--disable-router",
        action="store_true",
    )

    parser.add_argument(
        "--router-mode",
        type=str,
        default="softmax",
    )

    parser.add_argument(
        "--router-top-k",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--orthogonal-mode",
        type=str,
        default="other_slots",
    )

    parser.add_argument(
        "--orthogonal-strength",
        type=float,
        default=0.5,
    )

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
    )

    parser.add_argument(
        "--disable-amp",
        action="store_true",
    )

    args = parser.parse_args()

    if args.max_documents == -1:
        args.max_documents = None

    args.use_amp = (
        torch.cuda.is_available()
        and not args.disable_amp
    )

    return args


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------


def build_memory_config(
    args: argparse.Namespace,
) -> MemoryGPT2Config:
    """
    Build the architecture corresponding to the checkpoint.

    IMPORTANT:
    The architecture used here must match the architecture that created
    the checkpoint.
    """

    return MemoryGPT2Config(
        num_slots=args.num_slots,

        gate_type=args.gate_type,
        gate_mode=args.gate_mode,
        gate_init_bias=-2.0,

        router_enabled=not args.disable_router,
        router_mode=args.router_mode,
        router_top_k=(
            None
            if args.disable_router
            else args.router_top_k
        ),
        router_temperature=0.7,

        writer_mode="attention",
        writer_attention_heads=8,

        orthogonal_mode=args.orthogonal_mode,
        orthogonal_strength=args.orthogonal_strength,

        reader_mode="hybrid",
        reader_fusion="gated",
        reader_heads=8,
        reader_top_k=3,
        reader_temperature=0.8,

        # Loss coefficients do not affect inference.
        candidate_diversity_weight=0.0,
        update_orthogonality_weight=0.0,
        router_balance_weight=0.0,
        reader_balance_weight=0.0,
        head_diversity_weight=0.0,
        memory_collapse_weight=0.0,
        gate_sparsity_weight=0.0,
    )


def load_model(
    args: argparse.Namespace,
    device: torch.device,
) -> MemoryAugmentedGPT2LMHeadModel:

    memory_config = build_memory_config(args)

    print("Loading GPT-2 architecture...")

    model = (
        MemoryAugmentedGPT2LMHeadModel
        .from_pretrained(
            args.model_name,
            memory_config=memory_config,
        )
    )

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print(
        f"Loading checkpoint: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain model_state_dict."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch", "unknown"),
    )

    print(
        "Checkpoint global step:",
        checkpoint.get("global_step", "unknown"),
    )

    print(
        "Stored best validation loss:",
        checkpoint.get(
            "best_validation_loss",
            "unknown",
        ),
    )

    return model


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: MemoryAugmentedGPT2LMHeadModel,
    loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    use_amp: bool,
    max_documents: int | None,
) -> tuple[
    dict[str, float],
    list[dict[str, Any]],
]:

    total_lm_loss = 0.0
    total_auxiliary_loss = 0.0
    total_tokens = 0
    total_chunks = 0
    total_documents = 0

    # Chunk-level diagnostics.
    diagnostic_values: dict[
        str,
        list[float],
    ] = defaultdict(list)

    document_records: list[
        dict[str, Any]
    ] = []

    for document_index, document in enumerate(loader):

        if (
            max_documents is not None
            and document_index >= max_documents
        ):
            break

        memory_state = None

        document_lm_loss = 0.0
        document_tokens = 0
        document_chunks = 0

        # Important:
        # Memory starts fresh at the beginning of every document.
        for chunk_index, chunk in enumerate(
            document["chunks"]
        ):

            batch = prepare_chunk(
                chunk=chunk,
                tokenizer=tokenizer,
                device=device,
            )

            with get_autocast_context(
                device=device,
                enabled=use_amp,
            ):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    memory_state=memory_state,
                    update_memory=True,
                    return_diagnostics=True,
                )

            memory_state = (
                output.memory_state.detach()
            )

            sequence_tokens = int(
                batch[
                    "attention_mask"
                ].sum().item()
            )

            lm_loss = scalar_value(
                output.lm_loss
            )

            auxiliary_loss = scalar_value(
                output.auxiliary_loss
            )

            total_lm_loss += (
                lm_loss * sequence_tokens
            )

            total_auxiliary_loss += (
                auxiliary_loss
                * sequence_tokens
            )

            total_tokens += sequence_tokens
            total_chunks += 1

            document_lm_loss += (
                lm_loss * sequence_tokens
            )

            document_tokens += (
                sequence_tokens
            )

            document_chunks += 1

            diagnostics = output.diagnostics

            # -----------------------------------------------------
            # Chunk-level gate/router/read diagnostics
            # -----------------------------------------------------

            requested_chunk_metrics = (
                "gate/mean",
                "gate/std",
                "gate/within_sample_slot_variance",
                "gate/active_fraction",

                "router/routing_entropy",
                "router/normalized_routing_entropy",
                "router/active_slots_per_sample",
                "router/unused_slot_fraction",
                "router/slot_usage_variance",

                "reader/attention_entropy",
                "reader/normalized_attention_entropy",
                "reader/unused_slot_fraction",
                "reader/maximum_slot_usage",

                "writer/candidate_pairwise_cosine",
                "writer/candidate_effective_rank",

                "orthogonal/update_pairwise_cosine",
                "orthogonal/projection_ratio_mean",
            )

            for metric_name in requested_chunk_metrics:
                if metric_name in diagnostics:
                    diagnostic_values[
                        metric_name
                    ].append(
                        scalar_value(
                            diagnostics[
                                metric_name
                            ]
                        )
                    )

        # ---------------------------------------------------------
        # Final memory state of this document
        # ---------------------------------------------------------

        if (
            memory_state is None
            or document_tokens == 0
        ):
            continue

        total_documents += 1

        final_memory_metrics = (
            model.memory_bank
            .collapse_metrics(
                memory_state
            )
        )

        effective_rank = scalar_value(
            final_memory_metrics[
                "effective_rank"
            ]
        )

        stable_rank = scalar_value(
            final_memory_metrics[
                "stable_rank"
            ]
        )

        pairwise_cosine = scalar_value(
            final_memory_metrics[
                "pairwise_cosine"
            ]
        )

        unused_fraction = scalar_value(
            final_memory_metrics[
                "unused_slot_fraction"
            ]
        )

        mean_write_count = scalar_value(
            memory_state.write_count
            .float()
            .mean()
        )

        mean_read_count = scalar_value(
            memory_state.read_count
            .float()
            .mean()
        )

        mean_confidence = scalar_value(
            memory_state.confidence
            .float()
            .mean()
        )

        # Slot utilization:
        # fraction of slots written at least once.
        slot_utilization = (
            1.0 - unused_fraction
        )

        diagnostic_values[
            "memory/effective_rank"
        ].append(
            effective_rank
        )

        diagnostic_values[
            "memory/stable_rank"
        ].append(
            stable_rank
        )

        diagnostic_values[
            "memory/pairwise_cosine"
        ].append(
            pairwise_cosine
        )

        diagnostic_values[
            "memory/unused_slot_fraction"
        ].append(
            unused_fraction
        )

        diagnostic_values[
            "memory/slot_utilization"
        ].append(
            slot_utilization
        )

        diagnostic_values[
            "memory/mean_write_count"
        ].append(
            mean_write_count
        )

        diagnostic_values[
            "memory/mean_read_count"
        ].append(
            mean_read_count
        )

        diagnostic_values[
            "memory/mean_confidence"
        ].append(
            mean_confidence
        )

        document_mean_loss = (
            document_lm_loss
            / document_tokens
        )

        document_perplexity = math.exp(
            min(
                document_mean_loss,
                20.0,
            )
        )

        title = document.get(
            "title",
            f"document_{document_index}",
        )

        document_records.append(
            {
                "document_index": document_index,
                "title": title,
                "chunks": document_chunks,
                "tokens": document_tokens,
                "lm_loss": document_mean_loss,
                "perplexity": document_perplexity,
                "effective_rank": effective_rank,
                "stable_rank": stable_rank,
                "pairwise_cosine": pairwise_cosine,
                "slot_utilization": slot_utilization,
                "unused_slot_fraction": unused_fraction,
                "mean_write_count": mean_write_count,
                "mean_read_count": mean_read_count,
                "mean_confidence": mean_confidence,
            }
        )

        print(
            f"Document "
            f"{total_documents:04d} | "
            f"Chunks {document_chunks:03d} | "
            f"Loss {document_mean_loss:.4f} | "
            f"PPL {document_perplexity:.4f} | "
            f"EffRank {effective_rank:.3f} | "
            f"Cos {pairwise_cosine:.4f} | "
            f"Used {slot_utilization:.3f}"
        )

    if total_tokens == 0:
        raise RuntimeError(
            "Evaluation processed zero tokens."
        )

    mean_lm_loss = (
        total_lm_loss
        / total_tokens
    )

    summary: dict[str, float] = {
        "lm_loss": mean_lm_loss,
        "perplexity": math.exp(
            min(mean_lm_loss, 20.0)
        ),
        "auxiliary_loss": (
            total_auxiliary_loss
            / total_tokens
        ),
        "documents": float(
            total_documents
        ),
        "chunks": float(
            total_chunks
        ),
        "tokens": float(
            total_tokens
        ),
    }

    for metric_name, values in (
        diagnostic_values.items()
    ):
        summary[metric_name] = (
            mean(values)
        )

    return (
        summary,
        document_records,
    )


# ---------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------


def save_results(
    summary: dict[str, float],
    documents: list[dict[str, Any]],
    output_dir: Path,
    split: str,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # JSON summary
    # -------------------------------------------------------------

    json_path = (
        output_dir
        / f"{split}_summary.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    # -------------------------------------------------------------
    # CSV summary
    # -------------------------------------------------------------

    summary_csv_path = (
        output_dir
        / f"{split}_summary.csv"
    )

    with summary_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["metric", "value"]
        )

        for key, value in (
            summary.items()
        ):
            writer.writerow(
                [key, value]
            )

    # -------------------------------------------------------------
    # Per-document CSV
    # -------------------------------------------------------------

    documents_csv_path = (
        output_dir
        / f"{split}_documents.csv"
    )

    if documents:

        with documents_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    documents[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                documents
            )

    print("\nSaved:")
    print(json_path)
    print(summary_csv_path)

    if documents:
        print(documents_csv_path)


# ---------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------


def print_summary(
    summary: dict[str, float],
) -> None:

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    ordered_metrics = (
        ("LM Loss", "lm_loss"),
        ("Perplexity", "perplexity"),

        (
            "Effective Rank",
            "memory/effective_rank",
        ),
        (
            "Stable Rank",
            "memory/stable_rank",
        ),
        (
            "Pairwise Slot Cosine",
            "memory/pairwise_cosine",
        ),
        (
            "Slot Utilization",
            "memory/slot_utilization",
        ),
        (
            "Unused Slot Fraction",
            "memory/unused_slot_fraction",
        ),

        (
            "Gate Mean",
            "gate/mean",
        ),
        (
            "Gate Slot Variance",
            "gate/within_sample_slot_variance",
        ),

        (
            "Routing Entropy",
            "router/routing_entropy",
        ),
        (
            "Normalized Routing Entropy",
            "router/normalized_routing_entropy",
        ),

        (
            "Mean Write Count",
            "memory/mean_write_count",
        ),
        (
            "Mean Read Count",
            "memory/mean_read_count",
        ),
    )

    for label, key in ordered_metrics:

        if key not in summary:
            continue

        print(
            f"{label:<32}: "
            f"{summary[key]:.6f}"
        )

    print(
        f"{'Documents':<32}: "
        f"{int(summary['documents'])}"
    )

    print(
        f"{'Chunks':<32}: "
        f"{int(summary['chunks'])}"
    )

    print(
        f"{'Tokens':<32}: "
        f"{int(summary['tokens'])}"
    )

    print("=" * 80)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    args = parse_arguments()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("Memory GPT-2 Evaluation")
    print("=" * 80)

    print("Device:", device)
    print("Split:", args.split)
    print("Checkpoint:", args.checkpoint)
    print("Gate:", args.gate_type)
    print(
        "Router:",
        not args.disable_router,
    )

    if device.type != "cuda":
        print(
            "NOTE: CUDA is unavailable. "
            "Evaluation will run on CPU."
        )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.model_name
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print(
        "\nLoading dataset..."
    )

    dataset = (
        WikiText103DocumentDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            split=args.split,
            chunk_size=args.chunk_size,
            min_document_tokens=(
                args.min_document_tokens
            ),
            max_documents=(
                args.max_documents
            ),
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=document_collate_fn,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print(
        "Documents loaded:",
        len(dataset),
    )

    model = load_model(
        args=args,
        device=device,
    )

    summary, documents = evaluate(
        model=model,
        loader=loader,
        tokenizer=tokenizer,
        device=device,
        use_amp=args.use_amp,
        max_documents=args.max_documents,
    )

    print_summary(
        summary
    )

    output_dir = Path(
        args.output_dir
    )

    save_results(
        summary=summary,
        documents=documents,
        output_dir=output_dir,
        split=args.split,
    )


if __name__ == "__main__":
    main()