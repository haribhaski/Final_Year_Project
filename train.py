from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

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
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def get_autocast_context(
    device: torch.device,
    enabled: bool,
):
    if device.type == "cuda" and enabled:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return nullcontext()


def detach_memory_state(memory_state: Any) -> Any:
    """
    Detach the recurrent memory state from the previous computation graph.

    The custom MemoryState class is expected to provide detach().
    This fallback also supports plain tensors.
    """

    if memory_state is None:
        return None

    if hasattr(memory_state, "detach"):
        return memory_state.detach()

    if torch.is_tensor(memory_state):
        return memory_state.detach()

    raise TypeError(
        f"Cannot detach memory state of type {type(memory_state)}"
    )


def scalar_value(value: Any) -> float:
    if value is None:
        return 0.0

    if torch.is_tensor(value):
        return float(value.detach().cpu())

    return float(value)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def format_duration(seconds: float) -> str:
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


# ---------------------------------------------------------------------
# Model trainability
# ---------------------------------------------------------------------

def configure_trainable_parameters(
    model: MemoryAugmentedGPT2LMHeadModel,
    unfreeze_last_n: int,
) -> None:
    """
    Phase A:
        unfreeze_last_n = 0
        Freeze pretrained GPT-2 and train only memory modules.

    Phase B:
        unfreeze_last_n > 0
        Train memory modules and the final N GPT-2 blocks.
    """

    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
    else:
        # Fallback in case the wrapper does not expose freeze_backbone().
        for name, parameter in model.named_parameters():
            if not is_memory_parameter(name):
                parameter.requires_grad = False

    if unfreeze_last_n > 0:
        if hasattr(model, "set_trainable_backbone_layers"):
            model.set_trainable_backbone_layers(
                num_unfrozen_final_blocks=unfreeze_last_n
            )
        else:
            raise AttributeError(
                "The model does not provide "
                "set_trainable_backbone_layers(). "
                "Use --unfreeze-last-n 0 for memory-only training."
            )


def is_memory_parameter(parameter_name: str) -> bool:
    memory_keywords = (
        "memory",
        "reader",
        "writer",
        "write_",
        "confidence",
        "router",
        "gate",
        "orthogonal",
        "candidate",
        "slot",
    )

    lowered_name = parameter_name.lower()

    return any(
        keyword in lowered_name
        for keyword in memory_keywords
    )


def build_optimizer(
    model: torch.nn.Module,
    memory_learning_rate: float,
    backbone_learning_rate: float,
    weight_decay: float,
) -> AdamW:
    memory_parameters = []
    backbone_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if is_memory_parameter(name):
            memory_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    parameter_groups = []

    if memory_parameters:
        parameter_groups.append(
            {
                "params": memory_parameters,
                "lr": memory_learning_rate,
                "weight_decay": weight_decay,
                "group_name": "memory",
            }
        )

    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": backbone_learning_rate,
                "weight_decay": weight_decay,
                "group_name": "backbone",
            }
        )

    if not parameter_groups:
        raise RuntimeError(
            "No trainable parameters were found."
        )

    print(
        f"Memory parameter tensors: {len(memory_parameters)}"
    )
    print(
        f"Backbone parameter tensors: {len(backbone_parameters)}"
    )

    return AdamW(parameter_groups)


# ---------------------------------------------------------------------
# Checkpoint handling
# ---------------------------------------------------------------------

def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    args: argparse.Namespace,
) -> None:
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "scaler_state_dict": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
        "arguments": vars(args),
    }

    torch.save(payload, checkpoint_path)

    print(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and checkpoint.get("optimizer_state_dict") is not None
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if (
        scheduler is not None
        and checkpoint.get("scheduler_state_dict") is not None
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    if (
        scaler is not None
        and checkpoint.get("scaler_state_dict") is not None
    ):
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_validation_loss = float(
        checkpoint.get(
            "best_validation_loss",
            float("inf"),
        )
    )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Resuming from epoch: {start_epoch}")
    print(f"Resuming from global step: {global_step}")

    return (
        start_epoch,
        global_step,
        best_validation_loss,
    )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: MemoryAugmentedGPT2LMHeadModel,
    validation_loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    use_amp: bool,
    max_documents: int | None,
) -> dict[str, float]:
    model.eval()

    total_lm_loss = 0.0
    total_auxiliary_loss = 0.0
    total_tokens = 0
    total_chunks = 0
    total_documents = 0

    for document_index, document in enumerate(validation_loader):
        if (
            max_documents is not None
            and document_index >= max_documents
        ):
            break

        memory_state = None
        document_had_chunk = False

        for chunk in document["chunks"]:
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
                    return_diagnostics=False,
                )

            memory_state = detach_memory_state(
                output.memory_state
            )

            sequence_tokens = int(
                batch["attention_mask"].sum().item()
            )

            total_lm_loss += (
                scalar_value(output.lm_loss)
                * sequence_tokens
            )

            total_auxiliary_loss += (
                scalar_value(output.auxiliary_loss)
                * sequence_tokens
            )

            total_tokens += sequence_tokens
            total_chunks += 1
            document_had_chunk = True

        if document_had_chunk:
            total_documents += 1

    if total_tokens == 0:
        raise RuntimeError(
            "Validation processed zero tokens."
        )

    mean_lm_loss = total_lm_loss / total_tokens
    mean_auxiliary_loss = (
        total_auxiliary_loss / total_tokens
    )

    # Avoid overflow if the model is initially unstable.
    perplexity = math.exp(
        min(mean_lm_loss, 20.0)
    )

    return {
        "lm_loss": mean_lm_loss,
        "auxiliary_loss": mean_auxiliary_loss,
        "perplexity": perplexity,
        "documents": float(total_documents),
        "chunks": float(total_chunks),
        "tokens": float(total_tokens),
    }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_epoch(
    model: MemoryAugmentedGPT2LMHeadModel,
    train_loader: DataLoader,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> tuple[int, dict[str, float]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    epoch_start_time = time.time()

    total_lm_loss = 0.0
    total_auxiliary_loss = 0.0
    total_training_loss = 0.0
    total_tokens = 0
    total_chunks = 0
    total_documents = 0

    running_lm_loss = 0.0
    running_auxiliary_loss = 0.0
    running_training_loss = 0.0
    running_chunks = 0

    for document_index, document in enumerate(train_loader):
        if (
            args.max_train_documents is not None
            and document_index >= args.max_train_documents
        ):
            break

        memory_state = None
        document_chunks = document["chunks"]

        window_loss = None
        chunks_in_window = 0
        document_had_chunk = False

        for chunk_index, chunk in enumerate(document_chunks):
            batch = prepare_chunk(
                chunk=chunk,
                tokenizer=tokenizer,
                device=device,
            )

            with get_autocast_context(
                device=device,
                enabled=args.use_amp,
            ):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    memory_state=memory_state,
                    update_memory=True,
                    return_diagnostics=(
                        global_step % args.log_every == 0
                    ),
                )

                if output.loss is None:
                    raise RuntimeError(
                        "The model returned loss=None. "
                        "Ensure labels are supplied."
                    )

                current_loss = output.loss

            memory_state = output.memory_state

            if window_loss is None:
                window_loss = current_loss
            else:
                window_loss = window_loss + current_loss

            chunks_in_window += 1
            document_had_chunk = True

            lm_loss_value = scalar_value(output.lm_loss)
            auxiliary_loss_value = scalar_value(
                output.auxiliary_loss
            )
            training_loss_value = scalar_value(
                output.loss
            )

            sequence_tokens = int(
                batch["attention_mask"].sum().item()
            )

            total_lm_loss += (
                lm_loss_value * sequence_tokens
            )
            total_auxiliary_loss += (
                auxiliary_loss_value * sequence_tokens
            )
            total_training_loss += (
                training_loss_value * sequence_tokens
            )
            total_tokens += sequence_tokens
            total_chunks += 1

            running_lm_loss += lm_loss_value
            running_auxiliary_loss += auxiliary_loss_value
            running_training_loss += training_loss_value
            running_chunks += 1

            end_of_bptt_window = (
                chunks_in_window >= args.bptt_chunks
            )

            end_of_document = (
                chunk_index == len(document_chunks) - 1
            )

            if end_of_bptt_window or end_of_document:
                # Average loss over chunks in this truncated BPTT window.
                loss_for_backward = (
                    window_loss / chunks_in_window
                )

                if scaler is not None and scaler.is_enabled():
                    scaler.scale(
                        loss_for_backward
                    ).backward()

                    scaler.unscale_(optimizer)

                    gradient_norm = clip_grad_norm_(
                        model.parameters(),
                        max_norm=args.max_grad_norm,
                    )

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_for_backward.backward()

                    gradient_norm = clip_grad_norm_(
                        model.parameters(),
                        max_norm=args.max_grad_norm,
                    )

                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

                global_step += 1

                # Critical: break the computation graph after each
                # optimizer update while retaining memory contents.
                memory_state = detach_memory_state(
                    memory_state
                )

                window_loss = None
                chunks_in_window = 0

                if global_step % args.log_every == 0:
                    elapsed = (
                        time.time() - epoch_start_time
                    )

                    average_running_loss = (
                        running_training_loss
                        / max(running_chunks, 1)
                    )

                    average_running_lm_loss = (
                        running_lm_loss
                        / max(running_chunks, 1)
                    )

                    average_running_auxiliary = (
                        running_auxiliary_loss
                        / max(running_chunks, 1)
                    )

                    current_learning_rates = [
                        group["lr"]
                        for group in optimizer.param_groups
                    ]

                    print(
                        f"Epoch {epoch:02d} | "
                        f"Step {global_step:06d} | "
                        f"Document {document_index + 1:05d} | "
                        f"Chunk {chunk_index + 1:04d}/"
                        f"{len(document_chunks):04d} | "
                        f"Loss {average_running_loss:.4f} | "
                        f"LM {average_running_lm_loss:.4f} | "
                        f"Aux {average_running_auxiliary:.6f} | "
                        f"Grad {scalar_value(gradient_norm):.4f} | "
                        f"LR {current_learning_rates} | "
                        f"Time {format_duration(elapsed)}"
                    )

                    diagnostics = getattr(
                        output,
                        "diagnostics",
                        None,
                    )

                    if diagnostics:
                        diagnostic_items = []

                        for key in (
                            "memory/effective_rank",
                            "memory/stable_rank",
                            "memory/pairwise_cosine",
                            "memory/unused_slot_fraction",
                            "gate/mean",
                            "gate/within_sample_slot_variance",
                            "router/routing_entropy",
                            "router/unused_slot_fraction",
                            "reader/attention_entropy",
                        ):
                            if key in diagnostics:
                                diagnostic_items.append(
                                    f"{key}="
                                    f"{scalar_value(diagnostics[key]):.4f}"
                                )

                        if diagnostic_items:
                            print(
                                "  Diagnostics: "
                                + " | ".join(diagnostic_items)
                            )

                    running_lm_loss = 0.0
                    running_auxiliary_loss = 0.0
                    running_training_loss = 0.0
                    running_chunks = 0

                if (
                    args.max_steps is not None
                    and global_step >= args.max_steps
                ):
                    break

        if document_had_chunk:
            total_documents += 1

        if (
            args.max_steps is not None
            and global_step >= args.max_steps
        ):
            break

    if total_tokens == 0:
        raise RuntimeError(
            "Training processed zero tokens."
        )

    metrics = {
        "loss": total_training_loss / total_tokens,
        "lm_loss": total_lm_loss / total_tokens,
        "auxiliary_loss": (
            total_auxiliary_loss / total_tokens
        ),
        "perplexity": math.exp(
            min(total_lm_loss / total_tokens, 20.0)
        ),
        "documents": float(total_documents),
        "chunks": float(total_chunks),
        "tokens": float(total_tokens),
        "duration_seconds": (
            time.time() - epoch_start_time
        ),
    }

    return global_step, metrics


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train memory-augmented GPT-2 on WikiText-103."
        )
    )
    parser.add_argument(
        "--resume-weights-only",
        action="store_true",
        help=(
            "Load only model weights from --resume. "
            "Do not restore optimizer, scheduler, scaler, "
            "epoch, or global step."
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/wikitext-103",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/memory_gpt2",
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
        "--max-train-documents",
        type=int,
        default=100,
        help=(
            "Use 100 for the first real run. "
            "Pass -1 to use the complete training set."
        ),
    )
    parser.add_argument(
        "--max-validation-documents",
        type=int,
        default=25,
        help="Pass -1 to use the full validation set.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--bptt-chunks",
        type=int,
        default=2,
        help=(
            "Number of sequential chunks through which gradients "
            "are propagated before detaching memory."
        ),
    )
    parser.add_argument(
        "--memory-learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=0,
        help=(
            "0 trains only memory modules. "
            "Use 2 later for partial GPT-2 fine-tuning."
        ),
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Optional debugging limit on optimizer steps."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
    )
    # -------------------------------------------------------------
    # Memory architecture / ablation controls
    # -------------------------------------------------------------

    parser.add_argument(
        "--gate-type",
        type=str,
        choices=["scalar", "vector"],
        default="vector",
        help="Write gate type.",
    )

    parser.add_argument(
        "--gate-mode",
        type=str,
        choices=["sigmoid", "softmax", "gumbel_softmax"],
        default="sigmoid",
    )

    parser.add_argument(
        "--disable-router",
        action="store_true",
        help="Disable sparse slot routing.",
    )

    parser.add_argument(
        "--router-top-k",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--router-temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--orthogonal-mode",
        type=str,
        choices=[
            "none",
            "slot",
            "other_slots",
            "all_slots",
            "pairwise",
            "learned_basis",
        ],
        default="other_slots",
    )

    parser.add_argument(
        "--orthogonal-strength",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--candidate-diversity-weight",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--update-orthogonality-weight",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--router-balance-weight",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--reader-balance-weight",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--memory-collapse-weight",
        type=float,
        default=0.01,
    )

    args = parser.parse_args()

    if args.max_train_documents == -1:
        args.max_train_documents = None

    if args.max_validation_documents == -1:
        args.max_validation_documents = None

    args.use_amp = (
        torch.cuda.is_available()
        and not args.disable_amp
    )

    return args


def main() -> None:
    args = parse_arguments()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_dir / "training_arguments.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            vars(args),
            file,
            indent=2,
            default=str,
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("Memory-Augmented GPT-2 Training")
    print("=" * 80)
    print("Device:", device)
    print("Mixed precision:", args.use_amp)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )
        print(
            "CUDA version:",
            torch.version.cuda,
        )
    else:
        print(
            "WARNING: CUDA is unavailable. "
            "GPT-2 training on CPU will be very slow."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.save_pretrained(
        output_dir / "tokenizer"
    )

    print("\nLoading WikiText-103 datasets...")

    train_dataset = WikiText103DocumentDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        split="train",
        chunk_size=args.chunk_size,
        min_document_tokens=args.min_document_tokens,
        max_documents=args.max_train_documents,
    )

    validation_dataset = WikiText103DocumentDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        split="validation",
        chunk_size=args.chunk_size,
        min_document_tokens=args.min_document_tokens,
        max_documents=args.max_validation_documents,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=document_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=document_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    memory_config = MemoryGPT2Config(
        num_slots=8,

        gate_type=args.gate_type,
        gate_mode=args.gate_mode,
        gate_init_bias=-2.0,

        router_enabled=not args.disable_router,
        router_mode="softmax",
        router_top_k=(
            None
            if args.disable_router
            else args.router_top_k
        ),
        router_temperature=args.router_temperature,

        writer_mode="attention",
        writer_attention_heads=8,

        orthogonal_mode=args.orthogonal_mode,
        orthogonal_strength=args.orthogonal_strength,

        reader_mode="hybrid",
        reader_fusion="gated",
        reader_heads=8,
        reader_top_k=3,
        reader_temperature=0.8,

        candidate_diversity_weight=(
            args.candidate_diversity_weight
        ),
        update_orthogonality_weight=(
            args.update_orthogonality_weight
        ),
        router_balance_weight=(
            args.router_balance_weight
        ),
        reader_balance_weight=(
            args.reader_balance_weight
        ),
        memory_collapse_weight=(
            args.memory_collapse_weight
        ),
    )
    print("\nExperiment configuration")
    print("Gate type:", args.gate_type)
    print("Gate mode:", args.gate_mode)
    print("Router enabled:", not args.disable_router)
    print("Router top-k:", args.router_top_k)
    print("Orthogonal mode:", args.orthogonal_mode)
    print(
        "Candidate diversity weight:",
        args.candidate_diversity_weight,
    )
    print(
        "Update orthogonality weight:",
        args.update_orthogonality_weight,
    )
    print(
        "Memory collapse weight:",
        args.memory_collapse_weight,
    )
    print("\nLoading pretrained GPT-2...")

    model = (
        MemoryAugmentedGPT2LMHeadModel
        .from_pretrained(
            args.model_name,
            memory_config=memory_config,
        )
    )

    configure_trainable_parameters(
        model=model,
        unfreeze_last_n=args.unfreeze_last_n,
    )

    model.to(device)

    total_parameters, trainable_parameters = (
        count_parameters(model)
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )
    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )
    print(
        f"Trainable percentage: "
        f"{100.0 * trainable_parameters / total_parameters:.2f}%"
    )

    optimizer = build_optimizer(
        model=model,
        memory_learning_rate=args.memory_learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        weight_decay=args.weight_decay,
    )

    estimated_windows_per_epoch = 0

    for document in train_dataset.documents:
        number_of_chunks = len(document["chunks"])

        estimated_windows_per_epoch += math.ceil(
            number_of_chunks / args.bptt_chunks
        )

    total_training_steps = (
        estimated_windows_per_epoch
        * args.epochs
    )

    if args.max_steps is not None:
        total_training_steps = min(
            total_training_steps,
            args.max_steps,
        )

    warmup_steps = int(
        total_training_steps
        * args.warmup_ratio
    )

    print(
        f"Estimated optimizer steps: "
        f"{total_training_steps}"
    )
    print(
        f"Warmup steps: {warmup_steps}"
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(
            total_training_steps,
            1,
        ),
    )

    scaler = torch.amp.GradScaler(
    "cuda",
    enabled=args.use_amp,
)

    start_epoch = 1
    global_step = 0
    best_validation_loss = float("inf")

    if args.resume is not None:
        if args.resume_weights_only:
            checkpoint = torch.load(
                Path(args.resume),
                map_location=device,
            )

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            start_epoch = 1
            global_step = 0
            best_validation_loss = float("inf")

            print(
                f"Loaded model weights only from: {args.resume}"
            )
            print(
                "Optimizer, scheduler, scaler, epoch, "
                "and global step were reset."
            )

        else:
            (
                start_epoch,
                global_step,
                best_validation_loss,
            ) = load_checkpoint(
                checkpoint_path=Path(args.resume),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
            )

    print("\nRunning validation before training...")

    initial_validation_metrics = validate(
        model=model,
        validation_loader=validation_loader,
        tokenizer=tokenizer,
        device=device,
        use_amp=args.use_amp,
        max_documents=args.max_validation_documents,
    )

    print(
        f"Initial validation LM loss: "
        f"{initial_validation_metrics['lm_loss']:.4f}"
    )
    print(
        f"Initial validation perplexity: "
        f"{initial_validation_metrics['perplexity']:.4f}"
    )

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 80)

        global_step, training_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            global_step=global_step,
            args=args,
        )

        print("\nTraining summary")
        print(
            f"Loss: "
            f"{training_metrics['loss']:.4f}"
        )
        print(
            f"LM loss: "
            f"{training_metrics['lm_loss']:.4f}"
        )
        print(
            f"Auxiliary loss: "
            f"{training_metrics['auxiliary_loss']:.6f}"
        )
        print(
            f"Perplexity: "
            f"{training_metrics['perplexity']:.4f}"
        )
        print(
            f"Documents: "
            f"{int(training_metrics['documents'])}"
        )
        print(
            f"Chunks: "
            f"{int(training_metrics['chunks'])}"
        )
        print(
            f"Duration: "
            f"{format_duration(training_metrics['duration_seconds'])}"
        )

        validation_metrics = validate(
            model=model,
            validation_loader=validation_loader,
            tokenizer=tokenizer,
            device=device,
            use_amp=args.use_amp,
            max_documents=args.max_validation_documents,
        )

        print("\nValidation summary")
        print(
            f"LM loss: "
            f"{validation_metrics['lm_loss']:.4f}"
        )
        print(
            f"Auxiliary loss: "
            f"{validation_metrics['auxiliary_loss']:.6f}"
        )
        print(
            f"Perplexity: "
            f"{validation_metrics['perplexity']:.4f}"
        )

        latest_checkpoint_path = (
            output_dir / "checkpoint_latest.pt"
        )

        save_checkpoint(
            checkpoint_path=latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            args=args,
        )

        if args.save_every_epoch:
            save_checkpoint(
                checkpoint_path=(
                    output_dir
                    / f"checkpoint_epoch_{epoch}.pt"
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                args=args,
            )

        if (
            validation_metrics["lm_loss"]
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_metrics["lm_loss"]
            )

            save_checkpoint(
                checkpoint_path=(
                    output_dir / "checkpoint_best.pt"
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                args=args,
            )

            print(
                "New best validation checkpoint saved."
            )

        metrics_record = {
            "epoch": epoch,
            "global_step": global_step,
            "training": training_metrics,
            "validation": validation_metrics,
            "best_validation_loss": (
                best_validation_loss
            ),
        }

        with (
            output_dir / "metrics.jsonl"
        ).open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(metrics_record)
                + "\n"
            )

        if (
            args.max_steps is not None
            and global_step >= args.max_steps
        ):
            print(
                f"Reached max_steps={args.max_steps}."
            )
            break

    print("\nTraining completed.")
    print(
        f"Best validation LM loss: "
        f"{best_validation_loss:.4f}"
    )
    print(
        f"Best validation perplexity: "
        f"{math.exp(min(best_validation_loss, 20.0)):.4f}"
    )
    print(
        f"Outputs saved to: "
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()