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
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
)

from data.dataset import (
    WikiText103DocumentDataset,
    document_collate_fn,
)
from data.preprocessing import prepare_chunk


# =====================================================================
# Reproducibility
# =====================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# Utilities
# =====================================================================


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


def scalar_value(value: Any) -> float:
    if torch.is_tensor(value):
        return float(
            value.detach().float().cpu().item()
        )

    return float(value)


def count_parameters(
    model: torch.nn.Module,
) -> tuple[int, int]:

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def format_duration(
    seconds: float,
) -> str:

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"
    )


# =====================================================================
# GPT-2 trainability
# =====================================================================


def configure_partial_finetuning(
    model: GPT2LMHeadModel,
    unfreeze_last_n: int,
) -> None:
    """
    Freeze GPT-2 except:

    - final N transformer blocks
    - final LayerNorm
    - LM head

    This mirrors the backbone trainability strategy used by the
    memory-augmented model.
    """

    if unfreeze_last_n < 0:
        raise ValueError(
            "unfreeze_last_n cannot be negative."
        )

    blocks = model.transformer.h

    if unfreeze_last_n > len(blocks):
        raise ValueError(
            f"GPT-2 has only {len(blocks)} blocks."
        )

    # Freeze entire model first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Unfreeze final N transformer blocks.
    if unfreeze_last_n > 0:
        for block in blocks[-unfreeze_last_n:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    # Match MemoryAugmentedGPT2LMHeadModel's strategy.
    for parameter in (
        model.transformer.ln_f.parameters()
    ):
        parameter.requires_grad = True

    # GPT-2 ties lm_head.weight to transformer.wte.weight.
    # Making the LM head trainable therefore also makes the tied
    # token embedding matrix trainable.
    for parameter in model.lm_head.parameters():
        parameter.requires_grad = True


def print_trainable_components(
    model: GPT2LMHeadModel,
) -> None:

    print("\nTrainable GPT-2 components:")

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(
                f"  {name:<60} "
                f"{tuple(parameter.shape)}"
            )


# =====================================================================
# Causal language-model loss
# =====================================================================


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Use exactly the same next-token formulation used by the
    memory-augmented model.
    """

    shift_logits = (
        logits[..., :-1, :]
        .contiguous()
    )

    shift_labels = (
        labels[..., 1:]
        .contiguous()
    )

    return F.cross_entropy(
        shift_logits.view(
            -1,
            shift_logits.size(-1),
        ),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# =====================================================================
# Validation
# =====================================================================


@torch.no_grad()
def validate(
    model: GPT2LMHeadModel,
    validation_loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    use_amp: bool,
    max_documents: int | None,
) -> dict[str, float]:

    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_chunks = 0
    total_documents = 0

    for document_index, document in enumerate(
        validation_loader
    ):

        if (
            max_documents is not None
            and document_index >= max_documents
        ):
            break

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

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=(
                        batch["attention_mask"]
                    ),
                    use_cache=False,
                    return_dict=True,
                )

                loss = causal_lm_loss(
                    logits=outputs.logits,
                    labels=batch["labels"],
                )

            sequence_tokens = int(
                batch[
                    "attention_mask"
                ].sum().item()
            )

            total_loss += (
                scalar_value(loss)
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

    mean_loss = (
        total_loss / total_tokens
    )

    perplexity = math.exp(
        min(mean_loss, 20.0)
    )

    return {
        "lm_loss": mean_loss,
        "perplexity": perplexity,
        "documents": float(total_documents),
        "chunks": float(total_chunks),
        "tokens": float(total_tokens),
    }


# =====================================================================
# Training
# =====================================================================


def train_one_epoch(
    model: GPT2LMHeadModel,
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

    optimizer.zero_grad(
        set_to_none=True
    )

    epoch_start_time = time.time()

    total_loss = 0.0
    total_tokens = 0
    total_chunks = 0
    total_documents = 0

    running_loss = 0.0
    running_chunks = 0

    for document_index, document in enumerate(
        train_loader
    ):

        if (
            args.max_train_documents is not None
            and document_index
            >= args.max_train_documents
        ):
            break

        document_chunks = document["chunks"]

        # We retain the same update frequency as the memory model:
        # one optimizer step after every bptt_chunks chunks.
        window_loss = None
        chunks_in_window = 0

        document_had_chunk = False

        for chunk_index, chunk in enumerate(
            document_chunks
        ):

            batch = prepare_chunk(
                chunk=chunk,
                tokenizer=tokenizer,
                device=device,
            )

            with get_autocast_context(
                device=device,
                enabled=args.use_amp,
            ):

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=(
                        batch["attention_mask"]
                    ),
                    use_cache=False,
                    return_dict=True,
                )

                current_loss = causal_lm_loss(
                    logits=outputs.logits,
                    labels=batch["labels"],
                )

            if window_loss is None:
                window_loss = current_loss
            else:
                window_loss = (
                    window_loss
                    + current_loss
                )

            chunks_in_window += 1
            document_had_chunk = True

            sequence_tokens = int(
                batch[
                    "attention_mask"
                ].sum().item()
            )

            loss_value = scalar_value(
                current_loss
            )

            total_loss += (
                loss_value
                * sequence_tokens
            )

            total_tokens += sequence_tokens
            total_chunks += 1

            running_loss += loss_value
            running_chunks += 1

            end_of_window = (
                chunks_in_window
                >= args.bptt_chunks
            )

            end_of_document = (
                chunk_index
                == len(document_chunks) - 1
            )

            if (
                end_of_window
                or end_of_document
            ):

                # Match memory model:
                # average losses over chunks in window.
                loss_for_backward = (
                    window_loss
                    / chunks_in_window
                )

                if (
                    scaler is not None
                    and scaler.is_enabled()
                ):

                    scaler.scale(
                        loss_for_backward
                    ).backward()

                    scaler.unscale_(
                        optimizer
                    )

                    gradient_norm = (
                        clip_grad_norm_(
                            model.parameters(),
                            max_norm=(
                                args.max_grad_norm
                            ),
                        )
                    )

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    loss_for_backward.backward()

                    gradient_norm = (
                        clip_grad_norm_(
                            model.parameters(),
                            max_norm=(
                                args.max_grad_norm
                            ),
                        )
                    )

                    optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                if scheduler is not None:
                    scheduler.step()

                global_step += 1

                window_loss = None
                chunks_in_window = 0

                if (
                    global_step
                    % args.log_every
                    == 0
                ):

                    average_running_loss = (
                        running_loss
                        / max(
                            running_chunks,
                            1,
                        )
                    )

                    current_lr = (
                        optimizer
                        .param_groups[0]["lr"]
                    )

                    print(
                        f"Epoch {epoch:02d} | "
                        f"Step {global_step:06d} | "
                        f"Document "
                        f"{document_index + 1:05d} | "
                        f"Chunk "
                        f"{chunk_index + 1:04d}/"
                        f"{len(document_chunks):04d} | "
                        f"Loss "
                        f"{average_running_loss:.4f} | "
                        f"Grad "
                        f"{scalar_value(gradient_norm):.4f} | "
                        f"LR "
                        f"{current_lr:.8f} | "
                        f"Time "
                        f"{format_duration(time.time() - epoch_start_time)}"
                    )

                    running_loss = 0.0
                    running_chunks = 0

                if (
                    args.max_steps is not None
                    and global_step
                    >= args.max_steps
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

    mean_loss = (
        total_loss / total_tokens
    )

    return (
        global_step,
        {
            "lm_loss": mean_loss,
            "perplexity": math.exp(
                min(mean_loss, 20.0)
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
            "duration_seconds": (
                time.time()
                - epoch_start_time
            ),
        },
    )


# =====================================================================
# Checkpointing
# =====================================================================


def save_checkpoint(
    path: Path,
    model: GPT2LMHeadModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    args: argparse.Namespace,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "model_type": "plain_gpt2_baseline",
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_loss": (
            best_validation_loss
        ),
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
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

    torch.save(
        payload,
        path,
    )

    print(
        f"Saved checkpoint: {path}"
    )


# =====================================================================
# Arguments
# =====================================================================


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Fair plain GPT-2 baseline for "
            "memory-augmented GPT-2 experiments."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/wikitext-103",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "outputs/"
            "gpt2_baseline_partial_ft_1k"
        ),
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
        default=1000,
    )

    parser.add_argument(
        "--max-validation-documents",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--bptt-chunks",
        type=int,
        default=4,
        help=(
            "For the baseline this acts as gradient "
            "accumulation over sequential chunks, matching "
            "the memory model's optimizer-step frequency."
        ),
    )

    parser.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
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
        "--log-every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
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


# =====================================================================
# Main
# =====================================================================


def main() -> None:

    args = parse_arguments()

    set_seed(
        args.seed
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_dir
        / "training_arguments.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

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
    print("Plain GPT-2 Partial Fine-Tuning Baseline")
    print("=" * 80)

    print(
        "Device:",
        device,
    )

    print(
        "Mixed precision:",
        args.use_amp,
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
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

    tokenizer.save_pretrained(
        output_dir / "tokenizer"
    )

    print(
        "\nLoading WikiText-103..."
    )

    train_dataset = (
        WikiText103DocumentDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            split="train",
            chunk_size=args.chunk_size,
            min_document_tokens=(
                args.min_document_tokens
            ),
            max_documents=(
                args.max_train_documents
            ),
        )
    )

    validation_dataset = (
        WikiText103DocumentDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            split="validation",
            chunk_size=args.chunk_size,
            min_document_tokens=(
                args.min_document_tokens
            ),
            max_documents=(
                args.max_validation_documents
            ),
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=document_collate_fn,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=document_collate_fn,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print(
        "\nLoading pretrained GPT-2..."
    )

    model = (
        GPT2LMHeadModel
        .from_pretrained(
            args.model_name
        )
    )

    model.config.use_cache = False

    configure_partial_finetuning(
        model=model,
        unfreeze_last_n=(
            args.unfreeze_last_n
        ),
    )

    model.to(
        device
    )

    total_parameters, trainable_parameters = (
        count_parameters(
            model
        )
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

    # Uncomment temporarily if you want to inspect every
    # trainable tensor:
    #
    # print_trainable_components(model)

    optimizer_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        optimizer_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # -------------------------------------------------------------
    # Match number of optimizer updates used by memory model.
    # -------------------------------------------------------------

    estimated_windows_per_epoch = 0

    for document in train_dataset.documents:

        number_of_chunks = len(
            document["chunks"]
        )

        estimated_windows_per_epoch += (
            math.ceil(
                number_of_chunks
                / args.bptt_chunks
            )
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
        f"Warmup steps: "
        f"{warmup_steps}"
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=(
                warmup_steps
            ),
            num_training_steps=max(
                total_training_steps,
                1,
            ),
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.use_amp,
    )

    # -------------------------------------------------------------
    # Initial pretrained GPT-2 validation
    # -------------------------------------------------------------

    print(
        "\nRunning validation before fine-tuning..."
    )

    initial_metrics = validate(
        model=model,
        validation_loader=(
            validation_loader
        ),
        tokenizer=tokenizer,
        device=device,
        use_amp=args.use_amp,
        max_documents=(
            args.max_validation_documents
        ),
    )

    print(
        f"Initial validation LM loss: "
        f"{initial_metrics['lm_loss']:.4f}"
    )

    print(
        f"Initial validation perplexity: "
        f"{initial_metrics['perplexity']:.4f}"
    )

    # Save this too — useful for the pretrained baseline row.
    with (
        output_dir
        / "initial_validation.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            initial_metrics,
            file,
            indent=2,
        )

    global_step = 0
    best_validation_loss = float(
        "inf"
    )

    # -------------------------------------------------------------
    # Fine-tuning
    # -------------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"Epoch {epoch}/{args.epochs}"
        )

        print(
            "=" * 80
        )

        (
            global_step,
            training_metrics,
        ) = train_one_epoch(
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

        print(
            "\nTraining summary"
        )

        print(
            f"LM loss: "
            f"{training_metrics['lm_loss']:.4f}"
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
            validation_loader=(
                validation_loader
            ),
            tokenizer=tokenizer,
            device=device,
            use_amp=args.use_amp,
            max_documents=(
                args.max_validation_documents
            ),
        )

        print(
            "\nValidation summary"
        )

        print(
            f"LM loss: "
            f"{validation_metrics['lm_loss']:.4f}"
        )

        print(
            f"Perplexity: "
            f"{validation_metrics['perplexity']:.4f}"
        )

        save_checkpoint(
            path=(
                output_dir
                / "checkpoint_latest.pt"
            ),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_validation_loss=(
                best_validation_loss
            ),
            args=args,
        )

        if args.save_every_epoch:

            save_checkpoint(
                path=(
                    output_dir
                    / f"checkpoint_epoch_{epoch}.pt"
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_validation_loss=(
                    best_validation_loss
                ),
                args=args,
            )

        if (
            validation_metrics["lm_loss"]
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_metrics[
                    "lm_loss"
                ]
            )

            save_checkpoint(
                path=(
                    output_dir
                    / "checkpoint_best.pt"
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_validation_loss=(
                    best_validation_loss
                ),
                args=args,
            )

            print(
                "New best baseline checkpoint saved."
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
            output_dir
            / "metrics.jsonl"
        ).open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    metrics_record
                )
                + "\n"
            )

    print(
        "\nTraining completed."
    )

    print(
        f"Best validation LM loss: "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"Best validation perplexity: "
        f"{math.exp(best_validation_loss):.6f}"
    )

    print(
        f"Outputs saved to: "
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()