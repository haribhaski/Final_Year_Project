from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import AutoTokenizer

from models.gpt2_memory import (
    MemoryAugmentedGPT2LMHeadModel,
    MemoryGPT2Config,
)


# ================================================================
# Synthetic vocabulary
# ================================================================

ANSWERS = [
    "tiger",
    "apple",
    "blue",
    "horse",
    "green",
    "orange",
    "piano",
    "river",
    "chair",
    "lemon",
    "purple",
    "rabbit",
    "silver",
    "garden",
    "falcon",
    "banana",
]


DISTRACTOR_TEXTS = [
    (
        "Railway systems transformed transportation by allowing goods "
        "and passengers to move efficiently between distant locations."
    ),
    (
        "Photosynthesis allows plants to convert light energy into "
        "chemical energy using carbon dioxide and water."
    ),
    (
        "Computer networks exchange information using communication "
        "protocols and interconnected routing devices."
    ),
    (
        "Ocean currents transport heat around the planet and influence "
        "regional weather and long term climate patterns."
    ),
    (
        "Ancient civilizations developed agriculture, trade, writing, "
        "architecture, and increasingly complex systems of government."
    ),
    (
        "Machine learning models estimate patterns from data by adjusting "
        "parameters to reduce an objective function."
    ),
    (
        "Libraries preserve books, manuscripts, recordings, archives, "
        "and digital collections for future access."
    ),
    (
        "Volcanoes form when magma rises through weaknesses in the "
        "Earth's crust and reaches the surface."
    ),
]


# ================================================================
# Reproducibility
# ================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================================================================
# Synthetic examples
# ================================================================

def make_example(
    rng: random.Random,
    example_id: int,
    distances: list[int],
) -> dict[str, Any]:

    entity = f"Project-{example_id:06d}"

    answer = rng.choice(ANSWERS)

    distance = rng.choice(distances)

    fact = (
        f"The assigned keyword for {entity} is {answer}. "
        f"Remember that the keyword associated with {entity} is {answer}."
    )

    distractors = []

    for index in range(distance):
        distractor = DISTRACTOR_TEXTS[
            index % len(DISTRACTOR_TEXTS)
        ]

        distractors.append(
            f"Unrelated passage {index + 1}. {distractor}"
        )

    query = (
        f"The assigned keyword for {entity} is"
    )

    return {
        "id": example_id,
        "entity": entity,
        "answer": answer,
        "distance": distance,
        "fact": fact,
        "distractors": distractors,
        "query": query,
    }


def build_dataset(
    number_examples: int,
    distances: list[int],
    seed: int,
    start_id: int = 0,
) -> list[dict[str, Any]]:

    rng = random.Random(seed)

    return [
        make_example(
            rng=rng,
            example_id=start_id + index,
            distances=distances,
        )
        for index in range(number_examples)
    ]


# ================================================================
# Tokenization
# ================================================================

def tokenize_text(
    tokenizer: Any,
    text: str,
    device: torch.device,
    max_length: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:

    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    return (
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
    )


def prepare_query_with_answer(
    tokenizer: Any,
    query: str,
    answer: str,
    device: torch.device,
    max_length: int = 256,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    query_ids = tokenizer(
        query,
        add_special_tokens=False,
    )["input_ids"]

    # Leading space is important for GPT-2 tokenization.
    answer_ids = tokenizer(
        " " + answer,
        add_special_tokens=False,
    )["input_ids"]

    full_ids = query_ids + answer_ids

    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]

        answer_start = (
            len(full_ids)
            - len(answer_ids)
        )
    else:
        answer_start = len(query_ids)

    input_ids = torch.tensor(
        [full_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    labels = input_ids.clone()

    # Ignore the entire query.
    # Only answer tokens contribute to LM loss.
    labels[:, :answer_start] = -100

    return (
        input_ids,
        attention_mask,
        labels,
    )


# ================================================================
# Model configuration
# ================================================================

def build_memory_config() -> MemoryGPT2Config:
    """
    Match the full model architecture used by the best checkpoint.
    """

    return MemoryGPT2Config(
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

        # Keep training auxiliaries available.
        candidate_diversity_weight=0.01,
        update_orthogonality_weight=0.01,
        router_balance_weight=0.01,
        reader_balance_weight=0.01,
        memory_collapse_weight=0.01,
    )


def load_model(
    checkpoint_path: str,
    model_name: str,
    device: torch.device,
) -> MemoryAugmentedGPT2LMHeadModel:

    model = (
        MemoryAugmentedGPT2LMHeadModel
        .from_pretrained(
            model_name,
            memory_config=build_memory_config(),
        )
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)

    return model


# ================================================================
# Freeze GPT-2, train memory
# ================================================================

def configure_memory_only_training(
    model: MemoryAugmentedGPT2LMHeadModel,
) -> None:

    # Use your model's own backbone-freezing function.
    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone()

    else:
        for name, parameter in model.named_parameters():

            memory_keywords = (
                "memory",
                "reader",
                "writer",
                "router",
                "gate",
                "orthogonal",
                "candidate",
                "slot",
                "write_",
                "confidence",
            )

            parameter.requires_grad = any(
                keyword in name.lower()
                for keyword in memory_keywords
            )


def print_trainable_parameters(
    model: torch.nn.Module,
) -> None:

    total = 0
    trainable = 0

    for parameter in model.parameters():
        total += parameter.numel()

        if parameter.requires_grad:
            trainable += parameter.numel()

    print(
        f"Total parameters: {total:,}"
    )

    print(
        f"Trainable parameters: {trainable:,}"
    )

    print(
        f"Trainable percentage: "
        f"{100.0 * trainable / total:.2f}%"
    )


# ================================================================
# Forward memory construction
# ================================================================

def detach_memory(memory_state: Any) -> Any:

    if memory_state is None:
        return None

    return memory_state.detach()


def process_context_chunks(
    model: MemoryAugmentedGPT2LMHeadModel,
    tokenizer: Any,
    example: dict[str, Any],
    device: torch.device,
    keep_graph: bool,
):

    memory_state = None

    chunks = [
        example["fact"],
        *example["distractors"],
    ]

    for text in chunks:

        input_ids, attention_mask = tokenize_text(
            tokenizer,
            text,
            device,
        )

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memory_state=memory_state,
            update_memory=True,
            return_diagnostics=False,
        )

        memory_state = output.memory_state

        # Prevent graph explosion across many chunks.
        # For this first retrieval experiment we train memory behavior
        # through each chunk locally rather than full long BPTT.
        if not keep_graph:
            memory_state = detach_memory(
                memory_state
            )

    return memory_state


# ================================================================
# Retrieval loss
# ================================================================

def retrieval_loss(
    model: MemoryAugmentedGPT2LMHeadModel,
    tokenizer: Any,
    example: dict[str, Any],
    memory_state: Any,
    device: torch.device,
) -> torch.Tensor:

    (
        input_ids,
        attention_mask,
        labels,
    ) = prepare_query_with_answer(
        tokenizer=tokenizer,
        query=example["query"],
        answer=example["answer"],
        device=device,
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        memory_state=memory_state,

        # Do not write the answer/query into memory.
        update_memory=False,

        return_diagnostics=False,
    )

    if output.lm_loss is None:
        raise RuntimeError(
            "Model returned lm_loss=None."
        )

    return output.lm_loss


# ================================================================
# Validation
# ================================================================

@torch.no_grad()
def validate(
    model: MemoryAugmentedGPT2LMHeadModel,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:

    model.eval()

    total_loss = 0.0
    correct = 0

    for example in examples:

        memory_state = process_context_chunks(
            model=model,
            tokenizer=tokenizer,
            example=example,
            device=device,
            keep_graph=False,
        )

        answer_scores = {}

        # Correct answer + three negatives.
        candidates = [example["answer"]]

        negatives = [
            answer
            for answer in ANSWERS
            if answer != example["answer"]
        ]

        rng = random.Random(
            example["id"] + 999
        )

        candidates.extend(
            rng.sample(
                negatives,
                3,
            )
        )

        for candidate in candidates:

            temporary_example = dict(
                example
            )

            temporary_example[
                "answer"
            ] = candidate

            loss = retrieval_loss(
                model=model,
                tokenizer=tokenizer,
                example=temporary_example,
                memory_state=memory_state,
                device=device,
            )

            answer_scores[candidate] = float(
                loss.detach().cpu()
            )

        prediction = min(
            answer_scores,
            key=answer_scores.get,
        )

        if prediction == example["answer"]:
            correct += 1

        correct_loss = retrieval_loss(
            model=model,
            tokenizer=tokenizer,
            example=example,
            memory_state=memory_state,
            device=device,
        )

        total_loss += float(
            correct_loss.detach().cpu()
        )

    mean_loss = (
        total_loss / len(examples)
    )

    accuracy = (
        correct / len(examples)
    )

    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "examples": float(
            len(examples)
        ),
    }


# ================================================================
# Checkpoints
# ================================================================

def save_checkpoint(
    path: Path,
    model: MemoryAugmentedGPT2LMHeadModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "validation": (
                validation_metrics
            ),
            "arguments": vars(args),
        },
        path,
    )

    print(
        f"Saved checkpoint: {path}"
    )


# ================================================================
# Arguments
# ================================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Synthetic associative retrieval fine-tuning "
            "for memory-augmented GPT-2."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
    )

    parser.add_argument(
        "--train-examples",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--validation-examples",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
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
        "--log-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "outputs/"
            "synthetic_retrieval_memory"
        ),
    )

    return parser.parse_args()


# ================================================================
# Main training
# ================================================================

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
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("SYNTHETIC RETRIEVAL FINE-TUNING")
    print("=" * 80)

    print("Device:", device)
    print(
        "Starting checkpoint:",
        args.checkpoint,
    )
    print(
        "Training distances:",
        args.distances,
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
        "\nBuilding synthetic datasets..."
    )

    train_examples = build_dataset(
        number_examples=(
            args.train_examples
        ),
        distances=args.distances,
        seed=args.seed,
        start_id=0,
    )

    validation_examples = build_dataset(
        number_examples=(
            args.validation_examples
        ),
        distances=args.distances,
        seed=args.seed + 10000,
        start_id=100000,
    )

    print(
        "Training examples:",
        len(train_examples),
    )

    print(
        "Validation examples:",
        len(validation_examples),
    )

    print(
        "\nLoading memory model..."
    )

    model = load_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        device=device,
    )

    configure_memory_only_training(
        model
    )

    print_trainable_parameters(
        model
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_validation_accuracy = -1.0

    rng = random.Random(
        args.seed
    )

    # ------------------------------------------------------------
    # Initial validation
    # ------------------------------------------------------------

    print(
        "\nValidation before retrieval training..."
    )

    initial_validation = validate(
        model=model,
        tokenizer=tokenizer,
        examples=validation_examples,
        device=device,
    )

    print(
        f"Initial loss: "
        f"{initial_validation['loss']:.4f}"
    )

    print(
        f"Initial accuracy: "
        f"{100.0 * initial_validation['accuracy']:.2f}%"
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

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

        model.train()

        rng.shuffle(
            train_examples
        )

        running_loss = 0.0

        for step, example in enumerate(
            train_examples,
            start=1,
        ):

            optimizer.zero_grad(
                set_to_none=True
            )

            # Build memory from the fact and distractors.
            memory_state = (
                process_context_chunks(
                    model=model,
                    tokenizer=tokenizer,
                    example=example,
                    device=device,

                    # Keep graph from the final context chunk into
                    # the query. This keeps training manageable.
                    keep_graph=False,
                )
            )

            loss = retrieval_loss(
                model=model,
                tokenizer=tokenizer,
                example=example,
                memory_state=memory_state,
                device=device,
            )

            loss.backward()

            gradient_norm = (
                clip_grad_norm_(
                    trainable_parameters,
                    max_norm=(
                        args.max_grad_norm
                    ),
                )
            )

            optimizer.step()

            running_loss += float(
                loss.detach().cpu()
            )

            if (
                step
                % args.log_every
                == 0
            ):

                average_loss = (
                    running_loss
                    / args.log_every
                )

                print(
                    f"Epoch {epoch:02d} | "
                    f"Step {step:05d}/"
                    f"{len(train_examples):05d} | "
                    f"Loss {average_loss:.4f} | "
                    f"Grad "
                    f"{float(gradient_norm):.4f}"
                )

                running_loss = 0.0

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        validation_metrics = validate(
            model=model,
            tokenizer=tokenizer,
            examples=(
                validation_examples
            ),
            device=device,
        )

        print(
            "\nValidation summary"
        )

        print(
            f"Loss: "
            f"{validation_metrics['loss']:.4f}"
        )

        print(
            f"Accuracy: "
            f"{100.0 * validation_metrics['accuracy']:.2f}%"
        )

        save_checkpoint(
            path=(
                output_dir
                / "checkpoint_latest.pt"
            ),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            validation_metrics=(
                validation_metrics
            ),
            args=args,
        )

        if (
            validation_metrics[
                "accuracy"
            ]
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_metrics[
                    "accuracy"
                ]
            )

            save_checkpoint(
                path=(
                    output_dir
                    / "checkpoint_best.pt"
                ),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_metrics=(
                    validation_metrics
                ),
                args=args,
            )

            print(
                "New best retrieval checkpoint saved."
            )

    print(
        "\nTraining completed."
    )

    print(
        f"Best validation accuracy: "
        f"{100.0 * best_validation_accuracy:.2f}%"
    )

    print(
        f"Best checkpoint: "
        f"{output_dir / 'checkpoint_best.pt'}"
    )


if __name__ == "__main__":
    main()