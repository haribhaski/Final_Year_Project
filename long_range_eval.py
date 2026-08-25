from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2LMHeadModel

from models.gpt2_memory import (
    MemoryAugmentedGPT2LMHeadModel,
    MemoryGPT2Config,
)


# ================================================================
# Reproducibility
# ================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================================================================
# Synthetic data generation
# ================================================================


WORDS = [
    "RAVEN", "ORION", "TANGO", "NOVA",
    "EMBER", "SIGMA", "LUNAR", "DELTA",
    "KAPPA", "FALCON", "NEBULA", "ATLAS",
]


DISTRACTOR_TEXTS = [
    (
        "The development of railway systems changed transportation "
        "by allowing passengers and goods to travel efficiently "
        "between distant cities. Engineers improved track design, "
        "signalling systems, and locomotive technology over time."
    ),
    (
        "Photosynthesis is a biological process in which plants use "
        "light energy to produce chemical energy. Chlorophyll absorbs "
        "light while carbon dioxide and water participate in reactions "
        "that eventually produce carbohydrates."
    ),
    (
        "Ancient civilizations developed systems of agriculture, "
        "trade, administration, and architecture. Archaeological "
        "evidence provides information about their technologies, "
        "social organization, and economic activity."
    ),
    (
        "Computer networks allow devices to exchange information using "
        "communication protocols. Routers forward packets between "
        "networks while transport protocols support reliable or "
        "low-latency communication."
    ),
    (
        "Ocean currents influence climate by transporting heat across "
        "large distances. Differences in temperature, salinity, wind, "
        "and Earth's rotation contribute to complex circulation patterns."
    ),
    (
        "Machine learning systems estimate patterns from data. Training "
        "typically adjusts model parameters to reduce an objective "
        "function, while validation data is used to evaluate generalization."
    ),
    (
        "Volcanoes form when magma reaches the surface through weaknesses "
        "in Earth's crust. Their behavior depends on magma composition, "
        "gas concentration, pressure, and surrounding geological structure."
    ),
    (
        "Libraries preserve books, manuscripts, recordings, and digital "
        "collections. Cataloguing systems allow material to be organized "
        "and retrieved according to author, subject, title, and other metadata."
    ),
]


def make_code(rng: random.Random) -> str:
    word = rng.choice(WORDS)
    number = rng.randint(1000, 9999)
    return f"{word} {number}"


def make_example(
    rng: random.Random,
    distance: int,
    example_id: int,
) -> dict[str, Any]:

    entity = f"Project-{example_id:04d}"

    correct_code = make_code(rng)

    wrong_codes = set()

    while len(wrong_codes) < 3:
        candidate = make_code(rng)

        if candidate != correct_code:
            wrong_codes.add(candidate)

    choices = [correct_code] + list(wrong_codes)
    rng.shuffle(choices)

    fact_chunk = (
        f"This record contains an arbitrary temporary identifier. "
        f"The secret access code assigned to {entity} is "
        f"{correct_code}. Remember this exact association because "
        f"it may be requested later."
    )

    distractors = []

    for index in range(distance):
        text = DISTRACTOR_TEXTS[
            index % len(DISTRACTOR_TEXTS)
        ]

        distractors.append(
            f"Unrelated passage {index + 1}. {text}"
        )

    query = (
        f"Earlier, a secret access code was assigned to {entity}. "
        f"The secret access code assigned to {entity} is"
    )

    return {
        "id": example_id,
        "entity": entity,
        "correct": correct_code,
        "choices": choices,
        "fact_chunk": fact_chunk,
        "distractors": distractors,
        "query": query,
        "distance": distance,
    }


# ================================================================
# Tokenization
# ================================================================


def tokenize_chunk(
    tokenizer: Any,
    text: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        add_special_tokens=False,
    )

    return (
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
    )


# ================================================================
# Candidate continuation scoring
# ================================================================


def continuation_nll_plain(
    model: GPT2LMHeadModel,
    tokenizer: Any,
    prompt: str,
    answer: str,
    device: torch.device,
) -> float:

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    # Leading space matters for GPT-2 continuation tokenization.
    answer_ids = tokenizer(
        " " + answer,
        add_special_tokens=False,
    )["input_ids"]

    full_ids = prompt_ids + answer_ids

    if len(full_ids) > 256:
        full_ids = full_ids[-256:]

        # Recompute answer start after truncation.
        answer_start = len(full_ids) - len(answer_ids)
    else:
        answer_start = len(prompt_ids)

    input_ids = torch.tensor(
        [full_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(input_ids)

    labels = input_ids.clone()
    labels[:, :answer_start] = -100

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    shift_logits = output.logits[:, :-1]
    shift_labels = labels[:, 1:]

    loss = F.cross_entropy(
        shift_logits.reshape(
            -1,
            shift_logits.size(-1),
        ),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )

    number_answer_tokens = (
        shift_labels != -100
    ).sum().item()

    return float(
        loss.item()
        / max(number_answer_tokens, 1)
    )


def continuation_nll_memory(
    model: MemoryAugmentedGPT2LMHeadModel,
    tokenizer: Any,
    prompt: str,
    answer: str,
    memory_state: Any,
    device: torch.device,
) -> float:

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    answer_ids = tokenizer(
        " " + answer,
        add_special_tokens=False,
    )["input_ids"]

    full_ids = prompt_ids + answer_ids

    if len(full_ids) > 256:
        full_ids = full_ids[-256:]
        answer_start = len(full_ids) - len(answer_ids)
    else:
        answer_start = len(prompt_ids)

    input_ids = torch.tensor(
        [full_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(input_ids)

    labels = input_ids.clone()
    labels[:, :answer_start] = -100

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        memory_state=memory_state,
        update_memory=False,
        return_diagnostics=False,
    )

    return float(
        output.lm_loss.detach().float().cpu().item()
    )


# ================================================================
# Memory preparation
# ================================================================


@torch.no_grad()
def build_persistent_memory(
    model: MemoryAugmentedGPT2LMHeadModel,
    tokenizer: Any,
    example: dict[str, Any],
    device: torch.device,
):

    memory_state = None

    chunks = [
        example["fact_chunk"],
        *example["distractors"],
    ]

    for text in chunks:

        input_ids, attention_mask = tokenize_chunk(
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

        memory_state = output.memory_state.detach()

    return memory_state


# ================================================================
# Model loading
# ================================================================


def load_memory_model(
    checkpoint_path: str,
    model_name: str,
    device: torch.device,
) -> MemoryAugmentedGPT2LMHeadModel:

    # Match your trained full proposed model.
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

        # These weights affect training losses, not inference,
        # but matching configuration keeps the experiment explicit.
        candidate_diversity_weight=0.01,
        update_orthogonality_weight=0.01,
        router_balance_weight=0.01,
        reader_balance_weight=0.01,
        memory_collapse_weight=0.01,
    )

    model = (
        MemoryAugmentedGPT2LMHeadModel
        .from_pretrained(
            model_name,
            memory_config=config,
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
    model.eval()

    return model


def load_plain_model(
    checkpoint_path: str,
    model_name: str,
    device: torch.device,
) -> GPT2LMHeadModel:

    model = GPT2LMHeadModel.from_pretrained(
        model_name
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
    model.eval()
    model.config.use_cache = False

    return model


# ================================================================
# Evaluation
# ================================================================


@torch.no_grad()
def evaluate_example(
    example: dict[str, Any],
    memory_model: MemoryAugmentedGPT2LMHeadModel,
    plain_model: GPT2LMHeadModel,
    tokenizer: Any,
    device: torch.device,
) -> dict[str, Any]:

    # ------------------------------------------------------------
    # Condition A:
    # persistent memory
    # ------------------------------------------------------------

    persistent_state = build_persistent_memory(
        memory_model,
        tokenizer,
        example,
        device,
    )

    persistent_scores = {}

    for choice in example["choices"]:

        persistent_scores[choice] = (
            continuation_nll_memory(
                memory_model,
                tokenizer,
                example["query"],
                choice,
                persistent_state,
                device,
            )
        )

    persistent_prediction = min(
        persistent_scores,
        key=persistent_scores.get,
    )

    # ------------------------------------------------------------
    # Condition B:
    # reset memory before final query
    # ------------------------------------------------------------

    reset_scores = {}

    for choice in example["choices"]:

        reset_scores[choice] = (
            continuation_nll_memory(
                memory_model,
                tokenizer,
                example["query"],
                choice,
                None,
                device,
            )
        )

    reset_prediction = min(
        reset_scores,
        key=reset_scores.get,
    )

    # ------------------------------------------------------------
    # Condition C:
    # plain GPT-2
    # ------------------------------------------------------------

    plain_scores = {}

    for choice in example["choices"]:

        plain_scores[choice] = (
            continuation_nll_plain(
                plain_model,
                tokenizer,
                example["query"],
                choice,
                device,
            )
        )

    plain_prediction = min(
        plain_scores,
        key=plain_scores.get,
    )

    return {
        "id": example["id"],
        "distance": example["distance"],
        "correct": example["correct"],

        "persistent_prediction": persistent_prediction,
        "persistent_correct": int(
            persistent_prediction
            == example["correct"]
        ),

        "reset_prediction": reset_prediction,
        "reset_correct": int(
            reset_prediction
            == example["correct"]
        ),

        "plain_prediction": plain_prediction,
        "plain_correct": int(
            plain_prediction
            == example["correct"]
        ),
    }


# ================================================================
# Main
# ================================================================


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--memory-checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--plain-checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
    )

    parser.add_argument(
        "--examples-per-distance",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results/long_range",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_arguments()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("CONTROLLED LONG-RANGE MEMORY EVALUATION")
    print("=" * 80)

    print("Device:", device)
    print("Distances:", args.distances)
    print(
        "Examples per distance:",
        args.examples_per_distance,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading full memory model...")

    memory_model = load_memory_model(
        args.memory_checkpoint,
        args.model_name,
        device,
    )

    print("Loading plain GPT-2 baseline...")

    plain_model = load_plain_model(
        args.plain_checkpoint,
        args.model_name,
        device,
    )

    rng = random.Random(
        args.seed
    )

    results = []

    example_id = 0

    for distance in args.distances:

        print(
            f"\nDistance = {distance} chunks"
        )

        for local_index in range(
            args.examples_per_distance
        ):

            example_id += 1

            example = make_example(
                rng=rng,
                distance=distance,
                example_id=example_id,
            )

            record = evaluate_example(
                example=example,
                memory_model=memory_model,
                plain_model=plain_model,
                tokenizer=tokenizer,
                device=device,
            )

            results.append(record)

            if (
                local_index + 1
            ) % 10 == 0:

                print(
                    f"  Completed "
                    f"{local_index + 1}/"
                    f"{args.examples_per_distance}"
                )

    # ------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------

    grouped = defaultdict(
        lambda: {
            "n": 0,
            "persistent": 0,
            "reset": 0,
            "plain": 0,
        }
    )

    for row in results:

        distance = row["distance"]

        grouped[distance]["n"] += 1

        grouped[distance][
            "persistent"
        ] += row[
            "persistent_correct"
        ]

        grouped[distance][
            "reset"
        ] += row[
            "reset_correct"
        ]

        grouped[distance][
            "plain"
        ] += row[
            "plain_correct"
        ]

    summary = []

    print("\n" + "=" * 80)
    print("LONG-RANGE RESULTS")
    print("=" * 80)

    print(
        f"{'Distance':<12}"
        f"{'Plain':>12}"
        f"{'Reset':>12}"
        f"{'Persistent':>14}"
    )

    for distance in sorted(grouped):

        data = grouped[distance]

        n = data["n"]

        plain_acc = (
            data["plain"] / n
        )

        reset_acc = (
            data["reset"] / n
        )

        persistent_acc = (
            data["persistent"] / n
        )

        print(
            f"{distance:<12}"
            f"{plain_acc:>12.3f}"
            f"{reset_acc:>12.3f}"
            f"{persistent_acc:>14.3f}"
        )

        summary.append(
            {
                "distance": distance,
                "examples": n,
                "plain_accuracy": plain_acc,
                "reset_accuracy": reset_acc,
                "persistent_accuracy": persistent_acc,
            }
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
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    with (
        output_dir
        / "summary.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "distance",
                "examples",
                "plain_accuracy",
                "reset_accuracy",
                "persistent_accuracy",
            ],
        )

        writer.writeheader()
        writer.writerows(summary)

    with (
        output_dir
        / "examples.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:")
    print(
        output_dir
        / "summary.json"
    )
    print(
        output_dir
        / "summary.csv"
    )
    print(
        output_dir
        / "examples.csv"
    )


if __name__ == "__main__":
    main()