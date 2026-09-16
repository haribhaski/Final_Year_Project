from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2LMHeadModel


# ============================================================================
# PROJECT PATH
# ============================================================================

current_dir = Path(__file__).resolve().parent
candidates = (
    [current_dir]
    + list(current_dir.glob("**/models"))
    + list(current_dir.parent.glob("**/models"))
)

for c in candidates:
    target = c.parent if c.name == "models" else c
    if (target / "models").is_dir():
        if str(target) not in sys.path:
            sys.path.insert(0, str(target))
        break


from models.gpt2_memory import (
    MemoryAugmentedGPT2LMHeadModel,
    MemoryGPT2Config,
)


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# SYNTHETIC RETRIEVAL DATA
# ============================================================================

NAMES = [
    "Alpha",
    "Bravo",
    "Charlie",
    "Delta",
    "Echo",
    "Foxtrot",
    "Golf",
    "Hotel",
]


def generate_unique_codes(
    rng: random.Random,
    n: int,
) -> List[str]:

    codes = set()

    while len(codes) < n:
        codes.add(str(rng.randint(1000, 9999)))

    return list(codes)


def build_distractor(
    tokenizer,
    num_tokens: int,
    device: torch.device,
) -> Optional[torch.Tensor]:

    if num_tokens <= 0:
        return None

    filler_sentence = (
        "The regional weather in the valley remains unpredictable "
        "with light rain and moderate winds. "
    )

    filler = filler_sentence * (num_tokens // 10 + 10)

    tokens = tokenizer(
        filler,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids

    tokens = tokens[:, :num_tokens]

    return tokens.to(device)


def make_episode(
    tokenizer,
    num_entities: int,
    distractor_len: int,
    device: torch.device,
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Creates ONE episodic memory problem.

    IMPORTANT:
    Mapping changes every episode.

    Example:

        Episode 1:
            Alpha -> 5382
            Bravo -> 9174

        Episode 2:
            Alpha -> 2041
            Bravo -> 6619

    Therefore the model cannot solve the task by memorizing
    permanent name -> code associations in parameters.
    """

    if num_entities > len(NAMES):
        raise ValueError(
            f"num_entities={num_entities}, but only "
            f"{len(NAMES)} unique names exist."
        )

    selected_names = rng.sample(NAMES, num_entities)
    selected_codes = generate_unique_codes(rng, num_entities)

    mappings = list(zip(selected_names, selected_codes))

    # ------------------------------------------------------------
    # Randomly choose ONE entity to query
    # ------------------------------------------------------------

    target_name, target_code = rng.choice(mappings)

    # ------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------

    fact_texts = [
        f"Agent {name}'s secret passcode is {code}."
        for name, code in mappings
    ]

    fact_tokens = [
        tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)
        for text in fact_texts
    ]

    # ------------------------------------------------------------
    # Distractor
    # ------------------------------------------------------------

    distractor_tokens = build_distractor(
        tokenizer,
        distractor_len,
        device,
    )

    # ------------------------------------------------------------
    # Query
    # ------------------------------------------------------------

    query_text = (
        f"What is the secret passcode for Agent "
        f"{target_name}? The passcode is"
    )

    answer_text = f" {target_code}"

    query_tokens = tokenizer(
        query_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    answer_tokens = tokenizer(
        answer_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    # ------------------------------------------------------------
    # 4-way candidate set
    #
    # Correct + 3 random wrong codes
    #
    # Chance = 1 / 4 = 25%
    # ------------------------------------------------------------

    wrong_codes = set()

    while len(wrong_codes) < 3:
        wrong = str(rng.randint(1000, 9999))

        if wrong != target_code:
            wrong_codes.add(wrong)

    candidate_codes = [target_code] + list(wrong_codes)

    rng.shuffle(candidate_codes)

    return {
        "mappings": mappings,
        "target_name": target_name,
        "target_code": target_code,
        "fact_texts": fact_texts,
        "fact_tokens": fact_tokens,
        "distractor_tokens": distractor_tokens,
        "query_text": query_text,
        "query_tokens": query_tokens,
        "answer_text": answer_text,
        "answer_tokens": answer_tokens,
        "candidate_codes": candidate_codes,
        "distractor_len": distractor_len,
    }


# ============================================================================
# MODEL
# ============================================================================

def load_memory_model(
    config: MemoryGPT2Config,
    device: torch.device,
):

    print("Loading pretrained GPT-2 backbone...")

    backbone = GPT2LMHeadModel.from_pretrained("gpt2")

    model = MemoryAugmentedGPT2LMHeadModel(
        backbone=backbone,
        memory_config=config,
    )

    return model.to(device)


def freeze_backbone(model: nn.Module) -> None:
    """
    Train memory modules only.
    """

    for name, param in model.named_parameters():
        param.requires_grad = not name.startswith("backbone.")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================================
# MEMORY HELPERS
# ============================================================================

def get_slots(memory_state):

    if memory_state is None:
        return None

    slots = getattr(memory_state, "slots", None)

    if slots is None and isinstance(memory_state, tuple):
        slots = memory_state[0]

    return slots


def compute_slot_cosine_similarity(memory_state) -> float:

    slots = get_slots(memory_state)

    if slots is None:
        return 0.0

    if slots.ndim == 3:
        slots = slots[0]

    if slots.size(0) <= 1:
        return 0.0

    normalized = F.normalize(
        slots,
        p=2,
        dim=-1,
    )

    cosine = normalized @ normalized.T

    mask = ~torch.eye(
        cosine.size(0),
        dtype=torch.bool,
        device=cosine.device,
    )

    return float(cosine[mask].mean().item())


def write_episode_to_memory(
    model,
    episode: Dict[str, Any],
    detach: bool,
):
    """
    Writes:

        facts -> memory
        distractor -> memory

    During TRAINING:
        detach=False

    so retrieval loss can propagate through the memory write path.

    During EVALUATION:
        detach=True

    because gradients are unnecessary.
    """

    memory_state = None

    for fact_tokens in episode["fact_tokens"]:

        output = model(
            input_ids=fact_tokens,
            memory_state=memory_state,
            update_memory=True,
        )

        memory_state = output.memory_state

        if detach and hasattr(memory_state, "detach"):
            memory_state = memory_state.detach()

    distractor = episode["distractor_tokens"]

    if distractor is not None and distractor.size(1) > 0:

        output = model(
            input_ids=distractor,
            memory_state=memory_state,
            update_memory=True,
        )

        memory_state = output.memory_state

        if detach and hasattr(memory_state, "detach"):
            memory_state = memory_state.detach()

    return memory_state


# ============================================================================
# TRAINING LOSS
# ============================================================================

def retrieval_training_loss(
    model,
    episode: Dict[str, Any],
    memory_state,
):
    """
    Target-only teacher-forced LM loss.

    Query does NOT update memory.
    """

    query = episode["query_tokens"]
    answer = episode["answer_tokens"]

    sequence = torch.cat(
        [query, answer],
        dim=1,
    )

    labels = torch.full_like(
        sequence,
        -100,
    )

    answer_len = answer.size(1)

    labels[:, -answer_len:] = sequence[:, -answer_len:]

    output = model(
        input_ids=sequence,
        labels=labels,
        memory_state=memory_state,

        # CRITICAL:
        # query must only READ memory.
        update_memory=False,
    )

    return output


# ============================================================================
# CANDIDATE SCORING
# ============================================================================

def score_candidate(
    model,
    tokenizer,
    query_tokens: torch.Tensor,
    candidate_code: str,
    memory_state,
    device: torch.device,
) -> float:
    """
    Mean NLL over candidate answer tokens.

    Lower = better.

    This is used for 4-way forced-choice retrieval.
    """

    answer_text = f" {candidate_code}"

    answer_tokens = tokenizer(
        answer_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    sequence = torch.cat(
        [query_tokens, answer_tokens],
        dim=1,
    )

    labels = torch.full_like(
        sequence,
        -100,
    )

    answer_len = answer_tokens.size(1)

    labels[:, -answer_len:] = answer_tokens

    output = model(
        input_ids=sequence,
        labels=labels,
        memory_state=memory_state,
        update_memory=False,
    )

    if output.lm_loss is None:
        raise RuntimeError(
            "Model did not return lm_loss during candidate scoring."
        )

    return float(output.lm_loss.item())


def forced_choice_prediction(
    model,
    tokenizer,
    episode,
    memory_state,
    device,
):

    scores = {}

    for candidate in episode["candidate_codes"]:

        scores[candidate] = score_candidate(
            model=model,
            tokenizer=tokenizer,
            query_tokens=episode["query_tokens"],
            candidate_code=candidate,
            memory_state=memory_state,
            device=device,
        )

    prediction = min(
        scores,
        key=scores.get,
    )

    correct_code = episode["target_code"]

    correct_nll = scores[correct_code]

    wrong_nlls = [
        value
        for code, value in scores.items()
        if code != correct_code
    ]

    best_wrong_nll = min(wrong_nlls)

    # Positive margin = correct answer beats best wrong answer
    margin = best_wrong_nll - correct_nll

    return prediction, scores, margin


# ============================================================================
# AUTOREGRESSIVE GENERATION
# ============================================================================

def autoregressive_retrieve(
    model,
    tokenizer,
    query_tokens,
    memory_state,
    target_answer_tokens,
    device,
):
    """
    Greedily generate exactly the same number of tokens as
    the ground-truth answer.

    IMPORTANT:
    No ground-truth answer token is fed to the model.

    This is stricter than teacher-forced token accuracy.
    """

    generated = query_tokens.clone()

    answer_len = target_answer_tokens.size(1)

    generated_answer_ids = []

    for _ in range(answer_len):

        output = model(
            input_ids=generated,
            memory_state=memory_state,
            update_memory=False,
        )

        next_token = (
            output.logits[:, -1, :]
            .argmax(dim=-1, keepdim=True)
        )

        generated_answer_ids.append(
            int(next_token.item())
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

    target_ids = target_answer_tokens[0].tolist()

    exact_match = (
        generated_answer_ids == target_ids
    )

    generated_text = tokenizer.decode(
        generated_answer_ids
    )

    target_text = tokenizer.decode(
        target_ids
    )

    return {
        "exact_match": exact_match,
        "generated_ids": generated_answer_ids,
        "target_ids": target_ids,
        "generated_text": generated_text,
        "target_text": target_text,
    }


# ============================================================================
# SYNTHETIC TRAINING
# ============================================================================

def train_synthetic_task(
    model,
    tokenizer,
    args,
    device,
):

    print("\n" + "=" * 80)
    print("SYNTHETIC RETRIEVAL TRAINING")
    print("=" * 80)

    freeze_backbone(model)

    print(
        f"Trainable parameters: "
        f"{count_trainable_parameters(model):,}"
    )

    optimizer = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=args.lr,
    )

    train_rng = random.Random(args.seed)

    model.train()

    running_loss = 0.0

    for step in range(
        1,
        args.max_steps + 1,
    ):

        optimizer.zero_grad(set_to_none=True)

        episode = make_episode(
            tokenizer=tokenizer,
            num_entities=args.num_entities,
            distractor_len=args.distractor_len,
            device=device,
            rng=train_rng,
        )

        # --------------------------------------------------------
        # IMPORTANT:
        # DO NOT DETACH DURING TRAINING
        # --------------------------------------------------------

        memory_state = write_episode_to_memory(
            model=model,
            episode=episode,
            detach=False,
        )

        output = retrieval_training_loss(
            model=model,
            episode=episode,
            memory_state=memory_state,
        )

        if output.loss is None:
            raise RuntimeError(
                "Training output.loss is None."
            )

        loss = output.loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [
                p
                for p in model.parameters()
                if p.requires_grad
            ],
            max_norm=1.0,
        )

        optimizer.step()

        running_loss += float(loss.item())

        if (
            step == 1
            or step % args.log_interval == 0
        ):

            average_loss = (
                running_loss
                / min(step, args.log_interval)
            )

            slot_cos = compute_slot_cosine_similarity(
                memory_state
            )

            print(
                f"Step {step:05d}/{args.max_steps:05d} | "
                f"Loss {float(loss.item()):.4f} | "
                f"Recent Avg {average_loss:.4f} | "
                f"Slot Cos {slot_cos:.4f}"
            )

            running_loss = 0.0

        # --------------------------------------------------------
        # Periodic REAL validation
        # --------------------------------------------------------

        if (
            args.eval_every > 0
            and step % args.eval_every == 0
        ):

            evaluate_synthetic(
                model=model,
                tokenizer=tokenizer,
                args=args,
                device=device,
                num_examples=args.quick_eval_examples,
                distances=[args.distractor_len],
                seed=args.validation_seed,
                save_results=False,
                title=f"Validation at step {step}",
            )

            model.train()

    save_checkpoint(
        model=model,
        args=args,
    )


# ============================================================================
# SYNTHETIC EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_synthetic(
    model,
    tokenizer,
    args,
    device,
    num_examples: int,
    distances: List[int],
    seed: int,
    save_results: bool = True,
    title: str = "Final Synthetic Evaluation",
):

    print("\n" + "=" * 80)
    print(title.upper())
    print("=" * 80)

    model.eval()

    all_rows = []
    summaries = []

    for distance in distances:

        rng = random.Random(
            seed + distance * 100003
        )

        persistent_correct = 0
        reset_correct = 0

        persistent_generation_correct = 0
        reset_generation_correct = 0

        persistent_margin_sum = 0.0
        reset_margin_sum = 0.0

        for example_idx in range(num_examples):

            episode = make_episode(
                tokenizer=tokenizer,
                num_entities=args.num_entities,
                distractor_len=distance,
                device=device,
                rng=rng,
            )

            # ====================================================
            # CONDITION A:
            # Persistent memory
            # ====================================================

            persistent_memory = write_episode_to_memory(
                model=model,
                episode=episode,
                detach=True,
            )

            (
                persistent_prediction,
                persistent_scores,
                persistent_margin,
            ) = forced_choice_prediction(
                model=model,
                tokenizer=tokenizer,
                episode=episode,
                memory_state=persistent_memory,
                device=device,
            )

            persistent_is_correct = (
                persistent_prediction
                == episode["target_code"]
            )

            persistent_correct += int(
                persistent_is_correct
            )

            persistent_margin_sum += (
                persistent_margin
            )

            persistent_generation = (
                autoregressive_retrieve(
                    model=model,
                    tokenizer=tokenizer,
                    query_tokens=episode["query_tokens"],
                    memory_state=persistent_memory,
                    target_answer_tokens=episode[
                        "answer_tokens"
                    ],
                    device=device,
                )
            )

            persistent_generation_correct += int(
                persistent_generation[
                    "exact_match"
                ]
            )

            # ====================================================
            # CONDITION B:
            # Reset / no episodic memory
            # ====================================================

            (
                reset_prediction,
                reset_scores,
                reset_margin,
            ) = forced_choice_prediction(
                model=model,
                tokenizer=tokenizer,
                episode=episode,
                memory_state=None,
                device=device,
            )

            reset_is_correct = (
                reset_prediction
                == episode["target_code"]
            )

            reset_correct += int(
                reset_is_correct
            )

            reset_margin_sum += reset_margin

            reset_generation = (
                autoregressive_retrieve(
                    model=model,
                    tokenizer=tokenizer,
                    query_tokens=episode["query_tokens"],
                    memory_state=None,
                    target_answer_tokens=episode[
                        "answer_tokens"
                    ],
                    device=device,
                )
            )

            reset_generation_correct += int(
                reset_generation[
                    "exact_match"
                ]
            )

            # ====================================================
            # Save per-example information
            # ====================================================

            row = {
                "distance_tokens": distance,
                "example": example_idx,
                "target_name": episode[
                    "target_name"
                ],
                "target_code": episode[
                    "target_code"
                ],

                "persistent_prediction":
                    persistent_prediction,

                "persistent_correct":
                    int(persistent_is_correct),

                "persistent_margin":
                    persistent_margin,

                "persistent_generated":
                    persistent_generation[
                        "generated_text"
                    ],

                "persistent_generation_exact":
                    int(
                        persistent_generation[
                            "exact_match"
                        ]
                    ),

                "reset_prediction":
                    reset_prediction,

                "reset_correct":
                    int(reset_is_correct),

                "reset_margin":
                    reset_margin,

                "reset_generated":
                    reset_generation[
                        "generated_text"
                    ],

                "reset_generation_exact":
                    int(
                        reset_generation[
                            "exact_match"
                        ]
                    ),
            }

            all_rows.append(row)

        # ========================================================
        # Distance summary
        # ========================================================

        persistent_acc = (
            persistent_correct
            / num_examples
        )

        reset_acc = (
            reset_correct
            / num_examples
        )

        persistent_gen_acc = (
            persistent_generation_correct
            / num_examples
        )

        reset_gen_acc = (
            reset_generation_correct
            / num_examples
        )

        persistent_avg_margin = (
            persistent_margin_sum
            / num_examples
        )

        reset_avg_margin = (
            reset_margin_sum
            / num_examples
        )

        memory_gain = (
            persistent_acc
            - reset_acc
        )

        summary = {
            "distance_tokens": distance,
            "examples": num_examples,

            # 4-way forced choice
            "chance_accuracy": 0.25,
            "persistent_accuracy":
                persistent_acc,
            "reset_accuracy":
                reset_acc,
            "memory_gain":
                memory_gain,

            # Free autoregressive retrieval
            "persistent_generation_exact":
                persistent_gen_acc,
            "reset_generation_exact":
                reset_gen_acc,

            # Confidence / separation
            "persistent_mean_margin":
                persistent_avg_margin,
            "reset_mean_margin":
                reset_avg_margin,
        }

        summaries.append(summary)

        print(
            f"\nDistance = {distance} distractor tokens"
        )

        print(
            f"  4-way chance       : 25.00%"
        )

        print(
            f"  Persistent         : "
            f"{persistent_acc * 100:.2f}%"
        )

        print(
            f"  Reset              : "
            f"{reset_acc * 100:.2f}%"
        )

        print(
            f"  Memory gain        : "
            f"{memory_gain * 100:+.2f} pp"
        )

        print(
            f"  Persistent Gen EM  : "
            f"{persistent_gen_acc * 100:.2f}%"
        )

        print(
            f"  Reset Gen EM       : "
            f"{reset_gen_acc * 100:.2f}%"
        )

        print(
            f"  Persistent margin  : "
            f"{persistent_avg_margin:.4f}"
        )

    # ============================================================
    # Save results
    # ============================================================

    if save_results:

        result_dir = Path(
            args.output_dir
        ) / "synthetic_evaluation"

        result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        examples_path = (
            result_dir / "examples.csv"
        )

        summary_path = (
            result_dir / "summary.json"
        )

        summary_csv_path = (
            result_dir / "summary.csv"
        )

        if all_rows:

            with open(
                examples_path,
                "w",
                newline="",
                encoding="utf-8",
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=list(
                        all_rows[0].keys()
                    ),
                )

                writer.writeheader()
                writer.writerows(all_rows)

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                summaries,
                f,
                indent=2,
            )

        if summaries:

            with open(
                summary_csv_path,
                "w",
                newline="",
                encoding="utf-8",
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=list(
                        summaries[0].keys()
                    ),
                )

                writer.writeheader()
                writer.writerows(summaries)

        print(
            f"\nSaved evaluation to: "
            f"{result_dir}"
        )

    return summaries


# ============================================================================
# CHECKPOINTING
# ============================================================================

def config_to_dict(config) -> Dict[str, Any]:

    if is_dataclass(config):
        return asdict(config)

    if hasattr(config, "__dict__"):
        return {
            k: v
            for k, v in vars(config).items()
            if not k.startswith("_")
        }

    raise TypeError(
        "Could not serialize MemoryGPT2Config."
    )


def save_checkpoint(
    model,
    args,
):

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    save_path = os.path.join(
        args.output_dir,
        "memory_checkpoint.pt",
    )

    memory_config = getattr(
        model,
        "memory_config",
        None,
    )

    if memory_config is None:
        memory_config = getattr(
            model,
            "config",
            None,
        )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "training_args":
            vars(args),

        "model_name":
            "gpt2",
    }

    if memory_config is not None:

        try:
            checkpoint[
                "memory_config"
            ] = config_to_dict(
                memory_config
            )

        except Exception as exc:

            print(
                "[Warning] Could not serialize "
                f"memory config: {exc}"
            )

    torch.save(
        checkpoint,
        save_path,
    )

    print(
        f"\n[Saved] Full checkpoint: "
        f"{save_path}"
    )


# ============================================================================
# CLI
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "MemAttn Synthetic Retrieval "
            "Training + Evaluation"
        )
    )

    # ------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------

    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "eval",
            "train_eval",
        ],
        default="train_eval",
    )

    # ------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------

    parser.add_argument(
        "--num_slots",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--gate_type",
        choices=[
            "scalar",
            "vector",
        ],
        default="vector",
    )

    parser.add_argument(
        "--gate_init_bias",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--ortho_weight",
        type=float,
        default=0.05,
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    parser.add_argument(
        "--max_steps",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--num_entities",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--distractor_len",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--eval_every",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--quick_eval_examples",
        type=int,
        default=50,
    )

    # ------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------

    parser.add_argument(
        "--eval_examples",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--eval_distances",
        type=int,
        nargs="+",
        default=[
            0,
            32,
            64,
            128,
            256,
            512,
        ],
    )

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--validation_seed",
        type=int,
        default=2026,
    )

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "outputs/"
            "synthetic_retrieval_corrected"
        ),
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            "gpt2"
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # ============================================================
    # Memory architecture
    # ============================================================

    memory_config = MemoryGPT2Config(
        num_slots=args.num_slots,

        gate_type=args.gate_type,
        gate_mode="sigmoid",
        gate_init_bias=args.gate_init_bias,

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

        update_orthogonality_weight=(
            args.ortho_weight
        ),

        router_balance_weight=0.01,
        reader_balance_weight=0.01,
        memory_collapse_weight=0.01,

        detach_memory_between_steps=False,
    )

    model = load_memory_model(
        config=memory_config,
        device=device,
    )

    # ============================================================
    # Train
    # ============================================================

    if args.mode in [
        "train",
        "train_eval",
    ]:

        train_synthetic_task(
            model=model,
            tokenizer=tokenizer,
            args=args,
            device=device,
        )

    # ============================================================
    # Final evaluation
    # ============================================================

    if args.mode in [
        "eval",
        "train_eval",
    ]:

        evaluate_synthetic(
            model=model,
            tokenizer=tokenizer,
            args=args,
            device=device,
            num_examples=args.eval_examples,
            distances=args.eval_distances,
            seed=args.validation_seed,
            save_results=True,
            title="Final Synthetic Retrieval Evaluation",
        )


if __name__ == "__main__":
    main()
