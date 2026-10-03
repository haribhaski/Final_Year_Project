from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.gpt2_memory import (
    MemoryAugmentedGPT2LMHeadModel,
    MemoryGPT2Config,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "gpt2"

BASE_CHECKPOINT = (
    "outputs/retrieval_gradient_test/checkpoint_best.pt"
)

OUTPUT_CHECKPOINT = (
    "outputs/two_fact_occupancy_trained.pt"
)

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

SEED = 42

STEPS = 300
LEARNING_RATE = 5e-5
LOG_EVERY = 25

# ------------------------------------------------------------
# Two facts
# ------------------------------------------------------------

ENTITY_A = "Project-A"
ANSWER_A = "rabbit"

ENTITY_B = "Project-B"
ANSWER_B = "river"

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


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# MEMORY CONFIG
# ============================================================

def build_memory_config():

    return MemoryGPT2Config(
        num_slots=8,

        gate_type="vector",
        gate_mode="sigmoid",
        gate_init_bias=-2.0,

        # ====================================================
        # NEW ROUTER
        # ====================================================
        router_enabled=True,
        router_mode="occupancy",

        # One fact -> one slot for this diagnostic.
        router_top_k=1,
        router_temperature=0.7,

        writer_mode="attention",
        writer_attention_heads=8,

        orthogonal_mode="other_slots",
        orthogonal_strength=0.5,

        # Use token reader because our previous experiments
        # showed hybrid pooling was not necessary here.
        reader_mode="token",
        reader_fusion="gated",
        reader_heads=8,
        reader_top_k=3,
        reader_temperature=0.8,

        candidate_diversity_weight=0.01,
        update_orthogonality_weight=0.01,
        router_balance_weight=0.01,
        reader_balance_weight=0.01,
        memory_collapse_weight=0.01,

        detach_memory_between_steps=False,
    )


# ============================================================
# TEXT
# ============================================================

def make_fact(
    entity: str,
    answer: str,
) -> str:

    return (
        f"The assigned keyword for {entity} is {answer}. "
        f"Remember that the keyword associated with "
        f"{entity} is {answer}."
    )


def make_query(
    entity: str,
) -> str:

    return (
        f"The assigned keyword for {entity} is"
    )


FACT_A = make_fact(
    ENTITY_A,
    ANSWER_A,
)

FACT_B = make_fact(
    ENTITY_B,
    ANSWER_B,
)

QUERY_A = make_query(
    ENTITY_A,
)

QUERY_B = make_query(
    ENTITY_B,
)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(
    tokenizer,
    text: str,
) -> Tuple[torch.Tensor, torch.Tensor]:

    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    return (
        encoded["input_ids"].to(DEVICE),
        encoded["attention_mask"].to(DEVICE),
    )


def prepare_query_answer(
    tokenizer,
    query: str,
    answer: str,
):

    # GPT-2 answer must include leading space.
    answer_text = " " + answer

    query_ids = tokenizer(
        query,
        add_special_tokens=False,
    )["input_ids"]

    answer_ids = tokenizer(
        answer_text,
        add_special_tokens=False,
    )["input_ids"]

    input_ids = torch.tensor(
        [query_ids + answer_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    labels = torch.full_like(
        input_ids,
        -100,
    )

    labels[
        :,
        len(query_ids):
    ] = torch.tensor(
        answer_ids,
        dtype=torch.long,
        device=DEVICE,
    )

    return (
        input_ids,
        attention_mask,
        labels,
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        f"Loading base checkpoint: "
        f"{BASE_CHECKPOINT}"
    )

    model = (
        MemoryAugmentedGPT2LMHeadModel
        .from_pretrained(
            MODEL_NAME,
            memory_config=build_memory_config(),
        )
    )

    checkpoint = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu",
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    # --------------------------------------------------------
    # IMPORTANT
    #
    # The new occupancy router contains parameters that were
    # not present in the old checkpoint.
    #
    # strict=False lets us retain the useful pretrained
    # memory/writer/reader weights while initializing the new
    # router additions normally.
    # --------------------------------------------------------

    result = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print()
    print("Checkpoint load:")
    print(
        "Missing keys:",
        result.missing_keys,
    )
    print(
        "Unexpected keys:",
        result.unexpected_keys,
    )

    model = model.to(DEVICE)

    return model


# ============================================================
# FREEZE GPT-2
# ============================================================

def configure_training(
    model,
):

    if hasattr(
        model,
        "freeze_backbone",
    ):
        model.freeze_backbone()

    else:
        for parameter in (
            model.backbone.parameters()
        ):
            parameter.requires_grad = False

    # Router must definitely train.
    if model.router is not None:

        for parameter in (
            model.router.parameters()
        ):
            parameter.requires_grad = True

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,} / {total:,}"
    )


# ============================================================
# WRITE ONE FACT
# ============================================================

def write_fact(
    model,
    tokenizer,
    fact: str,
    memory_state=None,
):

    input_ids, attention_mask = tokenize(
        tokenizer,
        fact,
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        memory_state=memory_state,
        update_memory=True,
        return_diagnostics=True,
    )

    return output


# ============================================================
# WRITE A THEN B
# ============================================================

def build_two_fact_memory(
    model,
    tokenizer,
):

    # --------------------------------------------------------
    # FACT A
    # --------------------------------------------------------

    out_a = write_fact(
        model,
        tokenizer,
        FACT_A,
        memory_state=None,
    )

    memory_a = out_a.memory_state

    # --------------------------------------------------------
    # FACT B
    # --------------------------------------------------------

    out_b = write_fact(
        model,
        tokenizer,
        FACT_B,
        memory_state=memory_a,
    )

    memory_ab = out_b.memory_state

    return (
        memory_ab,
        out_a,
        out_b,
    )


# ============================================================
# RETRIEVAL LOSS
# ============================================================

def retrieval_loss(
    model,
    tokenizer,
    query: str,
    answer: str,
    memory_state,
):

    (
        input_ids,
        attention_mask,
        labels,
    ) = prepare_query_answer(
        tokenizer,
        query,
        answer,
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        memory_state=memory_state,
        update_memory=False,
        return_diagnostics=True,
    )

    return (
        output.lm_loss,
        output,
    )


# ============================================================
# SCORE ONE ANSWER
# ============================================================

@torch.no_grad()
def score_answer(
    model,
    tokenizer,
    query: str,
    answer: str,
    memory_state,
) -> float:

    (
        input_ids,
        attention_mask,
        labels,
    ) = prepare_query_answer(
        tokenizer,
        query,
        answer,
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        memory_state=memory_state,
        update_memory=False,
        return_diagnostics=False,
    )

    return float(
        output.lm_loss.item()
    )


# ============================================================
# ALL-16 EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_query(
    model,
    tokenizer,
    query: str,
    expected: str,
    memory_state,
):

    scores = {}

    for answer in ANSWERS:

        scores[answer] = score_answer(
            model,
            tokenizer,
            query,
            answer,
            memory_state,
        )

    ordered = sorted(
        scores.items(),
        key=lambda item: item[1],
    )

    prediction = ordered[0][0]

    rank = next(
        i + 1
        for i, (answer, _)
        in enumerate(ordered)
        if answer == expected
    )

    return {
        "prediction": prediction,
        "rank": rank,
        "loss": scores[expected],
        "scores": ordered,
    }


# ============================================================
# ROUTER INFO
# ============================================================

def route_vector(
    output,
):

    if output.routing_output is None:
        return None

    return (
        output.routing_output.weights[
            0
        ]
        .detach()
        .float()
        .cpu()
    )


def selected_slot(
    output,
):

    route = route_vector(
        output
    )

    if route is None:
        return None

    return int(
        route.argmax().item()
    )


# ============================================================
# READER DISTRIBUTION
# ============================================================

def reader_distribution(
    output,
):

    if output.read_output is None:
        return None

    read_output = output.read_output

    # Prefer slot_usage because this already exists in your
    # reader output and is used by memory_bank.record_reads().
    if hasattr(
        read_output,
        "slot_usage",
    ):

        usage = (
            read_output.slot_usage
            .detach()
            .float()
            .cpu()
        )

        if usage.dim() == 2:
            usage = usage[0]

        usage = usage / (
            usage.sum().clamp_min(1e-8)
        )

        return usage

    return None


# ============================================================
# QUERY READER INFO
# ============================================================

@torch.no_grad()
def query_reader(
    model,
    tokenizer,
    query: str,
    memory_state,
):

    input_ids, attention_mask = tokenize(
        tokenizer,
        query,
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        memory_state=memory_state,
        update_memory=False,
        return_diagnostics=True,
    )

    return reader_distribution(
        output
    )


# ============================================================
# PRETTY PRINT
# ============================================================

def round_vector(
    tensor,
    digits=4,
):

    if tensor is None:
        return None

    return [
        round(float(x), digits)
        for x in tensor.tolist()
    ]


def print_state(
    model,
    tokenizer,
    title: str,
):

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    model.eval()

    # --------------------------------------------------------
    # IMPORTANT:
    # This evaluation builds a NEW memory.
    # --------------------------------------------------------

    memory_ab, out_a, out_b = (
        build_two_fact_memory(
            model,
            tokenizer,
        )
    )

    slot_a = selected_slot(
        out_a
    )

    slot_b = selected_slot(
        out_b
    )

    route_a = route_vector(
        out_a
    )

    route_b = route_vector(
        out_b
    )

    result_a = evaluate_query(
        model,
        tokenizer,
        QUERY_A,
        ANSWER_A,
        memory_ab,
    )

    result_b = evaluate_query(
        model,
        tokenizer,
        QUERY_B,
        ANSWER_B,
        memory_ab,
    )

    read_a = query_reader(
        model,
        tokenizer,
        QUERY_A,
        memory_ab,
    )

    read_b = query_reader(
        model,
        tokenizer,
        QUERY_B,
        memory_ab,
    )

    print(
        f"A WRITE: slot {slot_a}"
    )

    print(
        "A route:",
        round_vector(route_a),
    )

    print()

    print(
        f"B WRITE: slot {slot_b}"
    )

    print(
        "B route:",
        round_vector(route_b),
    )

    print()

    print(
        f"A expected={ANSWER_A:<7} | "
        f"pred={result_a['prediction']:<7} | "
        f"rank={result_a['rank']:2d}/16 | "
        f"loss={result_a['loss']:.4f}"
    )

    print(
        f"B expected={ANSWER_B:<7} | "
        f"pred={result_b['prediction']:<7} | "
        f"rank={result_b['rank']:2d}/16 | "
        f"loss={result_b['loss']:.4f}"
    )

    print()

    print(
        "A READ:",
        round_vector(read_a),
    )

    print(
        "B READ:",
        round_vector(read_b),
    )

    print()

    print(
        "A Top-5:"
    )

    for answer, loss in (
        result_a["scores"][:5]
    ):
        print(
            f"  {answer:>8}: "
            f"{loss:.4f}"
        )

    print()

    print(
        "B Top-5:"
    )

    for answer, loss in (
        result_b["scores"][:5]
    ):
        print(
            f"  {answer:>8}: "
            f"{loss:.4f}"
        )

    print()

    if slot_a != slot_b:

        print(
            "WRITE COLLISION: NO "
            f"(A={slot_a}, B={slot_b})"
        )

    else:

        print(
            "WRITE COLLISION: YES "
            f"(both slot {slot_a})"
        )

    model.train()

    return {
        "memory": memory_ab,
        "slot_a": slot_a,
        "slot_b": slot_b,
        "result_a": result_a,
        "result_b": result_b,
        "read_a": read_a,
        "read_b": read_b,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print(
        "TWO-FACT OCCUPANCY ROUTER TEST"
    )
    print("=" * 90)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Base checkpoint: "
        f"{BASE_CHECKPOINT}"
    )

    print()

    print(
        f"A: {ENTITY_A} -> {ANSWER_A}"
    )

    print(
        f"B: {ENTITY_B} -> {ANSWER_B}"
    )

    print()

    print(
        "NO FORCED WRITE ADDRESS"
    )

    print(
        "NO FORCED READ ADDRESS"
    )

    print(
        "Router: occupancy-aware, top_k=1"
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model()

    configure_training(
        model
    )

    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------

    before = print_state(
        model,
        tokenizer,
        "BEFORE TRAINING",
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("TRAINING")
    print("=" * 90)

    model.train()

    for step in range(
        1,
        STEPS + 1,
    ):

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Sequentially write BOTH facts into SAME memory.
        #
        # No detach.
        # This preserves the computational graph through:
        #
        # A write -> B write -> retrieval losses
        # ----------------------------------------------------

        memory_ab, out_a, out_b = (
            build_two_fact_memory(
                model,
                tokenizer,
            )
        )

        # ----------------------------------------------------
        # Retrieve BOTH facts from SAME final memory.
        # ----------------------------------------------------

        loss_a, query_out_a = (
            retrieval_loss(
                model,
                tokenizer,
                QUERY_A,
                ANSWER_A,
                memory_ab,
            )
        )

        loss_b, query_out_b = (
            retrieval_loss(
                model,
                tokenizer,
                QUERY_B,
                ANSWER_B,
                memory_ab,
            )
        )

        retrieval = (
            loss_a + loss_b
        ) / 2.0

        # ----------------------------------------------------
        # For THIS diagnostic we intentionally use only the
        # normal model loss.
        #
        # We do NOT add:
        # collision loss
        # binding loss
        # forced read
        # forced write
        #
        # The occupancy mechanism must solve write allocation
        # naturally.
        # ----------------------------------------------------

        loss = retrieval

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter
                in model.parameters()
                if parameter.requires_grad
            ],
            max_norm=1.0,
        )

        optimizer.step()

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        if (
            step == 1
            or step % LOG_EVERY == 0
            or step == STEPS
        ):

            model.eval()

            with torch.no_grad():

                eval_memory, eval_a, eval_b = (
                    build_two_fact_memory(
                        model,
                        tokenizer,
                    )
                )

                result_a = evaluate_query(
                    model,
                    tokenizer,
                    QUERY_A,
                    ANSWER_A,
                    eval_memory,
                )

                result_b = evaluate_query(
                    model,
                    tokenizer,
                    QUERY_B,
                    ANSWER_B,
                    eval_memory,
                )

                slot_a = selected_slot(
                    eval_a
                )

                slot_b = selected_slot(
                    eval_b
                )

                route_a = route_vector(
                    eval_a
                )

                route_b = route_vector(
                    eval_b
                )

                read_a = query_reader(
                    model,
                    tokenizer,
                    QUERY_A,
                    eval_memory,
                )

                read_b = query_reader(
                    model,
                    tokenizer,
                    QUERY_B,
                    eval_memory,
                )

            print()
            print("-" * 90)

            print(
                f"STEP {step:03d} | "
                f"retrieval={retrieval.item():.6f} | "
                f"lossA={loss_a.item():.6f} | "
                f"lossB={loss_b.item():.6f}"
            )

            print(
                f"A: pred="
                f"{result_a['prediction']:>7} | "
                f"rank={result_a['rank']:2d}/16 | "
                f"loss={result_a['loss']:.4f} | "
                f"write_slot={slot_a}"
            )

            print(
                f"B: pred="
                f"{result_b['prediction']:>7} | "
                f"rank={result_b['rank']:2d}/16 | "
                f"loss={result_b['loss']:.4f} | "
                f"write_slot={slot_b}"
            )

            print(
                "A write:",
                round_vector(route_a),
            )

            print(
                "B write:",
                round_vector(route_b),
            )

            print(
                "A read :",
                round_vector(read_a),
            )

            print(
                "B read :",
                round_vector(read_b),
            )

            print(
                "Collision:",
                "YES"
                if slot_a == slot_b
                else "NO",
            )

            model.train()

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    final = print_state(
        model,
        tokenizer,
        "FINAL RESULT",
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("AUTOMATIC INTERPRETATION")
    print("=" * 90)

    different_slots = (
        final["slot_a"]
        != final["slot_b"]
    )

    both_correct = (
        final["result_a"]["rank"] == 1
        and
        final["result_b"]["rank"] == 1
    )

    if (
        different_slots
        and both_correct
    ):

        print(
            "SUCCESS."
        )

        print(
            "The occupancy-aware router allocated "
            "A and B to different slots, and the "
            "normal learned reader retrieved both "
            "associations at rank 1."
        )

        print()

        print(
            "This is evidence that occupancy-aware "
            "write allocation substantially fixes "
            "the two-fact addressing failure."
        )

    elif (
        different_slots
        and not both_correct
    ):

        print(
            "PARTIAL SUCCESS."
        )

        print(
            "The occupancy-aware router successfully "
            "prevented the write collision, but both "
            "facts were not retrieved correctly."
        )

        print()

        print(
            "This means WRITE allocation improved, "
            "but learned READ selection/binding is "
            "still the remaining bottleneck."
        )

    elif (
        not different_slots
    ):

        print(
            "WRITE ALLOCATION FAILURE."
        )

        print(
            "Despite occupancy awareness, both facts "
            "were routed to the same slot."
        )

        print()

        print(
            "Inspect write_count propagation and the "
            "occupancy penalty before changing the reader."
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            OUTPUT_CHECKPOINT
        ),
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "step":
                STEPS,

            "experiment":
                "two_fact_occupancy",

            "entity_a":
                ENTITY_A,

            "answer_a":
                ANSWER_A,

            "entity_b":
                ENTITY_B,

            "answer_b":
                ANSWER_B,

            "final_slot_a":
                final["slot_a"],

            "final_slot_b":
                final["slot_b"],

            "final_rank_a":
                final["result_a"]["rank"],

            "final_rank_b":
                final["result_b"]["rank"],
        },
        OUTPUT_CHECKPOINT,
    )

    print()
    print(
        f"Saved: {OUTPUT_CHECKPOINT}"
    )


if __name__ == "__main__":
    main()