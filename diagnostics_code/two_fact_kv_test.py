from __future__ import annotations

from pathlib import Path

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
# SETTINGS
# ================================================================

CHECKPOINT = "outputs/retrieval_gradient_test/checkpoint_best.pt"
OUTPUT = "outputs/two_fact_dynamic_kv.pt"

MODEL_NAME = "gpt2"

STEPS = 300
LEARNING_RATE = 5e-5
SEED = 42


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


A = {
    "entity": "Project-A",
    "answer": "rabbit",
}

B = {
    "entity": "Project-B",
    "answer": "river",
}


for example in (A, B):

    example["fact"] = (
        f"The assigned keyword for {example['entity']} "
        f"is {example['answer']}. "
        f"Remember that the keyword associated with "
        f"{example['entity']} is {example['answer']}."
    )

    example["query"] = (
        f"The assigned keyword for {example['entity']} is"
    )


# ================================================================
# SEED
# ================================================================

def set_seed(seed):

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================================================================
# CONFIG
# ================================================================

def build_config():

    return MemoryGPT2Config(

        num_slots=8,

        gate_type="vector",
        gate_mode="sigmoid",
        gate_init_bias=-2.0,

        # --------------------------------------------------------
        # OCCUPANCY WRITE ALLOCATION
        # --------------------------------------------------------

        router_enabled=True,
        router_mode="occupancy",
        router_top_k=1,
        router_temperature=0.7,

        # --------------------------------------------------------
        # WRITER
        # --------------------------------------------------------

        writer_mode="attention",
        writer_attention_heads=8,

        orthogonal_mode="other_slots",
        orthogonal_strength=0.5,

        # --------------------------------------------------------
        # READER
        # --------------------------------------------------------

        reader_mode="token",
        reader_fusion="gated",
        reader_heads=8,

        # IMPORTANT:
        # Let the dynamic keys decide which written slot to read.
        reader_top_k=None,

        reader_temperature=0.8,

        # --------------------------------------------------------
        # NO EXTRA LOSSES FOR THIS DIAGNOSTIC
        # --------------------------------------------------------

        candidate_diversity_weight=0.0,
        update_orthogonality_weight=0.0,
        router_balance_weight=0.0,
        reader_balance_weight=0.0,
        memory_collapse_weight=0.0,

        # Keep graph through sequential writes.
        detach_memory_between_steps=False,
    )


# ================================================================
# LOAD MODEL
# ================================================================

def load_model(device):

    print(
        "Loading base checkpoint:",
        CHECKPOINT,
    )

    model = (
        MemoryAugmentedGPT2LMHeadModel
        .from_pretrained(
            MODEL_NAME,
            memory_config=build_config(),
        )
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    old_state = checkpoint[
        "model_state_dict"
    ]

    current_state = model.state_dict()

    compatible_state = {}

    skipped_shape = []

    for name, tensor in old_state.items():

        if (
            name in current_state
            and
            current_state[name].shape
            == tensor.shape
        ):

            compatible_state[name] = tensor

        elif name in current_state:

            skipped_shape.append(
                (
                    name,
                    tuple(tensor.shape),
                    tuple(
                        current_state[
                            name
                        ].shape
                    ),
                )
            )

    result = model.load_state_dict(
        compatible_state,
        strict=False,
    )

    print()
    print("Checkpoint load:")
    print(
        "Loaded compatible tensors:",
        len(compatible_state),
    )

    print()
    print("Missing/new keys:")

    if len(result.missing_keys) == 0:

        print("  None")

    else:

        for name in result.missing_keys:

            print(
                " ",
                name,
            )

    if len(result.unexpected_keys) > 0:

        print()
        print("Unexpected keys:")

        for name in result.unexpected_keys:

            print(
                " ",
                name,
            )

    if len(skipped_shape) > 0:

        print()
        print("Shape-skipped keys:")

        for item in skipped_shape:

            print(
                " ",
                item,
            )

    model.to(
        device
    )

    # Freeze GPT-2.
    # Memory architecture stays trainable.
    model.freeze_backbone()

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print()
    print(
        f"Trainable parameters: "
        f"{trainable:,} / {total:,}"
    )

    return model


# ================================================================
# TOKENIZATION
# ================================================================

def tokenize(
    tokenizer,
    text,
    device,
):

    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = (
        encoded["input_ids"]
        .to(device)
    )

    attention_mask = (
        encoded["attention_mask"]
        .to(device)
    )

    return (
        input_ids,
        attention_mask,
    )


def prepare_query_answer(
    tokenizer,
    query,
    answer,
    device,
):

    query_ids = tokenizer(
        query,
        add_special_tokens=False,
    )["input_ids"]

    answer_ids = tokenizer(
        " " + answer,
        add_special_tokens=False,
    )["input_ids"]

    full_ids = (
        query_ids
        +
        answer_ids
    )

    input_ids = torch.tensor(
        [full_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = (
        torch.ones_like(
            input_ids
        )
    )

    labels = (
        input_ids.clone()
    )

    # Ignore query tokens.
    # Loss only on answer.
    labels[
        :,
        :len(query_ids)
    ] = -100

    return (
        input_ids,
        attention_mask,
        labels,
    )


# ================================================================
# WRITE ONE FACT
# ================================================================

def write_fact(
    model,
    tokenizer,
    fact,
    memory_state,
    device,
):

    (
        input_ids,
        attention_mask,
    ) = tokenize(
        tokenizer,
        fact,
        device,
    )

    output = model(

        input_ids=input_ids,

        attention_mask=attention_mask,

        memory_state=memory_state,

        update_memory=True,

        return_diagnostics=False,

        use_cache=False,
    )

    route = None

    slot = None

    if (
        output.routing_output
        is not None
    ):

        route = (
            output
            .routing_output
            .weights[0]
        )

        slot = int(
            route
            .argmax()
            .item()
        )

    return (
        output.memory_state,
        route,
        slot,
    )


# ================================================================
# WRITE BOTH FACTS INTO SAME MEMORY
# ================================================================

def build_two_fact_memory(
    model,
    tokenizer,
    device,
):

    memory_state = None

    (
        memory_state,
        route_a,
        slot_a,
    ) = write_fact(

        model=model,

        tokenizer=tokenizer,

        fact=A["fact"],

        memory_state=memory_state,

        device=device,
    )

    (
        memory_state,
        route_b,
        slot_b,
    ) = write_fact(

        model=model,

        tokenizer=tokenizer,

        fact=B["fact"],

        memory_state=memory_state,

        device=device,
    )

    return (
        memory_state,
        route_a,
        slot_a,
        route_b,
        slot_b,
    )


# ================================================================
# QUERY LOSS
# ================================================================

def query_loss(
    model,
    tokenizer,
    example,
    memory_state,
    device,
):

    (
        input_ids,
        attention_mask,
        labels,
    ) = prepare_query_answer(

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

        update_memory=False,

        return_diagnostics=False,

        use_cache=False,
    )

    return (
        output.lm_loss,
        output,
    )


# ================================================================
# CANDIDATE LOSS
# ================================================================

@torch.no_grad()
def candidate_loss(
    model,
    tokenizer,
    query,
    candidate,
    memory_state,
    device,
):

    (
        input_ids,
        attention_mask,
        labels,
    ) = prepare_query_answer(

        tokenizer=tokenizer,

        query=query,

        answer=candidate,

        device=device,
    )

    output = model(

        input_ids=input_ids,

        attention_mask=attention_mask,

        labels=labels,

        memory_state=memory_state,

        update_memory=False,

        return_diagnostics=False,

        use_cache=False,
    )

    return float(
        output
        .lm_loss
        .detach()
        .cpu()
    )


# ================================================================
# ALL-16 RANKING
# ================================================================

@torch.no_grad()
def rank_example(
    model,
    tokenizer,
    example,
    memory_state,
    device,
):

    scores = {}

    for answer in ANSWERS:

        scores[answer] = (
            candidate_loss(

                model=model,

                tokenizer=tokenizer,

                query=example["query"],

                candidate=answer,

                memory_state=memory_state,

                device=device,
            )
        )

    ordered = sorted(
        scores.items(),
        key=lambda item: item[1],
    )

    rank = None

    for index, (
        answer,
        _
    ) in enumerate(
        ordered
    ):

        if (
            answer
            ==
            example["answer"]
        ):

            rank = index + 1

            break

    return {

        "pred":
            ordered[0][0],

        "rank":
            rank,

        "loss":
            scores[
                example["answer"]
            ],

        "ordered":
            ordered,
    }


# ================================================================
# READ DISTRIBUTION
# ================================================================

def get_read_distribution(
    output,
):

    if (
        output.read_output
        is None
    ):

        return None

    weights = (
        output
        .read_output
        .attention_weights
    )

    if weights is None:

        return (
            output
            .read_output
            .slot_usage[0]
            .detach()
        )

    # Expected:
    # [B, H, T, N]
    #
    # Average over heads and tokens
    # to obtain one distribution over slots.

    distribution = (
        weights[0]
        .mean(dim=0)
        .mean(dim=0)
    )

    return (
        distribution
        .detach()
    )


# ================================================================
# QUERY -> STORED KEY SIMILARITY
# ================================================================

@torch.no_grad()
def query_key_similarity(
    model,
    tokenizer,
    query,
    memory_state,
    device,
):

    (
        input_ids,
        attention_mask,
    ) = tokenize(
        tokenizer,
        query,
        device,
    )

    # Get normal GPT-2 hidden states
    # before memory reading.

    hidden_states = (
        model
        .backbone
        .transformer(

            input_ids=input_ids,

            attention_mask=(
                attention_mask
            ),

            use_cache=False,

            return_dict=True,

        )
        .last_hidden_state
    )

    reader = model.reader

    token_mask = (
        attention_mask.bool()
    )

    queries = (
        reader
        ._build_queries(

            hidden_states=(
                hidden_states
            ),

            token_mask=(
                token_mask
            ),
        )
    )

    last_index = (
        int(
            attention_mask[0]
            .sum()
            .item()
        )
        - 1
    )

    query_vector = (
        queries[
            :,
            last_index:
            last_index + 1,
            :
        ]
    )

    # Reader query projection

    projected_query = (
        reader
        .query_projection(
            query_vector
        )
        .squeeze(1)
    )

    # IMPORTANT:
    # Dynamic semantic keys stored in memory_state.keys

    normalized_keys = (
        reader
        .memory_norm(
            memory_state.keys
        )
    )

    projected_keys = (
        reader
        .key_projection(
            normalized_keys
        )
        .squeeze(0)
    )

    projected_query = (
        F.normalize(
            projected_query,
            p=2,
            dim=-1,
        )
    )

    projected_keys = (
        F.normalize(
            projected_keys,
            p=2,
            dim=-1,
        )
    )

    similarities = (
        torch.matmul(
            projected_keys,
            projected_query[0],
        )
    )

    return (
        similarities
        .detach()
    )


# ================================================================
# RAW STORED KEY COSINE
# ================================================================

@torch.no_grad()
def stored_key_cosine(
    memory_state,
    slot_a,
    slot_b,
):

    key_a = (
        memory_state
        .keys[
            0,
            slot_a
        ]
    )

    key_b = (
        memory_state
        .keys[
            0,
            slot_b
        ]
    )

    cosine = (
        F.cosine_similarity(

            key_a.unsqueeze(0),

            key_b.unsqueeze(0),

            dim=-1,
        )
    )

    return float(
        cosine.item()
    )


# ================================================================
# FORMAT VECTOR
# ================================================================

def format_vector(
    vector,
    digits=4,
):

    if vector is None:

        return None

    return [

        round(
            float(value),
            digits,
        )

        for value
        in (
            vector
            .detach()
            .cpu()
            .tolist()
        )
    ]


# ================================================================
# FULL EVALUATION
# ================================================================

@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    device,
    title,
):

    model.eval()

    (
        memory_state,
        route_a,
        slot_a,
        route_b,
        slot_b,
    ) = build_two_fact_memory(

        model=model,

        tokenizer=tokenizer,

        device=device,
    )

    (
        _,
        output_a,
    ) = query_loss(

        model=model,

        tokenizer=tokenizer,

        example=A,

        memory_state=memory_state,

        device=device,
    )

    (
        _,
        output_b,
    ) = query_loss(

        model=model,

        tokenizer=tokenizer,

        example=B,

        memory_state=memory_state,

        device=device,
    )

    result_a = rank_example(

        model=model,

        tokenizer=tokenizer,

        example=A,

        memory_state=memory_state,

        device=device,
    )

    result_b = rank_example(

        model=model,

        tokenizer=tokenizer,

        example=B,

        memory_state=memory_state,

        device=device,
    )

    read_a = (
        get_read_distribution(
            output_a
        )
    )

    read_b = (
        get_read_distribution(
            output_b
        )
    )

    similarity_a = (
        query_key_similarity(

            model=model,

            tokenizer=tokenizer,

            query=A["query"],

            memory_state=(
                memory_state
            ),

            device=device,
        )
    )

    similarity_b = (
        query_key_similarity(

            model=model,

            tokenizer=tokenizer,

            query=B["query"],

            memory_state=(
                memory_state
            ),

            device=device,
        )
    )

    written_mask = (

        memory_state
        .write_count[0]

        > 0
    )

    print()
    print(
        "=" * 90
    )

    print(
        title
    )

    print(
        "=" * 90
    )

    print()

    print(
        f"A WRITE: slot "
        f"{slot_a}"
    )

    print(
        "A route:",
        format_vector(
            route_a
        ),
    )

    print()

    print(
        f"B WRITE: slot "
        f"{slot_b}"
    )

    print(
        "B route:",
        format_vector(
            route_b
        ),
    )

    print()

    print(
        "WRITE COUNT:",
        (
            memory_state
            .write_count[0]
            .detach()
            .cpu()
            .tolist()
        ),
    )

    print(
        "WRITTEN MASK:",
        (
            written_mask
            .detach()
            .cpu()
            .tolist()
        ),
    )

    print()

    if (
        slot_a is not None
        and
        slot_b is not None
    ):

        key_cosine = (
            stored_key_cosine(

                memory_state,

                slot_a,

                slot_b,
            )
        )

        print(
            "RAW STORED KEY COSINE "
            "K_A vs K_B:",
            round(
                key_cosine,
                6,
            ),
        )

    print()

    print(
        "A QUERY -> similarity "
        "to stored keys:"
    )

    print(
        format_vector(
            similarity_a
        )
    )

    print()

    print(
        "B QUERY -> similarity "
        "to stored keys:"
    )

    print(
        format_vector(
            similarity_b
        )
    )

    print()

    print(
        "A READ:",
        format_vector(
            read_a
        ),
    )

    print(
        "B READ:",
        format_vector(
            read_b
        ),
    )

    if (
        read_a is not None
        and
        read_b is not None
    ):

        read_cosine = (
            F.cosine_similarity(

                read_a.unsqueeze(0),

                read_b.unsqueeze(0),

                dim=-1,
            )
        )

        print(
            "READ cosine A vs B:",
            round(
                float(
                    read_cosine.item()
                ),
                6,
            ),
        )

    print()

    print(

        f"A expected="
        f"{A['answer']:<7} | "

        f"pred="
        f"{result_a['pred']:<7} | "

        f"rank="
        f"{result_a['rank']:2d}/16 | "

        f"loss="
        f"{result_a['loss']:.4f}"
    )

    print(

        f"B expected="
        f"{B['answer']:<7} | "

        f"pred="
        f"{result_b['pred']:<7} | "

        f"rank="
        f"{result_b['rank']:2d}/16 | "

        f"loss="
        f"{result_b['loss']:.4f}"
    )

    print()

    print(
        "A Top-5:"
    )

    for (
        answer,
        loss,
    ) in (
        result_a[
            "ordered"
        ][:5]
    ):

        print(
            f"  "
            f"{answer:>8}: "
            f"{loss:.4f}"
        )

    print()

    print(
        "B Top-5:"
    )

    for (
        answer,
        loss,
    ) in (
        result_b[
            "ordered"
        ][:5]
    ):

        print(
            f"  "
            f"{answer:>8}: "
            f"{loss:.4f}"
        )

    collision = (
        slot_a
        ==
        slot_b
    )

    print()

    print(

        "WRITE COLLISION: "

        + (
            "YES"
            if collision
            else "NO"
        )

        + f" "
        f"(A={slot_a}, "
        f"B={slot_b})"
    )

    return {

        "memory_state":
            memory_state,

        "slot_a":
            slot_a,

        "slot_b":
            slot_b,

        "read_a":
            read_a,

        "read_b":
            read_b,

        "result_a":
            result_a,

        "result_b":
            result_b,
    }


# ================================================================
# MAIN
# ================================================================

def main():

    set_seed(
        SEED
    )

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )

    print(
        "=" * 90
    )

    print(
        "TWO-FACT DYNAMIC "
        "KEY-VALUE MEMORY TEST"
    )

    print(
        "=" * 90
    )

    print(
        "Device:",
        device,
    )

    print(
        "Base checkpoint:",
        CHECKPOINT,
    )

    print()

    print(
        f"A: "
        f"{A['entity']} "
        f"-> "
        f"{A['answer']}"
    )

    print(
        f"B: "
        f"{B['entity']} "
        f"-> "
        f"{B['answer']}"
    )

    print()

    print(
        "NO FORCED WRITE ADDRESS"
    )

    print(
        "NO FORCED READ ADDRESS"
    )

    print(
        "Router: "
        "occupancy-aware, "
        "top_k=1"
    )

    print(
        "Memory: "
        "dynamic semantic keys "
        "+ stored values"
    )

    # ------------------------------------------------------------
    # TOKENIZER
    # ------------------------------------------------------------

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    if (
        tokenizer.pad_token
        is None
    ):

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    model = load_model(
        device
    )

    # ------------------------------------------------------------
    # BEFORE TRAINING
    # ------------------------------------------------------------

    evaluate(

        model=model,

        tokenizer=tokenizer,

        device=device,

        title="BEFORE TRAINING",
    )

    # ------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------

    trainable_parameters = [

        parameter

        for parameter
        in model.parameters()

        if parameter.requires_grad
    ]

    optimizer = AdamW(

        trainable_parameters,

        lr=LEARNING_RATE,

        weight_decay=0.0,
    )

    print()

    print(
        "=" * 90
    )

    print(
        "TRAINING"
    )

    print(
        "=" * 90
    )

    # ============================================================
    # TRAIN
    # ============================================================

    for step in range(
        1,
        STEPS + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        # --------------------------------------------------------
        # Fresh memory every step.
        #
        # BUT:
        # A and B are sequentially written into the SAME memory.
        # --------------------------------------------------------

        (
            memory_state,
            route_a,
            slot_a,
            route_b,
            slot_b,
        ) = build_two_fact_memory(

            model=model,

            tokenizer=tokenizer,

            device=device,
        )

        # --------------------------------------------------------
        # RETRIEVE A
        # --------------------------------------------------------

        (
            loss_a,
            _,
        ) = query_loss(

            model=model,

            tokenizer=tokenizer,

            example=A,

            memory_state=memory_state,

            device=device,
        )

        # --------------------------------------------------------
        # RETRIEVE B
        # --------------------------------------------------------

        (
            loss_b,
            _,
        ) = query_loss(

            model=model,

            tokenizer=tokenizer,

            example=B,

            memory_state=memory_state,

            device=device,
        )

        # --------------------------------------------------------
        # BOTH MUST WORK SIMULTANEOUSLY
        # --------------------------------------------------------

        retrieval_loss = (

            0.5

            * (
                loss_a
                +
                loss_b
            )
        )

        retrieval_loss.backward()

        gradient_norm = (
            clip_grad_norm_(

                trainable_parameters,

                max_norm=1.0,
            )
        )

        optimizer.step()

        # --------------------------------------------------------
        # LOGGING
        # --------------------------------------------------------

        if (
            step == 1
            or
            step % 25 == 0
        ):

            model.eval()

            with torch.no_grad():

                (
                    eval_memory,
                    eval_route_a,
                    eval_slot_a,
                    eval_route_b,
                    eval_slot_b,
                ) = (
                    build_two_fact_memory(

                        model=model,

                        tokenizer=(
                            tokenizer
                        ),

                        device=device,
                    )
                )

                (
                    _,
                    output_a,
                ) = query_loss(

                    model=model,

                    tokenizer=(
                        tokenizer
                    ),

                    example=A,

                    memory_state=(
                        eval_memory
                    ),

                    device=device,
                )

                (
                    _,
                    output_b,
                ) = query_loss(

                    model=model,

                    tokenizer=(
                        tokenizer
                    ),

                    example=B,

                    memory_state=(
                        eval_memory
                    ),

                    device=device,
                )

                result_a = (
                    rank_example(

                        model=model,

                        tokenizer=(
                            tokenizer
                        ),

                        example=A,

                        memory_state=(
                            eval_memory
                        ),

                        device=device,
                    )
                )

                result_b = (
                    rank_example(

                        model=model,

                        tokenizer=(
                            tokenizer
                        ),

                        example=B,

                        memory_state=(
                            eval_memory
                        ),

                        device=device,
                    )
                )

                read_a = (
                    get_read_distribution(
                        output_a
                    )
                )

                read_b = (
                    get_read_distribution(
                        output_b
                    )
                )

                similarity_a = (
                    query_key_similarity(

                        model=model,

                        tokenizer=(
                            tokenizer
                        ),

                        query=A[
                            "query"
                        ],

                        memory_state=(
                            eval_memory
                        ),

                        device=device,
                    )
                )

                similarity_b = (
                    query_key_similarity(

                        model=model,

                        tokenizer=(
                            tokenizer
                        ),

                        query=B[
                            "query"
                        ],

                        memory_state=(
                            eval_memory
                        ),

                        device=device,
                    )
                )

                print()

                print(
                    "-" * 90
                )

                print(

                    f"STEP "
                    f"{step:03d} | "

                    f"retrieval="
                    f"{float(retrieval_loss.detach()):.6f} | "

                    f"lossA="
                    f"{float(loss_a.detach()):.6f} | "

                    f"lossB="
                    f"{float(loss_b.detach()):.6f} | "

                    f"grad="
                    f"{float(gradient_norm):.4f}"
                )

                print(

                    f"A: "
                    f"pred="
                    f"{result_a['pred']:>7} | "

                    f"rank="
                    f"{result_a['rank']:2d}/16 | "

                    f"loss="
                    f"{result_a['loss']:.4f} | "

                    f"write_slot="
                    f"{eval_slot_a}"
                )

                print(

                    f"B: "
                    f"pred="
                    f"{result_b['pred']:>7} | "

                    f"rank="
                    f"{result_b['rank']:2d}/16 | "

                    f"loss="
                    f"{result_b['loss']:.4f} | "

                    f"write_slot="
                    f"{eval_slot_b}"
                )

                print(
                    "A write:",
                    format_vector(
                        eval_route_a
                    ),
                )

                print(
                    "B write:",
                    format_vector(
                        eval_route_b
                    ),
                )

                print(
                    "A key-sim:",
                    format_vector(
                        similarity_a
                    ),
                )

                print(
                    "B key-sim:",
                    format_vector(
                        similarity_b
                    ),
                )

                print(
                    "A read:",
                    format_vector(
                        read_a
                    ),
                )

                print(
                    "B read:",
                    format_vector(
                        read_b
                    ),
                )

                print(
                    "Collision:",
                    (
                        "YES"

                        if (
                            eval_slot_a
                            ==
                            eval_slot_b
                        )

                        else "NO"
                    ),
                )

    # ============================================================
    # FINAL EVALUATION
    # ============================================================

    final = evaluate(

        model=model,

        tokenizer=tokenizer,

        device=device,

        title="FINAL RESULT",
    )

    # ============================================================
    # SAVE
    # ============================================================

    Path(
        OUTPUT
    ).parent.mkdir(

        parents=True,

        exist_ok=True,
    )

    torch.save(

        {

            "model_state_dict":
                model.state_dict(),

            "steps":
                STEPS,

            "learning_rate":
                LEARNING_RATE,

            "fact_a":
                A,

            "fact_b":
                B,
        },

        OUTPUT,
    )

    # ============================================================
    # INTERPRETATION
    # ============================================================

    print()

    print(
        "=" * 90
    )

    print(
        "AUTOMATIC INTERPRETATION"
    )

    print(
        "=" * 90
    )

    a_correct = (

        final[
            "result_a"
        ][
            "rank"
        ]

        == 1
    )

    b_correct = (

        final[
            "result_b"
        ][
            "rank"
        ]

        == 1
    )

    no_collision = (

        final[
            "slot_a"
        ]

        !=

        final[
            "slot_b"
        ]
    )

    if (
        no_collision
        and
        a_correct
        and
        b_correct
    ):

        print(
            "SUCCESS."
        )

        print(
            "Both associations are rank-1 "
            "inside the SAME memory."
        )

        print(
            "Write addresses are different "
            "and learned read addressing "
            "successfully distinguishes them."
        )

        print()

        print(
            "NEXT TEST:"
        )

        print(
            "Train on many entities and "
            "evaluate on completely unseen "
            "entity names."
        )

    elif no_collision:

        print(
            "PARTIAL SUCCESS."
        )

        print(
            "Write allocation is separated, "
            "but dynamic key/read binding "
            "still does not retrieve both "
            "associations at rank-1."
        )

        print()

        print(
            "DO NOT immediately add another loss."
        )

        print(
            "Inspect A/B key similarity and "
            "A/B read distributions printed "
            "above first."
        )

    else:

        print(
            "FAILURE."
        )

        print(
            "Write collision returned."
        )

        print(
            "The occupancy allocator failed "
            "to keep the two facts in "
            "different slots."
        )

    print()

    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":

    main()