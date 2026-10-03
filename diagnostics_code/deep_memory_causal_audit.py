# ============================================================
# DEEP MEMORY CAUSAL AUDIT
#
# Purpose:
#   Diagnose exactly WHERE information is lost in the
#   Memory-Augmented GPT-2 pipeline.
#
# DOES NOT:
#   - train the model
#   - modify models/
#   - overwrite checkpoints
#
# Tests:
#   1. GPT-2 representation separation
#   2. Pooling collapse
#   3. Write routing
#   4. Writer candidate separation
#   5. Orthogonalization effects
#   6. Effective memory updates
#   7. Stored-slot separation
#   8. Reader Q/K/V geometry
#   9. Raw QK addressing
#  10. Confidence effects
#  11. Attention collapse
#  12. Context collapse
#  13. Fusion strength
#  14. Output-token ranks
#  15. Correct/wrong/zero/random/shuffled memory sensitivity
#  16. Gradient-flow audit
#
# Run:
#   python deep_memory_causal_audit.py 2>&1 | tee deep_memory_causal_audit.log
# ============================================================

import math
import random
from copy import deepcopy

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

CHECKPOINT = (
    "outputs/retrieval_gradient_test/"
    "checkpoint_best.pt"
)

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

A_ENTITY = "Project-A"
A_ANSWER = "rabbit"

B_ENTITY = "Project-B"
B_ANSWER = "river"


# ============================================================
# UTILITIES
# ============================================================

def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def subsection(title):
    print()
    print("-" * 100)
    print(title)
    print("-" * 100)


def cosine(a, b):

    a = a.float().reshape(-1)
    b = b.float().reshape(-1)

    return float(
        F.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0),
            dim=-1,
        ).item()
    )


def relative_delta(a, b):

    a = a.float()
    b = b.float()

    numerator = (
        a - b
    ).norm()

    denominator = (
        a.norm()
        + 1e-8
    )

    return float(
        numerator / denominator
    )


def tensor_norm(x):
    return float(
        x.float().norm().item()
    )


def fmt(x):
    return f"{x:.8f}"


def probability_list(x):
    return [
        round(float(v), 6)
        for v in x.detach().float().cpu().tolist()
    ]


def safe_mean(x):

    if x is None:
        return float("nan")

    return float(
        x.detach().float().mean().item()
    )


def safe_max(x):

    if x is None:
        return float("nan")

    return float(
        x.detach().float().max().item()
    )


def set_seed(seed):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# TEXT
# ============================================================

def fact_text(entity, answer):

    return (
        f"The assigned keyword for {entity} "
        f"is {answer}. Remember that the keyword "
        f"associated with {entity} is {answer}."
    )


def query_text(entity):

    return (
        f"The assigned keyword for {entity} is"
    )


FACT_A = fact_text(
    A_ENTITY,
    A_ANSWER,
)

FACT_B = fact_text(
    B_ENTITY,
    B_ANSWER,
)

QUERY_A = query_text(
    A_ENTITY,
)

QUERY_B = query_text(
    B_ENTITY,
)


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def encode(text):

    batch = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    )

    return {
        k: v.to(DEVICE)
        for k, v in batch.items()
    }


# ============================================================
# MODEL CONFIG
# ============================================================

def build_config():

    return MemoryGPT2Config(

        num_slots=8,

        gate_type="vector",

        gate_mode="sigmoid",

        gate_init_bias=-2.0,

        router_enabled=True,

        router_mode="occupancy",

        router_top_k=1,

        router_temperature=0.7,

        writer_mode="attention",

        writer_attention_heads=8,

        orthogonal_mode="other_slots",

        orthogonal_strength=0.5,

        reader_mode="token",

        reader_fusion="gated",

        reader_heads=8,

        reader_top_k=3,

        reader_temperature=0.8,

        candidate_diversity_weight=0.0,

        update_orthogonality_weight=0.0,

        router_balance_weight=0.0,

        reader_balance_weight=0.0,

        memory_collapse_weight=0.0,

        detach_memory_between_steps=False,
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    section("LOAD MODEL")

    print("Device:", DEVICE)

    print("Base:", MODEL_NAME)

    print(
        "Checkpoint:",
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
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:
        state = checkpoint[
            "state_dict"
        ]

    else:
        state = checkpoint

    current = model.state_dict()

    compatible = {}

    incompatible = []

    for name, value in state.items():

        if (
            name in current
            and current[name].shape
            == value.shape
        ):

            compatible[name] = value

        else:
            incompatible.append(name)

    result = model.load_state_dict(
        compatible,
        strict=False,
    )

    print(
        "Compatible tensors:",
        len(compatible),
    )

    print(
        "Missing:",
        len(result.missing_keys),
    )

    print(
        "Incompatible/skipped:",
        len(incompatible),
    )

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# INITIAL MEMORY
# ============================================================

def fresh_memory(model):

    dtype = next(
        model.parameters()
    ).dtype

    return model.initialize_memory(
        batch_size=1,
        device=DEVICE,
        dtype=dtype,
    )


# ============================================================
# GPT-2 HIDDEN STATES
# ============================================================

@torch.no_grad()
def backbone_hidden(
    model,
    text,
    output_hidden_states=False,
):

    batch = encode(text)

    out = model.backbone.transformer(

        input_ids=batch[
            "input_ids"
        ],

        attention_mask=batch[
            "attention_mask"
        ],

        use_cache=False,

        output_hidden_states=(
            output_hidden_states
        ),

        return_dict=True,
    )

    return batch, out


# ============================================================
# LAST TOKEN
# ============================================================

def last_token_hidden(
    hidden,
    attention_mask,
):

    index = (
        attention_mask
        .sum(dim=1)
        .long()
        - 1
    )

    batch_index = torch.arange(
        hidden.size(0),
        device=hidden.device,
    )

    return hidden[
        batch_index,
        index,
    ]


# ============================================================
# MASKED MEAN
# ============================================================

def masked_mean(
    hidden,
    mask,
):

    weights = (
        mask.unsqueeze(-1)
        .to(hidden.dtype)
    )

    return (
        hidden * weights
    ).sum(dim=1) / (
        weights.sum(dim=1)
        .clamp_min(1.0)
    )


# ============================================================
# TOKEN RANK
# ============================================================

def answer_token_id(answer):

    ids = tokenizer.encode(
        " " + answer,
        add_special_tokens=False,
    )

    if len(ids) == 0:
        raise RuntimeError(
            f"No token for {answer}"
        )

    return ids[0]


def rank_token(
    logits,
    token_id,
):

    values = logits.float()

    target = values[
        token_id
    ]

    rank = int(
        (values > target)
        .sum()
        .item()
    ) + 1

    probability = float(
        torch.softmax(
            values,
            dim=-1,
        )[token_id].item()
    )

    return rank, probability


# ============================================================
# REPRESENTATION AUDIT
# ============================================================

@torch.no_grad()
def representation_audit(model):

    section(
        "1. GPT-2 REPRESENTATION TRAJECTORY"
    )

    pairs = [

        (
            "FACT A vs FACT B",
            FACT_A,
            FACT_B,
        ),

        (
            "QUERY A vs QUERY B",
            QUERY_A,
            QUERY_B,
        ),
    ]

    for title, text_a, text_b in pairs:

        subsection(title)

        ba, oa = backbone_hidden(
            model,
            text_a,
            output_hidden_states=True,
        )

        bb, ob = backbone_hidden(
            model,
            text_b,
            output_hidden_states=True,
        )

        hidden_a = (
            oa.hidden_states
        )

        hidden_b = (
            ob.hidden_states
        )

        max_layers = min(
            len(hidden_a),
            len(hidden_b),
        )

        print(
            f"{'Layer':<10}"
            f"{'Last cosine':>18}"
            f"{'Mean cosine':>18}"
            f"{'Last rel Δ':>18}"
        )

        for layer in range(
            max_layers
        ):

            ha = hidden_a[layer]
            hb = hidden_b[layer]

            la = last_token_hidden(
                ha,
                ba["attention_mask"],
            )

            lb = last_token_hidden(
                hb,
                bb["attention_mask"],
            )

            ma = masked_mean(
                ha,
                ba["attention_mask"],
            )

            mb = masked_mean(
                hb,
                bb["attention_mask"],
            )

            print(
                f"{layer:<10}"
                f"{cosine(la, lb):>18.8f}"
                f"{cosine(ma, mb):>18.8f}"
                f"{relative_delta(la, lb):>18.8f}"
            )

        final_a = (
            oa.last_hidden_state
        )

        final_b = (
            ob.last_hidden_state
        )

        pooled_a = model._pool_hidden(
            final_a,
            ba["attention_mask"],
        )

        pooled_b = model._pool_hidden(
            final_b,
            bb["attention_mask"],
        )

        print()
        print(
            "MODEL _pool_hidden cosine:",
            fmt(
                cosine(
                    pooled_a,
                    pooled_b,
                )
            ),
        )

        print(
            "MODEL _pool_hidden relative Δ:",
            fmt(
                relative_delta(
                    pooled_a,
                    pooled_b,
                )
            ),
        )


# ============================================================
# WRITE ONE FACT WITH FULL INTERMEDIATES
# ============================================================

@torch.no_grad()
def inspect_write(
    model,
    memory,
    text,
):

    batch, out = backbone_hidden(
        model,
        text,
    )

    base_hidden = (
        out.last_hidden_state
    )

    summary = model._pool_hidden(
        base_hidden,
        batch["attention_mask"],
    )

    routing = model.router(
        query=summary,

        memory_slots=memory.slots,

        slot_mask=None,

        write_count=(
            memory.write_count
        ),
    )

    routing_weights = (
        routing.weights
    )

    writer = model.writer(

        summary=summary,

        memory_slots=(
            memory.slots
        ),

        token_states=(
            base_hidden
        ),

        attention_mask=(
            batch["attention_mask"]
        ),

        routing_weights=(
            routing_weights
        ),
    )

    orth = model.orthogonalizer(

        updates=writer.deltas,

        memory_slots=(
            memory.slots
        ),
    )

    projected_candidate = (
        memory.slots
        + orth.updates
    )

    raw_gate = (
        model.write_gate_module(
            summary,
            slot_mask=None,
        )
    )

    effective_gate = (
        raw_gate
        * routing_weights.unsqueeze(-1)
    )

    confidence = (
        model.write_confidence_head(
            projected_candidate
        ).squeeze(-1)
    )

    selected_slot = int(
        routing_weights[
            0
        ].argmax().item()
    )

    old_slot = (
        memory.slots[
            0,
            selected_slot,
        ].clone()
    )

    new_memory = model.memory_bank(

        state=memory,

        candidate=(
            projected_candidate
        ),

        write_gate=(
            effective_gate
        ),

        write_mask=(
            routing.mask
            .unsqueeze(-1)
        ),

        confidence=confidence,
    )

    new_slot = (
        new_memory.slots[
            0,
            selected_slot,
        ].clone()
    )

    effective_update = (
        new_slot
        - old_slot
    )

    return {

        "batch": batch,

        "base_hidden":
            base_hidden,

        "summary":
            summary,

        "routing":
            routing,

        "writer":
            writer,

        "orth":
            orth,

        "projected_candidate":
            projected_candidate,

        "raw_gate":
            raw_gate,

        "effective_gate":
            effective_gate,

        "confidence":
            confidence,

        "selected_slot":
            selected_slot,

        "old_slot":
            old_slot,

        "new_slot":
            new_slot,

        "effective_update":
            effective_update,

        "memory":
            new_memory,
    }


# ============================================================
# WRITE PATH AUDIT
# ============================================================

@torch.no_grad()
def write_audit(model):

    section(
        "2. WRITE PATH AUDIT"
    )

    memory = fresh_memory(
        model
    )

    A = inspect_write(
        model,
        memory,
        FACT_A,
    )

    memory_after_A = (
        A["memory"]
    )

    B = inspect_write(
        model,
        memory_after_A,
        FACT_B,
    )

    memory_after_AB = (
        B["memory"]
    )

    print(
        "A selected slot:",
        A["selected_slot"],
    )

    print(
        "A route:",
        probability_list(
            A["routing"].weights[0]
        ),
    )

    print()

    print(
        "B selected slot:",
        B["selected_slot"],
    )

    print(
        "B route:",
        probability_list(
            B["routing"].weights[0]
        ),
    )

    print()

    collision = (
        A["selected_slot"]
        == B["selected_slot"]
    )

    print(
        "WRITE COLLISION:",
        "YES" if collision
        else "NO",
    )

    subsection(
        "WRITE REPRESENTATION SEPARATION"
    )

    print(
        "Fact summary cosine:",
        fmt(
            cosine(
                A["summary"],
                B["summary"],
            )
        ),
    )

    print(
        "Fact summary relative Δ:",
        fmt(
            relative_delta(
                A["summary"],
                B["summary"],
            )
        ),
    )

    sa = A["selected_slot"]
    sb = B["selected_slot"]

    candidate_A = (
        A["writer"]
        .candidates[0, sa]
    )

    candidate_B = (
        B["writer"]
        .candidates[0, sb]
    )

    delta_A = (
        A["writer"]
        .deltas[0, sa]
    )

    delta_B = (
        B["writer"]
        .deltas[0, sb]
    )

    orth_A = (
        A["orth"]
        .updates[0, sa]
    )

    orth_B = (
        B["orth"]
        .updates[0, sb]
    )

    print(
        "Writer candidate cosine:",
        fmt(
            cosine(
                candidate_A,
                candidate_B,
            )
        ),
    )

    print(
        "Writer delta cosine:",
        fmt(
            cosine(
                delta_A,
                delta_B,
            )
        ),
    )

    print(
        "Orthogonal update cosine:",
        fmt(
            cosine(
                orth_A,
                orth_B,
            )
        ),
    )

    print(
        "Actual stored-slot cosine:",
        fmt(
            cosine(
                memory_after_AB.slots[
                    0,
                    sa,
                ],
                memory_after_AB.slots[
                    0,
                    sb,
                ],
            )
        ),
    )

    subsection(
        "WRITE MAGNITUDES"
    )

    print(
        "A raw gate mean:",
        fmt(
            safe_mean(
                A["raw_gate"][
                    0,
                    sa,
                ]
            )
        ),
    )

    print(
        "A effective gate mean:",
        fmt(
            safe_mean(
                A["effective_gate"][
                    0,
                    sa,
                ]
            )
        ),
    )

    print(
        "A effective update norm:",
        fmt(
            tensor_norm(
                A[
                    "effective_update"
                ]
            )
        ),
    )

    print()

    print(
        "B raw gate mean:",
        fmt(
            safe_mean(
                B["raw_gate"][
                    0,
                    sb,
                ]
            )
        ),
    )

    print(
        "B effective gate mean:",
        fmt(
            safe_mean(
                B["effective_gate"][
                    0,
                    sb,
                ]
            )
        ),
    )

    print(
        "B effective update norm:",
        fmt(
            tensor_norm(
                B[
                    "effective_update"
                ]
            )
        ),
    )

    print()

    print(
        "A confidence:",
        fmt(
            float(
                A["confidence"][
                    0,
                    sa,
                ].item()
            )
        ),
    )

    print(
        "B confidence:",
        fmt(
            float(
                B["confidence"][
                    0,
                    sb,
                ].item()
            )
        ),
    )

    print()

    print(
        "Final write count:",
        memory_after_AB
        .write_count[0]
        .detach()
        .cpu()
        .tolist(),
    )

    return (
        memory_after_AB,
        A,
        B,
    )


# ============================================================
# READER INTERNAL COMPUTATION
# ============================================================

@torch.no_grad()
def reader_internals(
    model,
    memory,
    query,
    use_confidence=True,
    use_top_k=True,
):

    batch, out = backbone_hidden(
        model,
        query,
    )

    hidden = (
        out.last_hidden_state
    )

    reader = model.reader

    batch_size = (
        hidden.size(0)
    )

    sequence_length = (
        hidden.size(1)
    )

    token_mask = (
        reader._prepare_attention_mask(
            batch["attention_mask"],
            batch_size=batch_size,
            sequence_length=(
                sequence_length
            ),
            device=hidden.device,
        )
    )

    queries = (
        reader._build_queries(
            hidden_states=hidden,
            token_mask=token_mask,
        )
    )

    normalized_memory = (
        reader.memory_norm(
            memory.slots
        )
    )

    q_projected = (
        reader.query_projection(
            queries
        )
    )

    k_projected = (
        reader.key_projection(
            normalized_memory
        )
    )

    v_projected = (
        reader.value_projection(
            normalized_memory
        )
    )

    q = reader._split_heads(
        q_projected
    )

    k = reader._split_heads(
        k_projected
    )

    v = reader._split_heads(
        v_projected
    )

    raw_scores = torch.einsum(
        "bhtd,bhnd->bhtn",
        q,
        k,
    )

    raw_scores = (
        raw_scores
        / math.sqrt(
            reader.head_dim
        )
    )

    raw_scores = (
        raw_scores
        / reader.temperature
    )

    confidence_scores = (
        raw_scores.clone()
    )

    if use_confidence:

        confidence = (
            memory.confidence
            .clamp_min(0.05)
        )

        confidence_scores = (
            confidence_scores
            + confidence
            .clamp_min(reader.eps)
            .log()
            .unsqueeze(1)
            .unsqueeze(2)
        )

    final_scores = (
        confidence_scores.clone()
    )

    selected_indices = None

    if (
        use_top_k
        and reader.top_k
        is not None
        and reader.top_k
        < reader.num_slots
    ):

        final_scores, selected_indices = (
            reader._apply_top_k(
                final_scores
            )
        )

    weights = torch.softmax(
        final_scores,
        dim=-1,
    )

    context_heads = (
        torch.einsum(
            "bhtn,bhnd->bhtd",
            weights,
            v,
        )
    )

    context = (
        reader._merge_heads(
            context_heads
        )
    )

    context = (
        reader.output_projection(
            context
        )
    )

    context = (
        reader.output_norm(
            context
        )
    )

    read_confidence = (
        reader._compute_read_confidence(
            context=context,
            weights=weights,
        )
    )

    fused = reader._fuse(
        hidden_states=hidden,
        context=context,
        read_confidence=(
            read_confidence
        ),
    )

    last_index = int(
        batch[
            "attention_mask"
        ].sum().item()
        - 1
    )

    return {

        "batch":
            batch,

        "hidden":
            hidden,

        "queries":
            queries,

        "q_projected":
            q_projected,

        "k_projected":
            k_projected,

        "v_projected":
            v_projected,

        "q":
            q,

        "k":
            k,

        "v":
            v,

        "raw_scores":
            raw_scores,

        "confidence_scores":
            confidence_scores,

        "final_scores":
            final_scores,

        "weights":
            weights,

        "context":
            context,

        "fused":
            fused,

        "read_confidence":
            read_confidence,

        "selected_indices":
            selected_indices,

        "last_index":
            last_index,
    }


# ============================================================
# SCORE PER SLOT FOR LAST QUERY TOKEN
# ============================================================

def mean_head_slot_scores(
    scores,
    token_index,
):

    return (
        scores[
            0,
            :,
            token_index,
            :,
        ]
        .mean(dim=0)
    )


# ============================================================
# READ PATH AUDIT
# ============================================================

@torch.no_grad()
def read_audit(
    model,
    memory,
    write_A,
    write_B,
):

    section(
        "3. READ PATH AUDIT"
    )

    A = reader_internals(
        model,
        memory,
        QUERY_A,
    )

    B = reader_internals(
        model,
        memory,
        QUERY_B,
    )

    ia = A["last_index"]
    ib = B["last_index"]

    qa = A["q_projected"][
        0,
        ia,
    ]

    qb = B["q_projected"][
        0,
        ib,
    ]

    print(
        "Raw query hidden cosine:",
        fmt(
            cosine(
                A["hidden"][
                    0,
                    ia,
                ],
                B["hidden"][
                    0,
                    ib,
                ],
            )
        ),
    )

    print(
        "Built query cosine:",
        fmt(
            cosine(
                A["queries"][
                    0,
                    ia,
                ],
                B["queries"][
                    0,
                    ib,
                ],
            )
        ),
    )

    print(
        "Projected Q cosine:",
        fmt(
            cosine(
                qa,
                qb,
            )
        ),
    )

    sa = (
        write_A[
            "selected_slot"
        ]
    )

    sb = (
        write_B[
            "selected_slot"
        ]
    )

    subsection(
        "KEY/VALUE GEOMETRY"
    )

    ka = A["k_projected"][
        0,
        sa,
    ]

    kb = A["k_projected"][
        0,
        sb,
    ]

    va = A["v_projected"][
        0,
        sa,
    ]

    vb = A["v_projected"][
        0,
        sb,
    ]

    print(
        "Stored key cosine:",
        fmt(
            cosine(
                ka,
                kb,
            )
        ),
    )

    print(
        "Stored value cosine:",
        fmt(
            cosine(
                va,
                vb,
            )
        ),
    )

    subsection(
        "RAW QK SCORE MATRIX"
    )

    score_A = (
        mean_head_slot_scores(
            A["raw_scores"],
            ia,
        )
    )

    score_B = (
        mean_head_slot_scores(
            B["raw_scores"],
            ib,
        )
    )

    print(
        "A query raw scores:",
        probability_list(
            score_A
        ),
    )

    print(
        "B query raw scores:",
        probability_list(
            score_B
        ),
    )

    print()

    print(
        f"A correct slot {sa}:",
        fmt(
            float(
                score_A[sa].item()
            )
        ),
    )

    print(
        f"A competing B slot {sb}:",
        fmt(
            float(
                score_A[sb].item()
            )
        ),
    )

    print(
        "A QK margin:",
        fmt(
            float(
                (
                    score_A[sa]
                    - score_A[sb]
                ).item()
            )
        ),
    )

    print()

    print(
        f"B correct slot {sb}:",
        fmt(
            float(
                score_B[sb].item()
            )
        ),
    )

    print(
        f"B competing A slot {sa}:",
        fmt(
            float(
                score_B[sa].item()
            )
        ),
    )

    print(
        "B QK margin:",
        fmt(
            float(
                (
                    score_B[sb]
                    - score_B[sa]
                ).item()
            )
        ),
    )

    subsection(
        "AFTER CONFIDENCE"
    )

    conf_A = (
        mean_head_slot_scores(
            A[
                "confidence_scores"
            ],
            ia,
        )
    )

    conf_B = (
        mean_head_slot_scores(
            B[
                "confidence_scores"
            ],
            ib,
        )
    )

    print(
        "A scores:",
        probability_list(
            conf_A
        ),
    )

    print(
        "B scores:",
        probability_list(
            conf_B
        ),
    )

    subsection(
        "FINAL ATTENTION"
    )

    attention_A = (
        A["weights"][
            0,
            :,
            ia,
            :,
        ]
        .mean(dim=0)
    )

    attention_B = (
        B["weights"][
            0,
            :,
            ib,
            :,
        ]
        .mean(dim=0)
    )

    print(
        "A attention:",
        probability_list(
            attention_A
        ),
    )

    print(
        "B attention:",
        probability_list(
            attention_B
        ),
    )

    print()

    print(
        "A correct-slot attention:",
        fmt(
            float(
                attention_A[
                    sa
                ].item()
            )
        ),
    )

    print(
        "B correct-slot attention:",
        fmt(
            float(
                attention_B[
                    sb
                ].item()
            )
        ),
    )

    print()

    print(
        "Attention A/B cosine:",
        fmt(
            cosine(
                attention_A,
                attention_B,
            )
        ),
    )

    subsection(
        "CONTEXT + FUSION"
    )

    context_A = (
        A["context"][
            0,
            ia,
        ]
    )

    context_B = (
        B["context"][
            0,
            ib,
        ]
    )

    hidden_A = (
        A["hidden"][
            0,
            ia,
        ]
    )

    hidden_B = (
        B["hidden"][
            0,
            ib,
        ]
    )

    fused_A = (
        A["fused"][
            0,
            ia,
        ]
    )

    fused_B = (
        B["fused"][
            0,
            ib,
        ]
    )

    print(
        "Context cosine:",
        fmt(
            cosine(
                context_A,
                context_B,
            )
        ),
    )

    print(
        "Base hidden cosine:",
        fmt(
            cosine(
                hidden_A,
                hidden_B,
            )
        ),
    )

    print(
        "Fused hidden cosine:",
        fmt(
            cosine(
                fused_A,
                fused_B,
            )
        ),
    )

    contribution_A = (
        (
            fused_A
            - hidden_A
        ).norm()
        / (
            hidden_A.norm()
            + 1e-8
        )
    )

    contribution_B = (
        (
            fused_B
            - hidden_B
        ).norm()
        / (
            hidden_B.norm()
            + 1e-8
        )
    )

    print(
        "Memory contribution A:",
        fmt(
            float(
                contribution_A.item()
            )
        ),
    )

    print(
        "Memory contribution B:",
        fmt(
            float(
                contribution_B.item()
            )
        ),
    )

    print(
        "Read confidence A:",
        fmt(
            float(
                A[
                    "read_confidence"
                ][
                    0,
                    ia,
                ].mean().item()
            )
        ),
    )

    print(
        "Read confidence B:",
        fmt(
            float(
                B[
                    "read_confidence"
                ][
                    0,
                    ib,
                ].mean().item()
            )
        ),
    )

    return A, B


# ============================================================
# MODEL OUTPUT UNDER GIVEN MEMORY
# ============================================================

@torch.no_grad()
def query_model(
    model,
    query,
    memory,
):

    batch = encode(
        query
    )

    output = model(

        input_ids=(
            batch["input_ids"]
        ),

        attention_mask=(
            batch[
                "attention_mask"
            ]
        ),

        memory_state=memory,

        update_memory=False,

        return_diagnostics=True,
    )

    index = int(
        batch[
            "attention_mask"
        ].sum().item()
        - 1
    )

    logits = (
        output.logits[
            0,
            index,
        ]
    )

    return output, logits


# ============================================================
# OUTPUT RANKS
# ============================================================

@torch.no_grad()
def output_audit(
    model,
    memory,
):

    section(
        "4. OUTPUT TOKEN AUDIT"
    )

    rabbit = answer_token_id(
        A_ANSWER
    )

    river = answer_token_id(
        B_ANSWER
    )

    print(
        "rabbit token:",
        rabbit,
        tokenizer.decode(
            [rabbit]
        ),
    )

    print(
        "river token:",
        river,
        tokenizer.decode(
            [river]
        ),
    )

    for label, query, target in [

        (
            "QUERY A",
            QUERY_A,
            rabbit,
        ),

        (
            "QUERY B",
            QUERY_B,
            river,
        ),
    ]:

        output, logits = (
            query_model(
                model,
                query,
                memory,
            )
        )

        rank, prob = (
            rank_token(
                logits,
                target,
            )
        )

        pred = int(
            logits.argmax().item()
        )

        print()
        print(label)

        print(
            "Target:",
            tokenizer.decode(
                [target]
            ),
        )

        print(
            "Target rank:",
            rank,
        )

        print(
            "Target probability:",
            fmt(prob),
        )

        print(
            "Top prediction:",
            repr(
                tokenizer.decode(
                    [pred]
                )
            ),
        )


# ============================================================
# MEMORY VARIANTS
# ============================================================

def clone_memory(memory):

    return deepcopy(memory)


def zero_memory(memory):

    x = clone_memory(
        memory
    )

    x.slots = torch.zeros_like(
        x.slots
    )

    return x


def random_memory(memory):

    x = clone_memory(
        memory
    )

    x.slots = torch.randn_like(
        x.slots
    )

    return x


def amplified_memory(
    memory,
    scale=10.0,
):

    x = clone_memory(
        memory
    )

    x.slots = (
        x.slots * scale
    )

    return x


def shuffled_memory(memory):

    x = clone_memory(
        memory
    )

    permutation = torch.randperm(
        x.slots.size(1),
        device=x.slots.device,
    )

    x.slots = (
        x.slots[
            :,
            permutation,
            :
        ]
    )

    x.confidence = (
        x.confidence[
            :,
            permutation,
        ]
    )

    x.write_count = (
        x.write_count[
            :,
            permutation,
        ]
    )

    x.read_count = (
        x.read_count[
            :,
            permutation,
        ]
    )

    x.age = (
        x.age[
            :,
            permutation,
        ]
    )

    return x


# ============================================================
# MEMORY SENSITIVITY
# ============================================================

@torch.no_grad()
def memory_sensitivity_audit(
    model,
    memory,
):

    section(
        "5. CAUSAL MEMORY SENSITIVITY"
    )

    variants = {

        "CORRECT":
            memory,

        "ZERO":
            zero_memory(
                memory
            ),

        "RANDOM":
            random_memory(
                memory
            ),

        "SHUFFLED":
            shuffled_memory(
                memory
            ),

        "10X":
            amplified_memory(
                memory,
                10.0,
            ),
    }

    rabbit = answer_token_id(
        A_ANSWER
    )

    river = answer_token_id(
        B_ANSWER
    )

    print(
        f"{'Memory':<14}"
        f"{'A rank':>10}"
        f"{'A prob':>14}"
        f"{'B rank':>10}"
        f"{'B prob':>14}"
    )

    baseline_A = None
    baseline_B = None

    for name, mem in (
        variants.items()
    ):

        _, logits_A = query_model(
            model,
            QUERY_A,
            mem,
        )

        _, logits_B = query_model(
            model,
            QUERY_B,
            mem,
        )

        rank_A, prob_A = (
            rank_token(
                logits_A,
                rabbit,
            )
        )

        rank_B, prob_B = (
            rank_token(
                logits_B,
                river,
            )
        )

        if name == "CORRECT":

            baseline_A = (
                logits_A
            )

            baseline_B = (
                logits_B
            )

        print(
            f"{name:<14}"
            f"{rank_A:>10}"
            f"{prob_A:>14.8f}"
            f"{rank_B:>10}"
            f"{prob_B:>14.8f}"
        )

    subsection(
        "LOGIT SENSITIVITY"
    )

    for name, mem in (
        variants.items()
    ):

        if name == "CORRECT":
            continue

        _, la = query_model(
            model,
            QUERY_A,
            mem,
        )

        _, lb = query_model(
            model,
            QUERY_B,
            mem,
        )

        print(
            f"{name:<14}"
            f"A cosine="
            f"{cosine(baseline_A, la):.8f} "
            f"A relΔ="
            f"{relative_delta(baseline_A, la):.8f} "
            f"B cosine="
            f"{cosine(baseline_B, lb):.8f} "
            f"B relΔ="
            f"{relative_delta(baseline_B, lb):.8f}"
        )


# ============================================================
# CONFIDENCE / TOP-K ABLATION
# ============================================================

@torch.no_grad()
def addressing_ablation(
    model,
    memory,
    write_A,
    write_B,
):

    section(
        "6. READER ADDRESSING ABLATIONS"
    )

    sa = write_A[
        "selected_slot"
    ]

    sb = write_B[
        "selected_slot"
    ]

    settings = [

        (
            "NORMAL",
            True,
            True,
        ),

        (
            "NO CONFIDENCE",
            False,
            True,
        ),

        (
            "NO TOP-K",
            True,
            False,
        ),

        (
            "NO CONF + NO TOPK",
            False,
            False,
        ),
    ]

    print(
        f"{'Setting':<22}"
        f"{'A correct attn':>18}"
        f"{'B correct attn':>18}"
        f"{'A/B attn cos':>18}"
    )

    for (
        name,
        confidence,
        topk,
    ) in settings:

        A = reader_internals(
            model,
            memory,
            QUERY_A,
            use_confidence=(
                confidence
            ),
            use_top_k=topk,
        )

        B = reader_internals(
            model,
            memory,
            QUERY_B,
            use_confidence=(
                confidence
            ),
            use_top_k=topk,
        )

        ia = A[
            "last_index"
        ]

        ib = B[
            "last_index"
        ]

        wa = (
            A["weights"][
                0,
                :,
                ia,
                :,
            ]
            .mean(dim=0)
        )

        wb = (
            B["weights"][
                0,
                :,
                ib,
                :,
            ]
            .mean(dim=0)
        )

        print(
            f"{name:<22}"
            f"{float(wa[sa]):>18.8f}"
            f"{float(wb[sb]):>18.8f}"
            f"{cosine(wa, wb):>18.8f}"
        )


# ============================================================
# ORACLE SLOT READ
# ============================================================

@torch.no_grad()
def oracle_context(
    model,
    memory,
    query,
    slot,
):

    batch, out = backbone_hidden(
        model,
        query,
    )

    hidden = (
        out.last_hidden_state
    )

    reader = model.reader

    normalized_memory = (
        reader.memory_norm(
            memory.slots
        )
    )

    values = (
        reader.value_projection(
            normalized_memory
        )
    )

    value = values[
        :,
        slot:slot + 1,
        :
    ]

    value = value.expand(
        -1,
        hidden.size(1),
        -1,
    )

    context = (
        reader.output_projection(
            value
        )
    )

    context = (
        reader.output_norm(
            context
        )
    )

    # Use reader's own fusion mechanism.
    fake_confidence = torch.ones(
        hidden.size(0),
        hidden.size(1),
        1,
        device=hidden.device,
        dtype=hidden.dtype,
    )

    fused = reader._fuse(
        hidden_states=hidden,
        context=context,
        read_confidence=(
            fake_confidence
        ),
    )

    index = int(
        batch[
            "attention_mask"
        ].sum().item()
        - 1
    )

    logits = (
        model.backbone.lm_head(
            fused
        )[
            0,
            index,
        ]
    )

    return logits


# ============================================================
# ORACLE TEST
# ============================================================

@torch.no_grad()
def oracle_audit(
    model,
    memory,
    write_A,
    write_B,
):

    section(
        "7. ORACLE READ INTERVENTION"
    )

    rabbit = answer_token_id(
        A_ANSWER
    )

    river = answer_token_id(
        B_ANSWER
    )

    sa = write_A[
        "selected_slot"
    ]

    sb = write_B[
        "selected_slot"
    ]

    _, normal_A = query_model(
        model,
        QUERY_A,
        memory,
    )

    _, normal_B = query_model(
        model,
        QUERY_B,
        memory,
    )

    oracle_A = oracle_context(
        model,
        memory,
        QUERY_A,
        sa,
    )

    oracle_B = oracle_context(
        model,
        memory,
        QUERY_B,
        sb,
    )

    wrong_A = oracle_context(
        model,
        memory,
        QUERY_A,
        sb,
    )

    wrong_B = oracle_context(
        model,
        memory,
        QUERY_B,
        sa,
    )

    cases = [

        (
            "A NORMAL",
            normal_A,
            rabbit,
        ),

        (
            "A ORACLE CORRECT SLOT",
            oracle_A,
            rabbit,
        ),

        (
            "A WRONG SLOT",
            wrong_A,
            rabbit,
        ),

        (
            "B NORMAL",
            normal_B,
            river,
        ),

        (
            "B ORACLE CORRECT SLOT",
            oracle_B,
            river,
        ),

        (
            "B WRONG SLOT",
            wrong_B,
            river,
        ),
    ]

    print(
        f"{'Case':<30}"
        f"{'Rank':>10}"
        f"{'Probability':>18}"
        f"{'Prediction':>20}"
    )

    for (
        name,
        logits,
        target,
    ) in cases:

        rank, prob = (
            rank_token(
                logits,
                target,
            )
        )

        pred = int(
            logits.argmax().item()
        )

        pred_text = (
            tokenizer.decode(
                [pred]
            )
        )

        print(
            f"{name:<30}"
            f"{rank:>10}"
            f"{prob:>18.8f}"
            f"{repr(pred_text):>20}"
        )


# ============================================================
# GRADIENT AUDIT
# ============================================================

def grad_norm(parameters):

    total = 0.0

    for p in parameters:

        if p.grad is not None:

            total += float(
                p.grad.detach()
                .float()
                .pow(2)
                .sum()
                .item()
            )

    return math.sqrt(total)


def gradient_audit(
    model,
    memory,
):

    section(
        "8. GRADIENT FLOW AUDIT"
    )

    model.train()

    model.zero_grad(
        set_to_none=True
    )

    query = (
        QUERY_A
        + " "
        + A_ANSWER
    )

    batch = encode(
        query
    )

    labels = (
        batch["input_ids"]
        .clone()
    )

    # Mask query prefix.
    prefix = encode(
        QUERY_A
    )

    prefix_length = (
        prefix[
            "input_ids"
        ].size(1)
    )

    labels[
        :,
        :prefix_length,
    ] = -100

    output = model(

        input_ids=(
            batch["input_ids"]
        ),

        attention_mask=(
            batch[
                "attention_mask"
            ]
        ),

        labels=labels,

        memory_state=memory,

        update_memory=False,

        return_diagnostics=True,
    )

    if output.loss is None:

        print(
            "No loss returned."
        )

        model.eval()

        return

    output.loss.backward()

    groups = {

        "GPT2 backbone":
            model.backbone
            .transformer
            .parameters(),

        "LM head":
            model.backbone
            .lm_head
            .parameters(),

        "Reader query":
            model.reader
            .query_projection
            .parameters(),

        "Reader key":
            model.reader
            .key_projection
            .parameters(),

        "Reader value":
            model.reader
            .value_projection
            .parameters(),

        "Reader output":
            model.reader
            .output_projection
            .parameters(),
    }

    if hasattr(
        model.reader,
        "fusion_gate",
    ):

        groups[
            "Reader fusion gate"
        ] = (
            model.reader
            .fusion_gate
            .parameters()
        )

    print(
        "Loss:",
        fmt(
            float(
                output.loss.item()
            )
        ),
    )

    print()

    print(
        f"{'Module':<30}"
        f"{'Gradient norm':>20}"
    )

    for name, params in (
        groups.items()
    ):

        print(
            f"{name:<30}"
            f"{grad_norm(params):>20.10f}"
        )

    model.zero_grad(
        set_to_none=True
    )

    model.eval()


# ============================================================
# QUERY SWAP / IDENTITY TEST
# ============================================================

@torch.no_grad()
def identity_swap_audit(
    model,
    memory,
):

    section(
        "9. QUERY IDENTITY / SHORTCUT AUDIT"
    )

    variants = [

        (
            "A",
            QUERY_A,
        ),

        (
            "B",
            QUERY_B,
        ),

        (
            "Unknown entity",
            query_text(
                "Completely-Unseen-999"
            ),
        ),

        (
            "Template only",
            "The assigned keyword is",
        ),
    ]

    rabbit = answer_token_id(
        A_ANSWER
    )

    river = answer_token_id(
        B_ANSWER
    )

    print(
        f"{'Query':<25}"
        f"{'rabbit rank':>15}"
        f"{'rabbit p':>15}"
        f"{'river rank':>15}"
        f"{'river p':>15}"
    )

    for name, query in variants:

        _, logits = query_model(
            model,
            query,
            memory,
        )

        rr, rp = rank_token(
            logits,
            rabbit,
        )

        vr, vp = rank_token(
            logits,
            river,
        )

        print(
            f"{name:<25}"
            f"{rr:>15}"
            f"{rp:>15.8f}"
            f"{vr:>15}"
            f"{vp:>15.8f}"
        )


# ============================================================
# SUMMARY
# ============================================================

@torch.no_grad()
def final_summary(
    model,
    memory,
    write_A,
    write_B,
):

    section(
        "10. AUTOMATIC FAILURE LOCALIZATION"
    )

    A = reader_internals(
        model,
        memory,
        QUERY_A,
    )

    B = reader_internals(
        model,
        memory,
        QUERY_B,
    )

    ia = A[
        "last_index"
    ]

    ib = B[
        "last_index"
    ]

    sa = write_A[
        "selected_slot"
    ]

    sb = write_B[
        "selected_slot"
    ]

    query_cos = cosine(

        A["hidden"][
            0,
            ia,
        ],

        B["hidden"][
            0,
            ib,
        ],
    )

    projected_q_cos = cosine(

        A["q_projected"][
            0,
            ia,
        ],

        B["q_projected"][
            0,
            ib,
        ],
    )

    key_cos = cosine(

        A["k_projected"][
            0,
            sa,
        ],

        A["k_projected"][
            0,
            sb,
        ],
    )

    attention_A = (

        A["weights"][
            0,
            :,
            ia,
            :,
        ]
        .mean(dim=0)
    )

    attention_B = (

        B["weights"][
            0,
            :,
            ib,
            :,
        ]
        .mean(dim=0)
    )

    attention_cos = cosine(
        attention_A,
        attention_B,
    )

    context_cos = cosine(

        A["context"][
            0,
            ia,
        ],

        B["context"][
            0,
            ib,
        ],
    )

    fused_cos = cosine(

        A["fused"][
            0,
            ia,
        ],

        B["fused"][
            0,
            ib,
        ],
    )

    print(
        "Query hidden cosine:",
        fmt(query_cos),
    )

    print(
        "Projected Q cosine:",
        fmt(projected_q_cos),
    )

    print(
        "Correct-slot key cosine:",
        fmt(key_cos),
    )

    print(
        "Attention cosine:",
        fmt(attention_cos),
    )

    print(
        "Context cosine:",
        fmt(context_cos),
    )

    print(
        "Fused hidden cosine:",
        fmt(fused_cos),
    )

    print()

    print(
        "A correct attention:",
        fmt(
            float(
                attention_A[
                    sa
                ].item()
            )
        ),
    )

    print(
        "B correct attention:",
        fmt(
            float(
                attention_B[
                    sb
                ].item()
            )
        ),
    )

    print()

    if (
        query_cos < 0.98
        and projected_q_cos > 0.995
    ):

        print(
            "LIKELY FAILURE:"
        )

        print(
            "Reader query projection "
            "is collapsing distinguishable "
            "GPT-2 query representations."
        )

    elif (
        projected_q_cos < 0.98
        and attention_cos > 0.995
    ):

        print(
            "LIKELY FAILURE:"
        )

        print(
            "Queries remain different, "
            "but Q/K addressing maps them "
            "to nearly identical memory reads."
        )

    elif (
        attention_cos < 0.98
        and context_cos > 0.995
    ):

        print(
            "LIKELY FAILURE:"
        )

        print(
            "Attention differs, but value "
            "projection/context construction "
            "collapses the retrieved values."
        )

    elif (
        context_cos < 0.98
        and fused_cos > 0.995
    ):

        print(
            "LIKELY FAILURE:"
        )

        print(
            "Retrieved contexts differ, "
            "but fusion suppresses the "
            "memory distinction."
        )

    elif attention_cos > 0.995:

        print(
            "LIKELY FAILURE:"
        )

        print(
            "Read addressing collapse: "
            "different queries produce "
            "nearly identical attention."
        )

    else:

        print(
            "No single collapse threshold "
            "triggered."
        )

        print(
            "Use the detailed metrics above "
            "to identify the earliest major "
            "loss of discriminability."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "MEMORY-AUGMENTED GPT-2 "
        "DEEP CAUSAL AUDIT"
    )

    print(
        "This script DOES NOT train "
        "or modify the model."
    )

    print()

    print(
        "Fact A:",
        FACT_A,
    )

    print(
        "Query A:",
        QUERY_A,
    )

    print(
        "Answer A:",
        A_ANSWER,
    )

    print()

    print(
        "Fact B:",
        FACT_B,
    )

    print(
        "Query B:",
        QUERY_B,
    )

    print(
        "Answer B:",
        B_ANSWER,
    )

    model = load_model()

    # --------------------------------------------------------
    # 1. REPRESENTATION TRAJECTORY
    # --------------------------------------------------------

    representation_audit(
        model
    )

    # --------------------------------------------------------
    # 2. WRITE AUDIT
    # --------------------------------------------------------

    (
        memory,
        write_A,
        write_B,
    ) = write_audit(
        model
    )

    # --------------------------------------------------------
    # 3. READ AUDIT
    # --------------------------------------------------------

    read_audit(
        model,
        memory,
        write_A,
        write_B,
    )

    # --------------------------------------------------------
    # 4. OUTPUT AUDIT
    # --------------------------------------------------------

    output_audit(
        model,
        memory,
    )

    # --------------------------------------------------------
    # 5. MEMORY SENSITIVITY
    # --------------------------------------------------------

    memory_sensitivity_audit(
        model,
        memory,
    )

    # --------------------------------------------------------
    # 6. ADDRESSING ABLATIONS
    # --------------------------------------------------------

    addressing_ablation(
        model,
        memory,
        write_A,
        write_B,
    )

    # --------------------------------------------------------
    # 7. ORACLE READ
    # --------------------------------------------------------

    oracle_audit(
        model,
        memory,
        write_A,
        write_B,
    )

    # --------------------------------------------------------
    # 8. GRADIENT AUDIT
    # --------------------------------------------------------

    gradient_audit(
        model,
        memory,
    )

    # --------------------------------------------------------
    # 9. SHORTCUT TEST
    # --------------------------------------------------------

    identity_swap_audit(
        model,
        memory,
    )

    # --------------------------------------------------------
    # 10. AUTOMATIC SUMMARY
    # --------------------------------------------------------

    final_summary(
        model,
        memory,
        write_A,
        write_B,
    )

    section(
        "AUDIT COMPLETE"
    )

    print(
        "No model parameters were updated."
    )

    print(
        "No checkpoint was overwritten."
    )

    print(
        "No files inside models/ "
        "were modified."
    )


if __name__ == "__main__":

    main()