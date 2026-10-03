# ============================================================
# WRITER ABLATION + ATTENTION FAILURE DIAGNOSIS
#
# PURPOSE
# ============================================================
#
# Previous experiment:
#
# GPT2_SUMMARY       = ~84% answer decodability
# ATTENDED_CONTEXT   = ~24%
# WRITER_CANDIDATE   = ~17%
# FINAL_SLOT         = ~18%
#
# Therefore:
#
# We want to answer TWO questions:
#
# 1. Can a SIMPLER writer representation preserve the answer
#    information already present in GPT-2 summary?
#
# 2. WHY is the current attention writer losing so much?
#
#
# REPRESENTATIONS TESTED
# ============================================================
#
# CONTROL:
#   PRE_WRITE_SLOT
#
# BASE:
#   GPT2_SUMMARY
#   SUMMARY_LAYERNORM
#
# CURRENT WRITER:
#   ATTENDED_CONTEXT
#   CURRENT_CANDIDATE
#   CURRENT_DELTA
#
# SIMPLE ALTERNATIVES:
#
#   DIRECT_SUMMARY
#
#       candidate = LayerNorm(summary)
#
#   SUMMARY_PLUS_010_CONTEXT
#
#       LayerNorm(summary + 0.10 * attended_context)
#
#   SUMMARY_PLUS_025_CONTEXT
#
#       LayerNorm(summary + 0.25 * attended_context)
#
#   SUMMARY_PLUS_050_CONTEXT
#
#       LayerNorm(summary + 0.50 * attended_context)
#
#   SUMMARY_PLUS_CONTEXT
#
#       LayerNorm(summary + attended_context)
#
#
# ATTENTION DIAGNOSTICS
# ============================================================
#
# Capture actual CandidateWriter cross-attention weights.
#
# Measure:
#
#   - attention mass on ANSWER tokens
#   - attention mass on ENTITY tokens
#   - attention mass elsewhere
#   - attention entropy
#   - where the top-attended token is
#
#
# IMPORTANT
# ============================================================
#
# NO changes to models/
# NO training of GPT-2
# NO training of writer
# NO training of gate
#
# Only tiny linear diagnostic probes are trained.
#
#
# RUN:
#
# python writer_ablation_and_attention_diagnosis.py \
#   2>&1 | tee writer_ablation_and_attention_diagnosis.log
#
# ============================================================


import math
import random
import string
from collections import OrderedDict, Counter

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

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

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SEED = 2030

NUM_SLOTS = 8

TRAIN_PER_CLASS = 100
VAL_PER_CLASS = 25
TEST_PER_CLASS = 50

PROBE_EPOCHS = 50
PROBE_BATCH_SIZE = 128
PROBE_LR = 1e-2
PROBE_WEIGHT_DECAY = 1e-4


# ============================================================
# ANSWERS
# ============================================================

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


ANSWER_TO_ID = {
    answer: idx
    for idx, answer in enumerate(ANSWERS)
}


# ============================================================
# TEMPLATES
# ============================================================

TRAIN_TEMPLATES = [

    "The assigned keyword for {entity} is {answer}.",

    "{entity} has been assigned the keyword {answer}.",

    "Remember that {answer} is associated with {entity}.",

    "Store the mapping {entity} to {answer}.",

    "The value assigned to {entity} is {answer}.",

    "For {entity}, the stored keyword is {answer}.",

    "{answer} is the keyword linked with {entity}.",

    "Please remember this association: {entity} means {answer}.",
]


HELDOUT_TEMPLATES = [

    "Record {answer} as the value belonging to {entity}.",

    "In memory, associate {entity} with {answer}.",
]


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# PRINT
# ============================================================

def section(title):

    print()
    print("=" * 125)
    print(title)
    print("=" * 125)


def fmt(x):

    return f"{float(x):.6f}"


# ============================================================
# MODEL CONFIG
# ============================================================

def build_config():

    return MemoryGPT2Config(

        num_slots=NUM_SLOTS,

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
# TOKENIZER
# ============================================================

section("TOKENIZER")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)


if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token


print(
    "Tokenizer:",
    tokenizer.__class__.__name__,
)

print(
    "Vocabulary:",
    len(tokenizer),
)


# ============================================================
# LOAD MODEL
# ============================================================

section("LOAD FROZEN MODEL")


print(
    "Device:",
    DEVICE,
)

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

    checkpoint_state = checkpoint[
        "model_state_dict"
    ]

elif "state_dict" in checkpoint:

    checkpoint_state = checkpoint[
        "state_dict"
    ]

else:

    checkpoint_state = checkpoint


current_state = model.state_dict()


compatible = {}


for name, value in checkpoint_state.items():

    if (
        name in current_state
        and
        current_state[name].shape == value.shape
    ):

        compatible[name] = value


load_result = model.load_state_dict(
    compatible,
    strict=False,
)


print(
    "Compatible tensors:",
    len(compatible),
)

print(
    "Missing tensors:",
    len(load_result.missing_keys),
)


model.to(DEVICE)

model.eval()


for parameter in model.parameters():

    parameter.requires_grad = False


print(
    "Entire model frozen."
)


# ============================================================
# DATA GENERATION
# ============================================================

def random_entity(
    rng,
    prefix,
):

    letters = "".join(

        rng.choice(
            string.ascii_uppercase
        )

        for _ in range(7)
    )


    digits = "".join(

        rng.choice(
            string.digits
        )

        for _ in range(6)
    )


    return (
        f"{prefix}-{letters}-{digits}"
    )


def build_examples(
    prefix,
    per_class,
    templates,
    seed,
):

    rng = random.Random(seed)

    examples = []

    used_entities = set()


    for answer in ANSWERS:

        for _ in range(
            per_class
        ):

            while True:

                entity = random_entity(
                    rng,
                    prefix,
                )

                if entity not in used_entities:

                    used_entities.add(
                        entity
                    )

                    break


            template = rng.choice(
                templates
            )


            text = template.format(

                entity=entity,

                answer=answer,
            )


            examples.append(

                {
                    "text": text,

                    "entity": entity,

                    "answer": answer,

                    "label":
                        ANSWER_TO_ID[
                            answer
                        ],
                }
            )


    rng.shuffle(
        examples
    )


    return examples


# ============================================================
# BUILD DATASETS
# ============================================================

section("BUILD DATASETS")


train_examples = build_examples(

    prefix="WriterTrain",

    per_class=TRAIN_PER_CLASS,

    templates=TRAIN_TEMPLATES,

    seed=SEED + 1,
)


val_examples = build_examples(

    prefix="WriterValid",

    per_class=VAL_PER_CLASS,

    templates=TRAIN_TEMPLATES,

    seed=SEED + 2,
)


test_examples = build_examples(

    prefix="WriterTest",

    per_class=TEST_PER_CLASS,

    templates=HELDOUT_TEMPLATES,

    seed=SEED + 3,
)


print(
    "Train:",
    len(train_examples),
)

print(
    "Validation:",
    len(val_examples),
)

print(
    "Held-out test:",
    len(test_examples),
)

print(
    "Chance:",
    f"{100 / len(ANSWERS):.2f}%"
)


# ============================================================
# MEMORY
# ============================================================

def initialize_memory():

    dtype = next(
        model.parameters()
    ).dtype


    return model.initialize_memory(

        batch_size=1,

        device=DEVICE,

        dtype=dtype,
    )


# ============================================================
# IDENTIFY WRITTEN SLOT
# ============================================================

def identify_written_slot(
    before,
    after,
):

    diff = (

        after.write_count[0]
        -
        before.write_count[0]
    )


    changed = (

        diff
        .gt(0)
        .nonzero(
            as_tuple=False
        )
        .flatten()
    )


    if len(changed) == 0:

        return None


    if len(changed) == 1:

        return int(
            changed[0].item()
        )


    return int(
        torch.argmax(
            diff
        ).item()
    )


# ============================================================
# TOKEN SPAN MASK
# ============================================================

def span_token_mask(
    text,
    substring,
    offsets,
):

    start = text.find(
        substring
    )


    if start == -1:

        raise RuntimeError(
            f"{substring!r} not found in {text!r}"
        )


    end = (
        start
        +
        len(substring)
    )


    mask = torch.zeros(
        offsets.shape[0],
        dtype=torch.bool,
    )


    for token_idx in range(
        offsets.shape[0]
    ):

        s = int(
            offsets[
                token_idx,
                0
            ].item()
        )

        e = int(
            offsets[
                token_idx,
                1
            ].item()
        )


        if e <= s:

            continue


        if (
            s < end
            and
            e > start
        ):

            mask[
                token_idx
            ] = True


    return mask


# ============================================================
# CROSS-ATTENTION CAPTURE
# ============================================================

ATTENTION_CAPTURE = {
    "weights": None
}


def attention_hook(
    module,
    inputs,
    output,
):

    # PyTorch MultiheadAttention normally returns:
    #
    # (
    #     attention_output,
    #     attention_weights
    # )

    if (
        isinstance(output, tuple)
        and
        len(output) >= 2
    ):

        weights = output[1]


        if weights is not None:

            ATTENTION_CAPTURE[
                "weights"
            ] = (
                weights
                .detach()
                .float()
                .cpu()
            )


attention_handle = (
    model.writer
    .cross_attention
    .register_forward_hook(
        attention_hook
    )
)


# ============================================================
# EXTRACT REPRESENTATIONS + ATTENTION
# ============================================================

@torch.no_grad()
def extract_one(
    example,
):

    text = example[
        "text"
    ]

    entity = example[
        "entity"
    ]

    answer = example[
        "answer"
    ]


    # --------------------------------------------------------
    # Fresh memory
    # --------------------------------------------------------

    memory_state = (
        initialize_memory()
    )


    pre_slots = (

        memory_state
        .slots
        .detach()
        .clone()
    )


    # --------------------------------------------------------
    # Tokenization WITH offsets
    # --------------------------------------------------------

    encoded = tokenizer(

        text,

        return_tensors="pt",

        return_offsets_mapping=True,
    )


    offsets = (
        encoded.pop(
            "offset_mapping"
        )[0]
    )


    input_ids = (
        encoded[
            "input_ids"
        ].to(
            DEVICE
        )
    )


    attention_mask = (
        encoded[
            "attention_mask"
        ].to(
            DEVICE
        )
    )


    # --------------------------------------------------------
    # Base GPT-2 representation
    # --------------------------------------------------------

    transformer_output = (
        model.backbone.transformer(

            input_ids=input_ids,

            attention_mask=attention_mask,

            use_cache=False,

            return_dict=True,
        )
    )


    base_hidden = (
        transformer_output
        .last_hidden_state
    )


    summary = model._pool_hidden(

        hidden_states=base_hidden,

        attention_mask=attention_mask,
    )


    summary_vector = (

        summary[
            0
        ]
        .detach()
        .float()
        .cpu()
    )


    # --------------------------------------------------------
    # Writer's own summary normalization
    # --------------------------------------------------------

    summary_normed = (

        model.writer
        .summary_norm(
            summary
        )[
            0
        ]
        .detach()
        .float()
        .cpu()
    )


    # --------------------------------------------------------
    # Run current architecture
    # --------------------------------------------------------

    ATTENTION_CAPTURE[
        "weights"
    ] = None


    output = model(

        input_ids=input_ids,

        attention_mask=attention_mask,

        memory_state=memory_state,

        update_memory=True,

        return_diagnostics=True,
    )


    after_state = (
        output.memory_state
    )


    slot = identify_written_slot(

        memory_state,

        after_state,
    )


    if slot is None:

        raise RuntimeError(
            "No slot was written."
        )


    # --------------------------------------------------------
    # Pre-write slot
    # --------------------------------------------------------

    pre_slot = (

        pre_slots[
            0,
            slot,
            :
        ]
        .detach()
        .float()
        .cpu()
    )


    # --------------------------------------------------------
    # Current attended context
    # --------------------------------------------------------

    attended = (
        output
        .writer_output
        .attended_context
    )


    if attended is None:

        raise RuntimeError(
            "Current writer did not return attended_context."
        )


    attended_vector = (

        attended[
            0,
            slot,
            :
        ]
        .detach()
        .float()
        .cpu()
    )


    # --------------------------------------------------------
    # Current candidate / delta
    # --------------------------------------------------------

    current_candidate = (

        output
        .writer_output
        .candidates[
            0,
            slot,
            :
        ]
        .detach()
        .float()
        .cpu()
    )


    current_delta = (

        output
        .writer_output
        .deltas[
            0,
            slot,
            :
        ]
        .detach()
        .float()
        .cpu()
    )


    # ========================================================
    # SIMPLE WRITER ALTERNATIVES
    # ========================================================

    # Direct semantic value.
    #
    # Use LayerNorm only so magnitude is controlled
    # but information isn't passed through the current
    # attention/fusion writer.

    direct_summary = F.layer_norm(

        summary_vector,

        normalized_shape=(
            summary_vector.shape[-1],
        ),
    )


    # --------------------------------------------------------
    # Residual alternatives
    #
    # Instead of REPLACING summary with attention output,
    # preserve summary and use attention as a correction.
    # --------------------------------------------------------

    def residual_value(alpha):

        x = (

            summary_normed
            +
            alpha
            *
            attended_vector
        )


        return F.layer_norm(

            x,

            normalized_shape=(
                x.shape[-1],
            ),
        )


    residual_010 = (
        residual_value(
            0.10
        )
    )


    residual_025 = (
        residual_value(
            0.25
        )
    )


    residual_050 = (
        residual_value(
            0.50
        )
    )


    residual_100 = (
        residual_value(
            1.00
        )
    )


    # ========================================================
    # ATTENTION ANALYSIS
    # ========================================================

    raw_attention = (
        ATTENTION_CAPTURE[
            "weights"
        ]
    )


    attention_info = None


    if raw_attention is not None:

        # MultiheadAttention commonly returns:
        #
        # [B, N, T]
        #
        # because average_attn_weights=True by default.
        #
        # Some variants may return [B,H,N,T].
        #
        # Handle both.

        if raw_attention.dim() == 4:

            # average heads

            weights = (
                raw_attention[
                    0,
                    :,
                    slot,
                    :
                ]
                .mean(
                    dim=0
                )
            )


        elif raw_attention.dim() == 3:

            weights = (
                raw_attention[
                    0,
                    slot,
                    :
                ]
            )


        else:

            weights = None


        if weights is not None:

            weights = (
                weights[
                    : offsets.shape[0]
                ]
            )


            weights = (
                weights
                /
                weights.sum()
                .clamp_min(
                    1e-12
                )
            )


            answer_mask = span_token_mask(

                text,

                answer,

                offsets,
            )


            entity_mask = span_token_mask(

                text,

                entity,

                offsets,
            )


            answer_mass = float(

                weights[
                    answer_mask
                ]
                .sum()
                .item()
            )


            entity_mass = float(

                weights[
                    entity_mask
                ]
                .sum()
                .item()
            )


            other_mask = ~(

                answer_mask
                |
                entity_mask
            )


            other_mass = float(

                weights[
                    other_mask
                ]
                .sum()
                .item()
            )


            entropy = float(

                (
                    -weights
                    *
                    torch.log(
                        weights.clamp_min(
                            1e-12
                        )
                    )
                )
                .sum()
                .item()
            )


            if len(
                weights
            ) > 1:

                normalized_entropy = (

                    entropy
                    /
                    math.log(
                        len(
                            weights
                        )
                    )
                )

            else:

                normalized_entropy = 0.0


            top_token_index = int(

                torch.argmax(
                    weights
                ).item()
            )


            top_token_id = int(

                input_ids[
                    0,
                    top_token_index
                ].item()
            )


            top_token = tokenizer.decode(

                [
                    top_token_id
                ]
            )


            if answer_mask[
                top_token_index
            ]:

                top_category = (
                    "ANSWER"
                )

            elif entity_mask[
                top_token_index
            ]:

                top_category = (
                    "ENTITY"
                )

            else:

                top_category = (
                    "OTHER"
                )


            attention_info = {

                "answer_mass":
                    answer_mass,

                "entity_mass":
                    entity_mass,

                "other_mass":
                    other_mass,

                "entropy":
                    entropy,

                "normalized_entropy":
                    normalized_entropy,

                "top_token":
                    top_token,

                "top_category":
                    top_category,

                "top_weight":
                    float(
                        weights[
                            top_token_index
                        ].item()
                    ),
            }


    representations = OrderedDict()


    representations[
        "PRE_WRITE_SLOT"
    ] = pre_slot


    representations[
        "GPT2_SUMMARY"
    ] = summary_vector


    representations[
        "SUMMARY_LAYERNORM"
    ] = summary_normed


    representations[
        "ATTENDED_CONTEXT"
    ] = attended_vector


    representations[
        "CURRENT_CANDIDATE"
    ] = current_candidate


    representations[
        "CURRENT_DELTA"
    ] = current_delta


    representations[
        "DIRECT_SUMMARY"
    ] = direct_summary


    representations[
        "SUMMARY_PLUS_010_CONTEXT"
    ] = residual_010


    representations[
        "SUMMARY_PLUS_025_CONTEXT"
    ] = residual_025


    representations[
        "SUMMARY_PLUS_050_CONTEXT"
    ] = residual_050


    representations[
        "SUMMARY_PLUS_CONTEXT"
    ] = residual_100


    return {

        "representations":
            representations,

        "attention":
            attention_info,

        "slot":
            slot,
    }


# ============================================================
# QUICK EXAMPLE
# ============================================================

section(
    "ONE EXAMPLE DIAGNOSTIC"
)


example_result = extract_one(
    test_examples[0]
)


print(
    "Text:"
)

print(
    test_examples[0]["text"]
)


print()

print(
    "Answer:",
    test_examples[0]["answer"]
)

print(
    "Slot:",
    example_result["slot"]
)


print()

print(
    "Representation norms:"
)


for name, vector in (
    example_result[
        "representations"
    ].items()
):

    print(

        f"{name:<30}"

        f"{vector.norm().item():.4f}"
    )


print()


if example_result[
    "attention"
] is not None:

    print(
        "Attention:"
    )

    for key, value in (
        example_result[
            "attention"
        ].items()
    ):

        print(
            f"{key}: {value}"
        )


else:

    print(
        "WARNING: Attention weights were not returned "
        "by PyTorch MultiheadAttention."
    )


# ============================================================
# EXTRACT DATASET
# ============================================================

@torch.no_grad()
def extract_dataset(
    examples,
    name,
):

    representation_storage = None

    labels = []

    attention_records = []

    slots = []


    for idx, example in enumerate(
        examples,
        start=1,
    ):

        result = extract_one(
            example
        )


        if representation_storage is None:

            representation_storage = OrderedDict(

                (
                    rep_name,
                    []
                )

                for rep_name
                in result[
                    "representations"
                ].keys()
            )


        for rep_name, vector in (
            result[
                "representations"
            ].items()
        ):

            representation_storage[
                rep_name
            ].append(
                vector
            )


        labels.append(
            example[
                "label"
            ]
        )


        slots.append(
            result[
                "slot"
            ]
        )


        if (
            result[
                "attention"
            ]
            is not None
        ):

            attention_records.append(

                result[
                    "attention"
                ]
            )


        if (
            idx % 100 == 0
            or
            idx == len(
                examples
            )
        ):

            print(

                f"{name}: "
                f"{idx}/{len(examples)}"
            )


    data = {}


    for rep_name, vectors in (
        representation_storage.items()
    ):

        data[
            rep_name
        ] = torch.stack(
            vectors,
            dim=0,
        )


    data[
        "labels"
    ] = torch.tensor(
        labels,
        dtype=torch.long,
    )


    data[
        "slots"
    ] = torch.tensor(
        slots,
        dtype=torch.long,
    )


    data[
        "attention_records"
    ] = attention_records


    return data


# ============================================================
# EXTRACT
# ============================================================

section(
    "EXTRACT TRAIN REPRESENTATIONS"
)


train_data = extract_dataset(

    train_examples,

    "TRAIN",
)


section(
    "EXTRACT VALIDATION REPRESENTATIONS"
)


val_data = extract_dataset(

    val_examples,

    "VALID",
)


section(
    "EXTRACT HELD-OUT TEST REPRESENTATIONS"
)


test_data = extract_dataset(

    test_examples,

    "TEST",
)


# ============================================================
# ATTENTION DIAGNOSTICS
# ============================================================

section(
    "ATTENTION WRITER DIAGNOSTICS"
)


records = (
    test_data[
        "attention_records"
    ]
)


if len(records) == 0:

    print(
        "No attention weights were captured."
    )

else:

    answer_mass = np.array(
        [
            x["answer_mass"]
            for x in records
        ]
    )


    entity_mass = np.array(
        [
            x["entity_mass"]
            for x in records
        ]
    )


    other_mass = np.array(
        [
            x["other_mass"]
            for x in records
        ]
    )


    entropy = np.array(
        [
            x[
                "normalized_entropy"
            ]
            for x in records
        ]
    )


    print(
        "Number of test examples:",
        len(records),
    )


    print()

    print(
        "Mean attention mass on ANSWER tokens:",
        f"{answer_mass.mean() * 100:.2f}%"
    )


    print(
        "Median attention mass on ANSWER tokens:",
        f"{np.median(answer_mass) * 100:.2f}%"
    )


    print(
        "Mean attention mass on ENTITY tokens:",
        f"{entity_mass.mean() * 100:.2f}%"
    )


    print(
        "Mean attention mass on OTHER tokens:",
        f"{other_mass.mean() * 100:.2f}%"
    )


    print(
        "Mean normalized attention entropy:",
        f"{entropy.mean():.4f}"
    )


    print()


    category_counts = Counter(

        x[
            "top_category"
        ]

        for x in records
    )


    print(
        "Top-attended-token category:"
    )


    for category in [

        "ANSWER",
        "ENTITY",
        "OTHER",
    ]:

        count = category_counts[
            category
        ]


        print(

            f"{category:<10}: "

            f"{count:4d} "

            f"({100 * count / len(records):.2f}%)"
        )


    print()

    print(
        "Most common top-attended tokens:"
    )


    token_counts = Counter(

        x[
            "top_token"
        ]

        for x in records
    )


    for token, count in (
        token_counts.most_common(
            15
        )
    ):

        printable = repr(
            token
        )


        print(

            f"{printable:<20} "

            f"{count:4d}"
        )


# ============================================================
# STANDARDIZATION
# ============================================================

def standardize(
    train_x,
    val_x,
    test_x,
):

    mean = train_x.mean(
        dim=0,
        keepdim=True,
    )


    std = train_x.std(
        dim=0,
        keepdim=True,
    )


    std = std.clamp_min(
        1e-5
    )


    return (

        (
            train_x - mean
        ) / std,

        (
            val_x - mean
        ) / std,

        (
            test_x - mean
        ) / std,
    )


# ============================================================
# LINEAR PROBE
# ============================================================

class Probe(nn.Module):

    def __init__(
        self,
        d_model,
    ):

        super().__init__()


        self.classifier = nn.Linear(

            d_model,

            len(ANSWERS),
        )


    def forward(
        self,
        x,
    ):

        return self.classifier(
            x
        )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    probe,
    x,
    y,
):

    probe.eval()


    x = x.to(
        DEVICE
    )


    y = y.to(
        DEVICE
    )


    logits = probe(
        x
    )


    prediction = logits.argmax(
        dim=-1
    )


    accuracy = (

        prediction
        .eq(y)
        .float()
        .mean()
        .item()
    )


    top3 = (

        logits
        .topk(
            3,
            dim=-1
        )
        .indices
        .eq(
            y.unsqueeze(
                -1
            )
        )
        .any(
            dim=-1
        )
        .float()
        .mean()
        .item()
    )


    correct_logits = (

        logits
        .gather(
            1,
            y.unsqueeze(
                1
            )
        )
        .squeeze(
            1
        )
    )


    ranks = (

        (
            logits
            >
            correct_logits.unsqueeze(
                1
            )
        )
        .sum(
            dim=1
        )
        + 1
    )


    mrr = (

        (
            1.0
            /
            ranks.float()
        )
        .mean()
        .item()
    )


    mean_rank = (

        ranks
        .float()
        .mean()
        .item()
    )


    return {

        "accuracy":
            accuracy,

        "top3":
            top3,

        "mrr":
            mrr,

        "mean_rank":
            mean_rank,
    }


# ============================================================
# TRAIN PROBE
# ============================================================

def train_probe(
    representation_name,
):

    section(
        f"PROBE: {representation_name}"
    )


    train_x = (
        train_data[
            representation_name
        ]
        .float()
    )


    val_x = (
        val_data[
            representation_name
        ]
        .float()
    )


    test_x = (
        test_data[
            representation_name
        ]
        .float()
    )


    train_x, val_x, test_x = (
        standardize(

            train_x,

            val_x,

            test_x,
        )
    )


    train_y = (
        train_data[
            "labels"
        ]
    )


    val_y = (
        val_data[
            "labels"
        ]
    )


    test_y = (
        test_data[
            "labels"
        ]
    )


    probe = Probe(

        train_x.shape[
            -1
        ]
    ).to(
        DEVICE
    )


    optimizer = torch.optim.AdamW(

        probe.parameters(),

        lr=PROBE_LR,

        weight_decay=PROBE_WEIGHT_DECAY,
    )


    loss_fn = (
        nn.CrossEntropyLoss()
    )


    loader = DataLoader(

        TensorDataset(
            train_x,
            train_y,
        ),

        batch_size=PROBE_BATCH_SIZE,

        shuffle=True,
    )


    best_val = -1.0

    best_epoch = None

    best_state = None


    for epoch in range(
        1,
        PROBE_EPOCHS + 1,
    ):

        probe.train()


        total_loss = 0.0

        count = 0


        for batch_x, batch_y in loader:

            batch_x = batch_x.to(
                DEVICE
            )


            batch_y = batch_y.to(
                DEVICE
            )


            logits = probe(
                batch_x
            )


            loss = loss_fn(
                logits,
                batch_y,
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            loss.backward()


            optimizer.step()


            total_loss += (

                float(
                    loss.item()
                )

                *
                batch_x.size(
                    0
                )
            )


            count += (
                batch_x.size(
                    0
                )
            )


        val_result = evaluate(

            probe,

            val_x,

            val_y,
        )


        if (
            val_result[
                "accuracy"
            ]
            >
            best_val
        ):

            best_val = (
                val_result[
                    "accuracy"
                ]
            )


            best_epoch = epoch


            best_state = {

                k:
                    v.detach()
                    .cpu()
                    .clone()

                for k, v
                in probe
                .state_dict()
                .items()
            }


        if (
            epoch == 1
            or
            epoch % 10 == 0
            or
            epoch == PROBE_EPOCHS
        ):

            print(

                f"Epoch "
                f"{epoch:02d}/{PROBE_EPOCHS}"

                f" | loss="
                f"{total_loss / max(count, 1):.4f}"

                f" | val acc="
                f"{val_result['accuracy'] * 100:.2f}%"

                f" | val MRR="
                f"{val_result['mrr']:.4f}"
            )


    probe.load_state_dict(
        best_state
    )


    train_result = evaluate(

        probe,

        train_x,

        train_y,
    )


    val_result = evaluate(

        probe,

        val_x,

        val_y,
    )


    test_result = evaluate(

        probe,

        test_x,

        test_y,
    )


    print()

    print(
        "Best epoch:",
        best_epoch
    )


    print()

    print(

        f"{'Split':<12}"
        f"{'Accuracy':>14}"
        f"{'Top-3':>12}"
        f"{'MRR':>12}"
        f"{'Mean Rank':>14}"
    )


    for split, result in [

        (
            "TRAIN",
            train_result,
        ),

        (
            "VALID",
            val_result,
        ),

        (
            "TEST",
            test_result,
        ),
    ]:

        print(

            f"{split:<12}"

            f"{result['accuracy'] * 100:>13.2f}%"

            f"{result['top3'] * 100:>11.2f}%"

            f"{result['mrr']:>12.4f}"

            f"{result['mean_rank']:>14.4f}"
        )


    return {

        "best_epoch":
            best_epoch,

        "train":
            train_result,

        "val":
            val_result,

        "test":
            test_result,
    }


# ============================================================
# RUN ABLATION
# ============================================================

section(
    "RUN WRITER REPRESENTATION ABLATION"
)


representation_names = [

    name

    for name in train_data.keys()

    if name not in {

        "labels",
        "slots",
        "attention_records",
    }
]


results = OrderedDict()


for representation_name in (
    representation_names
):

    results[
        representation_name
    ] = train_probe(
        representation_name
    )


# ============================================================
# FINAL TABLE
# ============================================================

section(
    "FINAL WRITER ABLATION RESULTS"
)


print(

    f"{'Representation':<32}"

    f"{'Test Acc':>14}"

    f"{'Top-3':>12}"

    f"{'MRR':>12}"

    f"{'Mean Rank':>14}"
)


for name, result in (
    results.items()
):

    test = (
        result[
            "test"
        ]
    )


    print(

        f"{name:<32}"

        f"{test['accuracy'] * 100:>13.2f}%"

        f"{test['top3'] * 100:>11.2f}%"

        f"{test['mrr']:>12.4f}"

        f"{test['mean_rank']:>14.4f}"
    )


print()

print(
    "Chance:",
    f"{100 / len(ANSWERS):.2f}%"
)


# ============================================================
# PRESERVATION RATIOS
# ============================================================

section(
    "SEMANTIC PRESERVATION"
)


summary_accuracy = (

    results[
        "GPT2_SUMMARY"
    ][
        "test"
    ][
        "accuracy"
    ]
)


for name in representation_names:

    accuracy = (

        results[
            name
        ][
            "test"
        ][
            "accuracy"
        ]
    )


    if summary_accuracy > 0:

        preservation = (

            accuracy
            /
            summary_accuracy
        )

    else:

        preservation = 0.0


    print(

        f"{name:<32}"

        f"accuracy="
        f"{accuracy * 100:6.2f}%"

        f" | preserve="
        f"{preservation * 100:6.2f}% "
        f"of summary probe accuracy"
    )


# ============================================================
# AUTOMATIC INTERPRETATION
# ============================================================

section(
    "AUTOMATIC INTERPRETATION"
)


current_attention_acc = (

    results[
        "ATTENDED_CONTEXT"
    ][
        "test"
    ][
        "accuracy"
    ]
)


current_candidate_acc = (

    results[
        "CURRENT_CANDIDATE"
    ][
        "test"
    ][
        "accuracy"
    ]
)


direct_acc = (

    results[
        "DIRECT_SUMMARY"
    ][
        "test"
    ][
        "accuracy"
    ]
)


res010_acc = (

    results[
        "SUMMARY_PLUS_010_CONTEXT"
    ][
        "test"
    ][
        "accuracy"
    ]
)


res025_acc = (

    results[
        "SUMMARY_PLUS_025_CONTEXT"
    ][
        "test"
    ][
        "accuracy"
    ]
)


res050_acc = (

    results[
        "SUMMARY_PLUS_050_CONTEXT"
    ][
        "test"
    ][
        "accuracy"
    ]
)


res100_acc = (

    results[
        "SUMMARY_PLUS_CONTEXT"
    ][
        "test"
    ][
        "accuracy"
    ]
)


print(
    "GPT2 summary:",
    f"{summary_accuracy * 100:.2f}%"
)

print(
    "Current attended context:",
    f"{current_attention_acc * 100:.2f}%"
)

print(
    "Current candidate:",
    f"{current_candidate_acc * 100:.2f}%"
)

print(
    "Direct summary:",
    f"{direct_acc * 100:.2f}%"
)

print(
    "Summary + 0.10 context:",
    f"{res010_acc * 100:.2f}%"
)

print(
    "Summary + 0.25 context:",
    f"{res025_acc * 100:.2f}%"
)

print(
    "Summary + 0.50 context:",
    f"{res050_acc * 100:.2f}%"
)

print(
    "Summary + 1.00 context:",
    f"{res100_acc * 100:.2f}%"
)


print()


best_alternative_name = max(

    [
        "DIRECT_SUMMARY",

        "SUMMARY_PLUS_010_CONTEXT",

        "SUMMARY_PLUS_025_CONTEXT",

        "SUMMARY_PLUS_050_CONTEXT",

        "SUMMARY_PLUS_CONTEXT",
    ],

    key=lambda name:
        results[
            name
        ][
            "test"
        ][
            "accuracy"
        ],
)


best_alternative_accuracy = (

    results[
        best_alternative_name
    ][
        "test"
    ][
        "accuracy"
    ]
)


print(
    "Best simple alternative:",
    best_alternative_name,
)

print(
    "Best alternative accuracy:",
    f"{best_alternative_accuracy * 100:.2f}%"
)


print()


if (
    best_alternative_accuracy
    >
    current_candidate_acc
    + 0.30
):

    print(
        "RESULT:"
    )

    print()

    print(
        "A simple summary-preserving writer retains "
        "substantially more answer information than "
        "the current attention CandidateWriter."
    )

    print()

    print(
        "This strongly supports redesigning the writer "
        "as a SEMANTIC-PRESERVING residual writer."
    )


else:

    print(
        "RESULT:"
    )

    print()

    print(
        "Simple summary preservation did not produce "
        "a sufficiently large improvement."
    )

    print()

    print(
        "Further analysis of the input representation "
        "and writer training objective is required."
    )


# ============================================================
# ATTENTION INTERPRETATION
# ============================================================

section(
    "WHY IS ATTENTION LOSING INFORMATION?"
)


if len(records) == 0:

    print(
        "Attention weights unavailable."
    )


else:

    mean_answer = float(
        answer_mass.mean()
    )


    mean_entity = float(
        entity_mass.mean()
    )


    mean_other = float(
        other_mass.mean()
    )


    top_answer_rate = (

        category_counts[
            "ANSWER"
        ]

        /
        len(
            records
        )
    )


    print(
        "Mean answer-token attention:",
        f"{mean_answer * 100:.2f}%"
    )


    print(
        "Mean entity-token attention:",
        f"{mean_entity * 100:.2f}%"
    )


    print(
        "Mean other-token attention:",
        f"{mean_other * 100:.2f}%"
    )


    print(
        "Answer is top-attended token in:",
        f"{top_answer_rate * 100:.2f}% "
        "of examples"
    )


    print(
        "Mean normalized entropy:",
        f"{entropy.mean():.4f}"
    )


    print()


    if mean_answer < 0.20:

        print(
            "DIAGNOSIS:"
        )

        print()

        print(
            "The CandidateWriter cross-attention assigns "
            "little weight to the actual answer token."
        )

        print()

        print(
            "Therefore the attended context is dominated "
            "by entity/template/other tokens instead of "
            "the value we want to remember."
        )


    elif entropy.mean() > 0.80:

        print(
            "DIAGNOSIS:"
        )

        print()

        print(
            "Attention is highly diffuse across the sentence."
        )

        print()

        print(
            "The answer signal is being averaged together "
            "with a large amount of template/context content."
        )


    else:

        print(
            "DIAGNOSIS:"
        )

        print()

        print(
            "Attention does look at the answer to some degree, "
            "so the larger loss may occur in the subsequent "
            "attention-fusion transformation."
        )


# ============================================================
# SAVE RESULTS
# ============================================================

section(
    "SAVE RESULTS"
)


save_data = {

    "answers":
        ANSWERS,

    "results":
        {},

    "attention":
        {},
}


for name, result in (
    results.items()
):

    save_data[
        "results"
    ][
        name
    ] = {

        "test_accuracy":
            result[
                "test"
            ][
                "accuracy"
            ],

        "test_mrr":
            result[
                "test"
            ][
                "mrr"
            ],

        "val_accuracy":
            result[
                "val"
            ][
                "accuracy"
            ],
    }


if len(records) > 0:

    save_data[
        "attention"
    ] = {

        "mean_answer_mass":
            float(
                answer_mass.mean()
            ),

        "mean_entity_mass":
            float(
                entity_mass.mean()
            ),

        "mean_other_mass":
            float(
                other_mass.mean()
            ),

        "mean_normalized_entropy":
            float(
                entropy.mean()
            ),

        "top_answer_rate":
            float(
                top_answer_rate
            ),
    }


output_path = (
    "outputs/"
    "writer_ablation_attention_diagnosis.pt"
)


torch.save(

    save_data,

    output_path,
)


print(
    "Saved:",
    output_path
)


# ============================================================
# CLEANUP
# ============================================================

attention_handle.remove()


# ============================================================
# COMPLETE
# ============================================================

section(
    "EXPERIMENT COMPLETE"
)


print(
    "GPT-2 frozen."
)

print(
    "Current CandidateWriter frozen."
)

print(
    "Vector gate frozen."
)

print(
    "Router frozen."
)

print(
    "Memory bank frozen."
)

print(
    "No architecture was modified."
)

print(
    "Only independent linear probes were trained."
)