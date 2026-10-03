# ============================================================
# STAGE-WISE MEMORY WRITER PROBE
#
# PURPOSE
# ============================================================
#
# We know:
#
#   Layer-1 addressing -> correct slot      WORKS
#   Correct slot -> correct answer          FAILS
#   Final stored-slot linear probe          ~chance
#
# Now we locate EXACTLY where answer information disappears
# in the WRITE pathway.
#
#
# STAGES PROBED
# ============================================================
#
# 1. GPT2_SUMMARY
#       Exact pooled GPT-2 representation used by writer/gate.
#
# 2. ATTENDED_CONTEXT
#       Writer's token-attention context for selected slot.
#
# 3. WRITER_CANDIDATE
#       writer_output.candidates[selected_slot]
#
# 4. WRITER_DELTA
#       writer_output.deltas[selected_slot]
#
# 5. ORTHOGONAL_UPDATE
#       orthogonal_output.updates[selected_slot]
#
# 6. PROJECTED_CANDIDATE
#       old_slot + orthogonal_update
#
# 7. GATED_PRENORM
#       old_slot + gate * orthogonal_update
#
#       This is the value BEFORE memory-bank normalization.
#
# 8. FINAL_STORED_SLOT
#       Actual value in memory_state.slots after memory_bank.
#
#
# CONTROL
# ============================================================
#
# PRE_WRITE_SLOT
#       Same selected slot BEFORE fact was written.
#
# Should be chance (~6.25%).
#
#
# IMPORTANT
# ============================================================
#
# - GPT-2 frozen
# - Writer frozen
# - Vector gate frozen
# - Router frozen
# - Orthogonalizer frozen
# - Memory bank frozen
#
# ONLY tiny linear probes are trained.
#
# No models/ files are changed.
# No original checkpoints are overwritten.
#
#
# RUN
# ============================================================
#
# python stagewise_writer_probe.py \
#   2>&1 | tee stagewise_writer_probe.log
#
# ============================================================


import os
import random
import string
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

SEED = 2027

NUM_SLOTS = 8

# ------------------------------------------------------------
# Balanced data
# ------------------------------------------------------------

TRAIN_PER_CLASS = 100
VAL_PER_CLASS = 25
TEST_PER_CLASS = 50

# 16 answers:
#
# train = 1600
# valid = 400
# test  = 800
#
# Chance = 6.25%


# ------------------------------------------------------------
# Probe training
# ------------------------------------------------------------

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
    answer: i
    for i, answer in enumerate(ANSWERS)
}


# ============================================================
# TRAIN / HELD-OUT TEMPLATES
# ============================================================

TRAIN_WRITE_TEMPLATES = [

    "The assigned keyword for {entity} is {answer}.",

    "{entity} has been assigned the keyword {answer}.",

    "Remember that {answer} is associated with {entity}.",

    "Store the mapping {entity} to {answer}.",

    "The value assigned to {entity} is {answer}.",

    "For {entity}, the stored keyword is {answer}.",

    "{answer} is the keyword linked with {entity}.",

    "Please remember this association: {entity} means {answer}.",
]


HELDOUT_WRITE_TEMPLATES = [

    "Record {answer} as the value belonging to {entity}.",

    "In memory, associate {entity} with {answer}.",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DISPLAY
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

section("LOAD FROZEN VECTOR-GATE MODEL")


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

skipped = []


for name, value in checkpoint_state.items():

    if (
        name in current_state
        and
        current_state[name].shape == value.shape
    ):

        compatible[name] = value

    else:

        skipped.append(name)


result = model.load_state_dict(
    compatible,
    strict=False,
)


print(
    "Compatible tensors:",
    len(compatible),
)

print(
    "Missing tensors:",
    len(result.missing_keys),
)

print(
    "Skipped tensors:",
    len(skipped),
)


model.to(DEVICE)

model.eval()


for parameter in model.parameters():

    parameter.requires_grad = False


print()

print(
    "Entire memory architecture frozen."
)

print(
    "Gate type = VECTOR"
)


# ============================================================
# ENTITY GENERATION
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


# ============================================================
# DATASET
# ============================================================

def build_examples(
    prefix,
    per_class,
    templates,
    seed,
):

    rng = random.Random(
        seed
    )


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

                    "text":
                        text,

                    "answer":
                        answer,

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
# BUILD DATA
# ============================================================

section("BUILD DATASETS")


train_examples = build_examples(

    prefix="StageTrain",

    per_class=TRAIN_PER_CLASS,

    templates=TRAIN_WRITE_TEMPLATES,

    seed=SEED + 1,
)


val_examples = build_examples(

    prefix="StageValid",

    per_class=VAL_PER_CLASS,

    templates=TRAIN_WRITE_TEMPLATES,

    seed=SEED + 2,
)


test_examples = build_examples(

    prefix="StageTest",

    per_class=TEST_PER_CLASS,

    # HELD-OUT sentence structures.

    templates=HELDOUT_WRITE_TEMPLATES,

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
    "Classes:",
    len(ANSWERS),
)

print(
    "Chance:",
    f"{100 / len(ANSWERS):.2f}%"
)


print()

print(
    "Training example:"
)

print(
    train_examples[0]["text"]
)


print()

print(
    "Held-out test example:"
)

print(
    test_examples[0]["text"]
)


# ============================================================
# MEMORY INITIALIZATION
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
    before_state,
    after_state,
):

    before = (
        before_state
        .write_count[0]
        .detach()
        .cpu()
    )


    after = (
        after_state
        .write_count[0]
        .detach()
        .cpu()
    )


    difference = (
        after
        - before
    )


    changed = (

        difference
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
            difference
        ).item()
    )


# ============================================================
# EXTRACT EXACT WRITE PIPELINE
# ============================================================

@torch.no_grad()
def extract_stages(
    text,
):

    # --------------------------------------------------------
    # Fresh memory.
    # --------------------------------------------------------

    memory_state = initialize_memory()


    pre_slots = (
        memory_state
        .slots
        .detach()
        .clone()
    )


    # --------------------------------------------------------
    # Tokenize fact.
    # --------------------------------------------------------

    encoded = tokenizer(

        text,

        return_tensors="pt",
    )


    input_ids = (
        encoded[
            "input_ids"
        ]
        .to(DEVICE)
    )


    attention_mask = (
        encoded[
            "attention_mask"
        ]
        .to(DEVICE)
    )


    # ========================================================
    # STAGE 1:
    # EXACT GPT-2 BASE HIDDEN + SUMMARY
    #
    # The model's writer receives:
    #
    # summary = _pool_hidden(base_hidden)
    #
    # ========================================================

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


    # ========================================================
    # RUN ACTUAL MODEL WRITE
    #
    # This gives us:
    #
    # writer_output
    # orthogonal_output
    # write_gate
    # final memory state
    #
    # ========================================================

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
            "No memory slot was written."
        )


    if output.writer_output is None:

        raise RuntimeError(
            "writer_output is None."
        )


    if output.orthogonal_output is None:

        raise RuntimeError(
            "orthogonal_output is None."
        )


    if output.write_gate is None:

        raise RuntimeError(
            "write_gate is None."
        )


    # ========================================================
    # CONTROL:
    # PRE-WRITE SLOT
    # ========================================================

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


    # ========================================================
    # STAGE 1:
    # GPT-2 SUMMARY
    # ========================================================

    summary_vector = (

        summary[
            0
        ]
        .detach()
        .float()
        .cpu()
    )


    # ========================================================
    # STAGE 2:
    # ATTENDED CONTEXT
    # ========================================================

    attended_context = (
        output
        .writer_output
        .attended_context
    )


    if attended_context is not None:

        attended_vector = (

            attended_context[
                0,
                slot,
                :
            ]
            .detach()
            .float()
            .cpu()
        )

    else:

        attended_vector = None


    # ========================================================
    # STAGE 3:
    # RAW WRITER CANDIDATE
    # ========================================================

    writer_candidate = (

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


    # ========================================================
    # STAGE 4:
    # RAW WRITER DELTA
    # ========================================================

    writer_delta = (

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
    # STAGE 5:
    # ORTHOGONALIZED UPDATE
    # ========================================================

    orthogonal_update = (

        output
        .orthogonal_output
        .updates[
            0,
            slot,
            :
        ]
        .detach()
        .float()
        .cpu()
    )


    # ========================================================
    # STAGE 6:
    # PROJECTED CANDIDATE
    #
    # Exact formula in _write_memory:
    #
    # projected_candidate =
    #       memory_state.slots
    #       +
    #       orthogonal_output.updates
    #
    # ========================================================

    projected_candidate = (

        pre_slot
        +
        orthogonal_update
    )


    # ========================================================
    # WRITE GATE
    # ========================================================

    gate = (

        output
        .write_gate[
            0,
            slot
        ]
        .detach()
        .float()
        .cpu()
    )


    # Gate may have shape [1].
    #
    # Convert it to scalar safely.

    gate_value = float(
        gate.reshape(
            -1
        )[0].item()
    )


    # ========================================================
    # STAGE 7:
    # GATED PRE-NORMALIZATION VALUE
    #
    # Memory bank conceptually performs:
    #
    # M_new =
    #     (1-g) M_old
    #     +
    #     g candidate
    #
    # Since:
    #
    # candidate =
    #     M_old + orthogonal_update
    #
    # therefore:
    #
    # M_new_pre_norm =
    #     M_old
    #     +
    #     g * orthogonal_update
    #
    # ========================================================

    gated_prenorm = (

        pre_slot
        +
        gate_value
        *
        orthogonal_update
    )


    # ========================================================
    # STAGE 8:
    # FINAL STORED SLOT
    # ========================================================

    final_slot = (

        after_state
        .slots[
            0,
            slot,
            :
        ]
        .detach()
        .float()
        .cpu()
    )


    # ========================================================
    # FINAL ACTUAL UPDATE
    # ========================================================

    final_delta = (

        final_slot
        -
        pre_slot
    )


    stages = OrderedDict()


    stages[
        "PRE_WRITE_SLOT"
    ] = pre_slot


    stages[
        "GPT2_SUMMARY"
    ] = summary_vector


    if attended_vector is not None:

        stages[
            "ATTENDED_CONTEXT"
        ] = attended_vector


    stages[
        "WRITER_CANDIDATE"
    ] = writer_candidate


    stages[
        "WRITER_DELTA"
    ] = writer_delta


    stages[
        "ORTHOGONAL_UPDATE"
    ] = orthogonal_update


    stages[
        "PROJECTED_CANDIDATE"
    ] = projected_candidate


    stages[
        "GATED_PRENORM"
    ] = gated_prenorm


    stages[
        "FINAL_STORED_SLOT"
    ] = final_slot


    stages[
        "FINAL_ACTUAL_DELTA"
    ] = final_delta


    return {

        "slot":
            slot,

        "gate":
            gate_value,

        "stages":
            stages,
    }


# ============================================================
# QUICK PIPELINE SANITY CHECK
# ============================================================

section(
    "PIPELINE SANITY CHECK"
)


example = extract_stages(
    train_examples[0]["text"]
)


print(
    "Selected slot:",
    example["slot"],
)

print(
    "Selected-slot gate:",
    fmt(
        example["gate"]
    ),
)


print()

print(
    "Available stages:"
)


for stage_name, vector in (
    example["stages"].items()
):

    print(

        f"{stage_name:<24}"

        f"shape={tuple(vector.shape)} "

        f"norm={vector.norm().item():.4f}"
    )


# ============================================================
# EXTRACT COMPLETE DATASET
# ============================================================

@torch.no_grad()
def extract_dataset(
    examples,
    dataset_name,
):

    # stage_name -> list[tensor]

    storage = None

    labels = []

    slots = []

    gates = []


    total = len(
        examples
    )


    for idx, example in enumerate(
        examples,
        start=1,
    ):

        result = extract_stages(
            example["text"]
        )


        if storage is None:

            storage = OrderedDict(

                (
                    stage_name,
                    []
                )

                for stage_name
                in result[
                    "stages"
                ].keys()
            )


        for stage_name, vector in (
            result[
                "stages"
            ].items()
        ):

            storage[
                stage_name
            ].append(
                vector
            )


        labels.append(
            example["label"]
        )


        slots.append(
            result["slot"]
        )


        gates.append(
            result["gate"]
        )


        if (
            idx % 100 == 0
            or
            idx == total
        ):

            print(

                f"{dataset_name}: "
                f"{idx}/{total}"
            )


    final = {}


    for stage_name, vectors in (
        storage.items()
    ):

        final[
            stage_name
        ] = torch.stack(
            vectors,
            dim=0,
        )


    final[
        "labels"
    ] = torch.tensor(
        labels,
        dtype=torch.long,
    )


    final[
        "slots"
    ] = torch.tensor(
        slots,
        dtype=torch.long,
    )


    final[
        "gates"
    ] = torch.tensor(
        gates,
        dtype=torch.float32,
    )


    return final


# ============================================================
# EXTRACT DATA
# ============================================================

section(
    "EXTRACT TRAIN WRITE STAGES"
)


train_data = extract_dataset(

    train_examples,

    "TRAIN",
)


section(
    "EXTRACT VALIDATION WRITE STAGES"
)


val_data = extract_dataset(

    val_examples,

    "VALID",
)


section(
    "EXTRACT HELD-OUT TEST WRITE STAGES"
)


test_data = extract_dataset(

    test_examples,

    "TEST",
)


# ============================================================
# SLOT / GATE DIAGNOSTICS
# ============================================================

section(
    "ROUTER / GATE DIAGNOSTICS"
)


for name, data in [

    (
        "TRAIN",
        train_data,
    ),

    (
        "VALID",
        val_data,
    ),

    (
        "TEST",
        test_data,
    ),
]:

    unique, counts = torch.unique(

        data["slots"],

        return_counts=True,
    )


    print()

    print(name)


    print(
        "Slot distribution:"
    )


    for slot, count in zip(

        unique.tolist(),

        counts.tolist(),
    ):

        print(

            f"  slot {slot}: "
            f"{count}"
        )


    print(

        "Gate mean:",
        fmt(
            data[
                "gates"
            ].mean()
        )
    )


    print(

        "Gate std:",
        fmt(
            data[
                "gates"
            ].std()
        )
    )


    print(

        "Gate min:",
        fmt(
            data[
                "gates"
            ].min()
        )
    )


    print(

        "Gate max:",
        fmt(
            data[
                "gates"
            ].max()
        )
    )


# ============================================================
# STAGE VECTOR DIAGNOSTICS
# ============================================================

section(
    "STAGE VECTOR DIAGNOSTICS"
)


stage_names = [

    key

    for key in train_data.keys()

    if key not in {
        "labels",
        "slots",
        "gates",
    }
]


print(

    f"{'Stage':<26}"
    f"{'Train norm':>14}"
    f"{'Test norm':>14}"
    f"{'Test std':>14}"
)


for stage_name in stage_names:

    train_vectors = (
        train_data[
            stage_name
        ]
    )


    test_vectors = (
        test_data[
            stage_name
        ]
    )


    train_norm = (

        train_vectors
        .norm(
            dim=-1
        )
        .mean()
        .item()
    )


    test_norm = (

        test_vectors
        .norm(
            dim=-1
        )
        .mean()
        .item()
    )


    test_std = (

        test_vectors
        .std(
            dim=0
        )
        .mean()
        .item()
    )


    print(

        f"{stage_name:<26}"

        f"{train_norm:>14.4f}"

        f"{test_norm:>14.4f}"

        f"{test_std:>14.6f}"
    )


# ============================================================
# LINEAR PROBE
# ============================================================

class LinearProbe(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes,
    ):

        super().__init__()


        self.classifier = nn.Linear(

            input_dim,

            num_classes,
        )


    def forward(
        self,
        x,
    ):

        return self.classifier(
            x
        )


# ============================================================
# STANDARDIZE FEATURES
#
# IMPORTANT:
#
# Fit mean/std ONLY on training data.
# Apply same transform to validation/test.
#
# This makes probe optimization more stable without leakage.
# ============================================================

def standardize_features(
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
# EVALUATE PROBE
# ============================================================

@torch.no_grad()
def evaluate_probe(
    probe,
    features,
    labels,
):

    probe.eval()


    features = features.to(
        DEVICE
    )


    labels = labels.to(
        DEVICE
    )


    logits = probe(
        features
    )


    predictions = (
        logits.argmax(
            dim=-1
        )
    )


    accuracy = (

        predictions
        .eq(labels)
        .float()
        .mean()
        .item()
    )


    # --------------------------------------------------------
    # Top-3 accuracy
    # --------------------------------------------------------

    top3 = (

        logits.topk(
            k=3,
            dim=-1,
        )
        .indices
    )


    top3_accuracy = (

        top3
        .eq(
            labels.unsqueeze(
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


    # --------------------------------------------------------
    # Rank / MRR
    # --------------------------------------------------------

    correct_logits = (

        logits.gather(

            1,

            labels.unsqueeze(
                1
            ),
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
            float(
                accuracy
            ),

        "top3":
            float(
                top3_accuracy
            ),

        "mrr":
            float(
                mrr
            ),

        "mean_rank":
            float(
                mean_rank
            ),
    }


# ============================================================
# TRAIN PROBE FOR ONE STAGE
# ============================================================

def train_probe_for_stage(
    stage_name,
):

    section(
        f"LINEAR PROBE: {stage_name}"
    )


    train_x = (
        train_data[
            stage_name
        ]
        .float()
    )


    val_x = (
        val_data[
            stage_name
        ]
        .float()
    )


    test_x = (
        test_data[
            stage_name
        ]
        .float()
    )


    (
        train_x,
        val_x,
        test_x,
    ) = standardize_features(

        train_x,

        val_x,

        test_x,
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


    input_dim = (
        train_x.shape[
            -1
        ]
    )


    probe = LinearProbe(

        input_dim=input_dim,

        num_classes=len(
            ANSWERS
        ),
    ).to(
        DEVICE
    )


    optimizer = torch.optim.AdamW(

        probe.parameters(),

        lr=PROBE_LR,

        weight_decay=PROBE_WEIGHT_DECAY,
    )


    criterion = (
        nn.CrossEntropyLoss()
    )


    dataset = TensorDataset(

        train_x,

        train_y,
    )


    loader = DataLoader(

        dataset,

        batch_size=PROBE_BATCH_SIZE,

        shuffle=True,
    )


    best_val_acc = -1.0

    best_state = None

    best_epoch = None


    for epoch in range(
        1,
        PROBE_EPOCHS + 1,
    ):

        probe.train()


        total_loss = 0.0

        total_examples = 0


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


            loss = criterion(

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


            total_examples += (
                batch_x.size(
                    0
                )
            )


        val_result = evaluate_probe(

            probe,

            val_x,

            val_y,
        )


        if (
            val_result[
                "accuracy"
            ]
            >
            best_val_acc
        ):

            best_val_acc = (
                val_result[
                    "accuracy"
                ]
            )


            best_epoch = epoch


            best_state = {

                key:
                    value
                    .detach()
                    .cpu()
                    .clone()

                for key, value
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

            mean_loss = (

                total_loss
                /
                max(
                    total_examples,
                    1
                )
            )


            print(

                f"Epoch "
                f"{epoch:02d}/{PROBE_EPOCHS}"

                f" | loss="
                f"{mean_loss:.4f}"

                f" | val acc="
                f"{val_result['accuracy'] * 100:.2f}%"

                f" | val MRR="
                f"{val_result['mrr']:.4f}"
            )


    probe.load_state_dict(
        best_state
    )


    probe.to(
        DEVICE
    )


    train_result = evaluate_probe(

        probe,

        train_x,

        train_y,
    )


    val_result = evaluate_probe(

        probe,

        val_x,

        val_y,
    )


    test_result = evaluate_probe(

        probe,

        test_x,

        test_y,
    )


    print()

    print(
        "Best epoch:",
        best_epoch,
    )


    print()

    print(

        f"{'Split':<12}"

        f"{'Accuracy':>14}"

        f"{'Top-3':>12}"

        f"{'MRR':>12}"

        f"{'Mean Rank':>14}"
    )


    for split_name, result in [

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

            f"{split_name:<12}"

            f"{result['accuracy'] * 100:>13.2f}%"

            f"{result['top3'] * 100:>11.2f}%"

            f"{result['mrr']:>12.4f}"

            f"{result['mean_rank']:>14.4f}"
        )


    return {

        "stage":
            stage_name,

        "best_epoch":
            best_epoch,

        "train":
            train_result,

        "val":
            val_result,

        "test":
            test_result,

        "probe_state":
            {
                key:
                    value.detach().cpu()

                for key, value
                in probe
                .state_dict()
                .items()
            },
    }


# ============================================================
# RUN ALL STAGE PROBES
# ============================================================

section(
    "RUN STAGE-WISE PROBES"
)


probe_results = OrderedDict()


for stage_name in stage_names:

    result = train_probe_for_stage(
        stage_name
    )


    probe_results[
        stage_name
    ] = result


# ============================================================
# FINAL TABLE
# ============================================================

section(
    "FINAL STAGE-WISE TEST RESULTS"
)


chance = (
    1.0
    /
    len(
        ANSWERS
    )
)


print(

    f"{'Stage':<26}"

    f"{'Test Acc':>14}"

    f"{'Top-3':>12}"

    f"{'MRR':>12}"

    f"{'Mean Rank':>14}"
)


for stage_name in stage_names:

    result = (

        probe_results[
            stage_name
        ][
            "test"
        ]
    )


    print(

        f"{stage_name:<26}"

        f"{result['accuracy'] * 100:>13.2f}%"

        f"{result['top3'] * 100:>11.2f}%"

        f"{result['mrr']:>12.4f}"

        f"{result['mean_rank']:>14.4f}"
    )


print()

print(
    "Random chance:",
    f"{chance * 100:.2f}%"
)


# ============================================================
# FIND LARGEST INFORMATION DROP
# ============================================================

section(
    "INFORMATION DROP ANALYSIS"
)


ordered_pipeline = [

    "GPT2_SUMMARY",

    "ATTENDED_CONTEXT",

    "WRITER_CANDIDATE",

    "WRITER_DELTA",

    "ORTHOGONAL_UPDATE",

    "PROJECTED_CANDIDATE",

    "GATED_PRENORM",

    "FINAL_STORED_SLOT",
]


ordered_pipeline = [

    stage

    for stage in ordered_pipeline

    if stage in probe_results
]


largest_drop = None


for previous_stage, current_stage in zip(

    ordered_pipeline[:-1],

    ordered_pipeline[1:],
):

    previous_accuracy = (

        probe_results[
            previous_stage
        ][
            "test"
        ][
            "accuracy"
        ]
    )


    current_accuracy = (

        probe_results[
            current_stage
        ][
            "test"
        ][
            "accuracy"
        ]
    )


    difference = (

        current_accuracy
        -
        previous_accuracy
    )


    print(

        f"{previous_stage:<24}"
        f" -> "
        f"{current_stage:<24}"

        f" : "

        f"{previous_accuracy * 100:6.2f}%"
        f" -> "
        f"{current_accuracy * 100:6.2f}%"

        f"   change="
        f"{difference * 100:+7.2f} points"
    )


    drop = (

        previous_accuracy
        -
        current_accuracy
    )


    if (
        largest_drop is None
        or
        drop
        >
        largest_drop[
            "drop"
        ]
    ):

        largest_drop = {

            "from":
                previous_stage,

            "to":
                current_stage,

            "drop":
                drop,
        }


print()

print(
    "Largest accuracy drop:"
)


print(

    largest_drop[
        "from"
    ],

    "->",

    largest_drop[
        "to"
    ],

    f"({largest_drop['drop'] * 100:+.2f} points)"
)


# ============================================================
# CORE DIAGNOSIS
# ============================================================

section(
    "AUTOMATIC DIAGNOSIS"
)


summary_acc = (

    probe_results[
        "GPT2_SUMMARY"
    ][
        "test"
    ][
        "accuracy"
    ]
)


writer_candidate_acc = (

    probe_results[
        "WRITER_CANDIDATE"
    ][
        "test"
    ][
        "accuracy"
    ]
)


writer_delta_acc = (

    probe_results[
        "WRITER_DELTA"
    ][
        "test"
    ][
        "accuracy"
    ]
)


orthogonal_acc = (

    probe_results[
        "ORTHOGONAL_UPDATE"
    ][
        "test"
    ][
        "accuracy"
    ]
)


gated_acc = (

    probe_results[
        "GATED_PRENORM"
    ][
        "test"
    ][
        "accuracy"
    ]
)


final_acc = (

    probe_results[
        "FINAL_STORED_SLOT"
    ][
        "test"
    ][
        "accuracy"
    ]
)


pre_acc = (

    probe_results[
        "PRE_WRITE_SLOT"
    ][
        "test"
    ][
        "accuracy"
    ]
)


print(
    "PRE-WRITE:",
    f"{pre_acc * 100:.2f}%"
)

print(
    "GPT2 SUMMARY:",
    f"{summary_acc * 100:.2f}%"
)

print(
    "WRITER CANDIDATE:",
    f"{writer_candidate_acc * 100:.2f}%"
)

print(
    "WRITER DELTA:",
    f"{writer_delta_acc * 100:.2f}%"
)

print(
    "ORTHOGONAL UPDATE:",
    f"{orthogonal_acc * 100:.2f}%"
)

print(
    "GATED PRE-NORM:",
    f"{gated_acc * 100:.2f}%"
)

print(
    "FINAL SLOT:",
    f"{final_acc * 100:.2f}%"
)

print()


# ============================================================
# CASE 1
# SUMMARY ALREADY FAILS
# ============================================================

if summary_acc < 0.20:

    print(
        "PRIMARY FINDING:"
    )

    print()

    print(
        "The exact pooled GPT-2 summary entering the writer "
        "already contains very weak linearly decodable "
        "answer identity."
    )

    print()

    print(
        "Therefore changing scalar/vector gate alone is "
        "unlikely to solve the problem."
    )

    print()

    print(
        "The write representation / pooling strategy "
        "must be investigated first."
    )


# ============================================================
# CASE 2
# SUMMARY GOOD, WRITER DESTROYS IT
# ============================================================

elif (
    summary_acc >= 0.50
    and
    writer_candidate_acc
    <
    summary_acc - 0.20
):

    print(
        "PRIMARY FINDING:"
    )

    print()

    print(
        "The GPT-2 summary contains answer information, "
        "but the CandidateWriter removes a large amount "
        "of that information."
    )

    print()

    print(
        "The CandidateWriter is the first major bottleneck."
    )

    print()

    print(
        "Do NOT blame the vector gate yet."
    )


# ============================================================
# CASE 3
# ORTHOGONALIZER DESTROYS IT
# ============================================================

elif (
    writer_delta_acc >= 0.50
    and
    orthogonal_acc
    <
    writer_delta_acc - 0.20
):

    print(
        "PRIMARY FINDING:"
    )

    print()

    print(
        "The writer delta contains answer information, "
        "but orthogonalization substantially removes it."
    )

    print()

    print(
        "NEXT EXPERIMENT:"
    )

    print(
        "Compare orthogonal update ON vs OFF."
    )


# ============================================================
# CASE 4
# GATE DESTROYS IT
# ============================================================

elif (
    orthogonal_acc >= 0.50
    and
    gated_acc
    <
    orthogonal_acc - 0.20
):

    print(
        "PRIMARY FINDING:"
    )

    print()

    print(
        "Answer information survives through the "
        "orthogonal update but collapses after gating."
    )

    print()

    print(
        "NOW scalar-vs-vector-vs-no-gate is the correct "
        "next ablation."
    )


# ============================================================
# CASE 5
# MEMORY BANK NORMALIZATION DESTROYS IT
# ============================================================

elif (
    gated_acc >= 0.50
    and
    final_acc
    <
    gated_acc - 0.20
):

    print(
        "PRIMARY FINDING:"
    )

    print()

    print(
        "The gated vector contains answer information, "
        "but the final memory-bank update/normalization "
        "removes it."
    )

    print()

    print(
        "NEXT EXPERIMENT:"
    )

    print(
        "Inspect memory normalization and old/new "
        "slot mixing."
    )


# ============================================================
# CASE 6
# INFORMATION SURVIVES ALL WRITING STAGES
# ============================================================

elif final_acc >= 0.70:

    print(
        "PRIMARY FINDING:"
    )

    print()

    print(
        "Answer identity remains strongly recoverable "
        "from the final stored memory."
    )

    print()

    print(
        "Therefore the write side is functioning."
    )

    print()

    print(
        "The next bottleneck is reader/value projection/fusion."
    )


# ============================================================
# PARTIAL / MIXED CASE
# ============================================================

else:

    print(
        "PRIMARY FINDING:"
    )

    print()

    print(
        "Answer information degrades gradually rather "
        "than disappearing at one obvious stage."
    )

    print()

    print(
        "Use the stage table and largest-drop analysis "
        "to choose the next ablation."
    )


# ============================================================
# PRE-WRITE CONTROL
# ============================================================

print()

if pre_acc > 0.20:

    print(
        "WARNING:"
    )

    print(
        "PRE_WRITE_SLOT probe is unexpectedly above chance."
    )

    print(
        "Do not trust conclusions until leakage is checked."
    )

else:

    print(
        "PRE_WRITE_SLOT control looks reasonable."
    )


# ============================================================
# SAVE EXPERIMENTAL RESULTS
# ============================================================

section(
    "SAVE RESULTS"
)


OUTPUT_PATH = (
    "outputs/"
    "stagewise_writer_probe.pt"
)


save_results = {

    "answers":
        ANSWERS,

    "chance":
        chance,

    "gate_mean":
        float(
            test_data[
                "gates"
            ].mean()
        ),

    "gate_std":
        float(
            test_data[
                "gates"
            ].std()
        ),

    "results":
        {},
}


for stage_name in stage_names:

    save_results[
        "results"
    ][
        stage_name
    ] = {

        "best_epoch":
            probe_results[
                stage_name
            ][
                "best_epoch"
            ],

        "train_accuracy":
            probe_results[
                stage_name
            ][
                "train"
            ][
                "accuracy"
            ],

        "val_accuracy":
            probe_results[
                stage_name
            ][
                "val"
            ][
                "accuracy"
            ],

        "test_accuracy":
            probe_results[
                stage_name
            ][
                "test"
            ][
                "accuracy"
            ],

        "test_mrr":
            probe_results[
                stage_name
            ][
                "test"
            ][
                "mrr"
            ],
    }


torch.save(

    save_results,

    OUTPUT_PATH,
)


print(
    "Saved to:",
    OUTPUT_PATH,
)


# ============================================================
# DONE
# ============================================================

section(
    "EXPERIMENT COMPLETE"
)


print(
    "GPT-2 frozen."
)

print(
    "Candidate writer frozen."
)

print(
    "Vector gate frozen."
)

print(
    "Router frozen."
)

print(
    "Orthogonalizer frozen."
)

print(
    "Memory bank frozen."
)

print(
    "Reader frozen."
)

print(
    "Only linear diagnostic probes were trained."
)

print(
    "No models/ source files modified."
)

print(
    "No original checkpoints overwritten."
)