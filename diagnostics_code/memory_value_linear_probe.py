# ============================================================
# MEMORY VALUE LINEAR PROBE
#
# PURPOSE
# ============================================================
#
# We already know:
#
#   Layer-1 address key -> correct memory slot
#
# works extremely well.
#
# But:
#
#   correct slot -> correct answer
#
# is currently near chance.
#
#
# THIS EXPERIMENT ASKS:
#
#   Does the ACTUAL VECTOR STORED INSIDE THE SLOT
#   contain information about the answer?
#
#
# We probe three representations:
#
#   1. PRE-WRITE SLOT
#
#          slot BEFORE seeing the fact
#
#      This is the control.
#      It should NOT predict the answer.
#
#
#   2. POST-WRITE SLOT
#
#          actual slot AFTER the current writer/gate updates it
#
#      If this predicts the answer well:
#
#          writer/storage DOES contain useful answer information
#
#
#   3. WRITE DELTA
#
#          POST_SLOT - PRE_SLOT
#
#      This isolates the information injected by the write.
#
#
# IMPORTANT
# ============================================================
#
# - Main model completely frozen.
# - No changes to models/
# - No original checkpoint overwritten.
# - Only tiny Linear(768 -> 16) probes are trained.
#
#
# INTERPRETATION
# ============================================================
#
# PRE ~ 6.25%, POST high, DELTA high
#     -> writer/gate stores answer information
#        reader/fusion is the remaining problem
#
# PRE ~ 6.25%, POST ~ 6.25%, DELTA ~ 6.25%
#     -> writer/storage path is failing
#        THEN test scalar vs vector gating
#
# DELTA high but POST weak
#     -> information is created by the write,
#        but slot update/normalization is damaging it
#
#
# RUN:
#
# python memory_value_linear_probe.py \
#   2>&1 | tee memory_value_linear_probe.log
#
# ============================================================


import copy
import math
import os
import random
import string

import numpy as np
import torch
import torch.nn as nn
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

SEED = 2026

NUM_SLOTS = 8

# ------------------------------------------------------------
# BALANCED DATASET
#
# Every answer gets exactly the same number of examples.
# ------------------------------------------------------------

TRAIN_PER_CLASS = 100

VAL_PER_CLASS = 25

TEST_PER_CLASS = 50


# Total:
#
# train = 1600
# val   = 400
# test  = 800
#
# Chance = 1/16 = 6.25%


# ------------------------------------------------------------
# LINEAR PROBE
# ------------------------------------------------------------

PROBE_EPOCHS = 60

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

    for i, answer
    in enumerate(
        ANSWERS
    )
}


# ============================================================
# TRAIN WRITE TEMPLATES
#
# The probe training data uses these.
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


# ============================================================
# HELD-OUT WRITE TEMPLATES
#
# These templates are NEVER used while fitting the probe.
#
# This makes the test stronger.
# ============================================================

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

        torch.cuda.manual_seed_all(
            seed
        )


set_seed(
    SEED
)


# ============================================================
# DISPLAY
# ============================================================

def section(title):

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def fmt(value):

    return f"{float(value):.6f}"


# ============================================================
# MODEL CONFIG
# ============================================================

def build_memory_config():

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

section(
    "TOKENIZER"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

print(
    "Tokenizer:",
    tokenizer.__class__.__name__,
)

print(
    "Vocabulary:",
    len(tokenizer),
)


# ============================================================
# LOAD FROZEN MODEL
# ============================================================

section(
    "LOAD CURRENT VECTOR-GATE MODEL"
)

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

        memory_config=build_memory_config(),
    )
)


checkpoint = torch.load(

    CHECKPOINT,

    map_location="cpu",

    weights_only=False,
)


if "model_state_dict" in checkpoint:

    checkpoint_state = (
        checkpoint[
            "model_state_dict"
        ]
    )

elif "state_dict" in checkpoint:

    checkpoint_state = (
        checkpoint[
            "state_dict"
        ]
    )

else:

    checkpoint_state = (
        checkpoint
    )


current_state = (
    model.state_dict()
)


compatible = {}

skipped = []


for name, value in checkpoint_state.items():

    if (
        name in current_state
        and
        current_state[name].shape
        ==
        value.shape
    ):

        compatible[
            name
        ] = value

    else:

        skipped.append(
            name
        )


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
    len(
        load_result.missing_keys
    ),
)

print(
    "Skipped tensors:",
    len(skipped),
)


model.to(
    DEVICE
)

model.eval()


for parameter in model.parameters():

    parameter.requires_grad = False


print()

print(
    "Entire memory model frozen."
)

print(
    "Gate type: VECTOR"
)


# ============================================================
# ENTITY GENERATION
# ============================================================

def make_random_entity(
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
# BUILD BALANCED EXAMPLES
# ============================================================

def build_dataset_examples(
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

                entity = (
                    make_random_entity(
                        rng,
                        prefix,
                    )
                )

                if (
                    entity
                    not in used_entities
                ):

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

                    "entity":
                        entity,

                    "answer":
                        answer,

                    "label":
                        ANSWER_TO_ID[
                            answer
                        ],

                    "text":
                        text,
                }
            )


    rng.shuffle(
        examples
    )


    return examples


# ============================================================
# DATASETS
# ============================================================

section(
    "BUILD BALANCED DATASETS"
)


train_examples = (
    build_dataset_examples(

        prefix="ProbeTrain",

        per_class=TRAIN_PER_CLASS,

        templates=TRAIN_WRITE_TEMPLATES,

        seed=SEED + 1,
    )
)


val_examples = (
    build_dataset_examples(

        prefix="ProbeValid",

        per_class=VAL_PER_CLASS,

        templates=TRAIN_WRITE_TEMPLATES,

        seed=SEED + 2,
    )
)


test_examples = (
    build_dataset_examples(

        prefix="ProbeTest",

        per_class=TEST_PER_CLASS,

        # IMPORTANT:
        # completely held-out phrasing

        templates=HELDOUT_WRITE_TEMPLATES,

        seed=SEED + 3,
    )
)


print(
    "Train examples:",
    len(train_examples),
)

print(
    "Validation examples:",
    len(val_examples),
)

print(
    "Test examples:",
    len(test_examples),
)

print()

print(
    "Classes:",
    len(ANSWERS),
)

print(
    "Chance accuracy:",
    f"{100 / len(ANSWERS):.2f}%"
)

print()

print(
    "Train example:"
)

print(
    train_examples[
        0
    ][
        "text"
    ]
)

print()

print(
    "HELD-OUT test example:"
)

print(
    test_examples[
        0
    ][
        "text"
    ]
)


# ============================================================
# INITIALIZE MEMORY
# ============================================================

def initialize_memory():

    dtype = next(
        model.parameters()
    ).dtype

    return (
        model.initialize_memory(

            batch_size=1,

            device=DEVICE,

            dtype=dtype,
        )
    )


# ============================================================
# IDENTIFY SLOT WRITTEN
# ============================================================

def identify_written_slot(
    before_state,
    after_state,
):

    before_count = (

        before_state
        .write_count[
            0
        ]
        .detach()
        .cpu()
    )


    after_count = (

        after_state
        .write_count[
            0
        ]
        .detach()
        .cpu()
    )


    difference = (

        after_count
        - before_count
    )


    changed = (

        difference
        .gt(0)
        .nonzero(
            as_tuple=False
        )
        .flatten()
    )


    if len(
        changed
    ) == 0:

        return None


    if len(
        changed
    ) == 1:

        return int(
            changed[
                0
            ].item()
        )


    return int(

        torch.argmax(
            difference
        ).item()
    )


# ============================================================
# EXTRACT SLOT VECTORS FROM ONE FACT
#
# Returns:
#
#   pre_slot
#   post_slot
#   delta
#
# Main model stays frozen.
# ============================================================

@torch.no_grad()
def extract_memory_vectors(
    text,
):

    memory_state = (
        initialize_memory()
    )


    # --------------------------------------------------------
    # Preserve memory BEFORE write.
    # --------------------------------------------------------

    pre_slots = (

        memory_state
        .slots
        .detach()
        .clone()
    )


    encoded = tokenizer(

        text,

        return_tensors="pt",
    )


    input_ids = (

        encoded[
            "input_ids"
        ]
        .to(
            DEVICE
        )
    )


    attention_mask = (

        encoded[
            "attention_mask"
        ]
        .to(
            DEVICE
        )
    )


    before_state = (
        memory_state
    )


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

        before_state,

        after_state,
    )


    if slot is None:

        raise RuntimeError(
            "Writer did not update any memory slot."
        )


    # --------------------------------------------------------
    # Get the exact same slot before and after write.
    # --------------------------------------------------------

    pre_vector = (

        pre_slots[
            0,
            slot,
            :
        ]
        .detach()
        .float()
        .cpu()
    )


    post_vector = (

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


    delta_vector = (

        post_vector
        - pre_vector
    )


    return (

        pre_vector,

        post_vector,

        delta_vector,

        slot,
    )


# ============================================================
# EXTRACT COMPLETE DATASET
# ============================================================

@torch.no_grad()
def extract_dataset(
    examples,
    dataset_name,
):

    pre_vectors = []

    post_vectors = []

    delta_vectors = []

    labels = []

    slots = []


    total = len(
        examples
    )


    for i, example in enumerate(
        examples,
        start=1,
    ):

        (
            pre,
            post,
            delta,
            slot,
        ) = extract_memory_vectors(

            example[
                "text"
            ]
        )


        pre_vectors.append(
            pre
        )

        post_vectors.append(
            post
        )

        delta_vectors.append(
            delta
        )

        labels.append(

            example[
                "label"
            ]
        )

        slots.append(
            slot
        )


        if (
            i % 100
            == 0
            or
            i == total
        ):

            print(

                f"{dataset_name}: "
                f"{i}/{total}"
            )


    return {

        "pre":
            torch.stack(
                pre_vectors,
                dim=0,
            ),

        "post":
            torch.stack(
                post_vectors,
                dim=0,
            ),

        "delta":
            torch.stack(
                delta_vectors,
                dim=0,
            ),

        "labels":
            torch.tensor(
                labels,
                dtype=torch.long,
            ),

        "slots":
            torch.tensor(
                slots,
                dtype=torch.long,
            ),
    }


# ============================================================
# EXTRACT ALL MEMORY VECTORS
# ============================================================

section(
    "EXTRACT TRAIN MEMORY VECTORS"
)

train_data = extract_dataset(

    train_examples,

    "TRAIN",
)


section(
    "EXTRACT VALIDATION MEMORY VECTORS"
)

val_data = extract_dataset(

    val_examples,

    "VALID",
)


section(
    "EXTRACT HELD-OUT TEST MEMORY VECTORS"
)

test_data = extract_dataset(

    test_examples,

    "TEST",
)


# ============================================================
# SLOT DISTRIBUTION CHECK
# ============================================================

section(
    "ACTUAL SLOT DISTRIBUTION"
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

        data[
            "slots"
        ],

        return_counts=True,
    )


    print()

    print(
        name
    )


    for slot, count in zip(
        unique.tolist(),
        counts.tolist(),
    ):

        print(
            f"slot {slot}: "
            f"{count}"
        )


# ============================================================
# BASIC VECTOR DIAGNOSTICS
# ============================================================

section(
    "VECTOR DIAGNOSTICS"
)


def mean_norm(
    tensor
):

    return float(

        tensor.norm(
            dim=-1
        )
        .mean()
        .item()
    )


print(
    "Mean PRE norm:",
    fmt(
        mean_norm(
            test_data[
                "pre"
            ]
        )
    ),
)

print(
    "Mean POST norm:",
    fmt(
        mean_norm(
            test_data[
                "post"
            ]
        )
    ),
)

print(
    "Mean DELTA norm:",
    fmt(
        mean_norm(
            test_data[
                "delta"
            ]
        )
    ),
)


relative_update = (

    test_data[
        "delta"
    ].norm(
        dim=-1
    )

    /

    test_data[
        "pre"
    ].norm(
        dim=-1
    ).clamp_min(
        1e-8
    )
)


print(
    "Mean relative update:",
    fmt(
        relative_update
        .mean()
        .item()
    ),
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
    # Top-3
    # --------------------------------------------------------

    top3 = (

        logits.topk(
            k=3,
            dim=-1,
        )
        .indices
    )


    top3_correct = (

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
    # Ranks / MRR
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
                top3_correct
            ),

        "mrr":
            float(
                mrr
            ),

        "mean_rank":
            float(
                mean_rank
            ),

        "predictions":
            predictions
            .detach()
            .cpu(),

        "ranks":
            ranks
            .detach()
            .cpu(),
    }


# ============================================================
# TRAIN ONE PROBE
# ============================================================

def train_probe(
    representation_name,
    train_features,
    train_labels,
    val_features,
    val_labels,
    test_features,
    test_labels,
):

    section(
        f"TRAIN LINEAR PROBE: "
        f"{representation_name}"
    )


    input_dim = (

        train_features
        .shape[
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


    train_dataset = (
        TensorDataset(

            train_features,

            train_labels,
        )
    )


    train_loader = (
        DataLoader(

            train_dataset,

            batch_size=PROBE_BATCH_SIZE,

            shuffle=True,
        )
    )


    best_val_accuracy = -1.0

    best_state = None

    best_epoch = -1


    for epoch in range(
        1,
        PROBE_EPOCHS + 1,
    ):

        probe.train()


        running_loss = 0.0

        total = 0


        for features, labels in train_loader:

            features = (
                features.to(
                    DEVICE
                )
            )

            labels = (
                labels.to(
                    DEVICE
                )
            )


            logits = probe(
                features
            )


            loss = criterion(

                logits,

                labels,
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            loss.backward()


            optimizer.step()


            running_loss += (

                float(
                    loss.item()
                )

                *
                features.size(
                    0
                )
            )


            total += (
                features.size(
                    0
                )
            )


        val_result = (
            evaluate_probe(

                probe,

                val_features,

                val_labels,
            )
        )


        mean_loss = (

            running_loss
            /
            max(
                total,
                1
            )
        )


        if (
            val_result[
                "accuracy"
            ]
            >
            best_val_accuracy
        ):

            best_val_accuracy = (

                val_result[
                    "accuracy"
                ]
            )

            best_epoch = (
                epoch
            )

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


    # --------------------------------------------------------
    # Restore best validation model.
    # --------------------------------------------------------

    probe.load_state_dict(
        best_state
    )


    probe.to(
        DEVICE
    )


    train_result = (
        evaluate_probe(

            probe,

            train_features,

            train_labels,
        )
    )


    val_result = (
        evaluate_probe(

            probe,

            val_features,

            val_labels,
        )
    )


    test_result = (
        evaluate_probe(

            probe,

            test_features,

            test_labels,
        )
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


    for name, result in [

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

            f"{name:<12}"

            f"{result['accuracy'] * 100:>13.2f}%"

            f"{result['top3'] * 100:>11.2f}%"

            f"{result['mrr']:>12.4f}"

            f"{result['mean_rank']:>14.4f}"
        )


    return {

        "probe":
            probe,

        "train":
            train_result,

        "val":
            val_result,

        "test":
            test_result,

        "best_epoch":
            best_epoch,
    }


# ============================================================
# RUN THREE INDEPENDENT PROBES
# ============================================================

pre_results = train_probe(

    representation_name="PRE-WRITE SLOT",

    train_features=train_data[
        "pre"
    ],

    train_labels=train_data[
        "labels"
    ],

    val_features=val_data[
        "pre"
    ],

    val_labels=val_data[
        "labels"
    ],

    test_features=test_data[
        "pre"
    ],

    test_labels=test_data[
        "labels"
    ],
)


post_results = train_probe(

    representation_name="POST-WRITE SLOT",

    train_features=train_data[
        "post"
    ],

    train_labels=train_data[
        "labels"
    ],

    val_features=val_data[
        "post"
    ],

    val_labels=val_data[
        "labels"
    ],

    test_features=test_data[
        "post"
    ],

    test_labels=test_data[
        "labels"
    ],
)


delta_results = train_probe(

    representation_name="WRITE DELTA",

    train_features=train_data[
        "delta"
    ],

    train_labels=train_data[
        "labels"
    ],

    val_features=val_data[
        "delta"
    ],

    val_labels=val_data[
        "labels"
    ],

    test_features=test_data[
        "delta"
    ],

    test_labels=test_data[
        "labels"
    ],
)


# ============================================================
# FINAL SUMMARY
# ============================================================

section(
    "FINAL MEMORY VALUE PROBE RESULTS"
)


chance = (

    1.0
    /
    len(
        ANSWERS
    )
)


print(
    f"{'Representation':<22}"
    f"{'Test Acc':>14}"
    f"{'Top-3':>12}"
    f"{'MRR':>12}"
    f"{'Mean Rank':>14}"
)


for name, result in [

    (
        "PRE-WRITE",
        pre_results[
            "test"
        ],
    ),

    (
        "POST-WRITE",
        post_results[
            "test"
        ],
    ),

    (
        "WRITE DELTA",
        delta_results[
            "test"
        ],
    ),
]:

    print(

        f"{name:<22}"

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
# AUTOMATIC DIAGNOSIS
# ============================================================

section(
    "AUTOMATIC DIAGNOSIS"
)


pre_acc = (

    pre_results[
        "test"
    ][
        "accuracy"
    ]
)


post_acc = (

    post_results[
        "test"
    ][
        "accuracy"
    ]
)


delta_acc = (

    delta_results[
        "test"
    ][
        "accuracy"
    ]
)


print(
    "PRE-WRITE accuracy:",
    f"{pre_acc * 100:.2f}%"
)

print(
    "POST-WRITE accuracy:",
    f"{post_acc * 100:.2f}%"
)

print(
    "WRITE-DELTA accuracy:",
    f"{delta_acc * 100:.2f}%"
)

print()


# ============================================================
# DIAGNOSIS CASE 1
# ============================================================

if (
    post_acc >= 0.75
    and
    delta_acc >= 0.75
):

    print(
        "RESULT: STORED MEMORY VALUE STRONGLY "
        "ENCODES THE ANSWER."
    )

    print()

    print(
        "The current vector-gated writer is successfully "
        "putting answer information into the slot."
    )

    print()

    print(
        "Therefore the vector gate should NOT be blamed "
        "at this stage."
    )

    print()

    print(
        "The main remaining bottleneck is likely:"
    )

    print()

    print(
        "MEMORY READER / VALUE PROJECTION / FUSION."
    )

    print()

    print(
        "NEXT EXPERIMENT:"
    )

    print(
        "Probe each stage of the read/fusion pathway "
        "to see where the answer information disappears."
    )


# ============================================================
# DIAGNOSIS CASE 2
# ============================================================

elif (
    delta_acc >= 0.75
    and
    post_acc < 0.50
):

    print(
        "RESULT: WRITE UPDATE CONTAINS THE ANSWER, "
        "BUT FINAL SLOT DOES NOT PRESERVE IT."
    )

    print()

    print(
        "This specifically points toward the memory update "
        "operation, gate strength, normalization, or "
        "old/new value mixing."
    )

    print()

    print(
        "NOW a scalar-vs-vector gate / update ablation "
        "would be justified."
    )


# ============================================================
# DIAGNOSIS CASE 3
# ============================================================

elif (
    post_acc <= 0.20
    and
    delta_acc <= 0.20
):

    print(
        "RESULT: MEMORY WRITE PATH DOES NOT "
        "PRESERVE ANSWER IDENTITY WELL."
    )

    print()

    print(
        "The stored slot and the actual write delta are "
        "both close to weak/chance decoding."
    )

    print()

    print(
        "The problem is upstream of the reader."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Compare candidate representation, scalar gate, "
        "vector gate, no gate, and orthogonal update."
    )


# ============================================================
# DIAGNOSIS CASE 4
# ============================================================

else:

    print(
        "RESULT: PARTIAL ANSWER INFORMATION IS PRESENT."
    )

    print()

    print(
        "The memory writer is not completely failing, "
        "but answer information is not cleanly preserved."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Run a writer-stage ablation before changing "
        "the main architecture."
    )


# ============================================================
# PRE-WRITE CONTROL WARNING
# ============================================================

print()

if pre_acc > 0.20:

    print(
        "WARNING:"
    )

    print(
        "PRE-WRITE control accuracy is unexpectedly high."
    )

    print(
        "Check for dataset leakage or slot-index leakage "
        "before trusting the probe result."
    )

else:

    print(
        "PRE-WRITE control behaves reasonably."
    )


# ============================================================
# SAVE ONLY PROBE RESULTS
# ============================================================

section(
    "SAVE EXPERIMENTAL PROBE RESULTS"
)


OUTPUT_PATH = (
    "outputs/"
    "memory_value_linear_probe.pt"
)


torch.save(

    {

        "answers":
            ANSWERS,

        "chance":
            chance,

        "pre_test_accuracy":
            pre_acc,

        "post_test_accuracy":
            post_acc,

        "delta_test_accuracy":
            delta_acc,

        "pre_probe_state":
            pre_results[
                "probe"
            ]
            .state_dict(),

        "post_probe_state":
            post_results[
                "probe"
            ]
            .state_dict(),

        "delta_probe_state":
            delta_results[
                "probe"
            ]
            .state_dict(),
    },

    OUTPUT_PATH,
)


print(
    "Saved experimental probe results to:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# COMPLETE
# ============================================================

section(
    "EXPERIMENT COMPLETE"
)

print(
    "Main GPT-2 model was frozen."
)

print(
    "Memory writer was frozen."
)

print(
    "Vector gate was frozen."
)

print(
    "Router was frozen."
)

print(
    "Reader was frozen."
)

print(
    "Only three tiny linear probes were trained."
)

print(
    "No source files were modified."
)

print(
    "No original checkpoints were overwritten."
)