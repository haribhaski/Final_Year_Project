# ============================================================
# STANDALONE KEY-VALUE MEMORY TEST
#
# KEY:
#   Layer-1 entity representation
#   -> trained contrastive address encoder
#
# VALUE:
#   normalized final GPT-2 summary of the FACT
#
#
# PURPOSE
# ============================================================
#
# Previous experiments showed:
#
#   Layer-1 address -> actual slot      ~100%
#
#   Direct GPT2 summary -> answer probe ~85%
#
# Current CandidateWriter destroys much of that value signal.
#
# Now test the clean abstraction:
#
#       KEY                     VALUE
#       ---                     -----
# Layer1 entity key      final semantic summary
#
#
# WRITE:
#
# Fact A -> key A + semantic value A
# Fact B -> key B + semantic value B
#
#
# READ:
#
# Query A
#    ↓
# address key
#    ↓
# cosine against stored keys
#    ↓
# retrieve corresponding semantic value
#    ↓
# frozen / trained linear value decoder
#    ↓
# answer class
#
#
# TEST:
#
#   2 facts
#   4 facts
#   8 facts
#
#
# IMPORTANT:
#
# - GPT-2 frozen
# - address encoder frozen
# - existing memory architecture untouched
# - only the small VALUE DECODER is trained
#
# This is NOT main-model integration.
#
#
# RUN:
#
# python standalone_key_value_memory_test.py \
#   2>&1 | tee standalone_key_value_memory_test.log
#
# ============================================================


import random
import string
from collections import OrderedDict

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

MODEL_CHECKPOINT = (
    "outputs/retrieval_gradient_test/"
    "checkpoint_best.pt"
)

ADDRESS_CHECKPOINT = (
    "outputs/contrastive_address_encoder.pt"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SEED = 2040

ADDRESS_LAYER = 1

ADDRESS_HIDDEN_DIM = 512
ADDRESS_DIM = 256

NUM_SLOTS = 8

FACT_COUNTS = [
    2,
    4,
    8,
]

EPISODES_PER_SIZE = 200


# ------------------------------------------------------------
# VALUE DECODER DATA
# ------------------------------------------------------------

VALUE_TRAIN_PER_CLASS = 150

VALUE_VAL_PER_CLASS = 30

VALUE_TEST_PER_CLASS = 50


# ------------------------------------------------------------
# VALUE DECODER TRAINING
# ------------------------------------------------------------

VALUE_DECODER_EPOCHS = 50

VALUE_DECODER_BATCH_SIZE = 128

VALUE_DECODER_LR = 1e-2

VALUE_DECODER_WEIGHT_DECAY = 1e-4


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

    answer:
        idx

    for idx, answer
    in enumerate(
        ANSWERS
    )
}


# ============================================================
# WRITING TEMPLATES
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
# QUERY TEMPLATES
# ============================================================

QUERY_TEMPLATES = [

    "What is the keyword for {entity}?",

    "Recall the keyword associated with {entity}.",

    "Which keyword was assigned to {entity}?",

    "Retrieve the value belonging to {entity}.",

    "What value belongs to {entity}?",

    "Give the stored keyword linked with {entity}.",
]


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# PRINT HELPERS
# ============================================================

def section(title):

    print()
    print("=" * 125)
    print(title)
    print("=" * 125)


def fmt(x):

    return f"{float(x):.6f}"


# ============================================================
# MEMORY MODEL CONFIG
#
# The memory architecture itself is NOT used for storage here.
#
# We only load this checkpoint because:
#
#   - it contains the trained GPT-2 backbone
#   - the address encoder was trained using it
#
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
# ADDRESS ENCODER
# ============================================================

class AddressEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout=0.1,
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.LayerNorm(
                input_dim
            ),

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                output_dim,
                bias=False,
            ),
        )


    def forward(
        self,
        x,
    ):

        x = self.net(
            x
        )

        return F.normalize(

            x,

            p=2,

            dim=-1,

            eps=1e-8,
        )


# ============================================================
# VALUE DECODER
# ============================================================

class ValueDecoder(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes,
    ):

        super().__init__()

        self.linear = nn.Linear(

            input_dim,

            num_classes,
        )


    def forward(
        self,
        x,
    ):

        return self.linear(
            x
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
# LOAD GPT-2 MEMORY CHECKPOINT
# ============================================================

section(
    "LOAD FROZEN GPT-2"
)


model = (
    MemoryAugmentedGPT2LMHeadModel
    .from_pretrained(

        MODEL_NAME,

        memory_config=build_config(),
    )
)


checkpoint = torch.load(

    MODEL_CHECKPOINT,

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


current_state = (
    model.state_dict()
)


compatible = {}


for name, value in state.items():

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


load_result = (
    model.load_state_dict(

        compatible,

        strict=False,
    )
)


print(
    "Compatible tensors:",
    len(compatible),
)

print(
    "Missing tensors:",
    len(load_result.missing_keys),
)


model.to(
    DEVICE
)

model.eval()


for parameter in model.parameters():

    parameter.requires_grad = False


print(
    "GPT-2 / memory checkpoint frozen."
)


# ============================================================
# LOAD CONTRASTIVE ADDRESS ENCODER
# ============================================================

section(
    "LOAD ADDRESS ENCODER"
)


address_checkpoint = torch.load(

    ADDRESS_CHECKPOINT,

    map_location="cpu",

    weights_only=False,
)


address_input_dim = (
    address_checkpoint.get(
        "input_dim",
        model.backbone.config.n_embd,
    )
)


address_hidden_dim = (
    address_checkpoint.get(
        "hidden_dim",
        ADDRESS_HIDDEN_DIM,
    )
)


address_dim = (
    address_checkpoint.get(
        "address_dim",
        ADDRESS_DIM,
    )
)


address_layer = (
    address_checkpoint.get(
        "address_layer",
        ADDRESS_LAYER,
    )
)


address_encoder = AddressEncoder(

    input_dim=address_input_dim,

    hidden_dim=address_hidden_dim,

    output_dim=address_dim,

    dropout=0.1,
)


address_encoder.load_state_dict(

    address_checkpoint[
        "encoder_state_dict"
    ]
)


address_encoder.to(
    DEVICE
)

address_encoder.eval()


for parameter in address_encoder.parameters():

    parameter.requires_grad = False


print(
    "Address layer:",
    address_layer,
)

print(
    "Address dimension:",
    address_dim,
)

print(
    "Previous address Recall@1:",
    address_checkpoint.get(
        "test_recall1",
        "unknown",
    ),
)

print(
    "Address encoder frozen."
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
# ENTITY MASK
# ============================================================

def build_entity_mask(
    text,
    entity,
    offsets,
):

    start = text.find(
        entity
    )


    if start == -1:

        raise RuntimeError(
            f"Entity {entity!r} not found in text."
        )


    end = (
        start
        +
        len(entity)
    )


    mask = torch.zeros(

        offsets.shape[0],

        dtype=torch.bool,
    )


    for token_idx in range(
        offsets.shape[0]
    ):

        token_start = int(

            offsets[
                token_idx,
                0
            ].item()
        )


        token_end = int(

            offsets[
                token_idx,
                1
            ].item()
        )


        if token_end <= token_start:

            continue


        if (
            token_start < end
            and
            token_end > start
        ):

            mask[
                token_idx
            ] = True


    if not mask.any():

        raise RuntimeError(
            f"No tokens found for entity {entity!r}"
        )


    return mask


# ============================================================
# GET ADDRESS KEY
# ============================================================

@torch.no_grad()
def get_address_key(
    text,
    entity,
):

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


    entity_mask = build_entity_mask(

        text,

        entity,

        offsets,
    ).to(
        DEVICE
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


    transformer_output = (
        model.backbone.transformer(

            input_ids=input_ids,

            attention_mask=attention_mask,

            output_hidden_states=True,

            use_cache=False,

            return_dict=True,
        )
    )


    hidden = (

        transformer_output
        .hidden_states[
            address_layer
        ]
    )


    weights = (
        entity_mask
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(
            hidden.dtype
        )
    )


    entity_repr = (

        (
            hidden
            *
            weights
        )
        .sum(
            dim=1
        )

        /

        weights
        .sum(
            dim=1
        )
        .clamp_min(
            1.0
        )
    )


    key = (
        address_encoder(
            entity_repr
        )
    )


    return (
        key[
            0
        ]
        .detach()
    )


# ============================================================
# GET DIRECT SEMANTIC VALUE
#
# This is the important new VALUE.
#
# We use the exact final GPT-2 pooled summary and simply
# normalize it.
# ============================================================

@torch.no_grad()
def get_semantic_value(
    text,
):

    encoded = tokenizer(

        text,

        return_tensors="pt",
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


    transformer_output = (
        model.backbone.transformer(

            input_ids=input_ids,

            attention_mask=attention_mask,

            use_cache=False,

            return_dict=True,
        )
    )


    hidden = (
        transformer_output
        .last_hidden_state
    )


    summary = (
        model._pool_hidden(

            hidden_states=hidden,

            attention_mask=attention_mask,
        )
    )


    # --------------------------------------------------------
    # Same idea that performed strongly in our ablation:
    #
    # DIRECT SUMMARY / normalized summary
    # --------------------------------------------------------

    value = F.layer_norm(

        summary,

        normalized_shape=(
            summary.shape[-1],
        ),
    )


    return (
        value[
            0
        ]
        .detach()
    )


# ============================================================
# BUILD VALUE DECODER DATA
# ============================================================

def build_value_examples(
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

                    "entity":
                        entity,

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
# EXTRACT VALUE FEATURES
# ============================================================

@torch.no_grad()
def extract_value_dataset(
    examples,
    name,
):

    values = []

    labels = []


    for idx, example in enumerate(
        examples,
        start=1,
    ):

        value = get_semantic_value(

            example[
                "text"
            ]
        )


        values.append(
            value.cpu()
        )


        labels.append(
            example[
                "label"
            ]
        )


        if (
            idx % 100 == 0
            or
            idx == len(examples)
        ):

            print(

                f"{name}: "
                f"{idx}/{len(examples)}"
            )


    return {

        "values":
            torch.stack(
                values,
                dim=0,
            ),

        "labels":
            torch.tensor(
                labels,
                dtype=torch.long,
            ),
    }


# ============================================================
# BUILD VALUE DECODER DATASETS
# ============================================================

section(
    "BUILD VALUE DECODER DATA"
)


value_train_examples = build_value_examples(

    prefix="KVTrain",

    per_class=VALUE_TRAIN_PER_CLASS,

    templates=TRAIN_WRITE_TEMPLATES,

    seed=SEED + 1,
)


value_val_examples = build_value_examples(

    prefix="KVValid",

    per_class=VALUE_VAL_PER_CLASS,

    templates=TRAIN_WRITE_TEMPLATES,

    seed=SEED + 2,
)


value_test_examples = build_value_examples(

    prefix="KVTest",

    per_class=VALUE_TEST_PER_CLASS,

    templates=HELDOUT_WRITE_TEMPLATES,

    seed=SEED + 3,
)


print(
    "Train:",
    len(value_train_examples),
)

print(
    "Validation:",
    len(value_val_examples),
)

print(
    "Held-out test:",
    len(value_test_examples),
)


# ============================================================
# EXTRACT VALUE FEATURES
# ============================================================

section(
    "EXTRACT VALUE TRAIN FEATURES"
)


value_train = extract_value_dataset(

    value_train_examples,

    "VALUE TRAIN",
)


section(
    "EXTRACT VALUE VALID FEATURES"
)


value_val = extract_value_dataset(

    value_val_examples,

    "VALUE VALID",
)


section(
    "EXTRACT VALUE TEST FEATURES"
)


value_test = extract_value_dataset(

    value_test_examples,

    "VALUE TEST",
)


# ============================================================
# STANDARDIZATION
# ============================================================

value_mean = (
    value_train[
        "values"
    ]
    .mean(
        dim=0,
        keepdim=True,
    )
)


value_std = (
    value_train[
        "values"
    ]
    .std(
        dim=0,
        keepdim=True,
    )
    .clamp_min(
        1e-5
    )
)


def standardize_value(
    x,
):

    return (

        x
        -
        value_mean.to(
            x.device
        )

    ) / value_std.to(
        x.device
    )


# ============================================================
# VALUE DECODER
# ============================================================

section(
    "TRAIN VALUE DECODER"
)


decoder = ValueDecoder(

    input_dim=
        value_train[
            "values"
        ].shape[
            -1
        ],

    num_classes=
        len(
            ANSWERS
        ),
).to(
    DEVICE
)


optimizer = torch.optim.AdamW(

    decoder.parameters(),

    lr=VALUE_DECODER_LR,

    weight_decay=
        VALUE_DECODER_WEIGHT_DECAY,
)


loss_fn = (
    nn.CrossEntropyLoss()
)


train_dataset = TensorDataset(

    value_train[
        "values"
    ],

    value_train[
        "labels"
    ],
)


train_loader = DataLoader(

    train_dataset,

    batch_size=
        VALUE_DECODER_BATCH_SIZE,

    shuffle=True,
)


@torch.no_grad()
def evaluate_decoder(
    values,
    labels,
):

    decoder.eval()


    values = (
        standardize_value(
            values.to(
                DEVICE
            )
        )
    )


    labels = labels.to(
        DEVICE
    )


    logits = decoder(
        values
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


    correct_logits = (
        logits
        .gather(
            1,
            labels.unsqueeze(
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


    return {

        "accuracy":
            accuracy,

        "mrr":
            mrr,

        "mean_rank":
            ranks
            .float()
            .mean()
            .item(),
    }


best_state = None

best_val_accuracy = -1.0

best_epoch = None


for epoch in range(
    1,
    VALUE_DECODER_EPOCHS + 1,
):

    decoder.train()


    total_loss = 0.0

    count = 0


    for batch_x, batch_y in (
        train_loader
    ):

        batch_x = (
            standardize_value(
                batch_x.to(
                    DEVICE
                )
            )
        )


        batch_y = batch_y.to(
            DEVICE
        )


        logits = decoder(
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


    val_result = evaluate_decoder(

        value_val[
            "values"
        ],

        value_val[
            "labels"
        ],
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


        best_epoch = epoch


        best_state = {

            k:
                v.detach()
                .cpu()
                .clone()

            for k, v
            in decoder
            .state_dict()
            .items()
        }


    if (
        epoch == 1
        or
        epoch % 10 == 0
        or
        epoch
        ==
        VALUE_DECODER_EPOCHS
    ):

        print(

            f"Epoch "
            f"{epoch:02d}/"
            f"{VALUE_DECODER_EPOCHS}"

            f" | loss="
            f"{total_loss / max(count,1):.4f}"

            f" | val acc="
            f"{val_result['accuracy'] * 100:.2f}%"

            f" | val MRR="
            f"{val_result['mrr']:.4f}"
        )


decoder.load_state_dict(
    best_state
)


decoder.to(
    DEVICE
)


train_result = evaluate_decoder(

    value_train[
        "values"
    ],

    value_train[
        "labels"
    ],
)


val_result = evaluate_decoder(

    value_val[
        "values"
    ],

    value_val[
        "labels"
    ],
)


test_result = evaluate_decoder(

    value_test[
        "values"
    ],

    value_test[
        "labels"
    ],
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

        f"{result['mrr']:>12.4f}"

        f"{result['mean_rank']:>14.4f}"
    )


# ============================================================
# FREEZE VALUE DECODER
# ============================================================

decoder.eval()


for parameter in decoder.parameters():

    parameter.requires_grad = False


# ============================================================
# PREDICT ANSWER FROM VALUE
# ============================================================

@torch.no_grad()
def decode_value(
    value,
):

    x = (
        standardize_value(
            value
            .unsqueeze(0)
            .to(
                DEVICE
            )
        )
    )


    logits = decoder(
        x
    )[0]


    predicted_id = int(

        torch.argmax(
            logits
        ).item()
    )


    return (
        ANSWERS[
            predicted_id
        ],
        logits,
    )


# ============================================================
# ONE KEY-VALUE MEMORY EPISODE
# ============================================================

@torch.no_grad()
def run_episode(
    num_facts,
    rng,
    verbose=False,
):

    stored_keys = []

    stored_values = []

    facts = []

    used_entities = set()

    used_answers = set()


    # ========================================================
    # WRITE
    # ========================================================

    for idx in range(
        num_facts
    ):

        while True:

            entity = random_entity(

                rng,

                prefix="Episode",
            )


            if entity not in used_entities:

                used_entities.add(
                    entity
                )

                break


        available_answers = [

            answer

            for answer in ANSWERS

            if answer not in used_answers
        ]


        answer = rng.choice(
            available_answers
        )


        used_answers.add(
            answer
        )


        write_template = rng.choice(
            HELDOUT_WRITE_TEMPLATES
        )


        query_template = rng.choice(
            QUERY_TEMPLATES
        )


        write_text = write_template.format(

            entity=entity,

            answer=answer,
        )


        query_text = query_template.format(

            entity=entity
        )


        # ----------------------------------------------------
        # KEY
        # ----------------------------------------------------

        key = get_address_key(

            write_text,

            entity,
        )


        # ----------------------------------------------------
        # VALUE
        # ----------------------------------------------------

        value = get_semantic_value(
            write_text
        )


        stored_keys.append(
            key
        )


        stored_values.append(
            value
        )


        facts.append(

            {

                "entity":
                    entity,

                "answer":
                    answer,

                "query":
                    query_text,

                "index":
                    idx,
            }
        )


    keys = torch.stack(

        stored_keys,

        dim=0,
    )


    keys = F.normalize(

        keys,

        p=2,

        dim=-1,

        eps=1e-8,
    )


    values = torch.stack(

        stored_values,

        dim=0,
    )


    # ========================================================
    # READ
    # ========================================================

    address_correct = 0

    answer_correct = 0

    oracle_answer_correct = 0

    wrong_answer_correct = 0

    address_rr_sum = 0.0

    records = []


    for fact in facts:

        query_key = get_address_key(

            fact[
                "query"
            ],

            fact[
                "entity"
            ],
        )


        query_key = F.normalize(

            query_key,

            p=2,

            dim=-1,

            eps=1e-8,
        )


        similarities = (

            keys
            @
            query_key
        )


        ordering = torch.argsort(

            similarities,

            descending=True,
        )


        predicted_index = int(

            ordering[
                0
            ].item()
        )


        correct_index = (
            fact[
                "index"
            ]
        )


        correct_score = (

            similarities[
                correct_index
            ]
        )


        rank = int(

            (
                similarities
                >
                correct_score
            )
            .sum()
            .item()
            + 1
        )


        address_rr_sum += (
            1.0
            /
            rank
        )


        if (
            predicted_index
            ==
            correct_index
        ):

            address_correct += 1


        # ====================================================
        # ADDRESSED VALUE
        # ====================================================

        addressed_value = (

            values[
                predicted_index
            ]
        )


        addressed_answer, _ = (
            decode_value(
                addressed_value
            )
        )


        if (
            addressed_answer
            ==
            fact[
                "answer"
            ]
        ):

            answer_correct += 1


        # ====================================================
        # ORACLE VALUE
        # ====================================================

        oracle_value = (

            values[
                correct_index
            ]
        )


        oracle_answer, _ = (
            decode_value(
                oracle_value
            )
        )


        if (
            oracle_answer
            ==
            fact[
                "answer"
            ]
        ):

            oracle_answer_correct += 1


        # ====================================================
        # WRONG VALUE
        # ====================================================

        wrong_indices = [

            i

            for i
            in range(
                num_facts
            )

            if i
            != correct_index
        ]


        wrong_index = (
            wrong_indices[
                0
            ]
        )


        wrong_value = (

            values[
                wrong_index
            ]
        )


        wrong_answer, _ = (
            decode_value(
                wrong_value
            )
        )


        if (
            wrong_answer
            ==
            fact[
                "answer"
            ]
        ):

            wrong_answer_correct += 1


        records.append(

            {

                "entity":
                    fact[
                        "entity"
                    ],

                "true_answer":
                    fact[
                        "answer"
                    ],

                "correct_index":
                    correct_index,

                "predicted_index":
                    predicted_index,

                "address_rank":
                    rank,

                "addressed_answer":
                    addressed_answer,

                "oracle_answer":
                    oracle_answer,

                "wrong_answer":
                    wrong_answer,
            }
        )


    total = (
        len(
            facts
        )
    )


    result = {

        "address_accuracy":
            address_correct
            /
            total,

        "address_mrr":
            address_rr_sum
            /
            total,

        "addressed_answer_accuracy":
            answer_correct
            /
            total,

        "oracle_answer_accuracy":
            oracle_answer_correct
            /
            total,

        "wrong_answer_accuracy":
            wrong_answer_correct
            /
            total,

        "records":
            records,
    }


    if verbose:

        print()

        print(
            "STORED FACTS"
        )


        for fact in facts:

            print(

                f"slot {fact['index']} "
                f"| {fact['entity']} "
                f"-> {fact['answer']}"
            )


        print()

        print(
            "QUERY RESULTS"
        )


        for record in records:

            print()

            print(
                "Entity:",
                record[
                    "entity"
                ]
            )

            print(
                "True:",
                record[
                    "true_answer"
                ]
            )

            print(
                "Correct slot:",
                record[
                    "correct_index"
                ]
            )

            print(
                "Addressed slot:",
                record[
                    "predicted_index"
                ]
            )

            print(
                "Address rank:",
                record[
                    "address_rank"
                ]
            )

            print(
                "Addressed value prediction:",
                record[
                    "addressed_answer"
                ]
            )

            print(
                "Oracle value prediction:",
                record[
                    "oracle_answer"
                ]
            )

            print(
                "Wrong value prediction:",
                record[
                    "wrong_answer"
                ]
            )


    return result


# ============================================================
# RUN 2 / 4 / 8 FACT TEST
# ============================================================

section(
    "STANDALONE KEY-VALUE MEMORY TEST"
)


print(
    "Fact counts:",
    FACT_COUNTS,
)

print(
    "Episodes per size:",
    EPISODES_PER_SIZE,
)

print(
    "Address key dimension:",
    address_dim,
)

print(
    "Value dimension:",
    model.backbone.config.n_embd,
)


all_results = {}


for num_facts in FACT_COUNTS:

    section(
        f"{num_facts}-FACT EPISODES"
    )


    rng = random.Random(

        SEED
        +
        num_facts
        *
        1000
    )


    address_correct = 0.0

    addressed_correct = 0.0

    oracle_correct = 0.0

    wrong_correct = 0.0

    mrr_sum = 0.0


    print(
        "Example episode:"
    )


    for episode_idx in range(
        EPISODES_PER_SIZE
    ):

        result = run_episode(

            num_facts=num_facts,

            rng=rng,

            verbose=(
                episode_idx == 0
            ),
        )


        address_correct += (
            result[
                "address_accuracy"
            ]
        )


        addressed_correct += (
            result[
                "addressed_answer_accuracy"
            ]
        )


        oracle_correct += (
            result[
                "oracle_answer_accuracy"
            ]
        )


        wrong_correct += (
            result[
                "wrong_answer_accuracy"
            ]
        )


        mrr_sum += (
            result[
                "address_mrr"
            ]
        )


        if (
            episode_idx + 1
        ) % 50 == 0:

            print(

                f"Completed "
                f"{episode_idx + 1}/"
                f"{EPISODES_PER_SIZE}"
            )


    mean_address_accuracy = (

        address_correct
        /
        EPISODES_PER_SIZE
    )


    mean_addressed_accuracy = (

        addressed_correct
        /
        EPISODES_PER_SIZE
    )


    mean_oracle_accuracy = (

        oracle_correct
        /
        EPISODES_PER_SIZE
    )


    mean_wrong_accuracy = (

        wrong_correct
        /
        EPISODES_PER_SIZE
    )


    mean_mrr = (

        mrr_sum
        /
        EPISODES_PER_SIZE
    )


    print()

    print(
        f"{num_facts}-FACT RESULTS"
    )


    print(
        "Address accuracy:",
        f"{mean_address_accuracy * 100:.2f}%"
    )


    print(
        "Address MRR:",
        fmt(
            mean_mrr
        )
    )


    print(
        "Addressed value answer accuracy:",
        f"{mean_addressed_accuracy * 100:.2f}%"
    )


    print(
        "Oracle value answer accuracy:",
        f"{mean_oracle_accuracy * 100:.2f}%"
    )


    print(
        "Wrong value answer accuracy:",
        f"{mean_wrong_accuracy * 100:.2f}%"
    )


    print(
        "Answer chance:",
        f"{100 / len(ANSWERS):.2f}%"
    )


    all_results[
        num_facts
    ] = {

        "address":
            mean_address_accuracy,

        "mrr":
            mean_mrr,

        "addressed":
            mean_addressed_accuracy,

        "oracle":
            mean_oracle_accuracy,

        "wrong":
            mean_wrong_accuracy,
    }


# ============================================================
# FINAL TABLE
# ============================================================

section(
    "FINAL KEY-VALUE MEMORY RESULTS"
)


print(

    f"{'Facts':<10}"

    f"{'Address':>12}"

    f"{'Addr MRR':>12}"

    f"{'Addressed Ans':>16}"

    f"{'Oracle Ans':>14}"

    f"{'Wrong Ans':>12}"

    f"{'Chance':>10}"
)


for num_facts in FACT_COUNTS:

    result = (
        all_results[
            num_facts
        ]
    )


    print(

        f"{num_facts:<10}"

        f"{result['address'] * 100:>11.2f}%"

        f"{result['mrr']:>12.4f}"

        f"{result['addressed'] * 100:>15.2f}%"

        f"{result['oracle'] * 100:>13.2f}%"

        f"{result['wrong'] * 100:>11.2f}%"

        f"{100 / len(ANSWERS):>9.2f}%"
    )


# ============================================================
# AUTOMATIC DIAGNOSIS
# ============================================================

section(
    "AUTOMATIC DIAGNOSIS"
)


mean_address = np.mean(

    [
        all_results[
            n
        ][
            "address"
        ]

        for n in FACT_COUNTS
    ]
)


mean_addressed = np.mean(

    [
        all_results[
            n
        ][
            "addressed"
        ]

        for n in FACT_COUNTS
    ]
)


mean_oracle = np.mean(

    [
        all_results[
            n
        ][
            "oracle"
        ]

        for n in FACT_COUNTS
    ]
)


mean_wrong = np.mean(

    [
        all_results[
            n
        ][
            "wrong"
        ]

        for n in FACT_COUNTS
    ]
)


print(
    "Mean address accuracy:",
    f"{mean_address * 100:.2f}%"
)

print(
    "Mean addressed answer accuracy:",
    f"{mean_addressed * 100:.2f}%"
)

print(
    "Mean oracle value accuracy:",
    f"{mean_oracle * 100:.2f}%"
)

print(
    "Mean wrong-value accuracy:",
    f"{mean_wrong * 100:.2f}%"
)


print()


if (
    mean_address >= 0.95
    and
    mean_oracle >= 0.75
    and
    mean_addressed >= 0.70
    and
    mean_wrong <= 0.20
):

    print(
        "RESULT: STRONG KEY-VALUE MEMORY SUCCESS."
    )

    print()

    print(
        "Layer-1 keys identify the correct memory item, "
        "and direct GPT-2 semantic summaries preserve "
        "the answer value well enough to decode."
    )

    print()

    print(
        "This supports the architecture:"
    )

    print()

    print(
        "KEY   = early-layer identity representation"
    )

    print(
        "VALUE = late-layer semantic summary"
    )

    print()

    print(
        "NEXT STEP:"
    )

    print(
        "Integrate this key/value separation into the "
        "actual memory architecture."
    )


elif (
    mean_address >= 0.95
    and
    mean_oracle >= 0.60
):

    print(
        "RESULT: KEY-VALUE DESIGN IS PROMISING."
    )

    print()

    print(
        "Addressing is strong and semantic values "
        "contain useful answer information."
    )

    print()

    print(
        "The value representation or decoder may "
        "still need refinement before integration."
    )


elif (
    mean_address >= 0.95
    and
    mean_oracle < 0.30
):

    print(
        "RESULT: ADDRESSING WORKS BUT DIRECT SUMMARY "
        "IS NOT ROBUST ENOUGH AS THE VALUE."
    )

    print()

    print(
        "Need a better semantic value representation."
    )


else:

    print(
        "RESULT: STANDALONE KEY-VALUE TEST DID NOT "
        "FULLY TRANSFER."
    )

    print()

    print(
        "Inspect address generalization and value "
        "generalization separately before integration."
    )


# ============================================================
# SAVE RESULTS
# ============================================================

section(
    "SAVE RESULTS"
)


OUTPUT_PATH = (
    "outputs/"
    "standalone_key_value_memory_test.pt"
)


torch.save(

    {

        "answers":
            ANSWERS,

        "value_decoder_state":
            decoder
            .state_dict(),

        "value_mean":
            value_mean,

        "value_std":
            value_std,

        "value_decoder_test_accuracy":
            test_result[
                "accuracy"
            ],

        "episode_results":
            all_results,
    },

    OUTPUT_PATH,
)


print(
    "Saved:",
    OUTPUT_PATH
)


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
    "Address encoder frozen."
)

print(
    "Original CandidateWriter not used for value storage."
)

print(
    "Vector/scalar gate not used."
)

print(
    "Only standalone value decoder was trained."
)

print(
    "No models/ files were modified."
)

print(
    "No original checkpoints were overwritten."
)