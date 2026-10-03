# ============================================================
# ACTUAL MEMORY SLOT ADDRESS RETRIEVAL TEST
#
# GOAL:
#
#   Test whether the learned Layer-1 contrastive address key
#   can identify the ACTUAL memory slot selected by the
#   occupancy router.
#
# IMPORTANT:
#
#   - Main architecture is NOT modified
#   - No model parameters are trained
#   - GPT-2 is frozen
#   - Memory model is frozen
#   - Previously trained contrastive address encoder is loaded
#
# TEST:
#
#   2 facts
#   4 facts
#   8 facts
#
# For each fact:
#
#   fact
#     ↓
#   actual occupancy router
#     ↓
#   selected slot
#
#   same fact
#     ↓
#   GPT-2 Layer 1 entity representation
#     ↓
#   trained address encoder
#     ↓
#   key stored beside selected slot
#
# At query:
#
#   query
#     ↓
#   Layer 1 entity representation
#     ↓
#   address encoder
#     ↓
#   compare against stored slot keys
#     ↓
#   predicted slot
#
# Metric:
#
#   SLOT RETRIEVAL ACCURACY
#
# If this works, then Layer-1-based keys can solve the
# slot-identification problem independently of value retrieval.
#
#
# RUN:
#
# python actual_slot_address_retrieval_test.py \
#   2>&1 | tee actual_slot_address_retrieval_test.log
#
# ============================================================


import os
import random
import string
from collections import defaultdict

import torch
import torch.nn as nn
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

MODEL_CHECKPOINT = (
    "outputs/retrieval_gradient_test/"
    "checkpoint_best.pt"
)

ADDRESS_ENCODER_CHECKPOINT = (
    "outputs/contrastive_address_encoder.pt"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SEED = 123

ADDRESS_LAYER = 1

# ------------------------------------------------------------
# TEST EPISODES
# ------------------------------------------------------------

FACT_COUNTS = [
    2,
    4,
    8,
]

EPISODES_PER_SIZE = 200

# ------------------------------------------------------------
# ADDRESS ENCODER CONFIG
#
# These values should match the earlier experiment.
# ------------------------------------------------------------

ADDRESS_HIDDEN_DIM = 512
ADDRESS_DIM = 256
DROPOUT = 0.1


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
# WRITE / QUERY TEMPLATES
# ============================================================

WRITE_TEMPLATES = [

    "The assigned keyword for {entity} is {answer}.",

    "{entity} has been assigned the keyword {answer}.",

    "Remember that {answer} is associated with {entity}.",

    "Store the mapping {entity} to {answer}.",

    "The value assigned to {entity} is {answer}.",

    "For {entity}, the stored keyword is {answer}.",

    "{answer} is the keyword linked with {entity}.",

    "Please remember this association: {entity} means {answer}.",

    "Record {answer} as the value belonging to {entity}.",

    "In memory, associate {entity} with {answer}.",
]


QUERY_TEMPLATES = [

    "What is the keyword for {entity}?",

    "Recall the keyword associated with {entity}:",

    "The stored value for {entity} is",

    "What value belongs to {entity}?",

    "Retrieve the mapping for {entity}:",

    "Which keyword was assigned to {entity}?",

    "For {entity}, what was the remembered keyword?",

    "Give the stored keyword linked with {entity}:",

    "What information was associated with {entity}?",

    "Retrieve the value belonging to {entity}:",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# PRINT HELPERS
# ============================================================

def section(title):

    print()

    print("=" * 120)

    print(title)

    print("=" * 120)


def fmt(x):

    return f"{float(x):.6f}"


# ============================================================
# MEMORY MODEL CONFIG
# ============================================================

def build_memory_config():

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
# ADDRESS ENCODER
# ============================================================

class AddressEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout,
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

    def forward(self, x):

        x = self.net(x)

        x = F.normalize(
            x,
            p=2,
            dim=-1,
            eps=1e-8,
        )

        return x


# ============================================================
# TOKENIZER
# ============================================================

section("TOKENIZER")

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
# LOAD MEMORY MODEL
# ============================================================

section("LOAD FROZEN MEMORY MODEL")

print(
    "Device:",
    DEVICE,
)

print(
    "Model checkpoint:",
    MODEL_CHECKPOINT,
)

model = (
    MemoryAugmentedGPT2LMHeadModel
    .from_pretrained(
        MODEL_NAME,
        memory_config=build_memory_config(),
    )
)

checkpoint = torch.load(
    MODEL_CHECKPOINT,
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
        and current_state[name].shape
        == value.shape
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
    "Memory model frozen."
)


# ============================================================
# LOAD ADDRESS ENCODER
# ============================================================

section("LOAD CONTRASTIVE ADDRESS ENCODER")

address_checkpoint = torch.load(
    ADDRESS_ENCODER_CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

input_dim = (
    address_checkpoint.get(
        "input_dim",
        model.backbone.config.n_embd,
    )
)

hidden_dim = (
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

    input_dim=input_dim,

    hidden_dim=hidden_dim,

    output_dim=address_dim,

    dropout=DROPOUT,
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
    "Previous test Recall@1:",
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
#
# Use completely new IDs not used in training.
# ============================================================

def random_entity(
    rng,
):

    letters = "".join(
        rng.choice(
            string.ascii_uppercase
        )
        for _ in range(5)
    )

    digits = "".join(
        rng.choice(
            string.digits
        )
        for _ in range(5)
    )

    return (
        f"MemoryEntity-{letters}-{digits}"
    )


# ============================================================
# ENTITY MASK
# ============================================================

def build_entity_mask(
    texts,
    entities,
    offset_mapping,
    attention_mask,
):

    batch_size = len(
        texts
    )

    sequence_length = (
        offset_mapping.size(1)
    )

    mask = torch.zeros(
        batch_size,
        sequence_length,
        dtype=torch.bool,
    )

    for batch_idx in range(
        batch_size
    ):

        text = texts[
            batch_idx
        ]

        entity = entities[
            batch_idx
        ]

        start = text.find(
            entity
        )

        if start == -1:

            raise RuntimeError(
                f"Entity {entity!r} "
                f"not found in text:\n{text}"
            )

        end = (
            start
            + len(entity)
        )

        found = 0

        for token_idx in range(
            sequence_length
        ):

            if not bool(
                attention_mask[
                    batch_idx,
                    token_idx,
                ]
            ):

                continue

            token_start = int(
                offset_mapping[
                    batch_idx,
                    token_idx,
                    0,
                ].item()
            )

            token_end = int(
                offset_mapping[
                    batch_idx,
                    token_idx,
                    1,
                ].item()
            )

            if token_end <= token_start:

                continue

            if (
                token_start < end
                and token_end > start
            ):

                mask[
                    batch_idx,
                    token_idx,
                ] = True

                found += 1

        if found == 0:

            raise RuntimeError(
                f"No entity tokens found "
                f"for {entity}"
            )

    return mask


# ============================================================
# ENTITY-SPAN AVERAGE
# ============================================================

def entity_average(
    hidden,
    mask,
):

    weights = (
        mask
        .unsqueeze(-1)
        .to(hidden.dtype)
    )

    numerator = (
        hidden
        * weights
    ).sum(
        dim=1
    )

    denominator = (
        weights
        .sum(dim=1)
        .clamp_min(1.0)
    )

    return (
        numerator
        / denominator
    )


# ============================================================
# EXTRACT ADDRESS VECTOR
# ============================================================

@torch.no_grad()
def get_address_key(
    text,
    entity,
):

    encoded = tokenizer(

        [text],

        padding=True,

        truncation=True,

        return_tensors="pt",

        return_offsets_mapping=True,
    )

    offsets = encoded.pop(
        "offset_mapping"
    )

    attention_mask_cpu = (
        encoded[
            "attention_mask"
        ].clone()
    )

    entity_mask = (
        build_entity_mask(

            texts=[text],

            entities=[entity],

            offset_mapping=offsets,

            attention_mask=attention_mask_cpu,
        )
        .to(
            DEVICE
        )
    )

    encoded = {

        key:
            value.to(
                DEVICE
            )

        for key, value
        in encoded.items()
    }

    output = (
        model.backbone.transformer(

            input_ids=encoded[
                "input_ids"
            ],

            attention_mask=encoded[
                "attention_mask"
            ],

            output_hidden_states=True,

            use_cache=False,

            return_dict=True,
        )
    )

    hidden = (
        output.hidden_states[
            address_layer
        ]
    )

    entity_representation = (
        entity_average(
            hidden,
            entity_mask,
        )
    )

    key = (
        address_encoder(
            entity_representation
        )
    )

    return key.squeeze(0)


# ============================================================
# FIND ACTUAL SLOT WRITTEN BY MODEL
#
# We detect which memory slot's write_count increased.
# ============================================================

def identify_written_slot(
    before_state,
    after_state,
):

    before = (
        before_state.write_count[
            0
        ]
        .detach()
        .cpu()
    )

    after = (
        after_state.write_count[
            0
        ]
        .detach()
        .cpu()
    )

    difference = (
        after
        - before
    )

    changed = (
        difference.gt(0)
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

    # If multiple slots changed,
    # choose the slot with the largest increment.

    best = torch.argmax(
        difference
    )

    return int(
        best.item()
    )


# ============================================================
# WRITE ONE FACT THROUGH ACTUAL MEMORY MODEL
# ============================================================

@torch.no_grad()
def write_fact(
    text,
    memory_state,
):

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

    before_state = memory_state

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

    selected_slot = (
        identify_written_slot(
            before_state,
            after_state,
        )
    )

    return (
        after_state,
        selected_slot,
        output,
    )


# ============================================================
# INITIALIZE FRESH MEMORY
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
# ONE EPISODE
# ============================================================

@torch.no_grad()
def run_episode(
    num_facts,
    rng,
    verbose=False,
):

    memory_state = (
        initialize_memory()
    )

    # Address keys stored by ACTUAL memory slot number.
    #
    # slot_keys[slot_id] = address vector

    slot_keys = {}

    facts = []

    used_entities = set()

    # --------------------------------------------------------
    # WRITE PHASE
    # --------------------------------------------------------

    for fact_idx in range(
        num_facts
    ):

        while True:

            entity = random_entity(
                rng
            )

            if entity not in used_entities:

                used_entities.add(
                    entity
                )

                break

        answer = ANSWERS[
            rng.randrange(
                len(ANSWERS)
            )
        ]

        write_template = rng.choice(
            WRITE_TEMPLATES
        )

        query_template = rng.choice(
            QUERY_TEMPLATES
        )

        write_text = (
            write_template.format(

                entity=entity,

                answer=answer,
            )
        )

        query_text = (
            query_template.format(
                entity=entity,
            )
        )

        # ----------------------------------------------------
        # Get address key from WRITE view.
        # ----------------------------------------------------

        write_key = (
            get_address_key(
                write_text,
                entity,
            )
        )

        # ----------------------------------------------------
        # Send fact through ACTUAL memory writer/router.
        # ----------------------------------------------------

        new_memory_state, slot, _ = (
            write_fact(
                write_text,
                memory_state,
            )
        )

        if slot is None:

            return {
                "valid":
                    False,

                "reason":
                    "no_slot_written",
            }

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Store the learned address key BESIDE the ACTUAL
        # slot selected by the model.
        # ----------------------------------------------------

        slot_keys[
            slot
        ] = (
            write_key
            .detach()
            .clone()
        )

        facts.append(
            {
                "entity":
                    entity,

                "answer":
                    answer,

                "write":
                    write_text,

                "query":
                    query_text,

                "slot":
                    slot,
            }
        )

        memory_state = (
            new_memory_state
        )

    # --------------------------------------------------------
    # Check whether occupancy produced unique slots.
    # --------------------------------------------------------

    slots = [
        fact[
            "slot"
        ]
        for fact in facts
    ]

    unique_slots = set(
        slots
    )

    collision_free = (
        len(unique_slots)
        == len(slots)
    )

    # --------------------------------------------------------
    # QUERY PHASE
    # --------------------------------------------------------

    correct = 0

    total = 0

    reciprocal_rank_sum = 0.0

    margins = []

    predictions = []

    occupied_slots = sorted(
        slot_keys.keys()
    )

    if len(
        occupied_slots
    ) == 0:

        return {
            "valid":
                False,

            "reason":
                "no_occupied_slots",
        }

    stored_keys = torch.stack(
        [
            slot_keys[
                slot
            ]
            for slot in occupied_slots
        ],
        dim=0,
    )

    stored_keys = F.normalize(

        stored_keys,

        p=2,

        dim=-1,

        eps=1e-8,
    )

    for fact in facts:

        query_key = (
            get_address_key(

                fact[
                    "query"
                ],

                fact[
                    "entity"
                ],
            )
        )

        query_key = F.normalize(

            query_key,

            p=2,

            dim=-1,

            eps=1e-8,
        )

        similarity = (
            stored_keys
            @ query_key
        )

        ordering = torch.argsort(
            similarity,
            descending=True,
        )

        predicted_index = int(
            ordering[
                0
            ].item()
        )

        predicted_slot = (
            occupied_slots[
                predicted_index
            ]
        )

        correct_slot = (
            fact[
                "slot"
            ]
        )

        correct_position = (
            occupied_slots.index(
                correct_slot
            )
        )

        correct_score = (
            similarity[
                correct_position
            ]
        )

        rank = int(
            (
                similarity
                > correct_score
            )
            .sum()
            .item()
            + 1
        )

        reciprocal_rank_sum += (
            1.0 / rank
        )

        if predicted_slot == correct_slot:

            correct += 1

        total += 1

        # ----------------------------------------------------
        # Margin
        # ----------------------------------------------------

        if len(
            occupied_slots
        ) > 1:

            temp = similarity.clone()

            temp[
                correct_position
            ] = -float(
                "inf"
            )

            hardest_negative = (
                temp.max()
            )

            margin = (
                correct_score
                - hardest_negative
            )

            margins.append(
                float(
                    margin.item()
                )
            )

        else:

            margins.append(
                1.0
            )

        predictions.append(
            {
                "entity":
                    fact[
                        "entity"
                    ],

                "correct_slot":
                    correct_slot,

                "predicted_slot":
                    predicted_slot,

                "rank":
                    rank,

                "correct_score":
                    float(
                        correct_score.item()
                    ),
            }
        )

    accuracy = (
        correct
        / max(
            total,
            1,
        )
    )

    mrr = (
        reciprocal_rank_sum
        / max(
            total,
            1,
        )
    )

    mean_margin = (
        sum(
            margins
        )
        / max(
            len(
                margins
            ),
            1,
        )
    )

    if verbose:

        print()

        print(
            "FACTS / SLOT ASSIGNMENTS"
        )

        for fact in facts:

            print(
                fact[
                    "entity"
                ],
                "-> slot",
                fact[
                    "slot"
                ],
                "|",
                fact[
                    "answer"
                ],
            )

        print()

        print(
            "QUERY RESULTS"
        )

        for prediction in predictions:

            print(
                prediction[
                    "entity"
                ],
                "| correct slot =",
                prediction[
                    "correct_slot"
                ],
                "| predicted =",
                prediction[
                    "predicted_slot"
                ],
                "| rank =",
                prediction[
                    "rank"
                ],
            )

    return {

        "valid":
            True,

        "accuracy":
            accuracy,

        "mrr":
            mrr,

        "margin":
            mean_margin,

        "collision_free":
            collision_free,

        "slots":
            slots,

        "predictions":
            predictions,
    }


# ============================================================
# MAIN TEST
# ============================================================

section(
    "ACTUAL SLOT ADDRESS RETRIEVAL TEST"
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
    "Memory slots:",
    8,
)

print(
    "Address layer:",
    address_layer,
)


results = {}


for num_facts in FACT_COUNTS:

    section(
        f"{num_facts}-FACT EPISODES"
    )

    rng = random.Random(
        SEED
        + num_facts
        * 1000
    )

    episode_accuracies = []

    episode_mrrs = []

    episode_margins = []

    valid_episodes = 0

    invalid_episodes = 0

    collision_free_count = 0

    total_queries = 0

    total_correct = 0

    all_query_ranks = []

    # --------------------------------------------------------
    # First episode verbose
    # --------------------------------------------------------

    print(
        "Example episode:"
    )

    example = run_episode(

        num_facts=num_facts,

        rng=rng,

        verbose=True,
    )

    if example[
        "valid"
    ]:

        episode_accuracies.append(
            example[
                "accuracy"
            ]
        )

        episode_mrrs.append(
            example[
                "mrr"
            ]
        )

        episode_margins.append(
            example[
                "margin"
            ]
        )

        valid_episodes += 1

        if example[
            "collision_free"
        ]:

            collision_free_count += 1

        for prediction in example[
            "predictions"
        ]:

            total_queries += 1

            all_query_ranks.append(
                prediction[
                    "rank"
                ]
            )

            if (
                prediction[
                    "correct_slot"
                ]
                ==
                prediction[
                    "predicted_slot"
                ]
            ):

                total_correct += 1

    else:

        invalid_episodes += 1

    # --------------------------------------------------------
    # Remaining episodes
    # --------------------------------------------------------

    for episode_idx in range(
        1,
        EPISODES_PER_SIZE,
    ):

        episode = run_episode(

            num_facts=num_facts,

            rng=rng,

            verbose=False,
        )

        if not episode[
            "valid"
        ]:

            invalid_episodes += 1

            continue

        valid_episodes += 1

        episode_accuracies.append(
            episode[
                "accuracy"
            ]
        )

        episode_mrrs.append(
            episode[
                "mrr"
            ]
        )

        episode_margins.append(
            episode[
                "margin"
            ]
        )

        if episode[
            "collision_free"
        ]:

            collision_free_count += 1

        for prediction in episode[
            "predictions"
        ]:

            total_queries += 1

            all_query_ranks.append(
                prediction[
                    "rank"
                ]
            )

            if (
                prediction[
                    "correct_slot"
                ]
                ==
                prediction[
                    "predicted_slot"
                ]
            ):

                total_correct += 1

        if (
            episode_idx % 50
            == 0
        ):

            print(
                f"Completed "
                f"{episode_idx}/"
                f"{EPISODES_PER_SIZE}"
            )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    mean_episode_accuracy = (
        sum(
            episode_accuracies
        )
        / max(
            len(
                episode_accuracies
            ),
            1,
        )
    )

    mean_episode_mrr = (
        sum(
            episode_mrrs
        )
        / max(
            len(
                episode_mrrs
            ),
            1,
        )
    )

    mean_margin = (
        sum(
            episode_margins
        )
        / max(
            len(
                episode_margins
            ),
            1,
        )
    )

    global_accuracy = (
        total_correct
        / max(
            total_queries,
            1,
        )
    )

    collision_free_rate = (
        collision_free_count
        / max(
            valid_episodes,
            1,
        )
    )

    mean_rank = (
        sum(
            all_query_ranks
        )
        / max(
            len(
                all_query_ranks
            ),
            1,
        )
    )

    print()

    print(
        f"{num_facts}-FACT RESULTS"
    )

    print(
        "Valid episodes:",
        valid_episodes,
    )

    print(
        "Invalid episodes:",
        invalid_episodes,
    )

    print(
        "Collision-free episodes:",
        f"{collision_free_rate * 100:.2f}%"
    )

    print(
        "Mean episode slot accuracy:",
        f"{mean_episode_accuracy * 100:.2f}%"
    )

    print(
        "Global query slot accuracy:",
        f"{global_accuracy * 100:.2f}%"
    )

    print(
        "MRR:",
        fmt(
            mean_episode_mrr
        ),
    )

    print(
        "Mean rank:",
        fmt(
            mean_rank
        ),
    )

    print(
        "Mean similarity margin:",
        fmt(
            mean_margin
        ),
    )

    print(
        "Chance accuracy:",
        f"{100.0 / num_facts:.2f}%"
    )

    results[
        num_facts
    ] = {

        "accuracy":
            global_accuracy,

        "mrr":
            mean_episode_mrr,

        "mean_rank":
            mean_rank,

        "margin":
            mean_margin,

        "collision_free_rate":
            collision_free_rate,

        "valid_episodes":
            valid_episodes,
    }


# ============================================================
# FINAL SUMMARY
# ============================================================

section(
    "FINAL SUMMARY"
)

print(
    f"{'Facts':<10}"
    f"{'Chance':>12}"
    f"{'Slot Acc':>14}"
    f"{'MRR':>12}"
    f"{'Mean Rank':>14}"
    f"{'Margin':>14}"
    f"{'No Collision':>16}"
)

for num_facts in FACT_COUNTS:

    result = results[
        num_facts
    ]

    print(
        f"{num_facts:<10}"
        f"{100.0 / num_facts:>11.2f}%"
        f"{result['accuracy'] * 100:>13.2f}%"
        f"{result['mrr']:>12.4f}"
        f"{result['mean_rank']:>14.4f}"
        f"{result['margin']:>14.5f}"
        f"{result['collision_free_rate'] * 100:>15.2f}%"
    )


# ============================================================
# AUTOMATIC INTERPRETATION
# ============================================================

section(
    "AUTOMATIC INTERPRETATION"
)

acc2 = results[
    2
][
    "accuracy"
]

acc4 = results[
    4
][
    "accuracy"
]

acc8 = results[
    8
][
    "accuracy"
]


print(
    "2-fact slot retrieval:",
    f"{acc2 * 100:.2f}%"
)

print(
    "4-fact slot retrieval:",
    f"{acc4 * 100:.2f}%"
)

print(
    "8-fact slot retrieval:",
    f"{acc8 * 100:.2f}%"
)

print()


if (
    acc2 >= 0.95
    and acc4 >= 0.90
    and acc8 >= 0.80
):

    print(
        "RESULT: STRONG SLOT-ADDRESSING SUCCESS."
    )

    print()

    print(
        "The learned Layer-1 address keys can "
        "identify the ACTUAL occupancy-selected "
        "memory slot across multi-fact episodes."
    )

    print()

    print(
        "NEXT STEP:"
    )

    print(
        "Use the retrieved slot to read its "
        "ACTUAL MEMORY VALUE and test whether "
        "GPT-2 can generate the correct answer."
    )


elif (
    acc2 >= 0.90
    and acc4 >= 0.75
    and acc8 >= 0.60
):

    print(
        "RESULT: PROMISING BUT NOT FULLY ROBUST."
    )

    print()

    print(
        "Addressing works substantially above "
        "chance, but 8-slot retrieval still "
        "needs improvement."
    )

    print()

    print(
        "Do not integrate into the main "
        "architecture yet."
    )


else:

    print(
        "RESULT: SLOT-ADDRESSING IS NOT YET "
        "ROBUST ENOUGH."
    )

    print()

    print(
        "The global contrastive retrieval result "
        "does not fully transfer to actual "
        "occupancy-selected memory slots."
    )

    print()

    print(
        "We should diagnose the slot-key binding "
        "before touching the main architecture."
    )


# ============================================================
# SAFETY CHECK
# ============================================================

section(
    "EXPERIMENT COMPLETE"
)

print(
    "GPT-2 was frozen."
)

print(
    "Memory architecture was frozen."
)

print(
    "Address encoder was frozen."
)

print(
    "No source files were modified."
)

print(
    "No checkpoints were overwritten."
)