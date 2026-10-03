# ============================================================
# ADDRESSED MEMORY VALUE RETRIEVAL TEST
# MULTI-TOKEN SAFE VERSION
#
# PURPOSE
# ------------------------------------------------------------
# Tests whether the VALUE stored in the correctly addressed
# slot helps predict the correct answer.
#
# CONDITIONS:
#
# 1. ADDRESSED SLOT
# 2. ORACLE SLOT
# 3. WRONG SLOT
# 4. NO MEMORY
#
# IMPORTANT:
#
# - Handles BOTH single-token and multi-token answers.
# - Scores full answer sequence using summed log-probability.
# - No training.
# - No model/source modification.
# - No checkpoint overwrite.
#
# RUN:
#
# python addressed_value_retrieval_test.py \
#   2>&1 | tee addressed_value_retrieval_test.log
# ============================================================


import random
import string

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

SEED = 777

FACT_COUNTS = [
    2,
    4,
    8,
]

EPISODES_PER_SIZE = 100

ADDRESS_LAYER = 1

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

    "The assigned keyword for {entity} is",

    "The stored keyword for {entity} is",

    "The value associated with {entity} is",

    "The remembered keyword for {entity} is",

    "The keyword belonging to {entity} is",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


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
# MEMORY CONFIG
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
# VERIFY ANSWER TOKENIZATION
# ============================================================

section(
    "ANSWER TOKENIZATION"
)

ANSWER_TOKEN_IDS = {}

for answer in ANSWERS:

    token_ids = tokenizer.encode(
        " " + answer,
        add_special_tokens=False,
    )

    ANSWER_TOKEN_IDS[
        answer
    ] = token_ids

    print(
        f"{answer:<10}",
        token_ids,
        "| tokens =",
        len(token_ids),
    )

print()

print(
    "Multi-token answers are allowed."
)


# ============================================================
# LOAD MEMORY MODEL
# ============================================================

section(
    "LOAD FROZEN MEMORY MODEL"
)

print(
    "Device:",
    DEVICE,
)

print(
    "Checkpoint:",
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
    len(load_result.missing_keys),
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

print(
    "Memory model frozen."
)


# ============================================================
# LOAD ADDRESS ENCODER
# ============================================================

section(
    "LOAD ADDRESS ENCODER"
)

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
    "Previous address test R@1:",
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
):

    letters = "".join(

        rng.choice(
            string.ascii_uppercase
        )

        for _ in range(6)
    )

    digits = "".join(

        rng.choice(
            string.digits
        )

        for _ in range(5)
    )

    return (
        f"ValueEntity-{letters}-{digits}"
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

            overlaps = (
                token_start < end
                and
                token_end > start
            )

            if overlaps:

                mask[
                    batch_idx,
                    token_idx,
                ] = True

                found += 1

        if found == 0:

            raise RuntimeError(
                f"No tokens found for entity "
                f"{entity!r}"
            )

    return mask


# ============================================================
# ENTITY AVERAGE
# ============================================================

def entity_average(
    hidden,
    mask,
):

    weights = (
        mask
        .unsqueeze(-1)
        .to(
            hidden.dtype
        )
    )

    numerator = (
        hidden
        * weights
    ).sum(
        dim=1
    )

    denominator = (
        weights
        .sum(
            dim=1
        )
        .clamp_min(
            1.0
        )
    )

    return (
        numerator
        /
        denominator
    )


# ============================================================
# ADDRESS KEY EXTRACTION
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

    entity_mask_cpu = (
        build_entity_mask(

            texts=[
                text
            ],

            entities=[
                entity
            ],

            offset_mapping=offsets,

            attention_mask=attention_mask_cpu,
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

    entity_mask = (
        entity_mask_cpu.to(
            DEVICE
        )
    )

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

    entity_repr = (
        entity_average(
            hidden,
            entity_mask,
        )
    )

    key = (
        address_encoder(
            entity_repr
        )
    )

    return (
        key.squeeze(0)
    )


# ============================================================
# INITIALIZE MEMORY
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
# WRITE FACT
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

    slot = (
        identify_written_slot(
            before_state,
            after_state,
        )
    )

    return (
        after_state,
        slot,
    )


# ============================================================
# SLOT MASK
# ============================================================

def make_slot_mask(
    slot,
):

    mask = torch.zeros(
        1,
        8,
        dtype=torch.bool,
        device=DEVICE,
    )

    mask[
        0,
        slot
    ] = True

    return mask


# ============================================================
# SEQUENCE LOG-PROBABILITY WITH MEMORY
#
# Scores:
#
#   P(answer | query, selected memory slot)
#
# for arbitrary-length answer token sequences.
# ============================================================

@torch.no_grad()
def score_sequence_with_memory(
    query,
    answer,
    memory_state,
    selected_slot,
):

    answer_ids = (
        ANSWER_TOKEN_IDS[
            answer
        ]
    )

    current_text = (
        query
    )

    total_logprob = 0.0

    token_logprobs = []

    memory_mask = (
        make_slot_mask(
            selected_slot
        )
    )

    for token_id in answer_ids:

        encoded = tokenizer(
            current_text,
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

        output = model(

            input_ids=input_ids,

            attention_mask=attention_mask,

            memory_state=memory_state,

            memory_mask=memory_mask,

            update_memory=False,

            return_diagnostics=True,
        )

        logits = (
            output.logits[
                0,
                -1,
                :
            ]
        )

        log_probs = F.log_softmax(
            logits.float(),
            dim=-1,
        )

        token_logprob = float(
            log_probs[
                token_id
            ].item()
        )

        token_logprobs.append(
            token_logprob
        )

        total_logprob += (
            token_logprob
        )

        token_text = tokenizer.decode(
            [
                token_id
            ]
        )

        current_text += (
            token_text
        )

    return {

        "total_logprob":
            total_logprob,

        "mean_logprob":
            total_logprob
            /
            len(
                answer_ids
            ),

        "token_logprobs":
            token_logprobs,
    }


# ============================================================
# SEQUENCE LOG-PROBABILITY WITHOUT EXTERNAL MEMORY
# ============================================================

@torch.no_grad()
def score_sequence_no_memory(
    query,
    answer,
):

    answer_ids = (
        ANSWER_TOKEN_IDS[
            answer
        ]
    )

    current_text = (
        query
    )

    total_logprob = 0.0

    token_logprobs = []

    for token_id in answer_ids:

        encoded = tokenizer(
            current_text,
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

        output = model.backbone(

            input_ids=input_ids,

            attention_mask=attention_mask,

            use_cache=False,

            return_dict=True,
        )

        logits = (
            output.logits[
                0,
                -1,
                :
            ]
        )

        log_probs = F.log_softmax(
            logits.float(),
            dim=-1,
        )

        token_logprob = float(
            log_probs[
                token_id
            ].item()
        )

        token_logprobs.append(
            token_logprob
        )

        total_logprob += (
            token_logprob
        )

        token_text = tokenizer.decode(
            [
                token_id
            ]
        )

        current_text += (
            token_text
        )

    return {

        "total_logprob":
            total_logprob,

        "mean_logprob":
            total_logprob
            /
            len(
                answer_ids
            ),

        "token_logprobs":
            token_logprobs,
    }


# ============================================================
# SCORE ALL 16 CANDIDATES WITH MEMORY
#
# IMPORTANT:
#
# We rank using MEAN log-probability, not total log-probability.
#
# Why?
#
# "falcon" has 2 tokens while most answers have 1.
#
# Summed log-probability would naturally penalize longer
# answers simply because they have more tokens.
#
# Mean log-probability gives a length-normalized comparison.
# ============================================================

@torch.no_grad()
def evaluate_candidates_with_memory(
    query,
    correct_answer,
    memory_state,
    selected_slot,
):

    scores = {}

    raw_scores = {}

    for candidate in ANSWERS:

        result = (
            score_sequence_with_memory(

                query=query,

                answer=candidate,

                memory_state=memory_state,

                selected_slot=selected_slot,
            )
        )

        scores[
            candidate
        ] = result[
            "mean_logprob"
        ]

        raw_scores[
            candidate
        ] = result[
            "total_logprob"
        ]

    ordered = sorted(

        ANSWERS,

        key=lambda answer:
            scores[
                answer
            ],

        reverse=True,
    )

    predicted_answer = (
        ordered[
            0
        ]
    )

    rank = (
        ordered.index(
            correct_answer
        )
        + 1
    )

    return {

        "predicted_answer":
            predicted_answer,

        "rank":
            rank,

        "rr":
            1.0
            / rank,

        "correct":
            (
                predicted_answer
                ==
                correct_answer
            ),

        "correct_score":
            scores[
                correct_answer
            ],

        "scores":
            scores,

        "raw_scores":
            raw_scores,
    }


# ============================================================
# SCORE ALL 16 CANDIDATES WITHOUT MEMORY
# ============================================================

@torch.no_grad()
def evaluate_candidates_no_memory(
    query,
    correct_answer,
):

    scores = {}

    raw_scores = {}

    for candidate in ANSWERS:

        result = (
            score_sequence_no_memory(

                query=query,

                answer=candidate,
            )
        )

        scores[
            candidate
        ] = result[
            "mean_logprob"
        ]

        raw_scores[
            candidate
        ] = result[
            "total_logprob"
        ]

    ordered = sorted(

        ANSWERS,

        key=lambda answer:
            scores[
                answer
            ],

        reverse=True,
    )

    predicted_answer = (
        ordered[
            0
        ]
    )

    rank = (
        ordered.index(
            correct_answer
        )
        + 1
    )

    return {

        "predicted_answer":
            predicted_answer,

        "rank":
            rank,

        "rr":
            1.0
            / rank,

        "correct":
            (
                predicted_answer
                ==
                correct_answer
            ),

        "correct_score":
            scores[
                correct_answer
            ],

        "scores":
            scores,

        "raw_scores":
            raw_scores,
    }


# ============================================================
# METRICS
# ============================================================

def empty_metrics():

    return {

        "correct":
            0,

        "total":
            0,

        "rr_sum":
            0.0,

        "rank_sum":
            0.0,

        "score_sum":
            0.0,
    }


def update_metrics(
    metrics,
    result,
):

    metrics[
        "total"
    ] += 1

    if result[
        "correct"
    ]:

        metrics[
            "correct"
        ] += 1

    metrics[
        "rr_sum"
    ] += result[
        "rr"
    ]

    metrics[
        "rank_sum"
    ] += result[
        "rank"
    ]

    metrics[
        "score_sum"
    ] += result[
        "correct_score"
    ]


def summarize_metrics(
    metrics,
):

    total = max(
        metrics[
            "total"
        ],
        1,
    )

    return {

        "accuracy":
            metrics[
                "correct"
            ]
            / total,

        "mrr":
            metrics[
                "rr_sum"
            ]
            / total,

        "mean_rank":
            metrics[
                "rank_sum"
            ]
            / total,

        "mean_score":
            metrics[
                "score_sum"
            ]
            / total,

        "total":
            metrics[
                "total"
            ],
    }


# ============================================================
# RUN ONE EPISODE
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

    facts = []

    slot_keys = {}

    used_entities = set()

    used_answers = set()


    # ========================================================
    # WRITE PHASE
    # ========================================================

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


        available_answers = [

            answer

            for answer in ANSWERS

            if answer
            not in used_answers
        ]

        answer = rng.choice(
            available_answers
        )

        used_answers.add(
            answer
        )


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
        # WRITE-VIEW ADDRESS KEY
        # ----------------------------------------------------

        write_key = (
            get_address_key(
                write_text,
                entity,
            )
        )


        # ----------------------------------------------------
        # ACTUAL MEMORY WRITE
        # ----------------------------------------------------

        new_memory_state, slot = (
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


    # ========================================================
    # COLLISION CHECK
    # ========================================================

    actual_slots = [

        fact[
            "slot"
        ]

        for fact in facts
    ]


    if len(
        set(
            actual_slots
        )
    ) != len(
        actual_slots
    ):

        return {

            "valid":
                False,

            "reason":
                "slot_collision",
        }


    # ========================================================
    # SLOT KEYS
    # ========================================================

    occupied_slots = sorted(
        slot_keys.keys()
    )


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


    # ========================================================
    # METRICS
    # ========================================================

    addressed_metrics = (
        empty_metrics()
    )

    oracle_metrics = (
        empty_metrics()
    )

    wrong_metrics = (
        empty_metrics()
    )

    no_memory_metrics = (
        empty_metrics()
    )


    addressing_correct = 0

    query_records = []


    # ========================================================
    # QUERY PHASE
    # ========================================================

    for fact in facts:

        entity = (
            fact[
                "entity"
            ]
        )

        query = (
            fact[
                "query"
            ]
        )

        correct_answer = (
            fact[
                "answer"
            ]
        )

        correct_slot = (
            fact[
                "slot"
            ]
        )


        # ----------------------------------------------------
        # ADDRESS QUERY
        # ----------------------------------------------------

        query_key = (
            get_address_key(
                query,
                entity,
            )
        )


        query_key = F.normalize(

            query_key,

            p=2,

            dim=-1,

            eps=1e-8,
        )


        similarities = (
            stored_keys
            @ query_key
        )


        predicted_index = int(
            torch.argmax(
                similarities
            ).item()
        )


        addressed_slot = (
            occupied_slots[
                predicted_index
            ]
        )


        if (
            addressed_slot
            ==
            correct_slot
        ):

            addressing_correct += 1


        # ----------------------------------------------------
        # WRONG SLOT
        # ----------------------------------------------------

        wrong_candidates = [

            slot

            for slot in occupied_slots

            if slot
            != correct_slot
        ]


        if len(
            wrong_candidates
        ) > 0:

            wrong_slot = (
                wrong_candidates[
                    0
                ]
            )

        else:

            wrong_slot = (
                correct_slot
            )


        # ====================================================
        # ADDRESSED CONDITION
        # ====================================================

        addressed_result = (
            evaluate_candidates_with_memory(

                query=query,

                correct_answer=correct_answer,

                memory_state=memory_state,

                selected_slot=addressed_slot,
            )
        )


        update_metrics(
            addressed_metrics,
            addressed_result,
        )


        # ====================================================
        # ORACLE CONDITION
        # ====================================================

        oracle_result = (
            evaluate_candidates_with_memory(

                query=query,

                correct_answer=correct_answer,

                memory_state=memory_state,

                selected_slot=correct_slot,
            )
        )


        update_metrics(
            oracle_metrics,
            oracle_result,
        )


        # ====================================================
        # WRONG SLOT CONDITION
        # ====================================================

        wrong_result = (
            evaluate_candidates_with_memory(

                query=query,

                correct_answer=correct_answer,

                memory_state=memory_state,

                selected_slot=wrong_slot,
            )
        )


        update_metrics(
            wrong_metrics,
            wrong_result,
        )


        # ====================================================
        # NO MEMORY CONDITION
        # ====================================================

        baseline_result = (
            evaluate_candidates_no_memory(

                query=query,

                correct_answer=correct_answer,
            )
        )


        update_metrics(
            no_memory_metrics,
            baseline_result,
        )


        query_records.append(

            {

                "entity":
                    entity,

                "answer":
                    correct_answer,

                "correct_slot":
                    correct_slot,

                "addressed_slot":
                    addressed_slot,

                "wrong_slot":
                    wrong_slot,

                "address_ok":
                    (
                        addressed_slot
                        ==
                        correct_slot
                    ),

                "addressed_pred":
                    addressed_result[
                        "predicted_answer"
                    ],

                "addressed_rank":
                    addressed_result[
                        "rank"
                    ],

                "oracle_pred":
                    oracle_result[
                        "predicted_answer"
                    ],

                "oracle_rank":
                    oracle_result[
                        "rank"
                    ],

                "wrong_pred":
                    wrong_result[
                        "predicted_answer"
                    ],

                "wrong_rank":
                    wrong_result[
                        "rank"
                    ],

                "baseline_pred":
                    baseline_result[
                        "predicted_answer"
                    ],

                "baseline_rank":
                    baseline_result[
                        "rank"
                    ],
            }
        )


    # ========================================================
    # VERBOSE EXAMPLE
    # ========================================================

    if verbose:

        print()

        print(
            "FACTS / ACTUAL SLOT ASSIGNMENTS"
        )

        for fact in facts:

            print(

                f"{fact['entity']:<30} "
                f"-> slot {fact['slot']} "
                f"| value={fact['answer']}"
            )


        print()

        print(
            "QUERY RESULTS"
        )


        for record in query_records:

            print()

            print(
                "ENTITY:",
                record[
                    "entity"
                ],
            )

            print(
                "TRUE ANSWER:",
                record[
                    "answer"
                ],
            )

            print(
                "Correct slot:",
                record[
                    "correct_slot"
                ],
            )

            print(
                "Addressed slot:",
                record[
                    "addressed_slot"
                ],
            )

            print(
                "Address correct:",
                record[
                    "address_ok"
                ],
            )

            print(
                "ADDRESSED ->",
                record[
                    "addressed_pred"
                ],
                "| rank",
                record[
                    "addressed_rank"
                ],
            )

            print(
                "ORACLE    ->",
                record[
                    "oracle_pred"
                ],
                "| rank",
                record[
                    "oracle_rank"
                ],
            )

            print(
                "WRONG     ->",
                record[
                    "wrong_pred"
                ],
                "| rank",
                record[
                    "wrong_rank"
                ],
            )

            print(
                "NO MEMORY ->",
                record[
                    "baseline_pred"
                ],
                "| rank",
                record[
                    "baseline_rank"
                ],
            )


    return {

        "valid":
            True,

        "address_correct":
            addressing_correct,

        "address_total":
            len(
                facts
            ),

        "addressed":
            addressed_metrics,

        "oracle":
            oracle_metrics,

        "wrong":
            wrong_metrics,

        "no_memory":
            no_memory_metrics,
    }


# ============================================================
# MAIN
# ============================================================

section(
    "ADDRESSED MEMORY VALUE RETRIEVAL"
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
    "Candidate answers:",
    len(
        ANSWERS
    ),
)

print(
    "Chance:",
    f"{100 / len(ANSWERS):.2f}%"
)


all_results = {}


# ============================================================
# TEST EACH MEMORY SIZE
# ============================================================

for num_facts in FACT_COUNTS:

    section(
        f"{num_facts}-FACT VALUE RETRIEVAL"
    )


    rng = random.Random(
        SEED
        +
        num_facts
        * 1000
    )


    aggregate = {

        "addressed":
            empty_metrics(),

        "oracle":
            empty_metrics(),

        "wrong":
            empty_metrics(),

        "no_memory":
            empty_metrics(),
    }


    address_correct = 0

    address_total = 0

    valid_episodes = 0

    invalid_episodes = 0


    for episode_idx in range(
        EPISODES_PER_SIZE
    ):

        episode = (
            run_episode(

                num_facts=num_facts,

                rng=rng,

                verbose=(
                    episode_idx == 0
                ),
            )
        )


        if not episode[
            "valid"
        ]:

            invalid_episodes += 1

            continue


        valid_episodes += 1


        address_correct += (
            episode[
                "address_correct"
            ]
        )


        address_total += (
            episode[
                "address_total"
            ]
        )


        for condition in [

            "addressed",

            "oracle",

            "wrong",

            "no_memory",
        ]:

            source = (
                episode[
                    condition
                ]
            )

            destination = (
                aggregate[
                    condition
                ]
            )


            destination[
                "correct"
            ] += source[
                "correct"
            ]

            destination[
                "total"
            ] += source[
                "total"
            ]

            destination[
                "rr_sum"
            ] += source[
                "rr_sum"
            ]

            destination[
                "rank_sum"
            ] += source[
                "rank_sum"
            ]

            destination[
                "score_sum"
            ] += source[
                "score_sum"
            ]


        if (
            episode_idx + 1
        ) % 25 == 0:

            print(

                f"Completed "
                f"{episode_idx + 1}/"
                f"{EPISODES_PER_SIZE}"
            )


    addressed_summary = (
        summarize_metrics(
            aggregate[
                "addressed"
            ]
        )
    )

    oracle_summary = (
        summarize_metrics(
            aggregate[
                "oracle"
            ]
        )
    )

    wrong_summary = (
        summarize_metrics(
            aggregate[
                "wrong"
            ]
        )
    )

    baseline_summary = (
        summarize_metrics(
            aggregate[
                "no_memory"
            ]
        )
    )


    address_accuracy = (
        address_correct
        /
        max(
            address_total,
            1,
        )
    )


    print()

    print(
        f"{num_facts}-FACT RESULTS"
    )

    print()

    print(
        "Valid episodes:",
        valid_episodes,
    )

    print(
        "Invalid episodes:",
        invalid_episodes,
    )

    print(
        "Address slot accuracy:",
        f"{address_accuracy * 100:.2f}%"
    )


    print()

    print(
        f"{'Condition':<16}"
        f"{'16-way Acc':>14}"
        f"{'MRR':>12}"
        f"{'Mean Rank':>14}"
        f"{'Mean LogP':>14}"
    )


    condition_data = [

        (
            "ADDRESSED",
            addressed_summary,
        ),

        (
            "ORACLE",
            oracle_summary,
        ),

        (
            "WRONG SLOT",
            wrong_summary,
        ),

        (
            "NO MEMORY",
            baseline_summary,
        ),
    ]


    for name, summary in condition_data:

        print(

            f"{name:<16}"

            f"{summary['accuracy'] * 100:>13.2f}%"

            f"{summary['mrr']:>12.4f}"

            f"{summary['mean_rank']:>14.4f}"

            f"{summary['mean_score']:>14.6f}"
        )


    all_results[
        num_facts
    ] = {

        "address_accuracy":
            address_accuracy,

        "addressed":
            addressed_summary,

        "oracle":
            oracle_summary,

        "wrong":
            wrong_summary,

        "no_memory":
            baseline_summary,
    }


# ============================================================
# FINAL TABLE
# ============================================================

section(
    "FINAL 16-WAY ANSWER ACCURACY"
)

print(

    f"{'Facts':<10}"

    f"{'Address':>12}"

    f"{'Addressed':>14}"

    f"{'Oracle':>12}"

    f"{'Wrong':>12}"

    f"{'NoMem':>12}"

    f"{'Chance':>12}"
)


for num_facts in FACT_COUNTS:

    result = (
        all_results[
            num_facts
        ]
    )

    print(

        f"{num_facts:<10}"

        f"{result['address_accuracy'] * 100:>11.2f}%"

        f"{result['addressed']['accuracy'] * 100:>13.2f}%"

        f"{result['oracle']['accuracy'] * 100:>11.2f}%"

        f"{result['wrong']['accuracy'] * 100:>11.2f}%"

        f"{result['no_memory']['accuracy'] * 100:>11.2f}%"

        f"{100 / len(ANSWERS):>11.2f}%"
    )


# ============================================================
# MEMORY EFFECT
# ============================================================

section(
    "MEMORY VALUE EFFECT"
)

for num_facts in FACT_COUNTS:

    result = (
        all_results[
            num_facts
        ]
    )

    addressed_acc = (
        result[
            "addressed"
        ][
            "accuracy"
        ]
    )

    oracle_acc = (
        result[
            "oracle"
        ][
            "accuracy"
        ]
    )

    wrong_acc = (
        result[
            "wrong"
        ][
            "accuracy"
        ]
    )

    baseline_acc = (
        result[
            "no_memory"
        ][
            "accuracy"
        ]
    )

    print()

    print(
        f"{num_facts} facts"
    )

    print(
        "Addressed - NoMem:",
        f"{(addressed_acc - baseline_acc) * 100:+.2f} points"
    )

    print(
        "Oracle - NoMem:",
        f"{(oracle_acc - baseline_acc) * 100:+.2f} points"
    )

    print(
        "Oracle - Wrong:",
        f"{(oracle_acc - wrong_acc) * 100:+.2f} points"
    )


# ============================================================
# AUTOMATIC DIAGNOSIS
# ============================================================

section(
    "AUTOMATIC DIAGNOSIS"
)


oracle_mean = sum(

    all_results[
        n
    ][
        "oracle"
    ][
        "accuracy"
    ]

    for n in FACT_COUNTS

) / len(
    FACT_COUNTS
)


wrong_mean = sum(

    all_results[
        n
    ][
        "wrong"
    ][
        "accuracy"
    ]

    for n in FACT_COUNTS

) / len(
    FACT_COUNTS
)


baseline_mean = sum(

    all_results[
        n
    ][
        "no_memory"
    ][
        "accuracy"
    ]

    for n in FACT_COUNTS

) / len(
    FACT_COUNTS
)


addressed_mean = sum(

    all_results[
        n
    ][
        "addressed"
    ][
        "accuracy"
    ]

    for n in FACT_COUNTS

) / len(
    FACT_COUNTS
)


print(
    "Mean addressed accuracy:",
    f"{addressed_mean * 100:.2f}%"
)

print(
    "Mean oracle accuracy:",
    f"{oracle_mean * 100:.2f}%"
)

print(
    "Mean wrong-slot accuracy:",
    f"{wrong_mean * 100:.2f}%"
)

print(
    "Mean no-memory accuracy:",
    f"{baseline_mean * 100:.2f}%"
)

print()


if (
    oracle_mean >= 0.70
    and
    oracle_mean > wrong_mean + 0.20
    and
    oracle_mean > baseline_mean + 0.20
):

    print(
        "RESULT: MEMORY VALUE PATH IS WORKING."
    )

    print()

    print(
        "Correctly selecting the slot makes "
        "the stored value strongly useful "
        "for answer retrieval."
    )

    print()

    print(
        "NEXT STEP:"
    )

    print(
        "Integrate Layer-1 addressing into "
        "the real reader architecture."
    )


elif (
    oracle_mean > wrong_mean + 0.05
    or
    oracle_mean > baseline_mean + 0.05
):

    print(
        "RESULT: MEMORY VALUE HAS SOME USEFUL SIGNAL, "
        "BUT VALUE/FUSION IS STILL WEAK."
    )

    print()

    print(
        "Addressing is no longer the main bottleneck."
    )

    print()

    print(
        "NEXT STEP:"
    )

    print(
        "Diagnose what the writer stores and "
        "how the selected memory value affects "
        "the LM output."
    )


else:

    print(
        "RESULT: CORRECT SLOT DOES NOT YET "
        "PRODUCE THE CORRECT ANSWER."
    )

    print()

    print(
        "Addressing works, but the current "
        "memory VALUE representation/read-fusion "
        "path is still insufficient."
    )

    print()

    print(
        "NEXT STEP:"
    )

    print(
        "Inspect the value written into each slot "
        "and redesign the value representation "
        "before changing addressing."
    )


# ============================================================
# DONE
# ============================================================

section(
    "EXPERIMENT COMPLETE"
)

print(
    "No parameters were trained."
)

print(
    "No source files were modified."
)

print(
    "No checkpoints were overwritten."
)