# ============================================================
# LOCOMO 2x2 WRITER-vs-READER RETRIEVAL ISOLATION TEST
#
# ============================================================
#
# PURPOSE
#
# We already know:
#
# EXISTING CandidateWriter + EXISTING MemoryReader
# performs much worse than direct representation retrieval.
#
# But that changed BOTH writing and reading simultaneously.
#
# This experiment isolates them.
#
#
#                     READER
#                 OLD          COSINE
#
# WRITER OLD       A              B
#
#        DIRECT    C              D
#
#
# A = EXISTING WRITER + EXISTING READER
#
# B = EXISTING WRITER + DIRECT COSINE
#
# C = DIRECT MEMORY + EXISTING READER
#
# D = DIRECT MEMORY + DIRECT COSINE
#
#
# Interpretation:
#
# A low, B high:
#     reader is the main problem
#
# A low, C high:
#     writer is the main problem
#
# B and C both improve:
#     both components hurt retrieval
#
# D is the clean direct-retrieval reference.
#
#
# SPEED IMPROVEMENT
# ============================================================
#
# OLD SCRIPT:
#
# 8 candidate GPT2 passes
# + question GPT2 pass
# per question
#
#
# THIS SCRIPT:
#
# 1 BATCHED candidate GPT2 pass
# + 1 question GPT2 pass
# per question
#
#
# CandidateWriter receives cached transformer states.
#
# No unnecessary LM-head computation.
# No answer decoder.
# No value classifier.
#
#
# RUN:
#
# python locomo_retrieval_benchmark.py \
#   --data data/locomo/locomo10.json \
#   --max-questions 300 \
#   2>&1 | tee locomo_2x2_retrieval.log
#
# ============================================================


import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

from models.gpt2_memory import (
    MemoryAugmentedGPT2LMHeadModel,
    MemoryGPT2Config,
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--data",
    type=str,
    default="data/locomo/locomo10.json",
)


parser.add_argument(
    "--checkpoint",
    type=str,
    default=(
        "outputs/"
        "retrieval_gradient_test/"
        "checkpoint_best.pt"
    ),
)


parser.add_argument(
    "--max-questions",
    type=int,
    default=300,
)


parser.add_argument(
    "--pool-size",
    type=int,
    default=8,
)


parser.add_argument(
    "--max-turn-tokens",
    type=int,
    default=96,
)


parser.add_argument(
    "--max-question-tokens",
    type=int,
    default=96,
)


parser.add_argument(
    "--seed",
    type=int,
    default=2080,
)


parser.add_argument(
    "--output",
    type=str,
    default=(
        "outputs/"
        "locomo_2x2_retrieval.pt"
    ),
)


args = parser.parse_args()


# ============================================================
# GLOBAL
# ============================================================

MODEL_NAME = "gpt2"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

NUM_SLOTS = args.pool_size

TOP_KS = [
    1,
    3,
    5,
]

SEED = args.seed


if NUM_SLOTS != 8:

    raise ValueError(
        "Current memory model has 8 slots. "
        "Use --pool-size 8."
    )


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
# PRINT
# ============================================================

def section(title):

    print()

    print(
        "=" * 125
    )

    print(
        title
    )

    print(
        "=" * 125
    )


# ============================================================
# MEMORY CONFIG
# ============================================================

def build_config():

    return MemoryGPT2Config(

        num_slots=8,

        gate_type="vector",

        gate_mode="sigmoid",

        gate_init_bias=-2.0,

        # ----------------------------------------------------
        # Occupancy routing prevents write collision.
        #
        # We are NOT testing the old softmax collision issue.
        # We are testing Writer vs Reader.
        # ----------------------------------------------------

        router_enabled=True,

        router_mode="occupancy",

        router_top_k=1,

        router_temperature=0.7,

        writer_mode="attention",

        writer_attention_heads=8,

        orthogonal_mode="other_slots",

        orthogonal_strength=0.5,

        # ----------------------------------------------------
        # Keep existing trained reader architecture.
        # ----------------------------------------------------

        reader_mode="hybrid",

        reader_fusion="gated",

        reader_heads=8,

        # Rank all eight slots.
        reader_top_k=8,

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
# LOAD MODEL
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
    args.checkpoint,
)


model = (
    MemoryAugmentedGPT2LMHeadModel
    .from_pretrained(

        MODEL_NAME,

        memory_config=
            build_config(),
    )
)


checkpoint = torch.load(

    args.checkpoint,

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

    checkpoint_state = checkpoint


current_state = (
    model.state_dict()
)


compatible = {}


for name, value in (
    checkpoint_state.items()
):

    if (
        name in current_state

        and

        current_state[
            name
        ].shape
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
    len(
        load_result.missing_keys
    ),
)


model.to(
    DEVICE
)

model.eval()


for parameter in model.parameters():

    parameter.requires_grad = False


print(
    "Model frozen."
)


# ============================================================
# LOAD LOCOMO
# ============================================================

section(
    "LOAD LOCOMO"
)


data_path = Path(
    args.data
)


if not data_path.exists():

    raise FileNotFoundError(

        f"LoCoMo not found:\n"
        f"{data_path}"
    )


with open(

    data_path,

    "r",

    encoding="utf-8",

) as handle:

    locomo = json.load(
        handle
    )


print(
    "Conversations:",
    len(locomo),
)


# ============================================================
# FLATTEN CONVERSATION
# ============================================================

def flatten_conversation(
    sample,
):

    conversation = sample.get(

        "conversation",

        {},
    )


    turns = []

    session_keys = []


    for key in (
        conversation.keys()
    ):

        match = re.fullmatch(

            r"session_(\d+)",

            key,
        )


        if match:

            session_keys.append(

                (
                    int(
                        match.group(
                            1
                        )
                    ),

                    key,
                )
            )


    session_keys.sort()


    for (
        session_number,
        session_key,
    ) in session_keys:

        session_turns = (

            conversation.get(

                session_key,

                [],
            )
        )


        for turn in session_turns:

            dia_id = str(

                turn.get(
                    "dia_id",
                    "",
                )
            ).strip()


            text = str(

                turn.get(
                    "text",
                    "",
                )
            ).strip()


            if not text:

                text = str(

                    turn.get(
                        "blip_caption",
                        "",
                    )
                ).strip()


            speaker = str(

                turn.get(
                    "speaker",
                    "",
                )
            ).strip()


            if (
                not dia_id
                or
                not text
            ):

                continue


            if speaker:

                memory_text = (

                    f"{speaker}: "
                    f"{text}"
                )

            else:

                memory_text = text


            turns.append(

                {

                    "dia_id":
                        dia_id,

                    "text":
                        memory_text,

                    "raw_text":
                        text,

                    "speaker":
                        speaker,

                    "session":
                        session_number,
                }
            )


    return turns


# ============================================================
# NORMALIZE EVIDENCE
# ============================================================

def normalize_evidence(
    evidence,
):

    if evidence is None:

        return []


    if isinstance(
        evidence,
        str,
    ):

        evidence = [
            evidence
        ]


    output = []


    def visit(item):

        if isinstance(
            item,
            str,
        ):

            value = item.strip()


            if value:

                output.append(
                    value
                )


        elif isinstance(
            item,
            list,
        ):

            for child in item:

                visit(
                    child
                )


    visit(
        evidence
    )


    seen = set()

    final = []


    for value in output:

        if value not in seen:

            seen.add(
                value
            )

            final.append(
                value
            )


    return final


# ============================================================
# BUILD QA ITEMS
# ============================================================

section(
    "BUILD RETRIEVAL QUESTIONS"
)


qa_items = []

category_counts = defaultdict(
    int
)


for conversation_index, sample in (
    enumerate(
        locomo
    )
):

    sample_id = str(

        sample.get(

            "sample_id",

            conversation_index,
        )
    )


    turns = flatten_conversation(
        sample
    )


    turn_by_id = {

        turn[
            "dia_id"
        ]:
            turn

        for turn in turns
    }


    qas = sample.get(
        "qa",
        [],
    )


    for qa_index, qa in (
        enumerate(
            qas
        )
    ):

        question = str(

            qa.get(
                "question",
                "",
            )
        ).strip()


        evidence = normalize_evidence(

            qa.get(
                "evidence",
                [],
            )
        )


        if (
            not question
            or
            not evidence
        ):

            continue


        if (
            len(
                evidence
            )
            >
            NUM_SLOTS
        ):

            continue


        if not all(

            evidence_id
            in
            turn_by_id

            for evidence_id
            in evidence
        ):

            continue


        if (
            len(turns)
            <
            NUM_SLOTS
        ):

            continue


        category = str(

            qa.get(
                "category",
                "unknown",
            )
        )


        qa_items.append(

            {

                "sample_id":
                    sample_id,

                "conversation_index":
                    conversation_index,

                "qa_index":
                    qa_index,

                "question":
                    question,

                "answer":
                    qa.get(
                        "answer",
                        "",
                    ),

                "category":
                    category,

                "evidence":
                    evidence,

                "turns":
                    turns,

                "turn_by_id":
                    turn_by_id,
            }
        )


        category_counts[
            category
        ] += 1


print(
    "Eligible questions:",
    len(qa_items),
)

print(
    "Categories:",
    dict(
        category_counts
    ),
)


# ============================================================
# FIX QUESTION SAMPLE
# ============================================================

rng = random.Random(
    SEED
)


rng.shuffle(
    qa_items
)


if args.max_questions > 0:

    qa_items = qa_items[
        :args.max_questions
    ]


print(
    "Questions evaluated:",
    len(qa_items),
)


# ============================================================
# BUILD 8-MEMORY POOL
# ============================================================

def build_candidate_pool(
    item,
    rng,
):

    gold_ids = set(

        item[
            "evidence"
        ]
    )


    gold_turns = [

        item[
            "turn_by_id"
        ][
            evidence_id
        ]

        for evidence_id
        in item[
            "evidence"
        ]
    ]


    distractors = [

        turn

        for turn
        in item[
            "turns"
        ]

        if (
            turn[
                "dia_id"
            ]
            not in
            gold_ids
        )
    ]


    required = (

        NUM_SLOTS

        -
        len(
            gold_turns
        )
    )


    if (
        len(
            distractors
        )
        <
        required
    ):

        return None


    chosen = (

        gold_turns

        +
        rng.sample(

            distractors,

            required,
        )
    )


    rng.shuffle(
        chosen
    )


    return chosen


# ============================================================
# BATCH ENCODE 8 CANDIDATE MEMORIES
#
# THIS IS THE MAIN SPEED IMPROVEMENT.
# ============================================================

@torch.inference_mode()
def encode_candidate_batch(
    candidates,
):

    texts = [

        candidate[
            "text"
        ]

        for candidate
        in candidates
    ]


    encoded = tokenizer(

        texts,

        padding=True,

        truncation=True,

        max_length=
            args.max_turn_tokens,

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

            input_ids=
                input_ids,

            attention_mask=
                attention_mask,

            output_hidden_states=False,

            use_cache=False,

            return_dict=True,
        )
    )


    hidden = (

        transformer_output
        .last_hidden_state
    )


    # This is the same pooling family used by the model.

    summaries = model._pool_hidden(

        hidden_states=
            hidden,

        attention_mask=
            attention_mask,
    )


    return {

        "hidden":
            hidden,

        "summaries":
            summaries,

        "attention_mask":
            attention_mask,
    }


# ============================================================
# ENCODE QUESTION ONCE
# ============================================================

@torch.inference_mode()
def encode_question(
    question,
):

    encoded = tokenizer(

        question,

        truncation=True,

        max_length=
            args.max_question_tokens,

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

            input_ids=
                input_ids,

            attention_mask=
                attention_mask,

            output_hidden_states=False,

            use_cache=False,

            return_dict=True,
        )
    )


    hidden = (

        transformer_output
        .last_hidden_state
    )


    summary = model._pool_hidden(

        hidden_states=
            hidden,

        attention_mask=
            attention_mask,
    )


    return {

        "hidden":
            hidden,

        "summary":
            summary,

        "attention_mask":
            attention_mask,
    }


# ============================================================
# IDENTIFY SLOT WRITTEN
# ============================================================

def identify_written_slot(
    before_state,
    after_state,
):

    delta_count = (

        after_state
        .write_count[
            0
        ]

        -

        before_state
        .write_count[
            0
        ]
    )


    changed = (

        delta_count
        .gt(
            0
        )
        .nonzero(
            as_tuple=False
        )
        .flatten()
    )


    if (
        len(
            changed
        )
        ==
        1
    ):

        return int(

            changed[
                0
            ].item()
        )


    # Fallback based on slot vector change.

    delta = (

        after_state
        .slots[
            0
        ]

        -

        before_state
        .slots[
            0
        ]
    )


    norms = (

        delta
        .float()
        .norm(
            dim=-1
        )
    )


    return int(

        torch.argmax(
            norms
        ).item()
    )


# ============================================================
# OLD WRITER
#
# IMPORTANT:
#
# We DO NOT rerun GPT2 here.
#
# Candidate transformer states were already computed in one
# batch.
#
# We invoke the actual project's _write_memory() directly.
# ============================================================

@torch.inference_mode()
def build_existing_writer_memory(
    candidates,
    candidate_encoding,
):

    dtype = next(
        model.parameters()
    ).dtype


    memory_state = (

        model.initialize_memory(

            batch_size=1,

            device=DEVICE,

            dtype=dtype,
        )
    )


    slot_to_dia = {}


    hidden = (

        candidate_encoding[
            "hidden"
        ]
    )


    summaries = (

        candidate_encoding[
            "summaries"
        ]
    )


    attention_mask = (

        candidate_encoding[
            "attention_mask"
        ]
    )


    for index, candidate in (
        enumerate(
            candidates
        )
    ):

        before_state = (
            memory_state
        )


        (
            memory_state,
            routing_output,
            writer_output,
            orthogonal_output,
            write_gate,

        ) = model._write_memory(

            summary=
                summaries[
                    index:
                    index + 1
                ],

            token_states=
                hidden[
                    index:
                    index + 1
                ],

            attention_mask=
                attention_mask[
                    index:
                    index + 1
                ],

            memory_state=
                memory_state,

            memory_mask=None,
        )


        slot_index = identify_written_slot(

            before_state,

            memory_state,
        )


        slot_to_dia[
            slot_index
        ] = candidate[
            "dia_id"
        ]


    return (
        memory_state,
        slot_to_dia,
    )


# ============================================================
# DIRECT MEMORY
#
# No CandidateWriter.
#
# Store final pooled GPT2 representations directly into
# memory slots.
#
# Candidate order == slot order.
#
# We normalize using the existing MemoryBank normalization so
# the vectors have the expected slot scale/geometry before
# being passed to the existing MemoryReader.
# ============================================================

@torch.inference_mode()
def build_direct_memory(
    candidates,
    candidate_encoding,
):

    dtype = next(
        model.parameters()
    ).dtype


    state = model.initialize_memory(

        batch_size=1,

        device=DEVICE,

        dtype=dtype,
    )


    direct_slots = (

        candidate_encoding[
            "summaries"
        ]
        .unsqueeze(
            0
        )
        .to(
            dtype=
                state.slots.dtype
        )
    )


    direct_slots = (

        model.memory_bank
        .normalize_slots(
            direct_slots
        )
    )


    state.slots = (
        direct_slots
    )


    # Tell reader these slots contain valid memories.

    state.write_count[:] = 1

    state.confidence[:] = 1.0


    slot_to_dia = {

        index:
            candidate[
                "dia_id"
            ]

        for index, candidate
        in enumerate(
            candidates
        )
    }


    return (
        state,
        slot_to_dia,
    )


# ============================================================
# SLOT RANKING HELPER
# ============================================================

def slots_to_dia_ranking(
    slot_scores,
    slot_to_dia,
    candidates,
):

    order = torch.argsort(

        slot_scores,

        descending=True,
    )


    ranking = []

    seen = set()


    for slot_tensor in order:

        slot = int(
            slot_tensor.item()
        )


        if slot in slot_to_dia:

            dia_id = (
                slot_to_dia[
                    slot
                ]
            )


            if dia_id not in seen:

                ranking.append(
                    dia_id
                )

                seen.add(
                    dia_id
                )


    # Safety fallback.

    for candidate in candidates:

        dia_id = candidate[
            "dia_id"
        ]


        if dia_id not in seen:

            ranking.append(
                dia_id
            )

            seen.add(
                dia_id
            )


    return ranking


# ============================================================
# EXISTING READER
#
# Use the project's actual trained MemoryReader.
#
# Current project _read_memory() passes hidden states +
# memory slots into self.reader and returns slot_usage.
# ============================================================

@torch.inference_mode()
def existing_reader_ranking(
    memory_state,
    slot_to_dia,
    candidates,
    question_encoding,
):

    read_output = model._read_memory(

        hidden_states=
            question_encoding[
                "hidden"
            ],

        memory_state=
            memory_state,

        attention_mask=
            question_encoding[
                "attention_mask"
            ],

        memory_mask=None,
    )


    slot_usage = (

        read_output
        .slot_usage[
            0
        ]
        .detach()
        .float()
    )


    ranking = slots_to_dia_ranking(

        slot_usage,

        slot_to_dia,

        candidates,
    )


    return (
        ranking,
        slot_usage,
    )


# ============================================================
# COSINE RETRIEVAL
#
# Query = direct final pooled GPT2 summary.
#
# Compare it against whichever memory representation is
# supplied:
#
#   B -> CandidateWriter-created slots
#
#   D -> direct final pooled memories
# ============================================================

@torch.inference_mode()
def cosine_reader_ranking(
    memory_state,
    slot_to_dia,
    candidates,
    question_encoding,
):

    query = (

        question_encoding[
            "summary"
        ][
            0
        ]
        .float()
    )


    query = F.normalize(

        query,

        p=2,

        dim=-1,

        eps=1e-8,
    )


    slots = (

        memory_state
        .slots[
            0
        ]
        .float()
    )


    slots = F.normalize(

        slots,

        p=2,

        dim=-1,

        eps=1e-8,
    )


    scores = (

        slots

        @

        query
    )


    ranking = slots_to_dia_ranking(

        scores,

        slot_to_dia,

        candidates,
    )


    return (
        ranking,
        scores,
    )


# ============================================================
# METRICS
# ============================================================

def retrieval_metrics(
    ranking,
    gold_evidence,
):

    gold = set(
        gold_evidence
    )


    first_gold_rank = None


    for rank, dia_id in enumerate(

        ranking,

        start=1,
    ):

        if dia_id in gold:

            first_gold_rank = rank

            break


    if first_gold_rank is None:

        mrr = 0.0

    else:

        mrr = (

            1.0

            /

            first_gold_rank
        )


    output = {

        "mrr":
            mrr
    }


    for k in TOP_KS:

        retrieved = set(

            ranking[
                :k
            ]
        )


        intersection = (

            retrieved

            &

            gold
        )


        output[
            f"hit@{k}"
        ] = (

            1.0

            if intersection

            else 0.0
        )


        output[
            f"recall@{k}"
        ] = (

            len(
                intersection
            )

            /

            len(
                gold
            )
        )


        output[
            f"all@{k}"
        ] = (

            1.0

            if gold.issubset(
                retrieved
            )

            else 0.0
        )


    return output


# ============================================================
# FOUR METHODS
# ============================================================

METHODS = [

    # A
    "A_OLD_WRITER_OLD_READER",

    # B
    "B_OLD_WRITER_COSINE",

    # C
    "C_DIRECT_MEMORY_OLD_READER",

    # D
    "D_DIRECT_MEMORY_COSINE",
]


def empty_store():

    keys = [
        "mrr",
    ]


    for k in TOP_KS:

        keys.extend(

            [
                f"hit@{k}",
                f"recall@{k}",
                f"all@{k}",
            ]
        )


    return {

        key: []

        for key in keys
    }


results = {

    method:
        empty_store()

    for method in METHODS
}


category_results = defaultdict(

    lambda: {

        method:
            empty_store()

        for method
        in METHODS
    }
)


# ============================================================
# RUN
# ============================================================

section(
    "RUN 2x2 RETRIEVAL ISOLATION"
)


print()

print(
    "A = Existing Writer + Existing Reader"
)

print(
    "B = Existing Writer + Cosine Reader"
)

print(
    "C = Direct Memory + Existing Reader"
)

print(
    "D = Direct Memory + Cosine Reader"
)

print()

print(
    "Memory slots:",
    NUM_SLOTS,
)

print(
    "Questions:",
    len(qa_items),
)

print()

print(
    "NO VALUE DECODER."
)

print(
    "NO WORD CLASSIFICATION."
)

print(
    "NO ANSWER GENERATION."
)

print(
    "ONLY GOLD-EVIDENCE RETRIEVAL."
)


successful = 0

example_printed = False


with torch.inference_mode():

    for index, item in enumerate(

        qa_items,

        start=1,
    ):

        local_rng = random.Random(

            SEED

            +

            index * 7919
        )


        candidates = build_candidate_pool(

            item,

            local_rng,
        )


        if candidates is None:

            continue


        # ====================================================
        # ONLY ONE GPT2 CALL FOR ALL 8 MEMORIES
        # ====================================================

        candidate_encoding = (
            encode_candidate_batch(
                candidates
            )
        )


        # ====================================================
        # ONLY ONE GPT2 CALL FOR QUESTION
        # ====================================================

        question_encoding = (
            encode_question(

                item[
                    "question"
                ]
            )
        )


        # ====================================================
        # BUILD OLD-WRITER MEMORY
        # ====================================================

        (
            old_memory,
            old_slot_map,

        ) = build_existing_writer_memory(

            candidates,

            candidate_encoding,
        )


        # ====================================================
        # BUILD DIRECT MEMORY
        # ====================================================

        (
            direct_memory,
            direct_slot_map,

        ) = build_direct_memory(

            candidates,

            candidate_encoding,
        )


        # ====================================================
        # A
        #
        # OLD WRITER
        # +
        # OLD READER
        # ====================================================

        ranking_a, scores_a = (
            existing_reader_ranking(

                old_memory,

                old_slot_map,

                candidates,

                question_encoding,
            )
        )


        # ====================================================
        # B
        #
        # OLD WRITER
        # +
        # DIRECT COSINE
        # ====================================================

        ranking_b, scores_b = (
            cosine_reader_ranking(

                old_memory,

                old_slot_map,

                candidates,

                question_encoding,
            )
        )


        # ====================================================
        # C
        #
        # DIRECT MEMORY
        # +
        # OLD READER
        # ====================================================

        ranking_c, scores_c = (
            existing_reader_ranking(

                direct_memory,

                direct_slot_map,

                candidates,

                question_encoding,
            )
        )


        # ====================================================
        # D
        #
        # DIRECT MEMORY
        # +
        # DIRECT COSINE
        # ====================================================

        ranking_d, scores_d = (
            cosine_reader_ranking(

                direct_memory,

                direct_slot_map,

                candidates,

                question_encoding,
            )
        )


        rankings = {

            "A_OLD_WRITER_OLD_READER":
                ranking_a,

            "B_OLD_WRITER_COSINE":
                ranking_b,

            "C_DIRECT_MEMORY_OLD_READER":
                ranking_c,

            "D_DIRECT_MEMORY_COSINE":
                ranking_d,
        }


        gold = item[
            "evidence"
        ]


        category = item[
            "category"
        ]


        # ====================================================
        # SCORE
        # ====================================================

        for method, ranking in (
            rankings.items()
        ):

            metric = retrieval_metrics(

                ranking,

                gold,
            )


            for (
                metric_name,
                value,
            ) in metric.items():

                results[
                    method
                ][
                    metric_name
                ].append(
                    value
                )


                category_results[
                    category
                ][
                    method
                ][
                    metric_name
                ].append(
                    value
                )


        successful += 1


        # ====================================================
        # PRINT FIRST EXAMPLE
        # ====================================================

        if not example_printed:

            example_printed = True


            print()

            print(
                "=" * 100
            )

            print(
                "EXAMPLE"
            )

            print(
                "=" * 100
            )


            print()

            print(
                "QUESTION:"
            )

            print(
                item[
                    "question"
                ]
            )


            print()

            print(
                "GOLD EVIDENCE:",
                gold,
            )


            print()

            print(
                "MEMORY CANDIDATES:"
            )


            for candidate_index, candidate in (
                enumerate(
                    candidates
                )
            ):

                flag = (

                    " <--- GOLD"

                    if (
                        candidate[
                            "dia_id"
                        ]
                        in
                        set(gold)
                    )

                    else ""
                )


                print(

                    f"{candidate_index}. "
                    f"{candidate['dia_id']}"
                    f"{flag}"
                )

                print(
                    candidate[
                        "text"
                    ]
                )

                print()


            print(
                "RANKINGS:"
            )


            for method in METHODS:

                print()

                print(
                    method
                )


                for rank, dia_id in enumerate(

                    rankings[
                        method
                    ],

                    start=1,
                ):

                    flag = (

                        " <--- GOLD"

                        if dia_id
                        in set(gold)

                        else ""
                    )


                    print(

                        f"  {rank}. "
                        f"{dia_id}"
                        f"{flag}"
                    )


        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            index % 25 == 0
            or
            index == len(
                qa_items
            )
        ):

            print(

                f"Completed "
                f"{index}/"
                f"{len(qa_items)}"
            )


# ============================================================
# AGGREGATION
# ============================================================

def mean_metric(values):

    if not values:

        return 0.0


    return float(

        np.mean(
            values
        )
    )


summary = {}


for method in METHODS:

    summary[
        method
    ] = {

        metric:
            mean_metric(
                values
            )

        for metric, values
        in results[
            method
        ].items()
    }


# ============================================================
# FINAL TABLE
# ============================================================

section(
    "FINAL 2x2 RETRIEVAL RESULTS"
)


print(
    "Questions:",
    successful,
)

print()


print(

    f"{'Method':<31}"

    f"{'Hit@1':>10}"

    f"{'Hit@3':>10}"

    f"{'Hit@5':>10}"

    f"{'Recall@3':>12}"

    f"{'Recall@5':>12}"

    f"{'MRR':>10}"
)


for method in METHODS:

    result = summary[
        method
    ]


    print(

        f"{method:<31}"

        f"{result['hit@1'] * 100:>9.2f}%"

        f"{result['hit@3'] * 100:>9.2f}%"

        f"{result['hit@5'] * 100:>9.2f}%"

        f"{result['recall@3'] * 100:>11.2f}%"

        f"{result['recall@5'] * 100:>11.2f}%"

        f"{result['mrr']:>10.4f}"
    )


# ============================================================
# 2x2 MATRIX
# ============================================================

section(
    "2x2 HIT@1 MATRIX"
)


A = summary[
    "A_OLD_WRITER_OLD_READER"
][
    "hit@1"
]


B = summary[
    "B_OLD_WRITER_COSINE"
][
    "hit@1"
]


C = summary[
    "C_DIRECT_MEMORY_OLD_READER"
][
    "hit@1"
]


D = summary[
    "D_DIRECT_MEMORY_COSINE"
][
    "hit@1"
]


print()

print(
    "                         READER"
)

print(
    "                 OLD             COSINE"
)

print()

print(

    f"OLD WRITER       "
    f"{A * 100:6.2f}%"
    f"          "
    f"{B * 100:6.2f}%"
)

print()

print(

    f"DIRECT MEMORY    "
    f"{C * 100:6.2f}%"
    f"          "
    f"{D * 100:6.2f}%"
)


# ============================================================
# COMPONENT EFFECTS
# ============================================================

section(
    "COMPONENT EFFECTS"
)


# ------------------------------------------------------------
# Effect of changing ONLY READER
#
# same old-written memories:
#
# A -> B
# ------------------------------------------------------------

reader_gain_on_old = (
    B - A
)


# ------------------------------------------------------------
# Effect of changing ONLY WRITER
#
# same old reader:
#
# A -> C
# ------------------------------------------------------------

writer_gain_with_old_reader = (
    C - A
)


# ------------------------------------------------------------
# Reader effect when memory is clean:
#
# C -> D
# ------------------------------------------------------------

reader_gain_on_direct = (
    D - C
)


# ------------------------------------------------------------
# Writer effect when reader is cosine:
#
# B -> D
# ------------------------------------------------------------

writer_gain_with_cosine = (
    D - B
)


print(
    "Change ONLY reader "
    "(Old Writer: A -> B):"
)

print(
    f"{reader_gain_on_old * 100:+.2f} points"
)


print()

print(
    "Change ONLY writer "
    "(Old Reader: A -> C):"
)

print(
    f"{writer_gain_with_old_reader * 100:+.2f} points"
)


print()

print(
    "Change reader with DIRECT memory "
    "(C -> D):"
)

print(
    f"{reader_gain_on_direct * 100:+.2f} points"
)


print()

print(
    "Change writer with COSINE reader "
    "(B -> D):"
)

print(
    f"{writer_gain_with_cosine * 100:+.2f} points"
)


# ============================================================
# AUTOMATIC DIAGNOSIS
# ============================================================

section(
    "AUTOMATIC DIAGNOSIS"
)


print(
    f"A OLD WRITER + OLD READER   : "
    f"{A * 100:.2f}%"
)

print(
    f"B OLD WRITER + COSINE       : "
    f"{B * 100:.2f}%"
)

print(
    f"C DIRECT MEMORY + OLD READER: "
    f"{C * 100:.2f}%"
)

print(
    f"D DIRECT MEMORY + COSINE    : "
    f"{D * 100:.2f}%"
)

print()


reader_effect = max(

    reader_gain_on_old,

    reader_gain_on_direct,
)


writer_effect = max(

    writer_gain_with_old_reader,

    writer_gain_with_cosine,
)


if (
    reader_effect > 0.08

    and

    writer_effect <= 0.05
):

    print(
        "PRIMARY DIAGNOSIS: READER BOTTLENECK"
    )

    print()

    print(
        "Changing the reader gives the major "
        "retrieval improvement."
    )


elif (
    writer_effect > 0.08

    and

    reader_effect <= 0.05
):

    print(
        "PRIMARY DIAGNOSIS: WRITER BOTTLENECK"
    )

    print()

    print(
        "Replacing the written memory representation "
        "gives the major retrieval improvement."
    )


elif (
    reader_effect > 0.05

    and

    writer_effect > 0.05
):

    print(
        "PRIMARY DIAGNOSIS: BOTH WRITER AND READER"
    )

    print()

    print(
        "Both components independently reduce "
        "retrieval quality."
    )


else:

    print(
        "PRIMARY DIAGNOSIS: INTERACTION / "
        "REPRESENTATION MISMATCH"
    )

    print()

    print(
        "Neither component alone explains the "
        "entire retrieval gap."
    )


# ============================================================
# CATEGORY BREAKDOWN
# ============================================================

section(
    "CATEGORY HIT@1 BREAKDOWN"
)


for category in sorted(

    category_results.keys()
):

    print()

    count = len(

        category_results[
            category
        ][
            METHODS[
                0
            ]
        ][
            "hit@1"
        ]
    )


    print(
        f"Category {category} "
        f"(n={count})"
    )


    for method in METHODS:

        value = mean_metric(

            category_results[
                category
            ][
                method
            ][
                "hit@1"
            ]
        )


        print(

            f"  {method:<31}"

            f"{value * 100:>7.2f}%"
        )


# ============================================================
# SAVE
# ============================================================

section(
    "SAVE RESULTS"
)


output_path = Path(
    args.output
)


output_path.parent.mkdir(

    parents=True,

    exist_ok=True,
)


torch.save(

    {

        "questions":
            successful,

        "methods":
            METHODS,

        "summary":
            summary,

        "A":
            A,

        "B":
            B,

        "C":
            C,

        "D":
            D,

        "reader_gain_on_old":
            reader_gain_on_old,

        "writer_gain_with_old_reader":
            writer_gain_with_old_reader,

        "reader_gain_on_direct":
            reader_gain_on_direct,

        "writer_gain_with_cosine":
            writer_gain_with_cosine,

        "raw_results":
            results,
    },

    output_path,
)


print(
    "Saved:",
    output_path
)


# ============================================================
# DONE
# ============================================================

section(
    "EXPERIMENT COMPLETE"
)


print(
    "This was a RETRIEVAL-ONLY experiment."
)

print(
    "No value decoder."
)

print(
    "No answer classifier."
)

print(
    "No GPT-2 training."
)

print(
    "No CandidateWriter training."
)

print(
    "No MemoryReader training."
)

print(
    "No models/ source files modified."
)