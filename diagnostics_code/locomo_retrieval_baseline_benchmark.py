# ============================================================
# LOCOMO RETRIEVAL BASELINE BENCHMARK
#
# Purpose:
#   Establish a strong RETRIEVAL ceiling before touching the
#   memory architecture again.
#
# Compared methods:
#
#   1. RAW GPT2 FINAL-MEAN COSINE
#   2. PREVIOUS TRAINED DUAL ENCODER
#   3. BM25
#   4. MiniLM
#   5. BGE-small
#   6. E5-base
#
#
# Evaluation:
#
#   A. SAME HELD-OUT TEST CONVERSATIONS
#      Uses the same 60/20/20 split and seed as the previous
#      training experiment.
#
#   B. ALL LOCOMO
#      Zero-shot robustness check across all conversations.
#
#
# Stability:
#
#   Multiple independently sampled 8-slot pools are created
#   for EACH question.
#
#   Default:
#       5 pools per question.
#
#
# Also evaluates FULL-CONVERSATION retrieval.
#
#
# IMPORTANT:
#
# NO MODEL TRAINING.
# NO CandidateWriter modification.
# NO MemoryReader modification.
# NO models/ modification.
#
#
# Run:
#
# python locomo_retrieval_baseline_benchmark.py \
#   --data data/locomo/locomo10.json \
#   --trials-per-question 5 \
#   2>&1 | tee locomo_retrieval_baselines.log
#
# ============================================================


import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

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
    "--dual-checkpoint",
    type=str,
    default=(
        "outputs/"
        "locomo_learned_retriever.pt"
    ),
)


parser.add_argument(
    "--trials-per-question",
    type=int,
    default=5,
)


parser.add_argument(
    "--pool-size",
    type=int,
    default=8,
)


parser.add_argument(
    "--seed",
    type=int,
    default=2090,
)


parser.add_argument(
    "--gpt-batch-size",
    type=int,
    default=64,
)


parser.add_argument(
    "--embed-batch-size",
    type=int,
    default=128,
)


parser.add_argument(
    "--max-memory-tokens",
    type=int,
    default=128,
)


parser.add_argument(
    "--max-question-tokens",
    type=int,
    default=96,
)


parser.add_argument(
    "--output",
    type=str,
    default=(
        "outputs/"
        "locomo_retrieval_baselines.pt"
    ),
)


args = parser.parse_args()


# ============================================================
# GLOBAL
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


MODEL_NAME = "gpt2"


TOP_KS = [
    1,
    3,
    5,
]


METHODS = [
    "GPT2_RAW_COSINE",
    "TRAINED_DUAL",
    "BM25",
    "MINILM",
    "BGE_SMALL",
    "E5_BASE",
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


set_seed(
    args.seed
)


def section(title):

    print()
    print("=" * 125)
    print(title)
    print("=" * 125)


# ============================================================
# GPT2 CONFIG
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

        reader_mode="hybrid",

        reader_fusion="gated",

        reader_heads=8,

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
    "LOAD GPT-2"
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = (
    MemoryAugmentedGPT2LMHeadModel
    .from_pretrained(
        MODEL_NAME,
        memory_config=build_config(),
    )
)


checkpoint = torch.load(
    args.checkpoint,
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


model.to(
    DEVICE
)

model.eval()


for parameter in model.parameters():
    parameter.requires_grad = False


GPT_DIM = model.backbone.config.n_embd


print(
    "Device:",
    DEVICE,
)

print(
    "GPT dim:",
    GPT_DIM,
)

print(
    "Compatible tensors:",
    len(compatible),
)

print(
    "Missing:",
    len(load_result.missing_keys),
)


# ============================================================
# PREVIOUS FAILED DUAL ENCODER
# ============================================================

class RetrievalEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout=0.0,
    ):

        super().__init__()

        self.network = nn.Sequential(

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

        x = self.network(
            x
        )

        return F.normalize(
            x,
            p=2,
            dim=-1,
            eps=1e-8,
        )


dual_available = False

dual_checkpoint_path = Path(
    args.dual_checkpoint
)


if dual_checkpoint_path.exists():

    section(
        "LOAD PREVIOUS TRAINED DUAL ENCODER"
    )


    dual_checkpoint = torch.load(
        dual_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


    hidden_dim = dual_checkpoint.get(
        "hidden_dim",
        512,
    )


    embedding_dim = dual_checkpoint.get(
        "embedding_dim",
        256,
    )


    query_encoder = RetrievalEncoder(

        GPT_DIM,

        hidden_dim,

        embedding_dim,

        0.0,

    ).to(
        DEVICE
    )


    memory_encoder = RetrievalEncoder(

        GPT_DIM,

        hidden_dim,

        embedding_dim,

        0.0,

    ).to(
        DEVICE
    )


    query_encoder.load_state_dict(

        dual_checkpoint[
            "query_encoder_state_dict"
        ]
    )


    memory_encoder.load_state_dict(

        dual_checkpoint[
            "memory_encoder_state_dict"
        ]
    )


    query_encoder.eval()

    memory_encoder.eval()


    for parameter in query_encoder.parameters():
        parameter.requires_grad = False

    for parameter in memory_encoder.parameters():
        parameter.requires_grad = False


    dual_available = True


    print(
        "Loaded:",
        args.dual_checkpoint,
    )

else:

    print(
        "WARNING: trained dual checkpoint not found."
    )


# ============================================================
# PRETRAINED SEMANTIC RETRIEVERS
# ============================================================

section(
    "LOAD PRETRAINED RETRIEVERS"
)


print(
    "Loading MiniLM..."
)


minilm = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=str(DEVICE),
)


print(
    "Loading BGE-small..."
)


bge = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    device=str(DEVICE),
)


print(
    "Loading E5-base..."
)


e5 = SentenceTransformer(
    "intfloat/e5-base-v2",
    device=str(DEVICE),
)


print(
    "Semantic retrievers loaded."
)


# ============================================================
# LOAD LOCOMO
# ============================================================

section(
    "LOAD LOCOMO"
)


with open(
    args.data,
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
# EVIDENCE PARSER
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


    result = []


    def visit(item):

        if isinstance(
            item,
            str,
        ):

            item = item.strip()

            if item:
                result.append(
                    item
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


    for item in result:

        if item not in seen:

            seen.add(
                item
            )

            final.append(
                item
            )


    return final


# ============================================================
# CONVERSATION PARSER
#
# IMPORTANT:
#
# TURN ONLY.
#
# NO radius=1 windows.
# ============================================================

def flatten_conversation(
    sample,
):

    conversation = sample.get(
        "conversation",
        {},
    )


    session_keys = []


    for key in conversation.keys():

        match = re.fullmatch(
            r"session_(\d+)",
            key,
        )


        if match:

            session_keys.append(
                (
                    int(match.group(1)),
                    key,
                )
            )


    session_keys.sort()


    turns = []


    for (
        session_number,
        session_key,
    ) in session_keys:

        for turn in conversation.get(
            session_key,
            [],
        ):

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

                formatted = (
                    f"{speaker}: {text}"
                )

            else:

                formatted = text


            turns.append(
                {
                    "dia_id":
                        dia_id,

                    "text":
                        formatted,

                    "session":
                        session_number,
                }
            )


    return turns


# ============================================================
# BUILD DATASET
# ============================================================

section(
    "PARSE DATASET"
)


conversations = []


for conversation_index, sample in enumerate(
    locomo
):

    turns = flatten_conversation(
        sample
    )


    if not turns:
        continue


    turn_by_id = {

        turn["dia_id"]:
            index

        for index, turn
        in enumerate(turns)
    }


    qas = []


    for qa_index, qa in enumerate(
        sample.get(
            "qa",
            [],
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


        if not all(
            evidence_id in turn_by_id
            for evidence_id in evidence
        ):
            continue


        if len(evidence) > args.pool_size:
            continue


        qas.append(
            {
                "qa_index":
                    qa_index,

                "question":
                    question,

                "evidence":
                    evidence,

                "category":
                    str(
                        qa.get(
                            "category",
                            "unknown",
                        )
                    ),
            }
        )


    sample_id = str(
        sample.get(
            "sample_id",
            conversation_index,
        )
    )


    conversations.append(
        {
            "sample_id":
                sample_id,

            "turns":
                turns,

            "turn_by_id":
                turn_by_id,

            "qas":
                qas,
        }
    )


    print(
        f"{sample_id:<12}"
        f" turns={len(turns):<5}"
        f" qa={len(qas)}"
    )


# ============================================================
# SAME SPLIT AS PREVIOUS TRAINING SCRIPT
# ============================================================

section(
    "REPRODUCE PREVIOUS SPLIT"
)


indices = list(
    range(
        len(conversations)
    )
)


split_rng = random.Random(
    args.seed
)


split_rng.shuffle(
    indices
)


n = len(
    indices
)


train_end = int(
    n * 0.60
)


valid_end = int(
    n * 0.80
)


train_ids = indices[
    :train_end
]


valid_ids = indices[
    train_end:
    valid_end
]


test_ids = indices[
    valid_end:
]


print(
    "TRAIN:",
    [
        conversations[i]["sample_id"]
        for i in train_ids
    ],
)


print(
    "VALID:",
    [
        conversations[i]["sample_id"]
        for i in valid_ids
    ],
)


print(
    "TEST:",
    [
        conversations[i]["sample_id"]
        for i in test_ids
    ],
)


# ============================================================
# GPT2 MEAN POOL
# ============================================================

def masked_mean(
    hidden,
    attention_mask,
):

    mask = (
        attention_mask
        .unsqueeze(-1)
        .to(hidden.dtype)
    )


    return (

        (hidden * mask).sum(
            dim=1
        )

        /

        mask.sum(
            dim=1
        ).clamp_min(
            1.0
        )
    )


# ============================================================
# GPT2 BATCH ENCODER
# ============================================================

@torch.inference_mode()
def gpt_encode(
    texts,
    max_tokens,
):

    vectors = []


    for start in range(
        0,
        len(texts),
        args.gpt_batch_size,
    ):

        batch = texts[
            start:
            start
            +
            args.gpt_batch_size
        ]


        encoded = tokenizer(

            batch,

            padding=True,

            truncation=True,

            max_length=max_tokens,

            return_tensors="pt",
        )


        ids = encoded[
            "input_ids"
        ].to(
            DEVICE
        )


        mask = encoded[
            "attention_mask"
        ].to(
            DEVICE
        )


        output = model.backbone.transformer(

            input_ids=ids,

            attention_mask=mask,

            output_hidden_states=False,

            use_cache=False,

            return_dict=True,
        )


        pooled = masked_mean(

            output.last_hidden_state,

            mask,
        )


        vectors.append(
            pooled
            .float()
            .cpu()
        )


    return torch.cat(
        vectors,
        dim=0,
    )


# ============================================================
# SEMANTIC EMBEDDING FUNCTIONS
# ============================================================

def encode_minilm(
    texts,
):

    values = minilm.encode(

        texts,

        batch_size=
            args.embed_batch_size,

        convert_to_numpy=True,

        normalize_embeddings=True,

        show_progress_bar=False,
    )


    return torch.tensor(
        values,
        dtype=torch.float32,
    )


def encode_bge(
    texts,
):

    values = bge.encode(

        texts,

        batch_size=
            args.embed_batch_size,

        convert_to_numpy=True,

        normalize_embeddings=True,

        show_progress_bar=False,
    )


    return torch.tensor(
        values,
        dtype=torch.float32,
    )


def encode_e5_memory(
    texts,
):

    texts = [
        "passage: " + text
        for text in texts
    ]


    values = e5.encode(

        texts,

        batch_size=
            args.embed_batch_size,

        convert_to_numpy=True,

        normalize_embeddings=True,

        show_progress_bar=False,
    )


    return torch.tensor(
        values,
        dtype=torch.float32,
    )


def encode_e5_query(
    texts,
):

    texts = [
        "query: " + text
        for text in texts
    ]


    values = e5.encode(

        texts,

        batch_size=
            args.embed_batch_size,

        convert_to_numpy=True,

        normalize_embeddings=True,

        show_progress_bar=False,
    )


    return torch.tensor(
        values,
        dtype=torch.float32,
    )


# ============================================================
# PRECOMPUTE EVERYTHING ONCE
# ============================================================

section(
    "PRECOMPUTE REPRESENTATIONS"
)


for conversation in conversations:

    print()

    print(
        conversation[
            "sample_id"
        ]
    )


    memory_texts = [

        turn["text"]

        for turn
        in conversation[
            "turns"
        ]
    ]


    question_texts = [

        qa["question"]

        for qa
        in conversation[
            "qas"
        ]
    ]


    # --------------------------------------------------------
    # GPT2
    # --------------------------------------------------------

    print(
        "  GPT2 memories..."
    )


    conversation[
        "gpt_memory"
    ] = gpt_encode(

        memory_texts,

        args.max_memory_tokens,
    )


    print(
        "  GPT2 questions..."
    )


    conversation[
        "gpt_query"
    ] = gpt_encode(

        question_texts,

        args.max_question_tokens,
    )


    # --------------------------------------------------------
    # MiniLM
    # --------------------------------------------------------

    print(
        "  MiniLM..."
    )


    conversation[
        "minilm_memory"
    ] = encode_minilm(
        memory_texts
    )


    conversation[
        "minilm_query"
    ] = encode_minilm(
        question_texts
    )


    # --------------------------------------------------------
    # BGE
    # --------------------------------------------------------

    print(
        "  BGE..."
    )


    conversation[
        "bge_memory"
    ] = encode_bge(
        memory_texts
    )


    conversation[
        "bge_query"
    ] = encode_bge(
        question_texts
    )


    # --------------------------------------------------------
    # E5
    # --------------------------------------------------------

    print(
        "  E5..."
    )


    conversation[
        "e5_memory"
    ] = encode_e5_memory(
        memory_texts
    )


    conversation[
        "e5_query"
    ] = encode_e5_query(
        question_texts
    )


    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    tokenized_memory = [

        re.findall(
            r"\b\w+\b",
            text.lower(),
        )

        for text
        in memory_texts
    ]


    conversation[
        "bm25"
    ] = BM25Okapi(
        tokenized_memory
    )


print()

print(
    "All representations cached."
)


# ============================================================
# DUAL REPRESENTATIONS
# ============================================================

if dual_available:

    section(
        "PRECOMPUTE PREVIOUS DUAL ENCODER"
    )


    with torch.inference_mode():

        for conversation in conversations:

            gpt_memory = conversation[
                "gpt_memory"
            ].to(
                DEVICE
            )


            gpt_query = conversation[
                "gpt_query"
            ].to(
                DEVICE
            )


            dual_memory_parts = []


            for start in range(
                0,
                len(gpt_memory),
                256,
            ):

                value = memory_encoder(

                    gpt_memory[
                        start:
                        start + 256
                    ]
                )

                dual_memory_parts.append(
                    value.cpu()
                )


            conversation[
                "dual_memory"
            ] = torch.cat(
                dual_memory_parts,
                dim=0,
            )


            conversation[
                "dual_query"
            ] = query_encoder(
                gpt_query
            ).cpu()


# ============================================================
# METRICS
# ============================================================

def init_metric_store():

    result = {
        "mrr": [],
    }


    for k in TOP_KS:

        result[
            f"hit@{k}"
        ] = []

        result[
            f"recall@{k}"
        ] = []


    return result


def score_ranking(
    ranking,
    gold_indices,
):

    gold = set(
        gold_indices
    )


    first_gold = None


    for rank, item in enumerate(
        ranking,
        start=1,
    ):

        if item in gold:

            first_gold = rank

            break


    if first_gold is None:

        mrr = 0.0

    else:

        mrr = (
            1.0
            /
            first_gold
        )


    result = {
        "mrr":
            mrr
    }


    for k in TOP_KS:

        retrieved = set(
            ranking[
                :k
            ]
        )


        overlap = (
            retrieved
            &
            gold
        )


        result[
            f"hit@{k}"
        ] = (
            1.0
            if overlap
            else 0.0
        )


        result[
            f"recall@{k}"
        ] = (

            len(
                overlap
            )

            /

            len(
                gold
            )
        )


    return result


def append_metrics(
    store,
    metrics,
):

    for key, value in metrics.items():

        store[
            key
        ].append(
            value
        )


def summarize(
    store,
):

    return {

        key:
            float(
                np.mean(values)
            )
            if values
            else 0.0

        for key, values
        in store.items()
    }


# ============================================================
# VECTOR RANKING
# ============================================================

def cosine_ranking(
    query,
    memories,
    candidate_indices,
):

    query = F.normalize(

        query.float(),

        p=2,

        dim=-1,
    )


    selected = memories[
        candidate_indices
    ]


    selected = F.normalize(

        selected.float(),

        p=2,

        dim=-1,
    )


    scores = (
        selected
        @
        query
    )


    order = torch.argsort(
        scores,
        descending=True,
    )


    return [

        candidate_indices[
            int(index.item())
        ]

        for index
        in order
    ]


# ============================================================
# BM25 RANKING
# ============================================================

def bm25_ranking(
    conversation,
    question,
    candidate_indices,
):

    query_tokens = re.findall(

        r"\b\w+\b",

        question.lower(),
    )


    scores = conversation[
        "bm25"
    ].get_scores(
        query_tokens
    )


    selected_scores = [

        (
            index,
            scores[index],
        )

        for index
        in candidate_indices
    ]


    selected_scores.sort(

        key=lambda x:
            x[1],

        reverse=True,
    )


    return [

        index

        for index, _
        in selected_scores
    ]


# ============================================================
# RETRIEVE USING A METHOD
# ============================================================

def rank_method(
    method,
    conversation,
    qa_position,
    candidate_indices,
):

    qa = conversation[
        "qas"
    ][
        qa_position
    ]


    if method == "GPT2_RAW_COSINE":

        return cosine_ranking(

            conversation[
                "gpt_query"
            ][
                qa_position
            ],

            conversation[
                "gpt_memory"
            ],

            candidate_indices,
        )


    if method == "TRAINED_DUAL":

        if not dual_available:

            return None


        return cosine_ranking(

            conversation[
                "dual_query"
            ][
                qa_position
            ],

            conversation[
                "dual_memory"
            ],

            candidate_indices,
        )


    if method == "MINILM":

        return cosine_ranking(

            conversation[
                "minilm_query"
            ][
                qa_position
            ],

            conversation[
                "minilm_memory"
            ],

            candidate_indices,
        )


    if method == "BGE_SMALL":

        return cosine_ranking(

            conversation[
                "bge_query"
            ][
                qa_position
            ],

            conversation[
                "bge_memory"
            ],

            candidate_indices,
        )


    if method == "E5_BASE":

        return cosine_ranking(

            conversation[
                "e5_query"
            ][
                qa_position
            ],

            conversation[
                "e5_memory"
            ],

            candidate_indices,
        )


    if method == "BM25":

        return bm25_ranking(

            conversation,

            qa[
                "question"
            ],

            candidate_indices,
        )


    raise RuntimeError(
        method
    )


# ============================================================
# CONTROLLED 8-WAY EVALUATION
#
# Multiple pools per question.
# ============================================================

def evaluate_controlled(
    conversation_ids,
    trials_per_question,
):

    stores = {

        method:
            init_metric_store()

        for method
        in METHODS

        if (
            method != "TRAINED_DUAL"
            or
            dual_available
        )
    }


    question_count = 0

    trial_count = 0


    for conversation_id in conversation_ids:

        conversation = conversations[
            conversation_id
        ]


        for qa_position, qa in enumerate(

            conversation[
                "qas"
            ]
        ):

            gold_indices = [

                conversation[
                    "turn_by_id"
                ][
                    evidence_id
                ]

                for evidence_id
                in qa[
                    "evidence"
                ]
            ]


            if (
                len(
                    gold_indices
                )
                >
                args.pool_size
            ):
                continue


            gold_set = set(
                gold_indices
            )


            negatives = [

                i

                for i in range(
                    len(
                        conversation[
                            "turns"
                        ]
                    )
                )

                if i not in gold_set
            ]


            needed = (

                args.pool_size

                -

                len(
                    gold_indices
                )
            )


            if len(negatives) < needed:

                continue


            question_count += 1


            for trial in range(
                trials_per_question
            ):

                # Stable but different candidate pool.

                trial_seed = (

                    args.seed
                    +
                    conversation_id * 1_000_003
                    +
                    qa_position * 10_007
                    +
                    trial * 97
                )


                rng = random.Random(
                    trial_seed
                )


                distractors = rng.sample(

                    negatives,

                    needed,
                )


                candidate_indices = (

                    gold_indices

                    +

                    distractors
                )


                rng.shuffle(
                    candidate_indices
                )


                for method in stores.keys():

                    ranking = rank_method(

                        method,

                        conversation,

                        qa_position,

                        candidate_indices,
                    )


                    metrics = score_ranking(

                        ranking,

                        gold_indices,
                    )


                    append_metrics(

                        stores[
                            method
                        ],

                        metrics,
                    )


                trial_count += 1


    summaries = {

        method:
            summarize(store)

        for method, store
        in stores.items()
    }


    return (

        summaries,

        question_count,

        trial_count,
    )


# ============================================================
# FULL CONVERSATION
# ============================================================

def evaluate_full(
    conversation_ids,
):

    stores = {

        method:
            init_metric_store()

        for method
        in METHODS

        if (
            method != "TRAINED_DUAL"
            or
            dual_available
        )
    }


    question_count = 0


    for conversation_id in conversation_ids:

        conversation = conversations[
            conversation_id
        ]


        all_candidates = list(

            range(
                len(
                    conversation[
                        "turns"
                    ]
                )
            )
        )


        for qa_position, qa in enumerate(

            conversation[
                "qas"
            ]
        ):

            gold_indices = [

                conversation[
                    "turn_by_id"
                ][
                    evidence_id
                ]

                for evidence_id
                in qa[
                    "evidence"
                ]
            ]


            question_count += 1


            for method in stores.keys():

                ranking = rank_method(

                    method,

                    conversation,

                    qa_position,

                    all_candidates,
                )


                metrics = score_ranking(

                    ranking,

                    gold_indices,
                )


                append_metrics(

                    stores[
                        method
                    ],

                    metrics,
                )


    summaries = {

        method:
            summarize(store)

        for method, store
        in stores.items()
    }


    return (
        summaries,
        question_count,
    )


# ============================================================
# DISPLAY
# ============================================================

def print_table(
    title,
    summaries,
):

    section(
        title
    )


    print(

        f"{'Method':<24}"

        f"{'Hit@1':>10}"

        f"{'Hit@3':>10}"

        f"{'Hit@5':>10}"

        f"{'Recall@5':>12}"

        f"{'MRR':>10}"
    )


    print(
        "-" * 76
    )


    ranking = sorted(

        summaries.items(),

        key=lambda x:
            x[1][
                "hit@1"
            ],

        reverse=True,
    )


    for method, result in ranking:

        print(

            f"{method:<24}"

            f"{result['hit@1'] * 100:>9.2f}%"

            f"{result['hit@3'] * 100:>9.2f}%"

            f"{result['hit@5'] * 100:>9.2f}%"

            f"{result['recall@5'] * 100:>11.2f}%"

            f"{result['mrr']:>10.4f}"
        )


# ============================================================
# 1. SAME HELD-OUT TEST
# ============================================================

section(
    "EVALUATE SAME HELD-OUT TEST"
)


(
    test_controlled,
    test_questions,
    test_trials,

) = evaluate_controlled(

    test_ids,

    args.trials_per_question,
)


print(
    "Held-out questions:",
    test_questions,
)

print(
    "8-way retrieval trials:",
    test_trials,
)


print_table(

    "HELD-OUT TEST — CONTROLLED 8-SLOT",

    test_controlled,
)


# ============================================================
# HELD-OUT FULL
# ============================================================

(
    test_full,
    test_full_questions,

) = evaluate_full(
    test_ids
)


print_table(

    "HELD-OUT TEST — FULL CONVERSATION",

    test_full,
)


# ============================================================
# 2. ALL LOCOMO
#
# Useful because these semantic encoders are ZERO-SHOT.
#
# This gives us a much larger sample.
# ============================================================

section(
    "EVALUATE ALL LOCOMO"
)


all_ids = list(

    range(
        len(
            conversations
        )
    )
)


(
    all_controlled,
    all_questions,
    all_trials,

) = evaluate_controlled(

    all_ids,

    args.trials_per_question,
)


print(
    "All eligible questions:",
    all_questions,
)

print(
    "Total 8-way trials:",
    all_trials,
)


print_table(

    "ALL LOCOMO — CONTROLLED 8-SLOT",

    all_controlled,
)


# ============================================================
# ALL FULL-CONVERSATION
# ============================================================

(
    all_full,
    all_full_questions,

) = evaluate_full(
    all_ids
)


print_table(

    "ALL LOCOMO — FULL CONVERSATION",

    all_full,
)


# ============================================================
# COMPARE AGAINST CURRENT GPT2 BASELINE
# ============================================================

section(
    "KEY RESULT"
)


baseline = test_controlled[
    "GPT2_RAW_COSINE"
]


best_name = max(

    [

        name

        for name in test_controlled.keys()

        if name
        not in {
            "GPT2_RAW_COSINE",
            "TRAINED_DUAL",
        }
    ],

    key=lambda name:
        test_controlled[
            name
        ][
            "hit@1"
        ],
)


best = test_controlled[
    best_name
]


print(
    "Current GPT-2 retrieval:"
)

print(
    f"  Hit@1 = "
    f"{baseline['hit@1'] * 100:.2f}%"
)

print(
    f"  MRR   = "
    f"{baseline['mrr']:.4f}"
)


print()

print(
    "Best pretrained retrieval baseline:"
)

print(
    " ",
    best_name,
)

print(
    f"  Hit@1 = "
    f"{best['hit@1'] * 100:.2f}%"
)

print(
    f"  MRR   = "
    f"{best['mrr']:.4f}"
)


gain = (

    best[
        "hit@1"
    ]

    -

    baseline[
        "hit@1"
    ]
)


print()

print(
    "Absolute Hit@1 gain:",
    f"{gain * 100:+.2f} points"
)


# ============================================================
# DIAGNOSIS
# ============================================================

section(
    "AUTOMATIC DIAGNOSIS"
)


if (
    best[
        "hit@1"
    ]
    >=
    0.75
):

    print(
        "STRONG RESULT"
    )

    print()

    print(
        "A pretrained semantic retrieval space "
        "solves most of the controlled retrieval task."
    )

    print()

    print(
        "This strongly supports separating the "
        "memory KEY encoder from GPT-2 VALUE storage."
    )


elif (
    best[
        "hit@1"
    ]
    >=
    0.60
):

    print(
        "GOOD RESULT"
    )

    print()

    print(
        "Semantic retrieval substantially improves "
        "over GPT-2 cosine, although further "
        "fine-tuning/reranking may still be needed."
    )


elif (
    best[
        "hit@1"
    ]
    >=
    baseline[
        "hit@1"
    ]
    +
    0.10
):

    print(
        "MEANINGFUL IMPROVEMENT"
    )

    print()

    print(
        "Retrieval-key quality is a major part "
        "of the current bottleneck."
    )


else:

    print(
        "NO LARGE SEMANTIC-ENCODER ADVANTAGE"
    )

    print()

    print(
        "Do not redesign the architecture yet."
    )

    print(
        "Inspect evidence granularity and "
        "candidate-pool construction next."
    )


# ============================================================
# SAVE
# ============================================================

section(
    "SAVE"
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
        "seed":
            args.seed,

        "trials_per_question":
            args.trials_per_question,

        "pool_size":
            args.pool_size,

        "train_conversations":
            [
                conversations[i]["sample_id"]
                for i in train_ids
            ],

        "valid_conversations":
            [
                conversations[i]["sample_id"]
                for i in valid_ids
            ],

        "test_conversations":
            [
                conversations[i]["sample_id"]
                for i in test_ids
            ],

        "test_controlled":
            test_controlled,

        "test_full":
            test_full,

        "all_controlled":
            all_controlled,

        "all_full":
            all_full,
    },

    output_path,
)


print(
    "Saved:",
    output_path
)


section(
    "EXPERIMENT COMPLETE"
)


print(
    "No model was trained."
)

print(
    "No memory architecture was modified."
)

print(
    "No models/ files were modified."
)

print(
    "This experiment only compares retrieval methods."
)