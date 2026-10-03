# ============================================================
# locomo_fullconv_improvements.py
#
# LONG-RANGE MEMORY RETRIEVAL BENCHMARK FOR MEMATTN
#
# ============================================================
#
# GOAL
# ------------------------------------------------------------
# We already established:
#
#   Controlled 8-slot retrieval:
#       E5-base ~79% Hit@1 on held-out LoCoMo conversations
#
#   Full-conversation retrieval:
#       E5-base ~28% Hit@1
#
# So the question is no longer:
#
#       "Can semantic retrieval work?"
#
# It CAN.
#
# The question now is:
#
#       "Why does retrieval degrade as the memory gets larger,
#        and how can we improve long-range retrieval?"
#
#
# THIS SCRIPT TESTS
# ============================================================
#
# 1. MEMORY-SIZE SCALING
#
#       8
#       16
#       32
#       64
#       128
#       256
#       FULL conversation
#
#
# 2. MEMORY SEGMENTATION
#
#       TURN
#       TURN + DATE
#       PREVIOUS + CURRENT   [SESSION SAFE]
#
#
# 3. RETRIEVAL METHODS
#
#       E5
#       BGE
#       BM25
#       E5 + BM25
#       E5 + BGE
#       E5 + BGE + BM25
#       RRF(E5, BM25)
#
#
# 4. CROSS-ENCODER RERANKING
#
#       best first-stage retriever
#              ↓
#            Top-K
#              ↓
#       BGE cross-encoder
#              ↓
#          reranked Top-K
#
#
# 5. LONG-RANGE / EVIDENCE-AGE ANALYSIS
#
#       0-20 turns back
#       21-50
#       51-100
#       101-200
#       201-400
#       400+
#
# Also:
#
#       same/recent session
#       1-2 sessions back
#       3-5 sessions back
#       6+ sessions back
#
#
# IMPORTANT EXPERIMENTAL RULES
# ============================================================
#
# - SAME conversation split as previous experiments.
#
# - TRAIN conversations are NOT used for tuning.
#
# - Fusion weights and reranker blend beta are selected ONLY
#   on VALID conversations.
#
# - TEST conversations remain untouched until final reporting.
#
# - For memory-size scaling, every retrieval method receives
#   EXACTLY the same candidate pools.
#
# - 5 random pools per question by default.
#
# - Larger pools are nested:
#
#       8 ⊂ 16 ⊂ 32 ⊂ ... ⊂ 256
#
#   so scaling comparisons are cleaner.
#
# - No model is trained.
#
# - No models/ source file is modified.
#
#
# INSTALL
# ============================================================
#
# pip install -U sentence-transformers
#
#
# RUN
# ============================================================
#
# python locomo_fullconv_improvements.py \
#   --data data/locomo/locomo10.json \
#   --trials-per-question 5 \
#   --rerank-k 30 \
#   2>&1 | tee locomo_fullconv_improvements.log
#
#
# FAST TEST WITHOUT CROSS-ENCODER
# ============================================================
#
# python locomo_fullconv_improvements.py \
#   --data data/locomo/locomo10.json \
#   --trials-per-question 5 \
#   --skip-rerank \
#   2>&1 | tee locomo_fullconv_improvements_fast.log
#
# ============================================================


import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
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
    "--seed",
    type=int,
    default=2090,
)


parser.add_argument(
    "--trials-per-question",
    type=int,
    default=5,
)


parser.add_argument(
    "--batch-size",
    type=int,
    default=96,
)


parser.add_argument(
    "--rerank-batch-size",
    type=int,
    default=32,
)


parser.add_argument(
    "--rerank-k",
    type=int,
    default=30,
)


parser.add_argument(
    "--reranker",
    type=str,
    default="BAAI/bge-reranker-base",
)


parser.add_argument(
    "--skip-rerank",
    action="store_true",
)


parser.add_argument(
    "--output",
    type=str,
    default=(
        "outputs/"
        "locomo_longrange_retrieval.json"
    ),
)


args = parser.parse_args()


# ============================================================
# GLOBAL
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


POOL_SIZES = [
    8,
    16,
    32,
    64,
    128,
    256,
]


BASE_KS = [
    1,
    3,
    5,
    10,
    20,
]


ALL_KS = sorted(
    set(
        BASE_KS
        +
        [
            args.rerank_k
        ]
    )
)


random.seed(
    args.seed
)

np.random.seed(
    args.seed
)

torch.manual_seed(
    args.seed
)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        args.seed
    )


# ============================================================
# DISPLAY
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
# EVIDENCE NORMALIZATION
# ============================================================

def normalize_evidence(
    evidence,
):

    output = []


    def visit(value):

        if isinstance(
            value,
            str,
        ):

            value = value.strip()

            if value:

                output.append(
                    value
                )


        elif isinstance(
            value,
            list,
        ):

            for child in value:

                visit(
                    child
                )


    visit(
        evidence
    )


    # Remove duplicates while preserving order.

    return list(
        dict.fromkeys(
            output
        )
    )


# ============================================================
# CONVERSATION PARSER
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
                    int(
                        match.group(
                            1
                        )
                    ),

                    key,
                )
            )


    session_keys.sort()


    turns = []


    for (
        session_number,
        session_key,
    ) in session_keys:

        date = str(

            conversation.get(

                f"session_"
                f"{session_number}"
                f"_date_time",

                "",
            )
        ).strip()


        session_turns = conversation.get(

            session_key,

            [],
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


            turns.append(

                {
                    "dia_id":
                        dia_id,

                    "speaker":
                        speaker,

                    "text":
                        text,

                    "session":
                        session_number,

                    "date":
                        date,
                }
            )


    return turns


# ============================================================
# MEMORY TEXT VIEWS
# ============================================================

def turn_line(
    turn,
):

    if turn[
        "speaker"
    ]:

        return (

            f"{turn['speaker']}: "
            f"{turn['text']}"
        )


    return turn[
        "text"
    ]


def build_turn_view(
    turns,
    index,
):

    return turn_line(
        turns[
            index
        ]
    )


def build_turn_date_view(
    turns,
    index,
):

    turn = turns[
        index
    ]


    text = turn_line(
        turn
    )


    if turn[
        "date"
    ]:

        return (

            f"[DATE: {turn['date']}]\n"
            f"{text}"
        )


    return text


def build_previous_current_view(
    turns,
    index,
):

    current = turns[
        index
    ]


    pieces = []


    # --------------------------------------------------------
    # SESSION-SAFE PREVIOUS TURN
    #
    # Never cross session boundaries.
    # --------------------------------------------------------

    if index > 0:

        previous = turns[
            index - 1
        ]


        if (
            previous[
                "session"
            ]
            ==
            current[
                "session"
            ]
        ):

            pieces.append(

                "[PREVIOUS]\n"

                +

                turn_line(
                    previous
                )
            )


    pieces.append(

        "[CURRENT]\n"

        +

        turn_line(
            current
        )
    )


    return "\n".join(
        pieces
    )


# ============================================================
# LOAD LOCOMO
# ============================================================

section(
    "LOAD + PARSE LOCOMO"
)


with open(

    args.data,

    "r",

    encoding="utf-8",

) as handle:

    raw_data = json.load(
        handle
    )


conversations = []


for conversation_index, sample in enumerate(

    raw_data
):

    turns = flatten_conversation(
        sample
    )


    if not turns:

        continue


    turn_by_id = {

        turn[
            "dia_id"
        ]:
            index

        for index, turn
        in enumerate(
            turns
        )
    }


    qa_items = []


    for qa in sample.get(
        "qa",
        [],
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

            evidence_id
            in
            turn_by_id

            for evidence_id
            in evidence
        ):

            continue


        # ----------------------------------------------------
        # Keep dataset compatible with our controlled
        # 8-slot benchmark.
        # ----------------------------------------------------

        if len(
            evidence
        ) > 8:

            continue


        gold_indices = [

            turn_by_id[
                evidence_id
            ]

            for evidence_id
            in evidence
        ]


        qa_items.append(

            {
                "question":
                    question,

                "gold":
                    gold_indices,

                "evidence":
                    evidence,

                "category":
                    str(
                        qa.get(
                            "category",
                            "?",
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


    views = {

        "turn": [

            build_turn_view(
                turns,
                index,
            )

            for index in range(
                len(
                    turns
                )
            )
        ],

        "turn_date": [

            build_turn_date_view(
                turns,
                index,
            )

            for index in range(
                len(
                    turns
                )
            )
        ],

        "prev_turn": [

            build_previous_current_view(
                turns,
                index,
            )

            for index in range(
                len(
                    turns
                )
            )
        ],
    }


    conversations.append(

        {
            "id":
                sample_id,

            "turns":
                turns,

            "turn_by_id":
                turn_by_id,

            "qas":
                qa_items,

            "views":
                views,
        }
    )


    print(

        f"{sample_id:<12}"

        f"turns="
        f"{len(turns):<5}"

        f"questions="
        f"{len(qa_items)}"
    )


print()

print(
    "Usable conversations:",
    len(
        conversations
    )
)


print(
    "Total eligible questions:",
    sum(
        len(
            conversation[
                "qas"
            ]
        )

        for conversation
        in conversations
    ),
)


# ============================================================
# SAME SPLIT AS EARLIER EXPERIMENTS
# ============================================================

section(
    "CONVERSATION-LEVEL SPLIT"
)


split_rng = random.Random(
    args.seed
)


order = list(

    range(
        len(
            conversations
        )
    )
)


split_rng.shuffle(
    order
)


train_end = max(

    1,

    int(
        len(
            order
        )
        *
        0.60
    ),
)


valid_end = max(

    train_end + 1,

    int(
        len(
            order
        )
        *
        0.80
    ),
)


TRAIN = order[
    :train_end
]


VALID = order[
    train_end:
    valid_end
]


TEST = order[
    valid_end:
]


ALL = list(

    range(
        len(
            conversations
        )
    )
)


print(
    "TRAIN:"
)

print(
    [
        conversations[
            index
        ][
            "id"
        ]

        for index
        in TRAIN
    ]
)


print()

print(
    "VALID — ONLY THIS SPLIT IS USED FOR TUNING:"
)

print(
    [
        conversations[
            index
        ][
            "id"
        ]

        for index
        in VALID
    ]
)


print()

print(
    "TEST — COMPLETELY HELD OUT:"
)

print(
    [
        conversations[
            index
        ][
            "id"
        ]

        for index
        in TEST
    ]
)


# ============================================================
# SCORE STORAGE
#
# scores[method][conversation_id]
#
# shape:
#
#       [num_questions, num_memory_turns]
#
# ============================================================

SCORES = {}


# ============================================================
# DENSE ENCODER CONFIG
# ============================================================

ENCODERS = {

    "e5": {

        "model":
            "intfloat/e5-base-v2",

        "query_prefix":
            "query: ",

        "document_prefix":
            "passage: ",
    },

    "bge": {

        "model":
            "BAAI/bge-small-en-v1.5",

        "query_prefix":
            (
                "Represent this sentence for "
                "searching relevant passages: "
            ),

        "document_prefix":
            "",
    },
}


# ============================================================
# ENCODE HELPER
# ============================================================

def encode_texts(
    model,
    texts,
    prefix,
):

    prepared = [

        prefix + text

        for text in texts
    ]


    return model.encode(

        prepared,

        batch_size=
            args.batch_size,

        convert_to_numpy=True,

        normalize_embeddings=True,

        show_progress_bar=False,
    ).astype(
        np.float32
    )


# ============================================================
# E5
# ============================================================

section(
    "ENCODE E5 REPRESENTATIONS"
)


e5_config = ENCODERS[
    "e5"
]


e5_model = SentenceTransformer(

    e5_config[
        "model"
    ],

    device=
        DEVICE,
)


e5_model.max_seq_length = 256


E5_VIEWS = [

    "turn",

    "turn_date",

    "prev_turn",
]


e5_queries = {}


for conversation_index, conversation in enumerate(

    conversations
):

    question_texts = [

        qa[
            "question"
        ]

        for qa
        in conversation[
            "qas"
        ]
    ]


    e5_queries[
        conversation_index
    ] = encode_texts(

        e5_model,

        question_texts,

        e5_config[
            "query_prefix"
        ],
    )


for view in E5_VIEWS:

    method_name = (
        f"e5:{view}"
    )


    SCORES[
        method_name
    ] = {}


    print()

    print(
        method_name
    )


    for conversation_index, conversation in enumerate(

        conversations
    ):

        memory_vectors = encode_texts(

            e5_model,

            conversation[
                "views"
            ][
                view
            ],

            e5_config[
                "document_prefix"
            ],
        )


        SCORES[
            method_name
        ][
            conversation_index
        ] = (

            e5_queries[
                conversation_index
            ]

            @

            memory_vectors.T
        ).astype(
            np.float32
        )


        print(

            f"  "
            f"{conversation['id']:<12}"

            f"{SCORES[method_name][conversation_index].shape}"
        )


del e5_model

del e5_queries


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# BGE
#
# Only TURN representation for the main multi-encoder test.
#
# We are not exploding the number of arbitrary combinations.
# ============================================================

section(
    "ENCODE BGE REPRESENTATIONS"
)


bge_config = ENCODERS[
    "bge"
]


bge_model = SentenceTransformer(

    bge_config[
        "model"
    ],

    device=
        DEVICE,
)


bge_model.max_seq_length = 256


SCORES[
    "bge:turn"
] = {}


for conversation_index, conversation in enumerate(

    conversations
):

    questions = encode_texts(

        bge_model,

        [
            qa[
                "question"
            ]

            for qa
            in conversation[
                "qas"
            ]
        ],

        bge_config[
            "query_prefix"
        ],
    )


    memories = encode_texts(

        bge_model,

        conversation[
            "views"
        ][
            "turn"
        ],

        bge_config[
            "document_prefix"
        ],
    )


    SCORES[
        "bge:turn"
    ][
        conversation_index
    ] = (

        questions

        @

        memories.T
    ).astype(
        np.float32
    )


    print(

        f"  "
        f"{conversation['id']:<12}"

        f"{SCORES['bge:turn'][conversation_index].shape}"
    )


del bge_model


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# BM25
# ============================================================

section(
    "BUILD BM25"
)


STOPWORDS = set(

    """
    a an the of to in on at is are was were be been being
    it its this that these those and or for with as by from
    do did does what when where who whom which how why
    has have had having he she they his her their theirs
    i you we me my your our ours about into out up down
    """.split()
)


def tokenize_bm25(
    text,
):

    words = re.findall(

        r"[a-z0-9']+",

        text.lower(),
    )


    return [

        word

        for word in words

        if word not in STOPWORDS
    ]


class BM25:

    def __init__(
        self,
        documents,
        k1=1.2,
        b=0.75,
    ):

        self.k1 = k1

        self.b = b

        self.N = len(
            documents
        )


        self.document_lengths = np.asarray(

            [
                len(
                    document
                )

                for document
                in documents
            ],

            dtype=np.float32,
        )


        self.average_length = max(

            float(
                self.document_lengths.mean()
            ),

            1e-9,
        )


        postings = defaultdict(
            list
        )


        for document_index, document in enumerate(

            documents
        ):

            counts = Counter(
                document
            )


            for word, count in counts.items():

                postings[
                    word
                ].append(

                    (
                        document_index,
                        count,
                    )
                )


        self.postings = {}


        for word, entries in postings.items():

            indices = np.asarray(

                [
                    entry[
                        0
                    ]

                    for entry
                    in entries
                ],

                dtype=np.int64,
            )


            frequencies = np.asarray(

                [
                    entry[
                        1
                    ]

                    for entry
                    in entries
                ],

                dtype=np.float32,
            )


            self.postings[
                word
            ] = (

                indices,

                frequencies,
            )


    def score(
        self,
        query,
    ):

        scores = np.zeros(

            self.N,

            dtype=np.float32,
        )


        for word in set(
            query
        ):

            if word not in self.postings:

                continue


            (
                indices,
                frequencies,

            ) = self.postings[
                word
            ]


            document_frequency = len(
                indices
            )


            idf = math.log(

                1.0

                +

                (
                    self.N
                    -
                    document_frequency
                    +
                    0.5
                )

                /

                (
                    document_frequency
                    +
                    0.5
                )
            )


            denominator = (

                frequencies

                +

                self.k1

                *

                (
                    1.0
                    -
                    self.b

                    +

                    self.b

                    *

                    self.document_lengths[
                        indices
                    ]

                    /

                    self.average_length
                )
            )


            scores[
                indices
            ] += (

                idf

                *

                frequencies

                *

                (
                    self.k1
                    +
                    1.0
                )

                /

                denominator
            )


        return scores


SCORES[
    "bm25:turn"
] = {}


for conversation_index, conversation in enumerate(

    conversations
):

    documents = [

        tokenize_bm25(
            text
        )

        for text
        in conversation[
            "views"
        ][
            "turn"
        ]
    ]


    bm25 = BM25(
        documents
    )


    question_scores = []


    for qa in conversation[
        "qas"
    ]:

        query_tokens = tokenize_bm25(

            qa[
                "question"
            ]
        )


        question_scores.append(

            bm25.score(
                query_tokens
            )
        )


    SCORES[
        "bm25:turn"
    ][
        conversation_index
    ] = np.stack(

        question_scores,

        axis=0,
    )


    print(

        f"  "
        f"{conversation['id']:<12}"

        f"{SCORES['bm25:turn'][conversation_index].shape}"
    )


# ============================================================
# METRICS
# ============================================================

def empty_metric_accumulator():

    result = {

        "n":
            0,

        "mrr":
            0.0,
    }


    for k in ALL_KS:

        result[
            f"hit@{k}"
        ] = 0.0

        result[
            f"recall@{k}"
        ] = 0.0


    return result


def ranking_metrics(
    ranking,
    gold_indices,
):

    gold = set(
        gold_indices
    )


    first_gold_rank = None


    for rank, index in enumerate(

        ranking,

        start=1,
    ):

        if index in gold:

            first_gold_rank = rank

            break


    result = {

        "mrr":
            (
                0.0

                if first_gold_rank is None

                else

                1.0
                /
                first_gold_rank
            )
    }


    for k in ALL_KS:

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


def add_metrics(
    accumulator,
    metrics,
):

    accumulator[
        "n"
    ] += 1


    accumulator[
        "mrr"
    ] += metrics[
        "mrr"
    ]


    for k in ALL_KS:

        accumulator[
            f"hit@{k}"
        ] += metrics[
            f"hit@{k}"
        ]


        accumulator[
            f"recall@{k}"
        ] += metrics[
            f"recall@{k}"
        ]


def finalize_metrics(
    accumulator,
):

    n = accumulator[
        "n"
    ]


    if n == 0:

        return accumulator


    result = {

        "n":
            n,

        "mrr":
            accumulator[
                "mrr"
            ]
            /
            n,
    }


    for k in ALL_KS:

        result[
            f"hit@{k}"
        ] = (

            accumulator[
                f"hit@{k}"
            ]

            /

            n
        )


        result[
            f"recall@{k}"
        ] = (

            accumulator[
                f"recall@{k}"
            ]

            /

            n
        )


    return result


# ============================================================
# FULL-CONVERSATION EVALUATION
# ============================================================

def evaluate_full(
    score_dict,
    conversation_ids,
):

    accumulator = (
        empty_metric_accumulator()
    )


    for conversation_index in (
        conversation_ids
    ):

        conversation = conversations[
            conversation_index
        ]


        score_matrix = score_dict[
            conversation_index
        ]


        for question_index, qa in enumerate(

            conversation[
                "qas"
            ]
        ):

            ranking = np.argsort(

                -score_matrix[
                    question_index
                ]
            ).tolist()


            metrics = ranking_metrics(

                ranking,

                qa[
                    "gold"
                ],
            )


            add_metrics(

                accumulator,

                metrics,
            )


    return finalize_metrics(
        accumulator
    )


# ============================================================
# FAST MRR FOR HYPERPARAMETER TUNING
#
# Does NOT sort every row.
# ============================================================

def validation_mrr(
    score_dict,
):

    total = 0.0

    count = 0


    for conversation_index in VALID:

        conversation = conversations[
            conversation_index
        ]


        matrix = score_dict[
            conversation_index
        ]


        for question_index, qa in enumerate(

            conversation[
                "qas"
            ]
        ):

            row = matrix[
                question_index
            ]


            best_gold_score = max(

                row[
                    gold_index
                ]

                for gold_index
                in qa[
                    "gold"
                ]
            )


            rank = (

                1

                +

                int(
                    np.sum(
                        row
                        >
                        best_gold_score
                    )
                )
            )


            total += (

                1.0

                /
                rank
            )


            count += 1


    return (

        total

        /

        max(
            count,
            1,
        )
    )


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def zscore_rows(
    matrix,
):

    mean = matrix.mean(

        axis=1,

        keepdims=True,
    )


    std = matrix.std(

        axis=1,

        keepdims=True,
    )


    return (

        matrix
        -
        mean

        /

        1.0

    ) if False else (

        (
            matrix
            -
            mean
        )

        /

        (
            std
            +
            1e-9
        )
    )


# ============================================================
# LINEAR FUSION
# ============================================================

def fuse_scores(
    names,
    weights,
):

    output = {}


    for conversation_index in range(

        len(
            conversations
        )
    ):

        fused = None


        for name, weight in zip(

            names,
            weights,
        ):

            if weight == 0.0:

                continue


            normalized = zscore_rows(

                SCORES[
                    name
                ][
                    conversation_index
                ]
            )


            contribution = (

                weight

                *

                normalized
            )


            if fused is None:

                fused = contribution.copy()

            else:

                fused += contribution


        if fused is None:

            raise RuntimeError(
                "Fusion received all-zero weights."
            )


        output[
            conversation_index
        ] = fused.astype(
            np.float32
        )


    return output


# ============================================================
# TWO-WAY WEIGHT SEARCH
#
# VALID ONLY.
# ============================================================

def tune_two_way(
    name_a,
    name_b,
):

    candidates = []


    for alpha in np.linspace(

        0.0,

        1.0,

        21,
    ):

        weights = [

            float(
                alpha
            ),

            float(
                1.0
                -
                alpha
            ),
        ]


        fused = fuse_scores(

            [
                name_a,
                name_b,
            ],

            weights,
        )


        mrr = validation_mrr(
            fused
        )


        candidates.append(

            (
                mrr,
                weights,
                fused,
            )
        )


    candidates.sort(

        key=lambda item:
            item[
                0
            ],

        reverse=True,
    )


    return candidates[
        0
    ]


# ============================================================
# THREE-WAY SIMPLEX SEARCH
#
# VALID ONLY.
# ============================================================

def tune_three_way(
    names,
):

    best = None


    grid = np.arange(

        0.0,

        1.0001,

        0.1,
    )


    for weight_a in grid:

        for weight_b in grid:

            weight_c = (

                1.0

                -
                weight_a

                -
                weight_b
            )


            if weight_c < -1e-9:

                continue


            weight_c = max(

                0.0,

                float(
                    weight_c
                ),
            )


            weights = [

                float(
                    weight_a
                ),

                float(
                    weight_b
                ),

                weight_c,
            ]


            fused = fuse_scores(

                names,

                weights,
            )


            mrr = validation_mrr(
                fused
            )


            candidate = (

                mrr,

                weights,

                fused,
            )


            if (
                best is None

                or

                mrr
                >
                best[
                    0
                ]
            ):

                best = candidate


    return best


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    names,
    rrf_k=60,
):

    output = {}


    for conversation_index in range(

        len(
            conversations
        )
    ):

        total = None


        for name in names:

            matrix = SCORES[
                name
            ][
                conversation_index
            ]


            order = np.argsort(

                -matrix,

                axis=1,
            )


            ranks = np.empty_like(
                order
            )


            row_ids = np.arange(

                matrix.shape[
                    0
                ]
            )[
                :,
                None
            ]


            ranks[
                row_ids,
                order
            ] = np.arange(

                matrix.shape[
                    1
                ]
            )[
                None,
                :
            ]


            contribution = (

                1.0

                /

                (
                    rrf_k

                    +

                    ranks

                    +

                    1
                )
            )


            if total is None:

                total = contribution

            else:

                total += contribution


        output[
            conversation_index
        ] = total.astype(
            np.float32
        )


    return output


# ============================================================
# BUILD PREDEFINED METHODS
# ============================================================

section(
    "BUILD RETRIEVAL METHODS"
)


METHODS = {

    "E5_TURN":
        SCORES[
            "e5:turn"
        ],

    "E5_TURN_DATE":
        SCORES[
            "e5:turn_date"
        ],

    "E5_PREV_TURN":
        SCORES[
            "e5:prev_turn"
        ],

    "BGE_TURN":
        SCORES[
            "bge:turn"
        ],

    "BM25_TURN":
        SCORES[
            "bm25:turn"
        ],
}


# ============================================================
# E5 + BM25
# ============================================================

(
    valid_mrr,
    weights_e5_bm25,
    fused_e5_bm25,

) = tune_two_way(

    "e5:turn",

    "bm25:turn",
)


METHODS[
    "HYBRID_E5_BM25"
] = fused_e5_bm25


print(
    "E5 + BM25 weights "
    "(VALID only):",
    weights_e5_bm25,
    "valid MRR=",
    f"{valid_mrr:.4f}",
)


# ============================================================
# E5 + BGE
# ============================================================

(
    valid_mrr,
    weights_e5_bge,
    fused_e5_bge,

) = tune_two_way(

    "e5:turn",

    "bge:turn",
)


METHODS[
    "HYBRID_E5_BGE"
] = fused_e5_bge


print(
    "E5 + BGE weights "
    "(VALID only):",
    weights_e5_bge,
    "valid MRR=",
    f"{valid_mrr:.4f}",
)


# ============================================================
# E5 + BGE + BM25
# ============================================================

(
    valid_mrr,
    weights_three_way,
    fused_three_way,

) = tune_three_way(

    [
        "e5:turn",
        "bge:turn",
        "bm25:turn",
    ]
)


METHODS[
    "HYBRID_E5_BGE_BM25"
] = fused_three_way


print(
    "E5 + BGE + BM25 weights "
    "(VALID only):",
    weights_three_way,
    "valid MRR=",
    f"{valid_mrr:.4f}",
)


# ============================================================
# RRF
# ============================================================

METHODS[
    "RRF_E5_BM25"
] = reciprocal_rank_fusion(

    [
        "e5:turn",
        "bm25:turn",
    ]
)


# ============================================================
# SEGMENTATION ABLATION
# ============================================================

section(
    "SEGMENTATION ABLATION — HELD-OUT TEST"
)


SEGMENTATION_METHODS = [

    "E5_TURN",

    "E5_TURN_DATE",

    "E5_PREV_TURN",
]


print(

    f"{'Representation':<24}"

    f"{'Hit@1':>10}"

    f"{'Hit@3':>10}"

    f"{'Hit@5':>10}"

    f"{'R@10':>10}"

    f"{'R@20':>10}"

    f"{'MRR':>10}"
)


print(
    "-" * 84
)


segmentation_results = {}


for method_name in SEGMENTATION_METHODS:

    metrics = evaluate_full(

        METHODS[
            method_name
        ],

        TEST,
    )


    segmentation_results[
        method_name
    ] = metrics


    print(

        f"{method_name:<24}"

        f"{metrics['hit@1'] * 100:>9.2f}%"

        f"{metrics['hit@3'] * 100:>9.2f}%"

        f"{metrics['hit@5'] * 100:>9.2f}%"

        f"{metrics['recall@10'] * 100:>9.2f}%"

        f"{metrics['recall@20'] * 100:>9.2f}%"

        f"{metrics['mrr']:>10.4f}"
    )


# ============================================================
# FIRST-STAGE FULL-CONVERSATION TABLE
# ============================================================

section(
    "FIRST-STAGE RETRIEVAL — HELD-OUT TEST"
)


FIRST_STAGE_ORDER = [

    "E5_TURN",

    "BM25_TURN",

    "BGE_TURN",

    "HYBRID_E5_BM25",

    "HYBRID_E5_BGE",

    "HYBRID_E5_BGE_BM25",

    "RRF_E5_BM25",
]


first_stage_results = {}


print(

    f"{'Method':<28}"

    f"{'Hit@1':>9}"

    f"{'Hit@3':>9}"

    f"{'Hit@5':>9}"

    f"{'R@10':>9}"

    f"{'R@20':>9}"

    f"{f'R@{args.rerank_k}':>9}"

    f"{'MRR':>9}"
)


print(
    "-" * 91
)


for method_name in FIRST_STAGE_ORDER:

    metrics = evaluate_full(

        METHODS[
            method_name
        ],

        TEST,
    )


    first_stage_results[
        method_name
    ] = metrics


    print(

        f"{method_name:<28}"

        f"{metrics['hit@1'] * 100:>8.2f}%"

        f"{metrics['hit@3'] * 100:>8.2f}%"

        f"{metrics['hit@5'] * 100:>8.2f}%"

        f"{metrics['recall@10'] * 100:>8.2f}%"

        f"{metrics['recall@20'] * 100:>8.2f}%"

        f"{metrics[f'recall@{args.rerank_k}'] * 100:>8.2f}%"

        f"{metrics['mrr']:>9.4f}"
    )


# ============================================================
# SELECT RERANK FIRST STAGE
#
# Small, PREDECLARED candidate set.
#
# Selection uses VALID only.
# ============================================================

RERANK_CANDIDATES = [

    "E5_TURN",

    "HYBRID_E5_BM25",

    "HYBRID_E5_BGE",

    "HYBRID_E5_BGE_BM25",
]


valid_first_stage_mrr = {

    method_name:
        validation_mrr(

            METHODS[
                method_name
            ]
        )

    for method_name
    in RERANK_CANDIDATES
}


best_first_stage = max(

    valid_first_stage_mrr,

    key=
        valid_first_stage_mrr.get,
)


section(
    "RERANKER FIRST-STAGE SELECTION"
)


for method_name in RERANK_CANDIDATES:

    print(

        f"{method_name:<28}"

        f"VALID MRR = "
        f"{valid_first_stage_mrr[method_name]:.4f}"
    )


print()

print(
    "Selected using VALID only:",
    best_first_stage,
)


best_first_stage_scores = METHODS[
    best_first_stage
]


# ============================================================
# CROSS-ENCODER RERANK
#
# Only VALID + TEST are processed.
#
# TRAIN is untouched and unnecessary.
# ============================================================

RERANKED_SCORES = None

rerank_beta = None

rerank_valid_result = None

rerank_test_result = None


if not args.skip_rerank:

    section(
        f"CROSS-ENCODER RERANK TOP-{args.rerank_k}"
    )


    print(
        "Reranker:",
        args.reranker,
    )


    cross_encoder = CrossEncoder(

        args.reranker,

        device=
            DEVICE,

        max_length=384,
    )


    ce_cache = {}


    rerank_conversation_ids = sorted(

        set(
            VALID
            +
            TEST
        )
    )


    for conversation_index in (
        rerank_conversation_ids
    ):

        conversation = conversations[
            conversation_index
        ]


        base_matrix = best_first_stage_scores[
            conversation_index
        ]


        num_questions = base_matrix.shape[
            0
        ]


        num_memories = base_matrix.shape[
            1
        ]


        local_k = min(

            args.rerank_k,

            num_memories,
        )


        top_indices = np.argsort(

            -base_matrix,

            axis=1,
        )[
            :,
            :local_k
        ]


        pairs = []


        for question_index, qa in enumerate(

            conversation[
                "qas"
            ]
        ):

            for memory_index in top_indices[
                question_index
            ]:

                pairs.append(

                    (
                        qa[
                            "question"
                        ],

                        conversation[
                            "views"
                        ][
                            "turn"
                        ][
                            int(
                                memory_index
                            )
                        ],
                    )
                )


        predictions = cross_encoder.predict(

            pairs,

            batch_size=
                args.rerank_batch_size,

            show_progress_bar=False,
        )


        predictions = np.asarray(

            predictions,

            dtype=np.float32,
        ).reshape(

            num_questions,

            local_k,
        )


        ce_cache[
            conversation_index
        ] = {

            "top_indices":
                top_indices,

            "scores":
                predictions,
        }


        print(

            f"  "
            f"{conversation['id']:<12}"

            f"questions="
            f"{num_questions:<4}"

            f"pairs="
            f"{len(pairs)}"
        )


    del cross_encoder


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    # ========================================================
    # BUILD RERANKED SCORE MATRIX
    # ========================================================

    def build_reranked(
        beta,
    ):

        output = {}


        for conversation_index in (
            rerank_conversation_ids
        ):

            base = best_first_stage_scores[
                conversation_index
            ]


            normalized_base = zscore_rows(
                base
            )


            final = normalized_base.copy()


            top_indices = ce_cache[
                conversation_index
            ][
                "top_indices"
            ]


            cross_scores = ce_cache[
                conversation_index
            ][
                "scores"
            ]


            for question_index in range(

                top_indices.shape[
                    0
                ]
            ):

                candidate_indices = top_indices[
                    question_index
                ]


                base_top = normalized_base[

                    question_index,

                    candidate_indices,
                ]


                base_top = (

                    base_top
                    -
                    base_top.mean()
                ) / (

                    base_top.std()
                    +
                    1e-9
                )


                cross_top = cross_scores[
                    question_index
                ]


                cross_top = (

                    cross_top
                    -
                    cross_top.mean()
                ) / (

                    cross_top.std()
                    +
                    1e-9
                )


                combined = (

                    (
                        1.0
                        -
                        beta
                    )

                    *

                    base_top

                    +

                    beta

                    *

                    cross_top
                )


                # ------------------------------------------------
                # Top-K must remain ahead of non-retrieved items.
                #
                # This makes first-stage Recall@K the actual
                # reranker ceiling.
                # ------------------------------------------------

                final[

                    question_index,

                    candidate_indices,

                ] = (

                    1000.0

                    +

                    combined
                )


            output[
                conversation_index
            ] = final


        return output


    # ========================================================
    # TUNE BETA ON VALID ONLY
    # ========================================================

    beta_candidates = [

        0.0,

        0.25,

        0.50,

        0.75,

        1.0,
    ]


    best_beta_tuple = None


    for beta in beta_candidates:

        candidate_scores = build_reranked(
            beta
        )


        valid_metric = evaluate_full(

            candidate_scores,

            VALID,
        )


        item = (

            valid_metric[
                "mrr"
            ],

            beta,

            candidate_scores,

            valid_metric,
        )


        if (
            best_beta_tuple is None

            or

            item[
                0
            ]
            >
            best_beta_tuple[
                0
            ]
        ):

            best_beta_tuple = item


    (
        _,
        rerank_beta,
        RERANKED_SCORES,
        rerank_valid_result,

    ) = best_beta_tuple


    rerank_test_result = evaluate_full(

        RERANKED_SCORES,

        TEST,
    )


    print()

    print(
        "Selected beta using VALID only:",
        rerank_beta,
    )


    print()

    first_stage_test = evaluate_full(

        best_first_stage_scores,

        TEST,
    )


    print(
        "FIRST-STAGE TEST:"
    )

    print(
        f"  Hit@1      = "
        f"{first_stage_test['hit@1'] * 100:.2f}%"
    )

    print(
        f"  Recall@{args.rerank_k:<2} = "
        f"{first_stage_test[f'recall@{args.rerank_k}'] * 100:.2f}%"
    )


    print()

    print(
        "RERANKED TEST:"
    )

    print(
        f"  Hit@1      = "
        f"{rerank_test_result['hit@1'] * 100:.2f}%"
    )

    print(
        f"  Hit@5      = "
        f"{rerank_test_result['hit@5'] * 100:.2f}%"
    )

    print(
        f"  MRR        = "
        f"{rerank_test_result['mrr']:.4f}"
    )


# ============================================================
# MEMORY-SIZE SCALING
#
# SAME candidate pools for all methods.
#
# Larger candidate pools contain smaller candidate pools.
# ============================================================

SCALING_METHOD_NAMES = [

    "E5_TURN",

    "BM25_TURN",

    "BGE_TURN",

    "HYBRID_E5_BM25",

    "HYBRID_E5_BGE_BM25",
]


def evaluate_scaling(
    conversation_ids,
    method_names,
    trials_per_question,
):

    stores = {

        method_name: {

            str(pool_size):
                empty_metric_accumulator()

            for pool_size
            in POOL_SIZES
        }

        for method_name
        in method_names
    }


    for method_name in method_names:

        stores[
            method_name
        ][
            "FULL"
        ] = empty_metric_accumulator()


    question_count = 0


    for conversation_index in (
        conversation_ids
    ):

        conversation = conversations[
            conversation_index
        ]


        number_of_memories = len(

            conversation[
                "turns"
            ]
        )


        all_indices = list(

            range(
                number_of_memories
            )
        )


        for question_index, qa in enumerate(

            conversation[
                "qas"
            ]
        ):

            gold = list(
                qa[
                    "gold"
                ]
            )


            gold_set = set(
                gold
            )


            negatives = [

                memory_index

                for memory_index
                in all_indices

                if memory_index
                not in
                gold_set
            ]


            question_count += 1


            # =================================================
            # FULL CONVERSATION
            # =================================================

            for method_name in method_names:

                row = METHODS[
                    method_name
                ][
                    conversation_index
                ][
                    question_index
                ]


                ranking = np.argsort(

                    -row

                ).tolist()


                metrics = ranking_metrics(

                    ranking,

                    gold,
                )


                add_metrics(

                    stores[
                        method_name
                    ][
                        "FULL"
                    ],

                    metrics,
                )


            # =================================================
            # CONTROLLED POOLS
            #
            # One shuffled negative order per trial.
            #
            # Therefore:
            #
            #   pool 8 subset pool 16 subset pool 32 ...
            # =================================================

            for trial in range(

                trials_per_question
            ):

                trial_seed = (

                    args.seed

                    +

                    conversation_index
                    *
                    1_000_003

                    +

                    question_index
                    *
                    10_007

                    +

                    trial
                    *
                    97
                )


                trial_rng = random.Random(
                    trial_seed
                )


                shuffled_negatives = negatives.copy()


                trial_rng.shuffle(
                    shuffled_negatives
                )


                for pool_size in POOL_SIZES:

                    if (
                        pool_size
                        >
                        number_of_memories
                    ):

                        continue


                    if (
                        len(
                            gold
                        )
                        >
                        pool_size
                    ):

                        continue


                    negatives_needed = (

                        pool_size

                        -

                        len(
                            gold
                        )
                    )


                    if (
                        negatives_needed
                        >
                        len(
                            shuffled_negatives
                        )
                    ):

                        continue


                    candidates = (

                        gold

                        +

                        shuffled_negatives[
                            :negatives_needed
                        ]
                    )


                    candidate_array = np.asarray(

                        candidates,

                        dtype=np.int64,
                    )


                    for method_name in method_names:

                        row = METHODS[
                            method_name
                        ][
                            conversation_index
                        ][
                            question_index
                        ]


                        selected_scores = row[
                            candidate_array
                        ]


                        local_order = np.argsort(

                            -selected_scores
                        )


                        ranking = [

                            candidates[
                                int(
                                    local_index
                                )
                            ]

                            for local_index
                            in local_order
                        ]


                        metrics = ranking_metrics(

                            ranking,

                            gold,
                        )


                        add_metrics(

                            stores[
                                method_name
                            ][
                                str(
                                    pool_size
                                )
                            ],

                            metrics,
                        )


    finalized = {}


    for method_name in method_names:

        finalized[
            method_name
        ] = {}


        for pool_label, accumulator in stores[
            method_name
        ].items():

            finalized[
                method_name
            ][
                pool_label
            ] = finalize_metrics(
                accumulator
            )


    return (
        finalized,
        question_count,
    )


# ============================================================
# HELD-OUT SCALING
# ============================================================

section(
    "LONG-RANGE SCALING — HELD-OUT TEST"
)


(
    test_scaling,
    test_question_count,

) = evaluate_scaling(

    TEST,

    SCALING_METHOD_NAMES,

    args.trials_per_question,
)


print(
    "Held-out questions:",
    test_question_count,
)

print(
    "Trials/question:",
    args.trials_per_question,
)


for method_name in SCALING_METHOD_NAMES:

    print()

    print(
        method_name
    )


    print(

        f"{'Memory':<10}"

        f"{'Hit@1':>10}"

        f"{'Hit@3':>10}"

        f"{'Hit@5':>10}"

        f"{'R@10':>10}"

        f"{'R@20':>10}"

        f"{'MRR':>10}"

        f"{'N':>10}"
    )


    print(
        "-" * 80
    )


    labels = [

        str(
            pool_size
        )

        for pool_size
        in POOL_SIZES

    ] + [

        "FULL"
    ]


    for label in labels:

        metrics = test_scaling[
            method_name
        ][
            label
        ]


        print(

            f"{label:<10}"

            f"{metrics['hit@1'] * 100:>9.2f}%"

            f"{metrics['hit@3'] * 100:>9.2f}%"

            f"{metrics['hit@5'] * 100:>9.2f}%"

            f"{metrics['recall@10'] * 100:>9.2f}%"

            f"{metrics['recall@20'] * 100:>9.2f}%"

            f"{metrics['mrr']:>10.4f}"

            f"{metrics['n']:>10}"
        )


# ============================================================
# COMPACT SCALING TABLE — HIT@1
# ============================================================

section(
    "COMPACT LONG-RANGE HIT@1 TABLE — TEST"
)


header = (

    f"{'Method':<24}"

    +

    "".join(

        f"{label:>9}"

        for label in [

            "8",
            "16",
            "32",
            "64",
            "128",
            "256",
            "FULL",
        ]
    )
)


print(
    header
)

print(
    "-" * len(
        header
    )
)


for method_name in SCALING_METHOD_NAMES:

    values = []


    for label in [

        "8",
        "16",
        "32",
        "64",
        "128",
        "256",
        "FULL",
    ]:

        value = (

            test_scaling[
                method_name
            ][
                label
            ][
                "hit@1"
            ]

            *
            100
        )


        values.append(
            value
        )


    print(

        f"{method_name:<24}"

        +

        "".join(

            f"{value:>8.2f}%"

            for value in values
        )
    )


# ============================================================
# ZERO-SHOT ALL-LOCOMO SCALING
#
# IMPORTANT:
#
# No tuned hybrids here.
#
# This table is descriptive only and gives us many more trials.
# ============================================================

section(
    "ALL LOCOMO ZERO-SHOT SCALING"
)


ZERO_SHOT_METHODS = [

    "E5_TURN",

    "BGE_TURN",

    "BM25_TURN",
]


(
    all_scaling,
    all_question_count,

) = evaluate_scaling(

    ALL,

    ZERO_SHOT_METHODS,

    args.trials_per_question,
)


print(
    "Questions:",
    all_question_count,
)

print(
    "Trials/question:",
    args.trials_per_question,
)


header = (

    f"{'Method':<20}"

    +

    "".join(

        f"{label:>9}"

        for label in [

            "8",
            "16",
            "32",
            "64",
            "128",
            "256",
            "FULL",
        ]
    )
)


print()

print(
    header
)

print(
    "-" * len(
        header
    )
)


for method_name in ZERO_SHOT_METHODS:

    print(

        f"{method_name:<20}"

        +

        "".join(

            f"{all_scaling[method_name][label]['hit@1'] * 100:>8.2f}%"

            for label in [

                "8",
                "16",
                "32",
                "64",
                "128",
                "256",
                "FULL",
            ]
        )
    )


# ============================================================
# CHOOSE FINAL FULL-CONVERSATION METHOD
#
# For distance analysis.
# ============================================================

if (
    RERANKED_SCORES is not None
):

    FINAL_METHOD_NAME = (

        "RERANK_"
        +
        best_first_stage
    )


    FINAL_SCORES = (
        RERANKED_SCORES
    )


else:

    FINAL_METHOD_NAME = (
        best_first_stage
    )


    FINAL_SCORES = (
        best_first_stage_scores
    )


# ============================================================
# EVIDENCE AGE BUCKETS
#
# Age = number of turns between the END of the conversation
# and the NEAREST annotated gold evidence.
#
# Since Hit@K counts retrieval of ANY annotated evidence,
# nearest-gold age is the consistent corresponding measure.
# ============================================================

TURN_AGE_BUCKETS = [

    (
        "0-20",
        0,
        20,
    ),

    (
        "21-50",
        21,
        50,
    ),

    (
        "51-100",
        51,
        100,
    ),

    (
        "101-200",
        101,
        200,
    ),

    (
        "201-400",
        201,
        400,
    ),

    (
        "400+",
        401,
        float(
            "inf"
        ),
    ),
]


def turn_age_bucket(
    age,
):

    for (
        label,
        low,
        high,

    ) in TURN_AGE_BUCKETS:

        if (
            age
            >=
            low

            and

            age
            <=
            high
        ):

            return label


    return "unknown"


def session_distance_bucket(
    distance,
):

    if distance <= 0:

        return "same/recent"

    if distance <= 2:

        return "1-2"

    if distance <= 5:

        return "3-5"

    return "6+"


# ============================================================
# DISTANCE ANALYSIS
# ============================================================

section(
    f"LONG-RANGE DISTANCE ANALYSIS — {FINAL_METHOD_NAME}"
)


turn_bucket_metrics = defaultdict(

    empty_metric_accumulator
)


session_bucket_metrics = defaultdict(

    empty_metric_accumulator
)


for conversation_index in TEST:

    conversation = conversations[
        conversation_index
    ]


    score_matrix = FINAL_SCORES[
        conversation_index
    ]


    final_turn_index = (

        len(
            conversation[
                "turns"
            ]
        )

        -
        1
    )


    final_session = max(

        turn[
            "session"
        ]

        for turn
        in conversation[
            "turns"
        ]
    )


    for question_index, qa in enumerate(

        conversation[
            "qas"
        ]
    ):

        gold = qa[
            "gold"
        ]


        nearest_turn_age = min(

            final_turn_index
            -
            gold_index

            for gold_index
            in gold
        )


        nearest_session_distance = min(

            final_session

            -

            conversation[
                "turns"
            ][
                gold_index
            ][
                "session"
            ]

            for gold_index
            in gold
        )


        ranking = np.argsort(

            -score_matrix[
                question_index
            ]
        ).tolist()


        metrics = ranking_metrics(

            ranking,

            gold,
        )


        add_metrics(

            turn_bucket_metrics[

                turn_age_bucket(
                    nearest_turn_age
                )
            ],

            metrics,
        )


        add_metrics(

            session_bucket_metrics[

                session_distance_bucket(
                    nearest_session_distance
                )
            ],

            metrics,
        )


print()

print(
    "TURN-DISTANCE BUCKETS"
)


print(

    f"{'Age':<14}"

    f"{'N':>8}"

    f"{'Hit@1':>10}"

    f"{'Hit@5':>10}"

    f"{'R@10':>10}"

    f"{'R@20':>10}"

    f"{'MRR':>10}"
)


print(
    "-" * 72
)


turn_distance_results = {}


for (
    label,
    _,
    _,

) in TURN_AGE_BUCKETS:

    metrics = finalize_metrics(

        turn_bucket_metrics[
            label
        ]
    )


    turn_distance_results[
        label
    ] = metrics


    print(

        f"{label:<14}"

        f"{metrics['n']:>8}"

        f"{metrics['hit@1'] * 100:>9.2f}%"

        f"{metrics['hit@5'] * 100:>9.2f}%"

        f"{metrics['recall@10'] * 100:>9.2f}%"

        f"{metrics['recall@20'] * 100:>9.2f}%"

        f"{metrics['mrr']:>10.4f}"
    )


print()

print(
    "SESSION-DISTANCE BUCKETS"
)


print(

    f"{'Sessions back':<18}"

    f"{'N':>8}"

    f"{'Hit@1':>10}"

    f"{'Hit@5':>10}"

    f"{'R@10':>10}"

    f"{'R@20':>10}"

    f"{'MRR':>10}"
)


print(
    "-" * 76
)


session_distance_results = {}


for label in [

    "same/recent",

    "1-2",

    "3-5",

    "6+",

]:

    metrics = finalize_metrics(

        session_bucket_metrics[
            label
        ]
    )


    session_distance_results[
        label
    ] = metrics


    print(

        f"{label:<18}"

        f"{metrics['n']:>8}"

        f"{metrics['hit@1'] * 100:>9.2f}%"

        f"{metrics['hit@5'] * 100:>9.2f}%"

        f"{metrics['recall@10'] * 100:>9.2f}%"

        f"{metrics['recall@20'] * 100:>9.2f}%"

        f"{metrics['mrr']:>10.4f}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

section(
    "FINAL SUMMARY"
)


e5_test = evaluate_full(

    METHODS[
        "E5_TURN"
    ],

    TEST,
)


best_first_test = evaluate_full(

    best_first_stage_scores,

    TEST,
)


print(
    "Baseline E5 TURN:"
)

print(
    f"  Hit@1 = "
    f"{e5_test['hit@1'] * 100:.2f}%"
)

print(
    f"  Hit@5 = "
    f"{e5_test['hit@5'] * 100:.2f}%"
)

print(
    f"  R@20  = "
    f"{e5_test['recall@20'] * 100:.2f}%"
)

print(
    f"  MRR   = "
    f"{e5_test['mrr']:.4f}"
)


print()

print(
    "Best first-stage selected on VALID:"
)

print(
    " ",
    best_first_stage,
)

print(
    f"  Hit@1 = "
    f"{best_first_test['hit@1'] * 100:.2f}%"
)

print(
    f"  Hit@5 = "
    f"{best_first_test['hit@5'] * 100:.2f}%"
)

print(
    f"  R@{args.rerank_k}  = "
    f"{best_first_test[f'recall@{args.rerank_k}'] * 100:.2f}%"
)

print(
    f"  MRR   = "
    f"{best_first_test['mrr']:.4f}"
)


if rerank_test_result is not None:

    print()

    print(
        "After cross-encoder reranking:"
    )

    print(
        f"  beta  = "
        f"{rerank_beta}"
    )

    print(
        f"  Hit@1 = "
        f"{rerank_test_result['hit@1'] * 100:.2f}%"
    )

    print(
        f"  Hit@5 = "
        f"{rerank_test_result['hit@5'] * 100:.2f}%"
    )

    print(
        f"  MRR   = "
        f"{rerank_test_result['mrr']:.4f}"
    )


print()

print(
    "Interpret reranking carefully:"
)

print(
    f"First-stage Recall@{args.rerank_k} is the "
    f"maximum fraction of queries for which the "
    f"reranker even receives a gold item."
)


# ============================================================
# SAVE JSON
# ============================================================

section(
    "SAVE RESULTS"
)


def clean_for_json(
    value,
):

    if isinstance(
        value,
        dict,
    ):

        return {

            str(
                key
            ):
                clean_for_json(
                    child
                )

            for key, child
            in value.items()
        }


    if isinstance(
        value,
        list,
    ):

        return [

            clean_for_json(
                child
            )

            for child
            in value
        ]


    if isinstance(
        value,
        tuple,
    ):

        return [

            clean_for_json(
                child
            )

            for child
            in value
        ]


    if isinstance(
        value,
        np.floating,
    ):

        return float(
            value
        )


    if isinstance(
        value,
        np.integer,
    ):

        return int(
            value
        )


    return value


output = {

    "seed":
        args.seed,

    "trials_per_question":
        args.trials_per_question,

    "rerank_k":
        args.rerank_k,

    "split": {

        "train": [

            conversations[
                index
            ][
                "id"
            ]

            for index
            in TRAIN
        ],

        "valid": [

            conversations[
                index
            ][
                "id"
            ]

            for index
            in VALID
        ],

        "test": [

            conversations[
                index
            ][
                "id"
            ]

            for index
            in TEST
        ],
    },

    "fusion_weights": {

        "e5_bm25":
            weights_e5_bm25,

        "e5_bge":
            weights_e5_bge,

        "e5_bge_bm25":
            weights_three_way,
    },

    "segmentation_test":
        segmentation_results,

    "first_stage_test":
        first_stage_results,

    "selected_first_stage":
        best_first_stage,

    "selected_first_stage_valid_mrr":
        valid_first_stage_mrr[
            best_first_stage
        ],

    "reranker": {

        "enabled":
            (
                not
                args.skip_rerank
            ),

        "model":
            args.reranker,

        "beta":
            rerank_beta,

        "valid":
            rerank_valid_result,

        "test":
            rerank_test_result,
    },

    "test_scaling":
        test_scaling,

    "all_zero_shot_scaling":
        all_scaling,

    "turn_distance_test":
        turn_distance_results,

    "session_distance_test":
        session_distance_results,
}


output = clean_for_json(
    output
)


output_path = Path(
    args.output
)


output_path.parent.mkdir(

    parents=True,

    exist_ok=True,
)


with open(

    output_path,

    "w",

    encoding="utf-8",

) as handle:

    json.dump(

        output,

        handle,

        indent=2,
    )


print(
    "Saved:",
    output_path
)


# ============================================================
# COMPLETE
# ============================================================

section(
    "EXPERIMENT COMPLETE"
)


print(
    "No model was trained."
)

print(
    "No GPT-2 parameter was modified."
)

print(
    "No CandidateWriter was modified."
)

print(
    "No MemoryReader was modified."
)

print(
    "No files under models/ were modified."
)

print()

print(
    "Fusion weights were selected using VALID conversations only."
)

print(
    "TEST conversations remained held out."
)

print()

print(
    "Main question answered by this experiment:"
)

print(
    "How does retrieval quality change as memory grows, "
    "and which retrieval strategy best preserves the "
    "~80% small-memory retrieval advantage?"
)