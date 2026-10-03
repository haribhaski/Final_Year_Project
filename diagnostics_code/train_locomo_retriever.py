# ============================================================
# TRAIN LOCOMO MEMORY RETRIEVER
#
# ============================================================
#
# PURPOSE
# ============================================================
#
# Train the thing we ACTUALLY care about:
#
#       QUESTION -> CORRECT HISTORICAL MEMORY
#
#
# We already established:
#
#   Existing Writer + Existing Reader:
#       Hit@1 ~17-18%
#
#   Direct final representation + cosine:
#       Hit@1 ~32%
#
#
# NOW:
#
# Learn a dedicated retrieval space using REAL LoCoMo
# question -> gold evidence supervision.
#
#
# ARCHITECTURE
# ============================================================
#
#                         QUERY
#                           |
#                     frozen GPT-2
#                           |
#                    final masked mean
#                           |
#                    QueryEncoder
#                    768 -> 512 -> 256
#                           |
#                         q key
#
#
#                    MEMORY CHUNK
#                           |
#                     frozen GPT-2
#                           |
#                    final masked mean
#                           |
#                    MemoryEncoder
#                    768 -> 512 -> 256
#                           |
#                         k_i
#
#
#                   score = cosine(q,k_i)
#
#
# TRAINING
# ============================================================
#
# - conversation-level split
# - positive = LoCoMo annotated evidence
# - negatives = other turns from SAME conversation
# - in-batch negatives
# - dynamic hard-negative mining
# - optional contextual memory:
#
#       previous turn
#       current turn
#       next turn
#
#
# EVALUATION
# ============================================================
#
# 1. CONTROLLED 8-SLOT
#
#    gold evidence + distractors
#
# 2. FULL CONVERSATION
#
#    query against every memory turn
#
#
# METHODS COMPARED
# ============================================================
#
# RAW_FINAL_COSINE
# TRAINED_DUAL_ENCODER
# TRAINED_DUAL_PLUS_RERANKER
#
#
# NOTHING in models/ is modified.
#
#
# RUN:
#
# python train_locomo_retriever.py \
#   --data data/locomo/locomo10.json \
#   --epochs 30 \
#   --hard-negatives 8 \
#   --context-radius 1 \
#   2>&1 | tee train_locomo_retriever.log
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

from torch.utils.data import (
    Dataset,
    DataLoader,
)

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
    "--output",
    type=str,
    default=(
        "outputs/"
        "locomo_learned_retriever.pt"
    ),
)


parser.add_argument(
    "--seed",
    type=int,
    default=2090,
)


parser.add_argument(
    "--epochs",
    type=int,
    default=30,
)


parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
)


parser.add_argument(
    "--learning-rate",
    type=float,
    default=2e-4,
)


parser.add_argument(
    "--weight-decay",
    type=float,
    default=1e-4,
)


parser.add_argument(
    "--temperature",
    type=float,
    default=0.07,
)


parser.add_argument(
    "--embedding-dim",
    type=int,
    default=256,
)


parser.add_argument(
    "--hidden-dim",
    type=int,
    default=512,
)


parser.add_argument(
    "--dropout",
    type=float,
    default=0.1,
)


parser.add_argument(
    "--hard-negatives",
    type=int,
    default=8,
)


parser.add_argument(
    "--hard-negative-weight",
    type=float,
    default=0.5,
)


parser.add_argument(
    "--margin",
    type=float,
    default=0.15,
)


parser.add_argument(
    "--context-radius",
    type=int,
    default=1,
    help=(
        "0=current turn only, "
        "1=previous+current+next"
    ),
)


parser.add_argument(
    "--max-memory-tokens",
    type=int,
    default=192,
)


parser.add_argument(
    "--max-question-tokens",
    type=int,
    default=96,
)


parser.add_argument(
    "--gpt-batch-size",
    type=int,
    default=32,
)


parser.add_argument(
    "--eval-pool-size",
    type=int,
    default=8,
)


parser.add_argument(
    "--reranker-epochs",
    type=int,
    default=15,
)


parser.add_argument(
    "--reranker-negatives",
    type=int,
    default=5,
)


parser.add_argument(
    "--disable-reranker",
    action="store_true",
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
# MEMORY CONFIG
#
# We only need GPT-2 representations.
# Memory architecture is NOT trained.
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
# LOAD FROZEN GPT-2
# ============================================================

section(
    "LOAD FROZEN GPT-2"
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


model.to(
    DEVICE
)

model.eval()


for parameter in model.parameters():

    parameter.requires_grad = False


GPT_DIM = (
    model.backbone.config.n_embd
)


print(
    "GPT dimension:",
    GPT_DIM,
)

print(
    "GPT-2 frozen."
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

        f"Cannot find LoCoMo:\n"
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

            item = item.strip()

            if item:

                output.append(
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


    for item in output:

        if item not in seen:

            seen.add(
                item
            )

            final.append(
                item
            )


    return final


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
                }
            )


    return turns


# ============================================================
# CONTEXTUAL MEMORY ENTRY
#
# radius=0:
#
#   CURRENT TURN ONLY
#
#
# radius=1:
#
#   PREVIOUS
#   CURRENT
#   NEXT
#
#
# Gold dia_id remains the CENTER/current turn.
# ============================================================

def build_memory_text(
    turns,
    center_index,
):

    radius = (
        args.context_radius
    )


    start = max(

        0,

        center_index - radius,
    )


    end = min(

        len(turns),

        center_index + radius + 1,
    )


    pieces = []


    for index in range(
        start,
        end,
    ):

        turn = turns[
            index
        ]


        if turn[
            "speaker"
        ]:

            prefix = (
                f"{turn['speaker']}: "
            )

        else:

            prefix = ""


        if index == center_index:

            marker = "[CURRENT] "

        elif index < center_index:

            marker = "[PREVIOUS] "

        else:

            marker = "[NEXT] "


        pieces.append(

            marker

            +

            prefix

            +

            turn[
                "text"
            ]
        )


    return "\n".join(
        pieces
    )


# ============================================================
# PARSE ALL CONVERSATIONS
# ============================================================

section(
    "PARSE LOCOMO"
)


conversations = []


for conversation_index, sample in (
    enumerate(
        locomo
    )
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


    memories = []


    for index, turn in enumerate(
        turns
    ):

        memories.append(

            {
                "dia_id":
                    turn[
                        "dia_id"
                    ],

                "text":
                    build_memory_text(
                        turns,
                        index,
                    ),
            }
        )


    qa_items = []


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

            evidence_id
            in
            turn_by_id

            for evidence_id
            in evidence
        ):

            continue


        qa_items.append(

            {
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
                    str(
                        qa.get(
                            "category",
                            "unknown",
                        )
                    ),

                "evidence":
                    evidence,
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

            "conversation_index":
                conversation_index,

            "turns":
                turns,

            "memories":
                memories,

            "turn_by_id":
                turn_by_id,

            "qas":
                qa_items,
        }
    )


print(
    "Usable conversations:",
    len(conversations),
)


for conversation in conversations:

    print(

        f"{conversation['sample_id']:<15}"

        f" turns="
        f"{len(conversation['memories']):<5}"

        f" questions="
        f"{len(conversation['qas'])}"
    )


# ============================================================
# CONVERSATION-LEVEL SPLIT
#
# CRITICAL:
#
# Same conversation NEVER appears in train and test.
# ============================================================

section(
    "CONVERSATION-LEVEL SPLIT"
)


rng = random.Random(
    args.seed
)


conversation_indices = list(

    range(
        len(conversations)
    )
)


rng.shuffle(
    conversation_indices
)


n_conv = len(
    conversation_indices
)


if n_conv < 5:

    raise RuntimeError(

        "Too few LoCoMo conversations."
    )


# ------------------------------------------------------------
# LoCoMo has 10 long conversations.
#
# 60 / 20 / 20
#
# = 6 train
# = 2 validation
# = 2 test
# ------------------------------------------------------------

train_end = max(

    1,

    int(
        n_conv * 0.60
    ),
)


valid_end = max(

    train_end + 1,

    int(
        n_conv * 0.80
    ),
)


train_ids = conversation_indices[
    :train_end
]


valid_ids = conversation_indices[
    train_end:
    valid_end
]


test_ids = conversation_indices[
    valid_end:
]


print(
    "TRAIN conversations:"
)

for idx in train_ids:

    print(
        " ",
        conversations[
            idx
        ][
            "sample_id"
        ]
    )


print()

print(
    "VALID conversations:"
)

for idx in valid_ids:

    print(
        " ",
        conversations[
            idx
        ][
            "sample_id"
        ]
    )


print()

print(
    "TEST conversations:"
)

for idx in test_ids:

    print(
        " ",
        conversations[
            idx
        ][
            "sample_id"
        ]
    )


# ============================================================
# GPT-2 MASKED MEAN
# ============================================================

def masked_mean(
    hidden,
    attention_mask,
):

    mask = (

        attention_mask
        .unsqueeze(
            -1
        )
        .to(
            hidden.dtype
        )
    )


    return (

        (
            hidden
            *
            mask
        )
        .sum(
            dim=1
        )

        /

        mask
        .sum(
            dim=1
        )
        .clamp_min(
            1.0
        )
    )


# ============================================================
# BATCH GPT-2 ENCODING
#
# Everything gets precomputed ONCE.
#
# Training after this is only tiny MLPs.
# ============================================================

@torch.inference_mode()
def encode_texts(
    texts,
    max_tokens,
    label,
):

    all_vectors = []


    total = len(
        texts
    )


    for start in range(

        0,

        total,

        args.gpt_batch_size,
    ):

        batch_texts = texts[

            start:
            start
            +
            args.gpt_batch_size
        ]


        encoded = tokenizer(

            batch_texts,

            padding=True,

            truncation=True,

            max_length=max_tokens,

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


        output = (

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


        vectors = masked_mean(

            output.last_hidden_state,

            attention_mask,
        )


        all_vectors.append(

            vectors
            .float()
            .cpu()
        )


        completed = min(

            start
            +
            args.gpt_batch_size,

            total,
        )


        if (
            completed % 250 < args.gpt_batch_size

            or

            completed == total
        ):

            print(

                f"{label}: "
                f"{completed}/"
                f"{total}"
            )


    return torch.cat(

        all_vectors,

        dim=0,
    )


# ============================================================
# PRECOMPUTE MEMORY REPRESENTATIONS
# ============================================================

section(
    "PRECOMPUTE MEMORY REPRESENTATIONS"
)


for conversation in conversations:

    texts = [

        memory[
            "text"
        ]

        for memory
        in conversation[
            "memories"
        ]
    ]


    conversation[
        "memory_base"
    ] = encode_texts(

        texts,

        args.max_memory_tokens,

        (
            "MEM "
            +
            conversation[
                "sample_id"
            ]
        ),
    )


print(
    "Memory representations cached."
)


# ============================================================
# PRECOMPUTE QUERY REPRESENTATIONS
# ============================================================

section(
    "PRECOMPUTE QUESTION REPRESENTATIONS"
)


for conversation in conversations:

    texts = [

        qa[
            "question"
        ]

        for qa
        in conversation[
            "qas"
        ]
    ]


    if texts:

        conversation[
            "question_base"
        ] = encode_texts(

            texts,

            args.max_question_tokens,

            (
                "QUERY "
                +
                conversation[
                    "sample_id"
                ]
            ),
        )


    else:

        conversation[
            "question_base"
        ] = torch.empty(

            0,

            GPT_DIM,
        )


print(
    "Question representations cached."
)


# ============================================================
# RETRIEVAL HEADS
# ============================================================

class RetrievalEncoder(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout,
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


query_encoder = RetrievalEncoder(

    GPT_DIM,

    args.hidden_dim,

    args.embedding_dim,

    args.dropout,

).to(
    DEVICE
)


memory_encoder = RetrievalEncoder(

    GPT_DIM,

    args.hidden_dim,

    args.embedding_dim,

    args.dropout,

).to(
    DEVICE
)


# ============================================================
# FLATTEN TRAINING QA REFERENCES
# ============================================================

def collect_question_refs(
    conversation_ids,
):

    refs = []


    for conversation_id in (
        conversation_ids
    ):

        conversation = conversations[
            conversation_id
        ]


        for qa_position, qa in enumerate(

            conversation[
                "qas"
            ]
        ):

            positive_indices = []


            for evidence_id in (

                qa[
                    "evidence"
                ]
            ):

                positive_indices.append(

                    conversation[
                        "turn_by_id"
                    ][
                        evidence_id
                    ]
                )


            if not positive_indices:

                continue


            refs.append(

                {
                    "conversation_id":
                        conversation_id,

                    "qa_position":
                        qa_position,

                    "positive_indices":
                        positive_indices,
                }
            )


    return refs


train_refs = collect_question_refs(
    train_ids
)


valid_refs = collect_question_refs(
    valid_ids
)


test_refs = collect_question_refs(
    test_ids
)


section(
    "RETRIEVAL SUPERVISION"
)


print(
    "Train questions:",
    len(train_refs),
)

print(
    "Valid questions:",
    len(valid_refs),
)

print(
    "Test questions:",
    len(test_refs),
)


# ============================================================
# HARD NEGATIVE CACHE
# ============================================================

hard_negative_cache = {}


@torch.inference_mode()
def mine_hard_negatives():

    section(
        "MINE HARD NEGATIVES"
    )


    query_encoder.eval()

    memory_encoder.eval()


    hard_negative_cache.clear()


    for ref_index, ref in enumerate(
        train_refs
    ):

        conversation = conversations[
            ref[
                "conversation_id"
            ]
        ]


        q_base = (

            conversation[
                "question_base"
            ][
                ref[
                    "qa_position"
                ]
            ]
            .unsqueeze(
                0
            )
            .to(
                DEVICE
            )
        )


        memory_base = (

            conversation[
                "memory_base"
            ]
            .to(
                DEVICE
            )
        )


        q = query_encoder(
            q_base
        )[0]


        keys = memory_encoder(
            memory_base
        )


        scores = (

            keys

            @

            q
        )


        positive_set = set(

            ref[
                "positive_indices"
            ]
        )


        order = torch.argsort(

            scores,

            descending=True,
        )


        negatives = []


        for index_tensor in order:

            index = int(

                index_tensor.item()
            )


            if index in positive_set:

                continue


            negatives.append(
                index
            )


            if (
                len(
                    negatives
                )
                >=
                args.hard_negatives
            ):

                break


        hard_negative_cache[
            ref_index
        ] = negatives


    print(
        "Hard negatives mined for",
        len(
            hard_negative_cache
        ),
        "training questions."
    )


# ============================================================
# TRAIN DATASET
# ============================================================

class RetrievalTrainingDataset(
    Dataset
):

    def __init__(
        self,
        refs,
    ):

        self.refs = refs


    def __len__(
        self,
    ):

        return len(
            self.refs
        )


    def __getitem__(
        self,
        index,
    ):

        ref = self.refs[
            index
        ]


        conversation = conversations[
            ref[
                "conversation_id"
            ]
        ]


        q_base = (

            conversation[
                "question_base"
            ][
                ref[
                    "qa_position"
                ]
            ]
        )


        # Random positive each epoch.
        #
        # Multi-evidence questions therefore expose all gold
        # evidence over training.

        positive_index = random.choice(

            ref[
                "positive_indices"
            ]
        )


        positive_base = (

            conversation[
                "memory_base"
            ][
                positive_index
            ]
        )


        # ----------------------------------------------------
        # Hard negatives
        # ----------------------------------------------------

        negatives = (
            hard_negative_cache.get(
                index,
                [],
            )
        )


        if not negatives:

            positive_set = set(

                ref[
                    "positive_indices"
                ]
            )


            candidate_negatives = [

                i

                for i in range(

                    len(
                        conversation[
                            "memories"
                        ]
                    )
                )

                if i not in positive_set
            ]


            random.shuffle(
                candidate_negatives
            )


            negatives = (
                candidate_negatives[
                    :args.hard_negatives
                ]
            )


        negative_vectors = [

            conversation[
                "memory_base"
            ][
                negative_index
            ]

            for negative_index
            in negatives
        ]


        while (
            len(
                negative_vectors
            )
            <
            args.hard_negatives
        ):

            negative_vectors.append(

                positive_base.clone()
                * 0.0
            )


        negatives_tensor = torch.stack(

            negative_vectors,

            dim=0,
        )


        return (

            q_base,

            positive_base,

            negatives_tensor,
        )


train_dataset = RetrievalTrainingDataset(

    train_refs
)


train_loader = DataLoader(

    train_dataset,

    batch_size=
        args.batch_size,

    shuffle=True,

    drop_last=False,
)


# ============================================================
# LOSS
#
# PART 1:
# In-batch InfoNCE
#
# Every positive memory of every OTHER query becomes a
# negative.
#
#
# PART 2:
# Explicit hard-negative margin loss
# ============================================================

def retrieval_loss(
    q_base,
    positive_base,
    negative_base,
):

    q_base = q_base.to(
        DEVICE
    )


    positive_base = positive_base.to(
        DEVICE
    )


    negative_base = negative_base.to(
        DEVICE
    )


    q = query_encoder(
        q_base
    )


    positive = memory_encoder(
        positive_base
    )


    # --------------------------------------------------------
    # IN-BATCH CONTRASTIVE LOSS
    # --------------------------------------------------------

    logits = (

        q

        @

        positive.T

        /

        args.temperature
    )


    labels = torch.arange(

        logits.shape[
            0
        ],

        device=
            DEVICE,
    )


    loss_q_to_k = (
        F.cross_entropy(

            logits,

            labels,
        )
    )


    loss_k_to_q = (
        F.cross_entropy(

            logits.T,

            labels,
        )
    )


    contrastive_loss = (

        0.5

        *

        (
            loss_q_to_k

            +

            loss_k_to_q
        )
    )


    # --------------------------------------------------------
    # HARD NEGATIVES
    # --------------------------------------------------------

    batch_size = (
        negative_base.shape[
            0
        ]
    )


    num_negatives = (
        negative_base.shape[
            1
        ]
    )


    negative_flat = (

        negative_base.reshape(

            batch_size
            *
            num_negatives,

            -1,
        )
    )


    negative_keys = memory_encoder(

        negative_flat
    )


    negative_keys = (

        negative_keys.reshape(

            batch_size,

            num_negatives,

            -1,
        )
    )


    positive_score = (

        q

        *

        positive
    ).sum(
        dim=-1
    )


    negative_scores = torch.einsum(

        "bd,bnd->bn",

        q,

        negative_keys,
    )


    hardest_negative = (

        negative_scores.max(
            dim=-1
        ).values
    )


    margin_loss = (

        F.relu(

            args.margin

            -

            positive_score

            +

            hardest_negative
        )
        .mean()
    )


    total_loss = (

        contrastive_loss

        +

        args.hard_negative_weight

        *

        margin_loss
    )


    return (

        total_loss,

        contrastive_loss.detach(),

        margin_loss.detach(),
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    rankings,
    gold_lists,
):

    totals = {

        "mrr":
            0.0,
    }


    for k in TOP_KS:

        totals[
            f"hit@{k}"
        ] = 0.0

        totals[
            f"recall@{k}"
        ] = 0.0

        totals[
            f"all@{k}"
        ] = 0.0


    n = len(
        rankings
    )


    for ranking, gold_list in zip(

        rankings,

        gold_lists,
    ):

        gold = set(
            gold_list
        )


        first_rank = None


        for rank, index in enumerate(

            ranking,

            start=1,
        ):

            if index in gold:

                first_rank = rank

                break


        if first_rank is not None:

            totals[
                "mrr"
            ] += (

                1.0

                /

                first_rank
            )


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


            totals[
                f"hit@{k}"
            ] += (

                1.0

                if overlap

                else 0.0
            )


            totals[
                f"recall@{k}"
            ] += (

                len(
                    overlap
                )

                /

                len(
                    gold
                )
            )


            totals[
                f"all@{k}"
            ] += (

                1.0

                if gold.issubset(
                    retrieved
                )

                else 0.0
            )


    if n == 0:

        return {

            key:
                0.0

            for key
            in totals
        }


    return {

        key:
            value / n

        for key, value
        in totals.items()
    }


# ============================================================
# FULL-CONVERSATION RETRIEVAL
# ============================================================

@torch.inference_mode()
def evaluate_full_conversation(
    conversation_ids,
    trained=True,
):

    query_encoder.eval()

    memory_encoder.eval()


    rankings = []

    gold_lists = []


    for conversation_id in (
        conversation_ids
    ):

        conversation = conversations[
            conversation_id
        ]


        memory_base = (

            conversation[
                "memory_base"
            ]
            .to(
                DEVICE
            )
        )


        if trained:

            memory_keys = memory_encoder(

                memory_base
            )

        else:

            memory_keys = F.normalize(

                memory_base,

                p=2,

                dim=-1,

                eps=1e-8,
            )


        for qa_position, qa in enumerate(

            conversation[
                "qas"
            ]
        ):

            q_base = (

                conversation[
                    "question_base"
                ][
                    qa_position
                ]
                .unsqueeze(
                    0
                )
                .to(
                    DEVICE
                )
            )


            if trained:

                q = query_encoder(
                    q_base
                )[0]

            else:

                q = F.normalize(

                    q_base[
                        0
                    ],

                    p=2,

                    dim=-1,

                    eps=1e-8,
                )


            scores = (

                memory_keys

                @

                q
            )


            order = torch.argsort(

                scores,

                descending=True,
            )


            ranking = [

                int(
                    index.item()
                )

                for index
                in order
            ]


            gold = [

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


            rankings.append(
                ranking
            )


            gold_lists.append(
                gold
            )


    return compute_metrics(

        rankings,

        gold_lists,
    )


# ============================================================
# CONTROLLED 8-SLOT RETRIEVAL
# ============================================================

@torch.inference_mode()
def evaluate_eight_slot(
    conversation_ids,
    trained=True,
):

    query_encoder.eval()

    memory_encoder.eval()


    rankings = []

    gold_lists = []


    eval_rng = random.Random(

        args.seed
        +
        9999
    )


    for conversation_id in (
        conversation_ids
    ):

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
                args.eval_pool_size
            ):

                continue


            gold_set = set(
                gold_indices
            )


            negatives = [

                index

                for index in range(

                    len(
                        conversation[
                            "memories"
                        ]
                    )
                )

                if index not in gold_set
            ]


            needed = (

                args.eval_pool_size

                -

                len(
                    gold_indices
                )
            )


            if (
                len(
                    negatives
                )
                <
                needed
            ):

                continue


            sampled_negatives = (

                eval_rng.sample(

                    negatives,

                    needed,
                )
            )


            candidate_indices = (

                gold_indices

                +

                sampled_negatives
            )


            eval_rng.shuffle(
                candidate_indices
            )


            candidate_base = (

                conversation[
                    "memory_base"
                ][
                    candidate_indices
                ]
                .to(
                    DEVICE
                )
            )


            q_base = (

                conversation[
                    "question_base"
                ][
                    qa_position
                ]
                .unsqueeze(
                    0
                )
                .to(
                    DEVICE
                )
            )


            if trained:

                q = query_encoder(
                    q_base
                )[0]


                keys = memory_encoder(

                    candidate_base
                )

            else:

                q = F.normalize(

                    q_base[
                        0
                    ],

                    p=2,

                    dim=-1,

                    eps=1e-8,
                )


                keys = F.normalize(

                    candidate_base,

                    p=2,

                    dim=-1,

                    eps=1e-8,
                )


            scores = (

                keys

                @

                q
            )


            order = torch.argsort(

                scores,

                descending=True,
            )


            # Ranking uses LOCAL pool indices.

            ranking = [

                int(
                    index.item()
                )

                for index
                in order
            ]


            local_gold = [

                local_index

                for local_index, global_index
                in enumerate(
                    candidate_indices
                )

                if global_index
                in gold_set
            ]


            rankings.append(
                ranking
            )


            gold_lists.append(
                local_gold
            )


    return compute_metrics(

        rankings,

        gold_lists,
    )


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    list(
        query_encoder.parameters()
    )

    +

    list(
        memory_encoder.parameters()
    ),

    lr=
        args.learning_rate,

    weight_decay=
        args.weight_decay,
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=
        args.epochs,

    eta_min=
        args.learning_rate
        *
        0.05,
)


# ============================================================
# BASELINE BEFORE TRAINING
# ============================================================

section(
    "BASELINE RETRIEVAL BEFORE TRAINING"
)


baseline_valid_8 = evaluate_eight_slot(

    valid_ids,

    trained=False,
)


baseline_test_8 = evaluate_eight_slot(

    test_ids,

    trained=False,
)


baseline_valid_full = (
    evaluate_full_conversation(

        valid_ids,

        trained=False,
    )
)


baseline_test_full = (
    evaluate_full_conversation(

        test_ids,

        trained=False,
    )
)


print(
    "RAW FINAL COSINE"
)


print(
    f"VALID 8-way Hit@1: "
    f"{baseline_valid_8['hit@1'] * 100:.2f}%"
)

print(
    f"TEST  8-way Hit@1: "
    f"{baseline_test_8['hit@1'] * 100:.2f}%"
)

print(
    f"VALID FULL Hit@1 : "
    f"{baseline_valid_full['hit@1'] * 100:.2f}%"
)

print(
    f"TEST  FULL Hit@1 : "
    f"{baseline_test_full['hit@1'] * 100:.2f}%"
)


# ============================================================
# INITIAL HARD NEGATIVES
#
# Randomly initialized encoders give weak hard negatives,
# but that is okay for epoch 1.
# ============================================================

mine_hard_negatives()


# ============================================================
# TRAIN
# ============================================================

section(
    "TRAIN DUAL RETRIEVER"
)


best_valid_mrr = -1.0

best_epoch = None

best_query_state = None

best_memory_state = None


for epoch in range(

    1,

    args.epochs + 1,
):

    query_encoder.train()

    memory_encoder.train()


    total_loss_sum = 0.0

    contrastive_sum = 0.0

    margin_sum = 0.0

    batches = 0


    for (
        q_base,
        positive_base,
        negative_base,

    ) in train_loader:

        (
            loss,
            contrastive_loss,
            margin_loss,

        ) = retrieval_loss(

            q_base,

            positive_base,

            negative_base,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            list(
                query_encoder.parameters()
            )

            +

            list(
                memory_encoder.parameters()
            ),

            max_norm=1.0,
        )


        optimizer.step()


        total_loss_sum += (
            loss.item()
        )


        contrastive_sum += (
            contrastive_loss.item()
        )


        margin_sum += (
            margin_loss.item()
        )


        batches += 1


    scheduler.step()


    # --------------------------------------------------------
    # Evaluate on HELD-OUT conversations.
    # --------------------------------------------------------

    valid_8 = evaluate_eight_slot(

        valid_ids,

        trained=True,
    )


    valid_full = (
        evaluate_full_conversation(

            valid_ids,

            trained=True,
        )
    )


    print()

    print(

        f"Epoch {epoch:02d}/"
        f"{args.epochs}"

        f" | loss="
        f"{total_loss_sum / max(batches, 1):.4f}"

        f" | InfoNCE="
        f"{contrastive_sum / max(batches, 1):.4f}"

        f" | margin="
        f"{margin_sum / max(batches, 1):.4f}"

        f" | Val8 H@1="
        f"{valid_8['hit@1'] * 100:.2f}%"

        f" | Val8 MRR="
        f"{valid_8['mrr']:.4f}"

        f" | ValFull H@1="
        f"{valid_full['hit@1'] * 100:.2f}%"

        f" | ValFull MRR="
        f"{valid_full['mrr']:.4f}"
    )


    # --------------------------------------------------------
    # Select model using FULL-conversation validation MRR.
    #
    # This prevents selecting purely for an easy 8-way task.
    # --------------------------------------------------------

    if (
        valid_full[
            "mrr"
        ]
        >
        best_valid_mrr
    ):

        best_valid_mrr = (

            valid_full[
                "mrr"
            ]
        )


        best_epoch = epoch


        best_query_state = {

            key:
                value
                .detach()
                .cpu()
                .clone()

            for key, value
            in query_encoder
            .state_dict()
            .items()
        }


        best_memory_state = {

            key:
                value
                .detach()
                .cpu()
                .clone()

            for key, value
            in memory_encoder
            .state_dict()
            .items()
        }


    # --------------------------------------------------------
    # Refresh hard negatives every 2 epochs.
    # --------------------------------------------------------

    if (
        epoch % 2 == 0

        and

        epoch
        <
        args.epochs
    ):

        mine_hard_negatives()


# ============================================================
# RESTORE BEST
# ============================================================

query_encoder.load_state_dict(

    best_query_state
)


memory_encoder.load_state_dict(

    best_memory_state
)


query_encoder.eval()

memory_encoder.eval()


section(
    "BEST DUAL ENCODER"
)


print(
    "Best epoch:",
    best_epoch,
)

print(
    "Best validation full MRR:",
    f"{best_valid_mrr:.4f}",
)


# ============================================================
# DUAL ENCODER FINAL EVALUATION
# ============================================================

dual_valid_8 = evaluate_eight_slot(

    valid_ids,

    trained=True,
)


dual_test_8 = evaluate_eight_slot(

    test_ids,

    trained=True,
)


dual_valid_full = evaluate_full_conversation(

    valid_ids,

    trained=True,
)


dual_test_full = evaluate_full_conversation(

    test_ids,

    trained=True,
)


# ============================================================
# RERANKER
#
# Receives:
#
# q
# k
# |q-k|
# q*k
#
# and predicts relevance.
#
# This is trained ONLY using TRAIN conversations.
# ============================================================

class PairReranker(
    nn.Module
):

    def __init__(
        self,
        embedding_dim,
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(

                embedding_dim * 4,

                512,
            ),

            nn.GELU(),

            nn.Dropout(
                0.15
            ),

            nn.Linear(
                512,
                128,
            ),

            nn.GELU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                128,
                1,
            ),
        )


    def forward(
        self,
        q,
        k,
    ):

        features = torch.cat(

            [
                q,

                k,

                torch.abs(
                    q - k
                ),

                q * k,
            ],

            dim=-1,
        )


        return (

            self.network(
                features
            )
            .squeeze(
                -1
            )
        )


reranker = PairReranker(

    args.embedding_dim

).to(
    DEVICE
)


# ============================================================
# BUILD RERANKER TRAIN PAIRS
#
# Positives = gold evidence
#
# Negatives = highest dual-encoder scoring WRONG turns
# ============================================================

@torch.inference_mode()
def build_reranker_pairs():

    query_encoder.eval()

    memory_encoder.eval()


    q_features = []

    k_features = []

    labels = []


    for ref in train_refs:

        conversation = conversations[
            ref[
                "conversation_id"
            ]
        ]


        q_base = (

            conversation[
                "question_base"
            ][
                ref[
                    "qa_position"
                ]
            ]
            .unsqueeze(
                0
            )
            .to(
                DEVICE
            )
        )


        memory_base = (

            conversation[
                "memory_base"
            ]
            .to(
                DEVICE
            )
        )


        q = query_encoder(
            q_base
        )[0]


        keys = memory_encoder(
            memory_base
        )


        scores = (
            keys
            @
            q
        )


        positive_set = set(

            ref[
                "positive_indices"
            ]
        )


        # ----------------------------------------------------
        # ALL positive evidence
        # ----------------------------------------------------

        for positive_index in (
            ref[
                "positive_indices"
            ]
        ):

            q_features.append(
                q.cpu()
            )

            k_features.append(

                keys[
                    positive_index
                ].cpu()
            )

            labels.append(
                1.0
            )


        # ----------------------------------------------------
        # Top wrong memories
        # ----------------------------------------------------

        order = torch.argsort(

            scores,

            descending=True,
        )


        added = 0


        for index_tensor in order:

            index = int(
                index_tensor.item()
            )


            if index in positive_set:

                continue


            q_features.append(
                q.cpu()
            )


            k_features.append(

                keys[
                    index
                ].cpu()
            )


            labels.append(
                0.0
            )


            added += 1


            if (
                added
                >=
                args.reranker_negatives
            ):

                break


    return (

        torch.stack(
            q_features
        ),

        torch.stack(
            k_features
        ),

        torch.tensor(
            labels,
            dtype=torch.float32,
        ),
    )


# ============================================================
# TRAIN RERANKER
# ============================================================

if not args.disable_reranker:

    section(
        "TRAIN TOP-K RERANKER"
    )


    (
        rerank_q,
        rerank_k,
        rerank_labels,

    ) = build_reranker_pairs()


    print(
        "Reranker pairs:",
        len(
            rerank_labels
        ),
    )


    print(
        "Positive pairs:",
        int(
            rerank_labels.sum().item()
        ),
    )


    rerank_dataset = torch.utils.data.TensorDataset(

        rerank_q,

        rerank_k,

        rerank_labels,
    )


    rerank_loader = DataLoader(

        rerank_dataset,

        batch_size=256,

        shuffle=True,
    )


    rerank_optimizer = torch.optim.AdamW(

        reranker.parameters(),

        lr=2e-4,

        weight_decay=1e-4,
    )


    positive_count = (

        rerank_labels.sum()
    )


    negative_count = (

        len(
            rerank_labels
        )

        -

        positive_count
    )


    pos_weight = (

        negative_count

        /

        positive_count.clamp_min(
            1.0
        )
    ).to(
        DEVICE
    )


    criterion = nn.BCEWithLogitsLoss(

        pos_weight=
            pos_weight
    )


    for epoch in range(

        1,

        args.reranker_epochs + 1,
    ):

        reranker.train()


        loss_sum = 0.0

        count = 0


        for (
            q_batch,
            k_batch,
            y_batch,

        ) in rerank_loader:

            q_batch = q_batch.to(
                DEVICE
            )


            k_batch = k_batch.to(
                DEVICE
            )


            y_batch = y_batch.to(
                DEVICE
            )


            logits = reranker(

                q_batch,

                k_batch,
            )


            loss = criterion(

                logits,

                y_batch,
            )


            rerank_optimizer.zero_grad(
                set_to_none=True
            )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(

                reranker.parameters(),

                1.0,
            )


            rerank_optimizer.step()


            loss_sum += (
                loss.item()
            )

            count += 1


        if (
            epoch == 1

            or

            epoch % 5 == 0

            or

            epoch
            ==
            args.reranker_epochs
        ):

            print(

                f"Reranker epoch "
                f"{epoch:02d}/"
                f"{args.reranker_epochs}"

                f" | loss="
                f"{loss_sum / max(count, 1):.4f}"
            )


else:

    print(
        "Reranker disabled."
    )


# ============================================================
# EVALUATE DUAL + RERANKER
#
# First:
#     dual encoder retrieves TOP-5
#
# Then:
#     reranker orders those candidates.
#
# ============================================================

@torch.inference_mode()
def evaluate_with_reranker(
    conversation_ids,
    pool_size=None,
):

    query_encoder.eval()

    memory_encoder.eval()

    reranker.eval()


    rankings = []

    gold_lists = []


    eval_rng = random.Random(

        args.seed
        +
        202020
    )


    for conversation_id in (
        conversation_ids
    ):

        conversation = conversations[
            conversation_id
        ]


        for qa_position, qa in enumerate(

            conversation[
                "qas"
            ]
        ):

            gold_global = [

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


            gold_set = set(
                gold_global
            )


            if pool_size is None:

                candidate_indices = list(

                    range(

                        len(
                            conversation[
                                "memories"
                            ]
                        )
                    )
                )


            else:

                if (
                    len(
                        gold_global
                    )
                    >
                    pool_size
                ):

                    continue


                negatives = [

                    index

                    for index in range(

                        len(
                            conversation[
                                "memories"
                            ]
                        )
                    )

                    if index not in gold_set
                ]


                needed = (

                    pool_size

                    -

                    len(
                        gold_global
                    )
                )


                if (
                    len(
                        negatives
                    )
                    <
                    needed
                ):

                    continue


                candidate_indices = (

                    gold_global

                    +

                    eval_rng.sample(

                        negatives,

                        needed,
                    )
                )


                eval_rng.shuffle(
                    candidate_indices
                )


            q_base = (

                conversation[
                    "question_base"
                ][
                    qa_position
                ]
                .unsqueeze(
                    0
                )
                .to(
                    DEVICE
                )
            )


            candidate_base = (

                conversation[
                    "memory_base"
                ][
                    candidate_indices
                ]
                .to(
                    DEVICE
                )
            )


            q = query_encoder(
                q_base
            )[0]


            keys = memory_encoder(

                candidate_base
            )


            dual_scores = (

                keys

                @

                q
            )


            # ------------------------------------------------
            # retrieve up to top 5
            # ------------------------------------------------

            retrieve_k = min(

                5,

                len(
                    candidate_indices
                ),
            )


            top_values, top_indices = torch.topk(

                dual_scores,

                k=
                    retrieve_k,
            )


            top_keys = keys[
                top_indices
            ]


            q_repeat = (

                q.unsqueeze(
                    0
                )
                .expand(

                    retrieve_k,

                    -1,
                )
            )


            rerank_scores = reranker(

                q_repeat,

                top_keys,
            )


            rerank_order = torch.argsort(

                rerank_scores,

                descending=True,
            )


            reranked_local = [

                int(

                    top_indices[
                        rerank_position
                    ].item()
                )

                for rerank_position
                in rerank_order
            ]


            # Append remaining dual candidates after reranked top5.

            dual_order = torch.argsort(

                dual_scores,

                descending=True,
            )


            used = set(
                reranked_local
            )


            final_local_ranking = list(
                reranked_local
            )


            for index_tensor in dual_order:

                local_index = int(

                    index_tensor.item()
                )


                if local_index not in used:

                    final_local_ranking.append(
                        local_index
                    )

                    used.add(
                        local_index
                    )


            if pool_size is None:

                ranking = [

                    candidate_indices[
                        local_index
                    ]

                    for local_index
                    in final_local_ranking
                ]


                gold = gold_global


            else:

                # Metrics expect local indices in controlled pool.

                ranking = (
                    final_local_ranking
                )


                gold = [

                    local_index

                    for local_index, global_index
                    in enumerate(
                        candidate_indices
                    )

                    if global_index
                    in gold_set
                ]


            rankings.append(
                ranking
            )


            gold_lists.append(
                gold
            )


    return compute_metrics(

        rankings,

        gold_lists,
    )


# ============================================================
# FINAL RERANKER EVALUATION
# ============================================================

if not args.disable_reranker:

    rerank_valid_8 = (
        evaluate_with_reranker(

            valid_ids,

            pool_size=
                args.eval_pool_size,
        )
    )


    rerank_test_8 = (
        evaluate_with_reranker(

            test_ids,

            pool_size=
                args.eval_pool_size,
        )
    )


    rerank_valid_full = (
        evaluate_with_reranker(

            valid_ids,

            pool_size=None,
        )
    )


    rerank_test_full = (
        evaluate_with_reranker(

            test_ids,

            pool_size=None,
        )
    )


else:

    rerank_valid_8 = (
        dual_valid_8
    )

    rerank_test_8 = (
        dual_test_8
    )

    rerank_valid_full = (
        dual_valid_full
    )

    rerank_test_full = (
        dual_test_full
    )


# ============================================================
# FINAL RESULTS
# ============================================================

section(
    "FINAL CONTROLLED 8-SLOT RETRIEVAL"
)


print(

    f"{'Method':<30}"

    f"{'Hit@1':>10}"

    f"{'Hit@3':>10}"

    f"{'Hit@5':>10}"

    f"{'MRR':>10}"
)


methods_8 = {

    "RAW_FINAL_COSINE":
        baseline_test_8,

    "TRAINED_DUAL_ENCODER":
        dual_test_8,

    "DUAL_PLUS_RERANKER":
        rerank_test_8,
}


for name, result in (
    methods_8.items()
):

    print(

        f"{name:<30}"

        f"{result['hit@1'] * 100:>9.2f}%"

        f"{result['hit@3'] * 100:>9.2f}%"

        f"{result['hit@5'] * 100:>9.2f}%"

        f"{result['mrr']:>10.4f}"
    )


# ============================================================
# FULL CONVERSATION
# ============================================================

section(
    "FINAL FULL-CONVERSATION RETRIEVAL"
)


print(

    f"{'Method':<30}"

    f"{'Hit@1':>10}"

    f"{'Hit@3':>10}"

    f"{'Hit@5':>10}"

    f"{'MRR':>10}"
)


methods_full = {

    "RAW_FINAL_COSINE":
        baseline_test_full,

    "TRAINED_DUAL_ENCODER":
        dual_test_full,

    "DUAL_PLUS_RERANKER":
        rerank_test_full,
}


for name, result in (
    methods_full.items()
):

    print(

        f"{name:<30}"

        f"{result['hit@1'] * 100:>9.2f}%"

        f"{result['hit@3'] * 100:>9.2f}%"

        f"{result['hit@5'] * 100:>9.2f}%"

        f"{result['mrr']:>10.4f}"
    )


# ============================================================
# VALIDATION RESULTS
# ============================================================

section(
    "VALIDATION CHECK"
)


print(
    "Best epoch:",
    best_epoch,
)


print()

print(
    "VALID 8-way:"
)

print(
    "Raw:",
    f"{baseline_valid_8['hit@1'] * 100:.2f}%"
)

print(
    "Dual:",
    f"{dual_valid_8['hit@1'] * 100:.2f}%"
)

print(
    "Reranked:",
    f"{rerank_valid_8['hit@1'] * 100:.2f}%"
)


print()

print(
    "VALID full:"
)

print(
    "Raw:",
    f"{baseline_valid_full['hit@1'] * 100:.2f}%"
)

print(
    "Dual:",
    f"{dual_valid_full['hit@1'] * 100:.2f}%"
)

print(
    "Reranked:",
    f"{rerank_valid_full['hit@1'] * 100:.2f}%"
)


# ============================================================
# IMPROVEMENT
# ============================================================

section(
    "IMPROVEMENT OVER RAW RETRIEVAL"
)


dual_gain_8 = (

    dual_test_8[
        "hit@1"
    ]

    -

    baseline_test_8[
        "hit@1"
    ]
)


rerank_gain_8 = (

    rerank_test_8[
        "hit@1"
    ]

    -

    baseline_test_8[
        "hit@1"
    ]
)


dual_gain_full = (

    dual_test_full[
        "hit@1"
    ]

    -

    baseline_test_full[
        "hit@1"
    ]
)


rerank_gain_full = (

    rerank_test_full[
        "hit@1"
    ]

    -

    baseline_test_full[
        "hit@1"
    ]
)


print(
    "8-slot dual gain:",
    f"{dual_gain_8 * 100:+.2f} points"
)

print(
    "8-slot reranked gain:",
    f"{rerank_gain_8 * 100:+.2f} points"
)

print()

print(
    "Full-conversation dual gain:",
    f"{dual_gain_full * 100:+.2f} points"
)

print(
    "Full-conversation reranked gain:",
    f"{rerank_gain_full * 100:+.2f} points"
)


# ============================================================
# TARGET CHECK
# ============================================================

section(
    "80% TARGET CHECK"
)


best_test_8 = max(

    dual_test_8[
        "hit@1"
    ],

    rerank_test_8[
        "hit@1"
    ],
)


print(
    "Best held-out TEST 8-slot Hit@1:",
    f"{best_test_8 * 100:.2f}%"
)


if best_test_8 >= 0.80:

    print()

    print(
        "TARGET REACHED:"
    )

    print(
        "Held-out conversation retrieval "
        "exceeded 80% Hit@1."
    )


elif best_test_8 >= 0.65:

    print()

    print(
        "RESULT:"
    )

    print(
        "Strong improvement, but below 80%."
    )

    print(
        "Next step should focus on better "
        "hard negatives / contextual memory "
        "rather than changing the memory bank."
    )


elif best_test_8 >= 0.50:

    print()

    print(
        "RESULT:"
    )

    print(
        "Retriever learned meaningful semantic "
        "matching, but more retrieval work is needed."
    )


else:

    print()

    print(
        "RESULT:"
    )

    print(
        "Generalization to unseen conversations "
        "is still weak."
    )

    print(
        "Do NOT integrate into the main architecture yet."
    )


# ============================================================
# SAVE
# ============================================================

section(
    "SAVE CHECKPOINT"
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
        "query_encoder_state_dict":
            query_encoder
            .state_dict(),

        "memory_encoder_state_dict":
            memory_encoder
            .state_dict(),

        "reranker_state_dict":
            (
                reranker.state_dict()

                if not args.disable_reranker

                else None
            ),

        "input_dim":
            GPT_DIM,

        "hidden_dim":
            args.hidden_dim,

        "embedding_dim":
            args.embedding_dim,

        "temperature":
            args.temperature,

        "context_radius":
            args.context_radius,

        "best_epoch":
            best_epoch,

        "train_conversations":
            [
                conversations[
                    index
                ][
                    "sample_id"
                ]

                for index
                in train_ids
            ],

        "valid_conversations":
            [
                conversations[
                    index
                ][
                    "sample_id"
                ]

                for index
                in valid_ids
            ],

        "test_conversations":
            [
                conversations[
                    index
                ][
                    "sample_id"
                ]

                for index
                in test_ids
            ],

        "baseline_test_8":
            baseline_test_8,

        "dual_test_8":
            dual_test_8,

        "rerank_test_8":
            rerank_test_8,

        "baseline_test_full":
            baseline_test_full,

        "dual_test_full":
            dual_test_full,

        "rerank_test_full":
            rerank_test_full,
    },

    output_path,
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
    "TRAINED:"
)

print(
    "  Query retrieval encoder"
)

print(
    "  Memory retrieval encoder"
)

if not args.disable_reranker:

    print(
        "  Pairwise reranker"
    )


print()

print(
    "FROZEN:"
)

print(
    "  GPT-2"
)

print(
    "  CandidateWriter"
)

print(
    "  MemoryReader"
)

print(
    "  Gates"
)

print(
    "  Router"
)

print(
    "  MemoryBank architecture"
)


print()

print(
    "NO models/ source files modified."
)

print(
    "Retrieval trained directly against "
    "LoCoMo gold evidence."
)