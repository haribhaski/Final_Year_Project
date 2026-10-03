# ============================================================
# MULTI-DATASET VALUE REPRESENTATION ABLATION
#
# PURPOSE
# ============================================================
#
# We have already found:
#
#   ADDRESS:
#       Layer-1 + contrastive encoder works very well.
#
#   VALUE:
#       final-layer masked mean works well on synthetic data,
#       but drops significantly on natural WikiText.
#
#
# NOW:
#
# Find a VALUE representation that works consistently across
# DIFFERENT benchmark domains.
#
#
# DATASETS
# ============================================================
#
# 1. WikiText-103
#       encyclopedia / article-style text
#
# 2. PG-19
#       long Project Gutenberg books
#
# 3. LoCoMo
#       long conversational-memory benchmark
#
#
# IMPORTANT
# ============================================================
#
# This is a REPRESENTATION DIAGNOSTIC.
#
# It is NOT claiming an official WikiText / PG19 / LoCoMo
# benchmark score.
#
# We use REAL text from each dataset and construct the same
# controlled content-retention task:
#
#       real sentence / utterance
#              ↓
#       frozen GPT-2 representation
#              ↓
#       small linear probe
#              ↓
#       which naturally occurring content word is present?
#
#
# This lets us compare representations fairly:
#
#   L1 masked mean
#   L4 masked mean
#   L8 masked mean
#   L10 masked mean
#   L11 masked mean
#   L12 masked mean
#   L12 last token
#   mean of layers 8-11
#   mean of layers 9-12
#
#
# NO architecture modifications.
# NO GPT-2 training.
# NO CandidateWriter.
# NO memory reader.
# NO scalar/vector gate.
#
# Only linear probes are trained.
#
#
# RUN ALL AVAILABLE:
#
# python multidataset_value_representation_ablation.py \
#     --datasets wikitext pg19 locomo \
#     2>&1 | tee multidataset_value_representation_ablation.log
#
#
# RUN ONLY WIKITEXT + PG19:
#
# python multidataset_value_representation_ablation.py \
#     --datasets wikitext pg19 \
#     2>&1 | tee multidataset_value_representation_ablation.log
#
# ============================================================


import os
import re
import csv
import json
import math
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict, OrderedDict

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
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(

    "--datasets",

    nargs="+",

    default=[
        "wikitext",
        "pg19",
        "locomo",
    ],

    choices=[
        "wikitext",
        "pg19",
        "locomo",
    ],
)


parser.add_argument(

    "--locomo-path",

    type=str,

    default=
        "data/locomo/locomo10.json",
)


parser.add_argument(

    "--output-dir",

    type=str,

    default=
        "outputs/"
        "multidataset_value_ablation",
)


args = parser.parse_args()


# ============================================================
# GLOBAL CONFIG
# ============================================================

MODEL_NAME = "gpt2"

CHECKPOINT = (
    "outputs/"
    "retrieval_gradient_test/"
    "checkpoint_best.pt"
)


DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)


SEED = 2060


# ------------------------------------------------------------
# Number of classes in diagnostic task
# ------------------------------------------------------------

NUM_CLASSES = 16


# ------------------------------------------------------------
# Desired balanced examples PER CLASS
# ------------------------------------------------------------

TRAIN_PER_CLASS = 100

VALID_PER_CLASS = 20

TEST_PER_CLASS = 20


# ------------------------------------------------------------
# Probe training
# ------------------------------------------------------------

PROBE_EPOCHS = 40

PROBE_BATCH_SIZE = 128

PROBE_LR = 1e-2

PROBE_WEIGHT_DECAY = 1e-4


# ------------------------------------------------------------
# Sentence constraints
# ------------------------------------------------------------

MIN_WORDS = 8

MAX_WORDS = 70


# ------------------------------------------------------------
# Maximum amount of source text processed.
#
# Keeps experiment manageable.
# ------------------------------------------------------------

MAX_TRAIN_SENTENCES = 40000

MAX_VALID_SENTENCES = 12000

MAX_TEST_SENTENCES = 12000


# ------------------------------------------------------------
# PG-19:
#
# Streaming prevents downloading the entire ~GB-scale
# training corpus.
# ------------------------------------------------------------

PG19_DATASET_ID = "emozilla/pg19"

PG19_MAX_TRAIN_DOCS = 250

PG19_MAX_VALID_DOCS = 50

PG19_MAX_TEST_DOCS = 100


# ============================================================
# REPRESENTATIONS TO TEST
#
# GPT-2 hidden_states:
#
# hidden_states[0]  = embedding output
# hidden_states[1]  = after block 1
# ...
# hidden_states[12] = after block 12
# ============================================================

REPRESENTATIONS = [

    "L1_MEAN",

    "L4_MEAN",

    "L8_MEAN",

    "L10_MEAN",

    "L11_MEAN",

    "L12_MEAN",

    "L12_LAST",

    "L8_TO_L11_MEAN",

    "L9_TO_L12_MEAN",
]


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {

    "the", "a", "an",
    "and", "or", "but",

    "if", "then", "than",

    "of", "to", "in",
    "on", "at", "by",

    "for", "with", "from",
    "as",

    "is", "was", "are",
    "were", "be", "been",
    "being",

    "has", "have", "had",

    "do", "does", "did",

    "this", "that",
    "these", "those",

    "it", "its",

    "he", "she",
    "they", "them",

    "his", "her",
    "their",

    "there", "here",

    "which", "who",
    "whom", "whose",

    "what", "when",
    "where", "why",
    "how",

    "not", "no", "yes",

    "also",

    "into", "over",
    "under", "after",
    "before", "during",
    "between", "through",

    "about", "against",
    "without", "within",

    "while", "although",
    "because", "so",

    "up", "down",
    "out", "off",

    "again", "further",
    "once",

    "more", "most",

    "other", "some",
    "such",

    "only", "own",
    "same", "very",

    "can", "could",
    "would", "should",
    "may", "might",
    "will", "shall",

    "him", "hers",
    "ours", "yours",

    "new",
}


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
# OUTPUT
# ============================================================

OUTPUT_DIR = Path(
    args.output_dir
)


OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True,
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
# MEMORY CONFIG
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
# LOAD FROZEN MODEL
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


for name, value in (
    checkpoint_state.items()
):

    if (
        name in current_state

        and

        current_state[
            name
        ].shape == value.shape
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


print(
    "Entire model frozen."
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(
        text
    )


    text = text.replace(

        "@-@",

        "-"
    )


    text = text.replace(

        "@,@",

        ","
    )


    text = text.replace(

        "@.@",

        "."
    )


    text = re.sub(

        r"\s+([,.;:!?])",

        r"\1",

        text,
    )


    text = re.sub(

        r"\s+",

        " ",

        text,
    )


    return text.strip()


# ============================================================
# SENTENCE SPLITTER
# ============================================================

def split_sentences(text):

    text = clean_text(
        text
    )


    pieces = re.split(

        r"(?<=[.!?])\s+",

        text,
    )


    output = []


    for piece in pieces:

        piece = piece.strip()


        if not piece:

            continue


        words = re.findall(

            r"[A-Za-z]+",

            piece,
        )


        if not (

            MIN_WORDS
            <=
            len(words)
            <=
            MAX_WORDS
        ):

            continue


        output.append(
            piece
        )


    return output


# ============================================================
# ELIGIBLE CONTENT WORDS
# ============================================================

def eligible_words(text):

    words = re.findall(

        r"[A-Za-z]+",

        text,
    )


    output = []


    for word in words:

        word = word.lower()


        if len(word) < 4:

            continue


        if word in STOPWORDS:

            continue


        token_ids = tokenizer.encode(

            " " + word,

            add_special_tokens=False,
        )


        # Keep single-GPT2-token words only.
        #
        # This keeps all 16 classes equally defined.

        if len(token_ids) != 1:

            continue


        output.append(
            word
        )


    return output


# ============================================================
# FIND WIKITEXT FILE
# ============================================================

def find_wikitext_file(split):

    candidates = [

        Path(
            "data/wikitext-103"
        )
        /
        f"wiki.{split}.tokens",

        Path(
            "wikitext-103"
        )
        /
        f"wiki.{split}.tokens",

        Path(
            f"wiki.{split}.tokens"
        ),
    ]


    for path in candidates:

        if path.exists():

            return path


    matches = list(

        Path(".").glob(

            f"**/wiki.{split}.tokens"
        )
    )


    if matches:

        return matches[
            0
        ]


    raise FileNotFoundError(

        f"Cannot find "
        f"wiki.{split}.tokens"
    )


# ============================================================
# WIKITEXT LOADER
# ============================================================

def load_wikitext_sentences(
    split,
    maximum,
):

    path = find_wikitext_file(
        split
    )


    sentences = []


    with open(

        path,

        "r",

        encoding="utf-8",

        errors="ignore",

    ) as handle:

        buffer = []


        for raw_line in handle:

            line = (
                raw_line.strip()
            )


            # Skip section/article headings.

            if (
                line.startswith("=")
                and
                line.endswith("=")
            ):

                continue


            if not line:

                continue


            buffer.append(
                line
            )


            if len(buffer) >= 3:

                text = " ".join(
                    buffer
                )


                buffer = []


                for sentence in split_sentences(
                    text
                ):

                    sentences.append(
                        sentence
                    )


                    if (
                        len(sentences)
                        >=
                        maximum
                    ):

                        return sentences


    return sentences


# ============================================================
# PG19 LOADER
#
# HuggingFace streaming so the full training corpus does
# NOT have to be downloaded.
# ============================================================

def load_pg19_sentences(
    split,
    maximum,
    max_docs,
):

    try:

        from datasets import (
            load_dataset
        )

    except ImportError:

        raise RuntimeError(

            "PG19 requires the datasets package.\n"
            "Run:\n"
            "pip install -U datasets pyarrow"
        )


    print(
        f"Streaming PG19 {split}..."
    )


    dataset = load_dataset(

        PG19_DATASET_ID,

        split=split,

        streaming=True,
    )


    sentences = []


    for document_index, row in (
        enumerate(
            dataset
        )
    ):

        if (
            document_index
            >=
            max_docs
        ):

            break


        text = row.get(
            "text",
            ""
        )


        for sentence in split_sentences(
            text
        ):

            sentences.append(
                sentence
            )


            if (
                len(sentences)
                >=
                maximum
            ):

                return sentences


    return sentences


# ============================================================
# LOCOMO LOADER
#
# Uses official LoCoMo schema:
#
# conversation:
#   session_1:
#       speaker / text / dia_id
#   session_2:
#       ...
#
#
# IMPORTANT:
#
# LoCoMo only contains a small number of long conversations.
#
# We therefore perform a CONVERSATION-LEVEL split, rather than
# randomly splitting utterances from the same conversation.
#
# This prevents direct conversation leakage.
# ============================================================

def load_locomo_conversations():

    path = Path(
        args.locomo_path
    )


    if not path.exists():

        return None


    with open(

        path,

        "r",

        encoding="utf-8",

    ) as handle:

        data = json.load(
            handle
        )


    conversations = []


    for sample in data:

        conversation = sample.get(

            "conversation",

            {},
        )


        utterances = []


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


        for _, key in session_keys:

            turns = conversation.get(

                key,

                [],
            )


            for turn in turns:

                text = str(

                    turn.get(
                        "text",
                        ""
                    )
                ).strip()


                if not text:

                    # Some multimodal LoCoMo turns contain
                    # captions rather than ordinary text.

                    text = str(

                        turn.get(
                            "blip_caption",
                            ""
                        )
                    ).strip()


                if not text:

                    continue


                speaker = str(

                    turn.get(
                        "speaker",
                        ""
                    )
                ).strip()


                if speaker:

                    text = (
                        f"{speaker}: "
                        f"{text}"
                    )


                utterances.extend(

                    split_sentences(
                        text
                    )
                )


        if utterances:

            conversations.append(
                utterances
            )


    return conversations


def split_locomo_sentences():

    conversations = (
        load_locomo_conversations()
    )


    if conversations is None:

        return None


    rng = random.Random(
        SEED
    )


    order = list(

        range(
            len(conversations)
        )
    )


    rng.shuffle(
        order
    )


    n = len(order)


    if n < 5:

        raise RuntimeError(

            "LoCoMo file contains too few conversations."
        )


    # Conversation-level split:
    #
    # 60% train
    # 20% validation
    # 20% test

    train_end = max(

        1,

        int(
            n * 0.6
        ),
    )


    valid_end = max(

        train_end + 1,

        int(
            n * 0.8
        ),
    )


    train_ids = order[
        :train_end
    ]


    valid_ids = order[
        train_end:
        valid_end
    ]


    test_ids = order[
        valid_end:
    ]


    def flatten(
        ids,
        maximum,
    ):

        result = []


        for idx in ids:

            result.extend(

                conversations[
                    idx
                ]
            )


            if (
                len(result)
                >=
                maximum
            ):

                break


        return result[
            :maximum
        ]


    return {

        "train":
            flatten(
                train_ids,
                MAX_TRAIN_SENTENCES,
            ),

        "validation":
            flatten(
                valid_ids,
                MAX_VALID_SENTENCES,
            ),

        "test":
            flatten(
                test_ids,
                MAX_TEST_SENTENCES,
            ),
    }


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset_sentences(
    dataset_name,
):

    section(

        f"LOAD {dataset_name.upper()} REAL TEXT"
    )


    if dataset_name == "wikitext":

        splits = {

            "train":
                load_wikitext_sentences(

                    "train",

                    MAX_TRAIN_SENTENCES,
                ),

            "validation":
                load_wikitext_sentences(

                    "valid",

                    MAX_VALID_SENTENCES,
                ),

            "test":
                load_wikitext_sentences(

                    "test",

                    MAX_TEST_SENTENCES,
                ),
        }


    elif dataset_name == "pg19":

        splits = {

            "train":
                load_pg19_sentences(

                    "train",

                    MAX_TRAIN_SENTENCES,

                    PG19_MAX_TRAIN_DOCS,
                ),

            "validation":
                load_pg19_sentences(

                    "validation",

                    MAX_VALID_SENTENCES,

                    PG19_MAX_VALID_DOCS,
                ),

            "test":
                load_pg19_sentences(

                    "test",

                    MAX_TEST_SENTENCES,

                    PG19_MAX_TEST_DOCS,
                ),
        }


    elif dataset_name == "locomo":

        splits = (
            split_locomo_sentences()
        )


        if splits is None:

            print()

            print(
                "LOCOMO SKIPPED."
            )

            print(
                "Official LoCoMo file not found at:"
            )

            print(
                args.locomo_path
            )

            print()

            print(
                "Place locomo10.json there and rerun."
            )


            return None


    else:

        raise ValueError(
            dataset_name
        )


    for split_name, sentences in (
        splits.items()
    ):

        print(

            f"{split_name:<12}: "
            f"{len(sentences)} sentences"
        )


    return splits


# ============================================================
# CHOOSE 16 NATURAL CONTENT CLASSES
#
# IMPORTANT IMPROVEMENT OVER LAST EXPERIMENT:
#
# We DO NOT simply choose the final word.
#
# 1. Count naturally occurring content words.
# 2. Find words present in train + validation + test.
# 3. Select the 16 with strongest cross-split support.
# 4. Keep sentences containing EXACTLY ONE selected class word.
#
# Therefore target position can occur ANYWHERE in the sentence.
# ============================================================

def choose_answer_vocabulary(
    splits,
):

    counts = {}


    for split_name in [

        "train",
        "validation",
        "test",
    ]:

        counter = Counter()


        for sentence in (
            splits[
                split_name
            ]
        ):

            # Count each word once per sentence.

            words = set(

                eligible_words(
                    sentence
                )
            )


            counter.update(
                words
            )


        counts[
            split_name
        ] = counter


    candidates = []


    for word, train_count in (
        counts[
            "train"
        ].items()
    ):

        valid_count = (

            counts[
                "validation"
            ][
                word
            ]
        )


        test_count = (

            counts[
                "test"
            ][
                word
            ]
        )


        if (
            valid_count == 0
            or
            test_count == 0
        ):

            continue


        minimum = min(

            train_count,

            valid_count,

            test_count,
        )


        total = (

            train_count
            +
            valid_count
            +
            test_count
        )


        candidates.append(

            (
                word,
                minimum,
                total,
                train_count,
                valid_count,
                test_count,
            )
        )


    candidates.sort(

        key=lambda item: (

            item[
                1
            ],

            item[
                2
            ],
        ),

        reverse=True,
    )


    if (
        len(candidates)
        <
        NUM_CLASSES
    ):

        raise RuntimeError(

            "Not enough common natural "
            "content words."
        )


    selected = candidates[
        :NUM_CLASSES
    ]


    answers = [

        item[
            0
        ]

        for item in selected
    ]


    print()

    print(
        "Selected natural answer classes:"
    )


    print(

        f"{'Word':<18}"
        f"{'Train':>10}"
        f"{'Valid':>10}"
        f"{'Test':>10}"
    )


    for item in selected:

        (
            word,
            minimum,
            total,
            train_count,
            valid_count,
            test_count,
        ) = item


        print(

            f"{word:<18}"

            f"{train_count:>10}"

            f"{valid_count:>10}"

            f"{test_count:>10}"
        )


    return answers


# ============================================================
# BUILD CONTROLLED NATURAL-TEXT TASK
# ============================================================

def create_examples(
    sentences,
    answers,
):

    answer_set = set(
        answers
    )


    answer_to_id = {

        answer:
            idx

        for idx, answer
        in enumerate(
            answers
        )
    }


    groups = defaultdict(
        list
    )


    for sentence in sentences:

        words = set(

            word.lower()

            for word in re.findall(

                r"[A-Za-z]+",

                sentence,
            )
        )


        matches = (

            words
            &
            answer_set
        )


        # Exactly one target-class word.
        #
        # This prevents ambiguous labels.

        if len(matches) != 1:

            continue


        target = next(
            iter(
                matches
            )
        )


        groups[
            target
        ].append(

            {

                "text":
                    sentence,

                "answer":
                    target,

                "label":
                    answer_to_id[
                        target
                    ],
            }
        )


    return groups


# ============================================================
# BALANCE
# ============================================================

def balanced_examples(
    groups,
    answers,
    desired_per_class,
    seed,
):

    rng = random.Random(
        seed
    )


    available = min(

        len(
            groups[
                answer
            ]
        )

        for answer in answers
    )


    per_class = min(

        desired_per_class,

        available,
    )


    if per_class < 8:

        raise RuntimeError(

            "Not enough balanced examples. "
            f"Only {per_class} per class."
        )


    output = []


    for answer in answers:

        samples = list(

            groups[
                answer
            ]
        )


        rng.shuffle(
            samples
        )


        output.extend(

            samples[
                :per_class
            ]
        )


    rng.shuffle(
        output
    )


    return (
        output,
        per_class,
    )


# ============================================================
# POOLING
# ============================================================

def masked_mean(
    hidden,
    attention_mask,
):

    weights = (

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


def last_token(
    hidden,
    attention_mask,
):

    indices = (

        attention_mask
        .long()
        .sum(
            dim=-1
        )
        .sub(
            1
        )
        .clamp_min(
            0
        )
    )


    batch_indices = torch.arange(

        hidden.shape[
            0
        ],

        device=
            hidden.device,
    )


    return hidden[

        batch_indices,

        indices,
    ]


# ============================================================
# EXTRACT ALL REPRESENTATIONS IN ONE GPT2 PASS
# ============================================================

@torch.no_grad()
def extract_representations(
    text,
):

    encoded = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        max_length=256,
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


    output = (
        model.backbone.transformer(

            input_ids=input_ids,

            attention_mask=attention_mask,

            output_hidden_states=True,

            use_cache=False,

            return_dict=True,
        )
    )


    hs = (
        output.hidden_states
    )


    reps = OrderedDict()


    # --------------------------------------------------------
    # Single-layer means
    # --------------------------------------------------------

    reps[
        "L1_MEAN"
    ] = masked_mean(

        hs[1],

        attention_mask,
    )[0]


    reps[
        "L4_MEAN"
    ] = masked_mean(

        hs[4],

        attention_mask,
    )[0]


    reps[
        "L8_MEAN"
    ] = masked_mean(

        hs[8],

        attention_mask,
    )[0]


    reps[
        "L10_MEAN"
    ] = masked_mean(

        hs[10],

        attention_mask,
    )[0]


    reps[
        "L11_MEAN"
    ] = masked_mean(

        hs[11],

        attention_mask,
    )[0]


    reps[
        "L12_MEAN"
    ] = masked_mean(

        hs[12],

        attention_mask,
    )[0]


    # --------------------------------------------------------
    # Final last-token
    # --------------------------------------------------------

    reps[
        "L12_LAST"
    ] = last_token(

        hs[12],

        attention_mask,
    )[0]


    # --------------------------------------------------------
    # Layer ensemble 8-11
    # --------------------------------------------------------

    layer_8_11 = torch.stack(

        [
            hs[8],
            hs[9],
            hs[10],
            hs[11],
        ],

        dim=0,
    ).mean(
        dim=0
    )


    reps[
        "L8_TO_L11_MEAN"
    ] = masked_mean(

        layer_8_11,

        attention_mask,
    )[0]


    # --------------------------------------------------------
    # Layer ensemble 9-12
    # --------------------------------------------------------

    layer_9_12 = torch.stack(

        [
            hs[9],
            hs[10],
            hs[11],
            hs[12],
        ],

        dim=0,
    ).mean(
        dim=0
    )


    reps[
        "L9_TO_L12_MEAN"
    ] = masked_mean(

        layer_9_12,

        attention_mask,
    )[0]


    for name in reps:

        reps[
            name
        ] = (
            reps[
                name
            ]
            .detach()
            .float()
            .cpu()
        )


    return reps


# ============================================================
# EXTRACT DATASET FEATURES
# ============================================================

@torch.no_grad()
def extract_features(
    examples,
    split_name,
):

    storage = {

        name: []

        for name in REPRESENTATIONS
    }


    labels = []


    total = len(
        examples
    )


    for index, example in (
        enumerate(
            examples,
            start=1,
        )
    ):

        reps = (
            extract_representations(

                example[
                    "text"
                ]
            )
        )


        for name in (
            REPRESENTATIONS
        ):

            storage[
                name
            ].append(

                reps[
                    name
                ]
            )


        labels.append(

            example[
                "label"
            ]
        )


        if (
            index % 100 == 0
            or
            index == total
        ):

            print(

                f"{split_name}: "
                f"{index}/{total}"
            )


    output = {}


    for name in (
        REPRESENTATIONS
    ):

        output[
            name
        ] = torch.stack(

            storage[
                name
            ],

            dim=0,
        )


    output[
        "labels"
    ] = torch.tensor(

        labels,

        dtype=torch.long,
    )


    return output


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
# STANDARDIZE USING TRAIN ONLY
# ============================================================

def standardize(
    train_x,
    valid_x,
    test_x,
):

    mean = train_x.mean(

        dim=0,

        keepdim=True,
    )


    std = train_x.std(

        dim=0,

        keepdim=True,
    ).clamp_min(
        1e-5
    )


    return (

        (
            train_x
            -
            mean
        )
        /
        std,

        (
            valid_x
            -
            mean
        )
        /
        std,

        (
            test_x
            -
            mean
        )
        /
        std,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    probe,
    features,
    labels,
):

    probe.eval()


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


    predictions = (

        logits.argmax(
            dim=-1
        )
    )


    accuracy = (

        predictions
        .eq(
            labels
        )
        .float()
        .mean()
        .item()
    )


    top3 = (

        logits
        .topk(
            3,
            dim=-1,
        )
        .indices
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


    correct_logits = (

        logits
        .gather(

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
            accuracy,

        "top3":
            top3,

        "mrr":
            mrr,

        "mean_rank":
            mean_rank,
    }


# ============================================================
# TRAIN ONE REPRESENTATION PROBE
# ============================================================

def train_probe(
    representation,
    train_data,
    valid_data,
    test_data,
):

    train_x = (
        train_data[
            representation
        ].float()
    )


    valid_x = (
        valid_data[
            representation
        ].float()
    )


    test_x = (
        test_data[
            representation
        ].float()
    )


    (
        train_x,
        valid_x,
        test_x,
    ) = standardize(

        train_x,

        valid_x,

        test_x,
    )


    train_y = (
        train_data[
            "labels"
        ]
    )


    valid_y = (
        valid_data[
            "labels"
        ]
    )


    test_y = (
        test_data[
            "labels"
        ]
    )


    probe = LinearProbe(

        input_dim=
            train_x.shape[
                -1
            ],

        num_classes=
            NUM_CLASSES,
    ).to(
        DEVICE
    )


    optimizer = torch.optim.AdamW(

        probe.parameters(),

        lr=
            PROBE_LR,

        weight_decay=
            PROBE_WEIGHT_DECAY,
    )


    criterion = (
        nn.CrossEntropyLoss()
    )


    loader = DataLoader(

        TensorDataset(

            train_x,

            train_y,
        ),

        batch_size=
            PROBE_BATCH_SIZE,

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


        for batch_x, batch_y in loader:

            batch_x = (
                batch_x.to(
                    DEVICE
                )
            )


            batch_y = (
                batch_y.to(
                    DEVICE
                )
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


        valid_result = evaluate(

            probe,

            valid_x,

            valid_y,
        )


        if (
            valid_result[
                "accuracy"
            ]
            >
            best_val
        ):

            best_val = (

                valid_result[
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


    probe.load_state_dict(
        best_state
    )


    train_result = evaluate(

        probe,

        train_x,

        train_y,
    )


    valid_result = evaluate(

        probe,

        valid_x,

        valid_y,
    )


    test_result = evaluate(

        probe,

        test_x,

        test_y,
    )


    return {

        "best_epoch":
            best_epoch,

        "train":
            train_result,

        "validation":
            valid_result,

        "test":
            test_result,
    }


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def run_dataset(
    dataset_name,
):

    splits = load_dataset_sentences(

        dataset_name
    )


    if splits is None:

        return None


    # ========================================================
    # Select natural labels
    # ========================================================

    section(

        f"{dataset_name.upper()} "
        "SELECT NATURAL CONTENT CLASSES"
    )


    answers = choose_answer_vocabulary(

        splits
    )


    # ========================================================
    # Convert into controlled diagnostic examples
    # ========================================================

    groups_train = create_examples(

        splits[
            "train"
        ],

        answers,
    )


    groups_valid = create_examples(

        splits[
            "validation"
        ],

        answers,
    )


    groups_test = create_examples(

        splits[
            "test"
        ],

        answers,
    )


    train_examples, train_pc = (
        balanced_examples(

            groups_train,

            answers,

            TRAIN_PER_CLASS,

            SEED + 1,
        )
    )


    valid_examples, valid_pc = (
        balanced_examples(

            groups_valid,

            answers,

            VALID_PER_CLASS,

            SEED + 2,
        )
    )


    test_examples, test_pc = (
        balanced_examples(

            groups_test,

            answers,

            TEST_PER_CLASS,

            SEED + 3,
        )
    )


    section(

        f"{dataset_name.upper()} "
        "FINAL DIAGNOSTIC DATA"
    )


    print(
        "Train:",
        len(
            train_examples
        ),
        f"({train_pc}/class)"
    )


    print(
        "Valid:",
        len(
            valid_examples
        ),
        f"({valid_pc}/class)"
    )


    print(
        "Test :",
        len(
            test_examples
        ),
        f"({test_pc}/class)"
    )


    print()

    print(
        "Example text:"
    )


    print(
        train_examples[
            0
        ][
            "text"
        ]
    )


    print(
        "Target:",
        train_examples[
            0
        ][
            "answer"
        ]
    )


    # ========================================================
    # Extract all frozen GPT2 representations
    # ========================================================

    section(

        f"{dataset_name.upper()} "
        "EXTRACT TRAIN REPRESENTATIONS"
    )


    train_data = extract_features(

        train_examples,

        "TRAIN",
    )


    section(

        f"{dataset_name.upper()} "
        "EXTRACT VALIDATION REPRESENTATIONS"
    )


    valid_data = extract_features(

        valid_examples,

        "VALID",
    )


    section(

        f"{dataset_name.upper()} "
        "EXTRACT TEST REPRESENTATIONS"
    )


    test_data = extract_features(

        test_examples,

        "TEST",
    )


    # ========================================================
    # Train probes
    # ========================================================

    section(

        f"{dataset_name.upper()} "
        "VALUE REPRESENTATION ABLATION"
    )


    results = OrderedDict()


    for representation in (
        REPRESENTATIONS
    ):

        print()

        print(
            "Testing:",
            representation
        )


        result = train_probe(

            representation,

            train_data,

            valid_data,

            test_data,
        )


        results[
            representation
        ] = result


        print(

            f"  TEST ACC = "
            f"{result['test']['accuracy'] * 100:.2f}%"

            f" | MRR = "
            f"{result['test']['mrr']:.4f}"

            f" | Mean Rank = "
            f"{result['test']['mean_rank']:.4f}"
        )


    # ========================================================
    # Print ranking
    # ========================================================

    ranking = sorted(

        REPRESENTATIONS,

        key=lambda name:

            results[
                name
            ][
                "test"
            ][
                "accuracy"
            ],

        reverse=True,
    )


    section(

        f"{dataset_name.upper()} RESULTS"
    )


    print(

        f"{'Representation':<24}"
        f"{'Test Acc':>12}"
        f"{'Top-3':>12}"
        f"{'MRR':>12}"
        f"{'Mean Rank':>14}"
    )


    for representation in ranking:

        result = (

            results[
                representation
            ][
                "test"
            ]
        )


        print(

            f"{representation:<24}"

            f"{result['accuracy'] * 100:>11.2f}%"

            f"{result['top3'] * 100:>11.2f}%"

            f"{result['mrr']:>12.4f}"

            f"{result['mean_rank']:>14.4f}"
        )


    print()

    print(
        "BEST REPRESENTATION:",
        ranking[
            0
        ]
    )


    print(
        "BEST TEST ACCURACY:",
        f"{results[ranking[0]]['test']['accuracy'] * 100:.2f}%"
    )


    return {

        "dataset":
            dataset_name,

        "answers":
            answers,

        "results":
            results,

        "ranking":
            ranking,
    }


# ============================================================
# RUN REQUESTED DATASETS
# ============================================================

all_results = {}


for dataset_name in (
    args.datasets
):

    try:

        result = run_dataset(

            dataset_name
        )


        if result is not None:

            all_results[
                dataset_name
            ] = result


    except Exception as error:

        section(

            f"{dataset_name.upper()} FAILED"
        )


        print(
            type(
                error
            ).__name__,
            ":",
            error,
        )


        print()

        print(
            "Continuing with other datasets."
        )


# ============================================================
# CROSS-DATASET SUMMARY
# ============================================================

section(
    "CROSS-DATASET VALUE REPRESENTATION SUMMARY"
)


if not all_results:

    raise RuntimeError(

        "No dataset completed successfully."
    )


print(

    f"{'Representation':<24}"

    +
    "".join(

        f"{dataset.upper():>14}"

        for dataset
        in all_results.keys()
    )

    +

    f"{'MEAN':>14}"
)


cross_dataset_scores = {}


for representation in (
    REPRESENTATIONS
):

    scores = []


    row = (
        f"{representation:<24}"
    )


    for dataset_name, result in (
        all_results.items()
    ):

        accuracy = (

            result[
                "results"
            ][
                representation
            ][
                "test"
            ][
                "accuracy"
            ]
        )


        scores.append(
            accuracy
        )


        row += (

            f"{accuracy * 100:>13.2f}%"
        )


    mean_accuracy = float(

        np.mean(
            scores
        )
    )


    cross_dataset_scores[
        representation
    ] = mean_accuracy


    row += (

        f"{mean_accuracy * 100:>13.2f}%"
    )


    print(
        row
    )


# ============================================================
# ROBUST RANKING
# ============================================================

section(
    "ROBUST CROSS-DATASET RANKING"
)


ranking = sorted(

    REPRESENTATIONS,

    key=lambda name:

        cross_dataset_scores[
            name
        ],

    reverse=True,
)


for index, representation in (
    enumerate(
        ranking,
        start=1,
    )
):

    print(

        f"{index:02d}. "

        f"{representation:<24}"

        f" mean accuracy = "

        f"{cross_dataset_scores[representation] * 100:.2f}%"
    )


best_representation = (
    ranking[
        0
    ]
)


print()

print(
    "ROBUST VALUE CANDIDATE:"
)

print(
    best_representation
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (

    OUTPUT_DIR
    /
    "value_representation_results.csv"
)


with open(

    csv_path,

    "w",

    newline="",

    encoding="utf-8",

) as handle:

    writer = csv.writer(
        handle
    )


    writer.writerow(

        [
            "dataset",
            "representation",
            "test_accuracy",
            "test_top3",
            "test_mrr",
            "test_mean_rank",
        ]
    )


    for dataset_name, dataset_result in (
        all_results.items()
    ):

        for representation in (
            REPRESENTATIONS
        ):

            test_result = (

                dataset_result[
                    "results"
                ][
                    representation
                ][
                    "test"
                ]
            )


            writer.writerow(

                [
                    dataset_name,

                    representation,

                    test_result[
                        "accuracy"
                    ],

                    test_result[
                        "top3"
                    ],

                    test_result[
                        "mrr"
                    ],

                    test_result[
                        "mean_rank"
                    ],
                ]
            )


print()

print(
    "Saved CSV:"
)

print(
    csv_path
)


# ============================================================
# SAVE TORCH RESULTS
# ============================================================

torch_path = (

    OUTPUT_DIR
    /
    "value_representation_results.pt"
)


torch.save(

    {

        "datasets":
            list(
                all_results.keys()
            ),

        "representations":
            REPRESENTATIONS,

        "cross_dataset_scores":
            cross_dataset_scores,

        "best_representation":
            best_representation,

        "results":
            all_results,
    },

    torch_path,
)


print(
    "Saved detailed results:"
)

print(
    torch_path
)


# ============================================================
# FINAL DIAGNOSIS
# ============================================================

section(
    "FINAL DIAGNOSIS"
)


print(
    "Best cross-dataset VALUE representation:"
)

print(
    best_representation
)


print()


for dataset_name, result in (
    all_results.items()
):

    winner = (
        result[
            "ranking"
        ][
            0
        ]
    )


    winner_accuracy = (

        result[
            "results"
        ][
            winner
        ][
            "test"
        ][
            "accuracy"
        ]
    )


    print(

        f"{dataset_name.upper():<12}"

        f"best = "
        f"{winner:<20}"

        f"accuracy = "
        f"{winner_accuracy * 100:.2f}%"
    )


print()


if len(
    all_results
) >= 2:

    winners = [

        result[
            "ranking"
        ][
            0
        ]

        for result
        in all_results.values()
    ]


    if len(
        set(
            winners
        )
    ) == 1:

        print(
            "RESULT:"
        )

        print(
            "The same VALUE representation wins "
            "across all completed datasets."
        )

        print()

        print(
            "That is a strong candidate for the "
            "real memory-value architecture."
        )


    else:

        print(
            "RESULT:"
        )

        print(
            "Different datasets prefer different "
            "representations."
        )

        print()

        print(
            "Use the cross-dataset mean ranking rather "
            "than optimizing only for WikiText."
        )


print()

print(
    "NEXT AFTER THIS EXPERIMENT:"
)

print(
    "Use the selected VALUE representation with the "
    "Layer-1 KEY mechanism and evaluate the actual "
    "memory retrieval task on benchmark-native QA "
    "datasets such as LoCoMo."
)


# ============================================================
# COMPLETE
# ============================================================

section(
    "EXPERIMENT COMPLETE"
)


print(
    "No GPT-2 parameters trained."
)

print(
    "No memory architecture parameters trained."
)

print(
    "No models/ files modified."
)

print(
    "Only linear value probes were trained."
)