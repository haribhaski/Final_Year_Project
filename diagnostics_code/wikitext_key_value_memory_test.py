# ============================================================
# WIKITEXT KEY-VALUE MEMORY TEST
#
# PURPOSE
# ============================================================
#
# Move from synthetic:
#
#       Entity-A -> rabbit
#
# toward REAL WikiText text while preserving a controlled
# ground truth.
#
#
# MEMORY DESIGN
# ============================================================
#
# KEY:
#   Layer-1 representation of the WikiText article title
#   -> frozen contrastive address encoder
#
# VALUE:
#   direct normalized final GPT-2 summary of a real
#   WikiText sentence
#
#
# TARGET:
#   A real recurring content word from the WikiText sentence.
#
# The target word IS inside the fact. The point is NOT
# question answering/world knowledge.
#
# The point is:
#
#       Does the memory preserve WHICH real WikiText fact
#       was stored and WHAT content that fact contained?
#
#
# TESTS
# ============================================================
#
# 2 simultaneously stored facts
# 4 simultaneously stored facts
# 8 simultaneously stored facts
#
#
# Metrics:
#
# - Address Recall@1
# - Address MRR
# - Addressed value answer accuracy
# - Oracle value answer accuracy
# - Wrong-value answer accuracy
#
#
# IMPORTANT
# ============================================================
#
# No models/ source files modified.
#
# GPT-2 frozen.
# Address encoder frozen.
#
# Only a small linear VALUE decoder is trained.
#
#
# EXPECTED LOCAL DATA
# ============================================================
#
# The script auto-detects paths like:
#
# data/wikitext-103/wiki.train.tokens
# data/wikitext-103/wiki.valid.tokens
# data/wikitext-103/wiki.test.tokens
#
#
# RUN
# ============================================================
#
# python wikitext_key_value_memory_test.py \
#   2>&1 | tee wikitext_key_value_memory_test.log
#
# ============================================================


import os
import re
import random
from pathlib import Path
from collections import (
    Counter,
    defaultdict,
)

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

SEED = 2050

NUM_SLOTS = 8

ADDRESS_LAYER_DEFAULT = 1
ADDRESS_HIDDEN_DIM = 512
ADDRESS_DIM = 256


# ------------------------------------------------------------
# Final memory test
# ------------------------------------------------------------

FACT_COUNTS = [
    2,
    4,
    8,
]

EPISODES_PER_SIZE = 200


# ------------------------------------------------------------
# Number of answer classes
#
# Same 16-way setup as our previous synthetic benchmark.
# ------------------------------------------------------------

NUM_ANSWER_CLASSES = 16


# ------------------------------------------------------------
# Desired samples PER CLASS.
#
# Script automatically reduces these if WikiText support
# is lower.
# ------------------------------------------------------------

DESIRED_TRAIN_PER_CLASS = 100
DESIRED_VALID_PER_CLASS = 20
DESIRED_TEST_PER_CLASS = 30


# ------------------------------------------------------------
# Decoder
# ------------------------------------------------------------

DECODER_EPOCHS = 50
DECODER_BATCH_SIZE = 128
DECODER_LR = 1e-2
DECODER_WEIGHT_DECAY = 1e-4


# ------------------------------------------------------------
# Sentence filtering
# ------------------------------------------------------------

MIN_SENTENCE_WORDS = 8
MAX_SENTENCE_WORDS = 60

MIN_TITLE_LENGTH = 2
MAX_TITLE_LENGTH = 80


# ============================================================
# STOPWORDS
#
# We do not want the 16 answer classes to become:
#
# "the", "and", "of", ...
#
# ============================================================

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "than",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "he",
    "she",
    "they",
    "them",
    "his",
    "her",
    "their",
    "there",
    "here",
    "which",
    "who",
    "whom",
    "whose",
    "what",
    "when",
    "where",
    "why",
    "how",
    "not",
    "no",
    "yes",
    "also",
    "into",
    "over",
    "under",
    "after",
    "before",
    "during",
    "between",
    "through",
    "about",
    "against",
    "without",
    "within",
    "while",
    "although",
    "because",
    "so",
    "up",
    "down",
    "out",
    "off",
    "again",
    "further",
    "once",
    "more",
    "most",
    "other",
    "some",
    "such",
    "only",
    "own",
    "same",
    "very",
    "can",
    "could",
    "would",
    "should",
    "may",
    "might",
    "will",
    "shall",
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
# PRINT
# ============================================================

def section(title):

    print()
    print("=" * 125)
    print(title)
    print("=" * 125)


def fmt(x):

    return f"{float(x):.6f}"


# ============================================================
# FIND WIKITEXT
# ============================================================

def find_wikitext_file(split):

    possible_roots = [

        Path(
            "data/wikitext-103"
        ),

        Path(
            "wikitext-103"
        ),

        Path(
            "./data"
        ),

        Path(
            "."
        ),
    ]


    names = [

        f"wiki.{split}.tokens",

        f"wikitext-103/wiki.{split}.tokens",

        f"data/wikitext-103/wiki.{split}.tokens",
    ]


    # --------------------------------------------------------
    # Direct guesses
    # --------------------------------------------------------

    for root in possible_roots:

        candidate = (
            root
            /
            f"wiki.{split}.tokens"
        )


        if candidate.exists():

            return candidate


    # --------------------------------------------------------
    # Recursive fallback
    # --------------------------------------------------------

    for name in names:

        candidate = Path(
            name
        )


        if candidate.exists():

            return candidate


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

        f"Could not find "
        f"wiki.{split}.tokens"
    )


TRAIN_FILE = find_wikitext_file(
    "train"
)

VALID_FILE = find_wikitext_file(
    "valid"
)

TEST_FILE = find_wikitext_file(
    "test"
)


section(
    "WIKITEXT FILES"
)


print(
    "Train:",
    TRAIN_FILE,
)

print(
    "Valid:",
    VALID_FILE,
)

print(
    "Test :",
    TEST_FILE,
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
# MODEL CONFIG
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
# LOAD GPT2
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
    MODEL_CHECKPOINT,
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
        current_state[name].shape
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


print(
    "GPT-2 frozen."
)


# ============================================================
# LOAD ADDRESS ENCODER
# ============================================================

section(
    "LOAD FROZEN ADDRESS ENCODER"
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

        ADDRESS_LAYER_DEFAULT,
    )
)


address_encoder = AddressEncoder(

    input_dim=
        address_input_dim,

    hidden_dim=
        address_hidden_dim,

    output_dim=
        address_dim,

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


for parameter in (
    address_encoder.parameters()
):

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
    "Previous synthetic Recall@1:",
    address_checkpoint.get(

        "test_recall1",

        "unknown",
    ),
)

print(
    "Address encoder frozen."
)


# ============================================================
# CLEAN WIKITEXT
# ============================================================

def clean_wikitext_text(text):

    # WikiText token files contain spacing around punctuation
    # and some residual markup.

    text = text.strip()


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

        r"\(\s+",

        "(",

        text,
    )


    text = re.sub(

        r"\s+\)",

        ")",

        text,
    )


    text = re.sub(

        r"\s+",

        " ",

        text,
    )


    return text.strip()


# ============================================================
# PARSE ARTICLES
# ============================================================

HEADING_PATTERN = re.compile(

    r"^=\s*([^=]+?)\s*=$"
)


def read_articles(path):

    articles = []


    current_title = None

    current_lines = []


    with open(

        path,

        "r",

        encoding="utf-8",

        errors="ignore",

    ) as handle:

        for raw_line in handle:

            line = raw_line.strip()


            heading = (
                HEADING_PATTERN.match(
                    line
                )
            )


            if heading:

                if (
                    current_title
                    and
                    current_lines
                ):

                    text = clean_wikitext_text(

                        " ".join(
                            current_lines
                        )
                    )


                    articles.append(

                        (
                            current_title,
                            text,
                        )
                    )


                current_title = (
                    clean_wikitext_text(
                        heading.group(
                            1
                        )
                    )
                )


                current_lines = []

                continue


            if (
                current_title
                and
                line
            ):

                current_lines.append(
                    line
                )


    if (
        current_title
        and
        current_lines
    ):

        text = clean_wikitext_text(

            " ".join(
                current_lines
            )
        )


        articles.append(

            (
                current_title,
                text,
            )
        )


    return articles


# ============================================================
# SENTENCE SPLIT
# ============================================================

def split_sentences(text):

    # Simple deterministic sentence splitter.
    #
    # Good enough for a diagnostic benchmark.

    parts = re.split(

        r"(?<=[.!?])\s+",

        text,
    )


    sentences = []


    for part in parts:

        part = part.strip()


        if not part:

            continue


        word_count = len(

            re.findall(

                r"[A-Za-z]+",

                part,
            )
        )


        if not (
            MIN_SENTENCE_WORDS
            <=
            word_count
            <=
            MAX_SENTENCE_WORDS
        ):

            continue


        sentences.append(
            part
        )


    return sentences


# ============================================================
# FIND TARGET CONTENT WORD
#
# We use the LAST eligible content word in each real sentence.
#
# Example:
#
# "The battle was fought in northern France."
#
# target -> "France"
#
# The word must:
#
# - be alphabetic
# - not be a stopword
# - have length >= 3
# - tokenize as ONE GPT-2 token with a leading space
#
# Single-token restriction makes the 16-way value decoder
# clean and comparable.
# ============================================================

def get_target_word(sentence):

    words = re.findall(

        r"[A-Za-z]+",

        sentence,
    )


    for word in reversed(
        words
    ):

        normalized = (
            word.lower()
        )


        if (
            len(normalized) < 3
            or
            normalized in STOPWORDS
        ):

            continue


        token_ids = tokenizer.encode(

            " " + normalized,

            add_special_tokens=False,
        )


        if len(token_ids) != 1:

            continue


        return normalized


    return None


# ============================================================
# BUILD RAW EXAMPLES
# ============================================================

def build_raw_examples(
    path,
    split_name,
):

    section(
        f"PARSE {split_name.upper()}"
    )


    articles = read_articles(
        path
    )


    print(
        "Articles:",
        len(articles),
    )


    examples = []


    for title, article_text in (
        articles
    ):

        if not (

            MIN_TITLE_LENGTH
            <=
            len(title)
            <=
            MAX_TITLE_LENGTH
        ):

            continue


        sentences = split_sentences(
            article_text
        )


        for sentence_index, sentence in (
            enumerate(
                sentences
            )
        ):

            target = get_target_word(
                sentence
            )


            if target is None:

                continue


            # ------------------------------------------------
            # IMPORTANT:
            #
            # Ensure the title itself occurs explicitly in
            # the stored text so that we can extract its
            # Layer-1 span as the address identity.
            # ------------------------------------------------

            write_text = (

                f"{title}: "
                f"{sentence}"
            )


            examples.append(

                {

                    "title":
                        title,

                    "sentence":
                        sentence,

                    "write_text":
                        write_text,

                    "answer":
                        target,

                    "sentence_index":
                        sentence_index,
                }
            )


    print(
        "Usable sentence examples:",
        len(examples),
    )


    return examples


# ============================================================
# RAW WIKITEXT EXAMPLES
# ============================================================

train_raw = build_raw_examples(

    TRAIN_FILE,

    "train",
)


valid_raw = build_raw_examples(

    VALID_FILE,

    "valid",
)


test_raw = build_raw_examples(

    TEST_FILE,

    "test",
)


# ============================================================
# ANSWER VOCABULARY
#
# Pick words that occur in ALL THREE splits.
# ============================================================

section(
    "SELECT WIKITEXT ANSWER VOCABULARY"
)


train_counts = Counter(

    x[
        "answer"
    ]

    for x in train_raw
)


valid_counts = Counter(

    x[
        "answer"
    ]

    for x in valid_raw
)


test_counts = Counter(

    x[
        "answer"
    ]

    for x in test_raw
)


common_words = []


for word, train_count in (
    train_counts.items()
):

    valid_count = (
        valid_counts[
            word
        ]
    )


    test_count = (
        test_counts[
            word
        ]
    )


    if (
        train_count > 0
        and
        valid_count > 0
        and
        test_count > 0
    ):

        # Score by weakest split support.
        #
        # This prevents selecting a word with 1000 train
        # samples but only one test sample.

        minimum_support = min(

            train_count,

            valid_count,

            test_count,
        )


        total_support = (

            train_count
            +
            valid_count
            +
            test_count
        )


        common_words.append(

            (
                word,
                minimum_support,
                total_support,
                train_count,
                valid_count,
                test_count,
            )
        )


common_words.sort(

    key=lambda x: (
        x[1],
        x[2],
    ),

    reverse=True,
)


if (
    len(common_words)
    <
    NUM_ANSWER_CLASSES
):

    raise RuntimeError(

        "Could not find enough answer words "
        "appearing in train/valid/test."
    )


selected_info = (
    common_words[
        :NUM_ANSWER_CLASSES
    ]
)


ANSWERS = [

    x[0]

    for x in selected_info
]


ANSWER_TO_ID = {

    answer:
        idx

    for idx, answer
    in enumerate(
        ANSWERS
    )
}


print(

    f"{'Answer':<18}"
    f"{'Train':>10}"
    f"{'Valid':>10}"
    f"{'Test':>10}"
    f"{'Min':>10}"
)


for (
    word,
    minimum_support,
    total_support,
    train_count,
    valid_count,
    test_count,
) in selected_info:

    print(

        f"{word:<18}"

        f"{train_count:>10}"

        f"{valid_count:>10}"

        f"{test_count:>10}"

        f"{minimum_support:>10}"
    )


print()

print(
    "Selected answers:"
)

print(
    ANSWERS
)

print()

print(
    "Chance:",
    f"{100 / len(ANSWERS):.2f}%"
)


# ============================================================
# FILTER TO SELECTED ANSWERS
# ============================================================

def group_by_answer(
    examples,
):

    groups = defaultdict(
        list
    )


    for example in examples:

        if (
            example[
                "answer"
            ]
            in ANSWER_TO_ID
        ):

            groups[
                example[
                    "answer"
                ]
            ].append(
                example
            )


    return groups


train_groups = group_by_answer(
    train_raw
)

valid_groups = group_by_answer(
    valid_raw
)

test_groups = group_by_answer(
    test_raw
)


# ============================================================
# DETERMINE BALANCED SAMPLE COUNTS
# ============================================================

train_per_class = min(

    DESIRED_TRAIN_PER_CLASS,

    min(
        len(
            train_groups[
                answer
            ]
        )

        for answer in ANSWERS
    ),
)


valid_per_class = min(

    DESIRED_VALID_PER_CLASS,

    min(
        len(
            valid_groups[
                answer
            ]
        )

        for answer in ANSWERS
    ),
)


test_per_class = min(

    DESIRED_TEST_PER_CLASS,

    min(
        len(
            test_groups[
                answer
            ]
        )

        for answer in ANSWERS
    ),
)


print()

print(
    "Balanced samples per class:"
)

print(
    "Train:",
    train_per_class,
)

print(
    "Valid:",
    valid_per_class,
)

print(
    "Test :",
    test_per_class,
)


if train_per_class < 10:

    raise RuntimeError(

        "Too few balanced training examples. "
        "Need at least 10 per class."
    )


# ============================================================
# BALANCED DATASET
# ============================================================

def make_balanced_dataset(
    groups,
    per_class,
    seed,
):

    rng = random.Random(
        seed
    )


    final = []


    for answer in ANSWERS:

        candidates = list(

            groups[
                answer
            ]
        )


        rng.shuffle(
            candidates
        )


        selected = candidates[
            :per_class
        ]


        for example in selected:

            item = dict(
                example
            )


            item[
                "label"
            ] = (
                ANSWER_TO_ID[
                    answer
                ]
            )


            final.append(
                item
            )


    rng.shuffle(
        final
    )


    return final


train_examples = make_balanced_dataset(

    train_groups,

    train_per_class,

    SEED + 1,
)


valid_examples = make_balanced_dataset(

    valid_groups,

    valid_per_class,

    SEED + 2,
)


test_examples = make_balanced_dataset(

    test_groups,

    test_per_class,

    SEED + 3,
)


section(
    "FINAL WIKITEXT DATASETS"
)


print(
    "Train:",
    len(train_examples),
)

print(
    "Valid:",
    len(valid_examples),
)

print(
    "Test :",
    len(test_examples),
)


print()

print(
    "Example:"
)

print(
    "TITLE:"
)

print(
    train_examples[
        0
    ][
        "title"
    ]
)

print()

print(
    "SENTENCE:"
)

print(
    train_examples[
        0
    ][
        "sentence"
    ]
)

print()

print(
    "TARGET:"
)

print(
    train_examples[
        0
    ][
        "answer"
    ]
)


# ============================================================
# ENTITY/TITLE SPAN MASK
# ============================================================

def build_span_mask(
    text,
    substring,
    offsets,
):

    start = text.find(
        substring
    )


    if start == -1:

        raise RuntimeError(

            f"Could not find "
            f"{substring!r} "
            f"in text."
        )


    end = (
        start
        +
        len(substring)
    )


    mask = torch.zeros(

        offsets.shape[
            0
        ],

        dtype=torch.bool,
    )


    for idx in range(
        offsets.shape[
            0
        ]
    ):

        s = int(

            offsets[
                idx,
                0
            ].item()
        )


        e = int(

            offsets[
                idx,
                1
            ].item()
        )


        if e <= s:

            continue


        if (
            s < end
            and
            e > start
        ):

            mask[
                idx
            ] = True


    if not mask.any():

        raise RuntimeError(
            "Span contains no tokens."
        )


    return mask


# ============================================================
# GET ADDRESS KEY
# ============================================================

@torch.no_grad()
def get_address_key(
    text,
    title,
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


    title_mask = build_span_mask(

        text,

        title,

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


    output = (
        model.backbone.transformer(

            input_ids=input_ids,

            attention_mask=attention_mask,

            output_hidden_states=True,

            use_cache=False,

            return_dict=True,
        )
    )


    hidden = (

        output
        .hidden_states[
            address_layer
        ]
    )


    weights = (

        title_mask
        .unsqueeze(
            0
        )
        .unsqueeze(
            -1
        )
        .to(
            hidden.dtype
        )
    )


    title_representation = (

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


    key = address_encoder(

        title_representation
    )


    return (
        key[
            0
        ]
        .detach()
    )


# ============================================================
# GET DIRECT VALUE
# ============================================================

@torch.no_grad()
def get_value(
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


    output = (
        model.backbone.transformer(

            input_ids=input_ids,

            attention_mask=attention_mask,

            use_cache=False,

            return_dict=True,
        )
    )


    hidden = (
        output
        .last_hidden_state
    )


    summary = model._pool_hidden(

        hidden_states=hidden,

        attention_mask=attention_mask,
    )


    value = F.layer_norm(

        summary,

        normalized_shape=(
            summary.shape[
                -1
            ],
        ),
    )


    return (
        value[
            0
        ]
        .detach()
    )


# ============================================================
# EXTRACT VALUE DATA
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

        value = get_value(

            example[
                "write_text"
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
            idx == len(
                examples
            )
        ):

            print(

                f"{name}: "
                f"{idx}/"
                f"{len(examples)}"
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
# EXTRACT
# ============================================================

section(
    "EXTRACT WIKITEXT TRAIN VALUES"
)


train_data = extract_value_dataset(

    train_examples,

    "TRAIN",
)


section(
    "EXTRACT WIKITEXT VALID VALUES"
)


valid_data = extract_value_dataset(

    valid_examples,

    "VALID",
)


section(
    "EXTRACT WIKITEXT TEST VALUES"
)


test_data = extract_value_dataset(

    test_examples,

    "TEST",
)


# ============================================================
# STANDARDIZATION
# ============================================================

value_mean = (

    train_data[
        "values"
    ]
    .mean(
        dim=0,
        keepdim=True,
    )
)


value_std = (

    train_data[
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


def standardize(
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
    "TRAIN WIKITEXT VALUE DECODER"
)


decoder = ValueDecoder(

    input_dim=
        train_data[
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

    lr=DECODER_LR,

    weight_decay=
        DECODER_WEIGHT_DECAY,
)


loss_fn = nn.CrossEntropyLoss()


loader = DataLoader(

    TensorDataset(

        train_data[
            "values"
        ],

        train_data[
            "labels"
        ],
    ),

    batch_size=
        DECODER_BATCH_SIZE,

    shuffle=True,
)


@torch.no_grad()
def evaluate_decoder(
    values,
    labels,
):

    decoder.eval()


    x = standardize(

        values.to(
            DEVICE
        )
    )


    y = labels.to(
        DEVICE
    )


    logits = decoder(
        x
    )


    prediction = (
        logits.argmax(
            dim=-1
        )
    )


    accuracy = (

        prediction
        .eq(y)
        .float()
        .mean()
        .item()
    )


    correct_logits = (

        logits
        .gather(
            1,
            y.unsqueeze(
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


    mean_rank = (

        ranks
        .float()
        .mean()
        .item()
    )


    return {

        "accuracy":
            accuracy,

        "mrr":
            mrr,

        "mean_rank":
            mean_rank,
    }


best_state = None

best_val_accuracy = -1.0

best_epoch = None


for epoch in range(
    1,
    DECODER_EPOCHS + 1,
):

    decoder.train()


    running_loss = 0.0

    total = 0


    for batch_x, batch_y in (
        loader
    ):

        batch_x = standardize(

            batch_x.to(
                DEVICE
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


        running_loss += (

            float(
                loss.item()
            )

            *
            batch_x.size(
                0
            )
        )


        total += (
            batch_x.size(
                0
            )
        )


    val_result = evaluate_decoder(

        valid_data[
            "values"
        ],

        valid_data[
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

            key:
                value
                .detach()
                .cpu()
                .clone()

            for key, value
            in decoder
            .state_dict()
            .items()
        }


    if (
        epoch == 1
        or
        epoch % 10 == 0
        or
        epoch == DECODER_EPOCHS
    ):

        print(

            f"Epoch "
            f"{epoch:02d}/"
            f"{DECODER_EPOCHS}"

            f" | loss="
            f"{running_loss / max(total,1):.4f}"

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

decoder.eval()


train_result = evaluate_decoder(

    train_data[
        "values"
    ],

    train_data[
        "labels"
    ],
)


valid_result = evaluate_decoder(

    valid_data[
        "values"
    ],

    valid_data[
        "labels"
    ],
)


test_result = evaluate_decoder(

    test_data[
        "values"
    ],

    test_data[
        "labels"
    ],
)


section(
    "VALUE DECODER RESULTS"
)


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


for split, result in [

    (
        "TRAIN",
        train_result,
    ),

    (
        "VALID",
        valid_result,
    ),

    (
        "TEST",
        test_result,
    ),
]:

    print(

        f"{split:<12}"

        f"{result['accuracy'] * 100:>13.2f}%"

        f"{result['mrr']:>12.4f}"

        f"{result['mean_rank']:>14.4f}"
    )


for parameter in decoder.parameters():

    parameter.requires_grad = False


# ============================================================
# DECODE ONE VALUE
# ============================================================

@torch.no_grad()
def decode_value(
    value,
):

    x = standardize(

        value
        .unsqueeze(
            0
        )
        .to(
            DEVICE
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


    return ANSWERS[
        predicted_id
    ]


# ============================================================
# QUERY TEMPLATES
# ============================================================

QUERY_TEMPLATES = [

    "What information is stored about {title}?",

    "Recall the stored information for {title}.",

    "Retrieve the memory associated with {title}.",

    "What was remembered about {title}?",

    "Find the stored fact for {title}.",

    "Which information belongs to {title}?",
]


# ============================================================
# PREPARE EPISODE POOL
#
# We use HELD-OUT WikiText test examples.
#
# Need unique title identities inside each episode.
# ============================================================

episode_pool = test_examples


# ============================================================
# SAMPLE EPISODE
# ============================================================

def sample_episode_examples(
    num_facts,
    rng,
):

    # Try repeatedly until:
    #
    # - unique titles
    # - unique answer classes
    #
    # Unique answers make WRONG-VALUE accuracy easier to
    # interpret causally.

    for _ in range(
        1000
    ):

        candidates = rng.sample(

            episode_pool,

            k=min(
                len(
                    episode_pool
                ),
                max(
                    num_facts * 8,
                    32,
                ),
            ),
        )


        selected = []

        titles = set()

        answers = set()


        for example in candidates:

            title = example[
                "title"
            ]

            answer = example[
                "answer"
            ]


            if (
                title in titles
                or
                answer in answers
            ):

                continue


            selected.append(
                example
            )


            titles.add(
                title
            )


            answers.add(
                answer
            )


            if len(
                selected
            ) == num_facts:

                return selected


    raise RuntimeError(

        f"Could not construct "
        f"{num_facts}-fact episode "
        "with unique titles/answers."
    )


# ============================================================
# RUN ONE EPISODE
# ============================================================

@torch.no_grad()
def run_episode(
    num_facts,
    rng,
    verbose=False,
):

    facts = sample_episode_examples(

        num_facts,

        rng,
    )


    keys = []

    values = []


    # ========================================================
    # WRITE
    # ========================================================

    for example in facts:

        key = get_address_key(

            example[
                "write_text"
            ],

            example[
                "title"
            ],
        )


        value = get_value(

            example[
                "write_text"
            ]
        )


        keys.append(
            key
        )


        values.append(
            value
        )


    keys = torch.stack(

        keys,

        dim=0,
    )


    keys = F.normalize(

        keys,

        p=2,

        dim=-1,

        eps=1e-8,
    )


    values = torch.stack(

        values,

        dim=0,
    )


    # ========================================================
    # READ
    # ========================================================

    address_correct = 0

    address_rr = 0.0

    addressed_answer_correct = 0

    oracle_answer_correct = 0

    wrong_answer_correct = 0


    records = []


    for correct_index, example in (
        enumerate(
            facts
        )
    ):

        query_template = rng.choice(

            QUERY_TEMPLATES
        )


        query_text = query_template.format(

            title=
                example[
                    "title"
                ]
        )


        query_key = get_address_key(

            query_text,

            example[
                "title"
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


        order = torch.argsort(

            similarities,

            descending=True,
        )


        predicted_index = int(

            order[
                0
            ].item()
        )


        correct_similarity = (

            similarities[
                correct_index
            ]
        )


        rank = int(

            (
                similarities
                >
                correct_similarity
            )
            .sum()
            .item()
            + 1
        )


        address_rr += (
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
        # ADDRESS-SELECTED VALUE
        # ====================================================

        addressed_prediction = decode_value(

            values[
                predicted_index
            ]
        )


        if (
            addressed_prediction
            ==
            example[
                "answer"
            ]
        ):

            addressed_answer_correct += 1


        # ====================================================
        # ORACLE VALUE
        # ====================================================

        oracle_prediction = decode_value(

            values[
                correct_index
            ]
        )


        if (
            oracle_prediction
            ==
            example[
                "answer"
            ]
        ):

            oracle_answer_correct += 1


        # ====================================================
        # WRONG VALUE
        # ====================================================

        wrong_index = (

            (
                correct_index
                +
                1
            )

            %
            num_facts
        )


        wrong_prediction = decode_value(

            values[
                wrong_index
            ]
        )


        if (
            wrong_prediction
            ==
            example[
                "answer"
            ]
        ):

            wrong_answer_correct += 1


        records.append(

            {

                "title":
                    example[
                        "title"
                    ],

                "sentence":
                    example[
                        "sentence"
                    ],

                "answer":
                    example[
                        "answer"
                    ],

                "correct_slot":
                    correct_index,

                "addressed_slot":
                    predicted_index,

                "rank":
                    rank,

                "addressed_prediction":
                    addressed_prediction,

                "oracle_prediction":
                    oracle_prediction,

                "wrong_prediction":
                    wrong_prediction,
            }
        )


    total = len(
        facts
    )


    result = {

        "address_accuracy":
            address_correct
            /
            total,

        "address_mrr":
            address_rr
            /
            total,

        "addressed_accuracy":
            addressed_answer_correct
            /
            total,

        "oracle_accuracy":
            oracle_answer_correct
            /
            total,

        "wrong_accuracy":
            wrong_answer_correct
            /
            total,

        "records":
            records,
    }


    if verbose:

        print()

        print(
            "EXAMPLE REAL WIKITEXT EPISODE"
        )


        for idx, record in enumerate(
            records
        ):

            print()

            print(
                "-" * 90
            )


            print(
                "MEMORY SLOT:",
                idx,
            )


            print(
                "TITLE:"
            )

            print(
                record[
                    "title"
                ]
            )


            print()

            print(
                "REAL WIKITEXT SENTENCE:"
            )

            print(
                record[
                    "sentence"
                ]
            )


            print()

            print(
                "TARGET CONTENT WORD:",
                record[
                    "answer"
                ]
            )


            print(
                "ADDRESSED SLOT:",
                record[
                    "addressed_slot"
                ]
            )


            print(
                "ADDRESS RANK:",
                record[
                    "rank"
                ]
            )


            print(
                "ADDRESSED VALUE PREDICTION:",
                record[
                    "addressed_prediction"
                ]
            )


            print(
                "ORACLE VALUE PREDICTION:",
                record[
                    "oracle_prediction"
                ]
            )


            print(
                "WRONG VALUE PREDICTION:",
                record[
                    "wrong_prediction"
                ]
            )


    return result


# ============================================================
# RUN MEMORY EXPERIMENT
# ============================================================

section(
    "REAL WIKITEXT KEY-VALUE MEMORY"
)


print(
    "Facts:",
    FACT_COUNTS,
)

print(
    "Episodes per size:",
    EPISODES_PER_SIZE,
)

print(
    "Answer classes:",
    len(
        ANSWERS
    ),
)

print(
    "Answer chance:",
    f"{100 / len(ANSWERS):.2f}%"
)


all_results = {}


for num_facts in FACT_COUNTS:

    section(

        f"{num_facts}-FACT "
        f"WIKITEXT MEMORY"
    )


    rng = random.Random(

        SEED
        +
        1000
        *
        num_facts
    )


    total_address = 0.0

    total_mrr = 0.0

    total_addressed = 0.0

    total_oracle = 0.0

    total_wrong = 0.0


    for episode_idx in range(
        EPISODES_PER_SIZE
    ):

        result = run_episode(

            num_facts,

            rng,

            verbose=(
                episode_idx
                ==
                0
            ),
        )


        total_address += (

            result[
                "address_accuracy"
            ]
        )


        total_mrr += (

            result[
                "address_mrr"
            ]
        )


        total_addressed += (

            result[
                "addressed_accuracy"
            ]
        )


        total_oracle += (

            result[
                "oracle_accuracy"
            ]
        )


        total_wrong += (

            result[
                "wrong_accuracy"
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


    result = {

        "address":
            total_address
            /
            EPISODES_PER_SIZE,

        "mrr":
            total_mrr
            /
            EPISODES_PER_SIZE,

        "addressed":
            total_addressed
            /
            EPISODES_PER_SIZE,

        "oracle":
            total_oracle
            /
            EPISODES_PER_SIZE,

        "wrong":
            total_wrong
            /
            EPISODES_PER_SIZE,
    }


    all_results[
        num_facts
    ] = result


    print()

    print(
        f"{num_facts}-FACT RESULTS"
    )


    print(
        "Address accuracy:",
        f"{result['address'] * 100:.2f}%"
    )


    print(
        "Address MRR:",
        fmt(
            result[
                "mrr"
            ]
        )
    )


    print(
        "Addressed value accuracy:",
        f"{result['addressed'] * 100:.2f}%"
    )


    print(
        "Oracle value accuracy:",
        f"{result['oracle'] * 100:.2f}%"
    )


    print(
        "Wrong-value accuracy:",
        f"{result['wrong'] * 100:.2f}%"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

section(
    "FINAL WIKITEXT KEY-VALUE RESULTS"
)


chance = (
    1.0
    /
    len(
        ANSWERS
    )
)


print(

    f"{'Facts':<10}"

    f"{'Address':>12}"

    f"{'Addr MRR':>12}"

    f"{'Addressed':>14}"

    f"{'Oracle':>12}"

    f"{'Wrong':>12}"

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

        f"{result['address'] * 100:>11.2f}%"

        f"{result['mrr']:>12.4f}"

        f"{result['addressed'] * 100:>13.2f}%"

        f"{result['oracle'] * 100:>11.2f}%"

        f"{result['wrong'] * 100:>11.2f}%"

        f"{chance * 100:>11.2f}%"
    )


# ============================================================
# GAP ANALYSIS
# ============================================================

section(
    "ADDRESS VS VALUE GAP"
)


for num_facts in FACT_COUNTS:

    result = (
        all_results[
            num_facts
        ]
    )


    addressing_loss = (

        result[
            "oracle"
        ]

        -
        result[
            "addressed"
        ]
    )


    value_gain = (

        result[
            "oracle"
        ]

        -
        result[
            "wrong"
        ]
    )


    print()

    print(
        f"{num_facts} facts"
    )


    print(
        "Oracle - Addressed:",
        f"{addressing_loss * 100:+.2f} points"
    )


    print(
        "Oracle - Wrong:",
        f"{value_gain * 100:+.2f} points"
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
    "Mean addressed accuracy:",
    f"{mean_addressed * 100:.2f}%"
)

print(
    "Mean oracle accuracy:",
    f"{mean_oracle * 100:.2f}%"
)

print(
    "Mean wrong-value accuracy:",
    f"{mean_wrong * 100:.2f}%"
)


print()


if (
    mean_address >= 0.80
    and
    mean_oracle >= 0.60
    and
    mean_addressed >= 0.55
):

    print(
        "RESULT: KEY-VALUE DESIGN TRANSFERS "
        "TO REAL WIKITEXT TEXT."
    )


    print()

    print(
        "The synthetic result was not purely dependent "
        "on artificial entity-value sentences."
    )


    print()

    print(
        "NEXT:"
    )


    print(
        "Integrate the separated key/value memory into "
        "a sequential WikiText language-model experiment."
    )


elif (
    mean_oracle >= 0.60
    and
    mean_address < 0.50
):

    print(
        "RESULT: WIKITEXT VALUES WORK, "
        "BUT ADDRESS ENCODER DOES NOT GENERALIZE "
        "WELL TO NATURAL ARTICLE TITLES."
    )


    print()

    print(
        "Next train/test the address encoder on "
        "natural entity/title identities."
    )


elif (
    mean_address >= 0.80
    and
    mean_oracle < 0.40
):

    print(
        "RESULT: NATURAL-TEXT ADDRESSING WORKS, "
        "BUT THE DIRECT VALUE REPRESENTATION "
        "IS TOO WEAK FOR WIKITEXT."
    )


    print()

    print(
        "Next improve semantic value extraction."
    )


else:

    print(
        "RESULT: PARTIAL TRANSFER."
    )


    print()

    print(
        "Inspect ADDRESS and ORACLE separately."
    )


    print(
        "Do not modify the main architecture yet."
    )


# ============================================================
# SAVE
# ============================================================

section(
    "SAVE RESULTS"
)


OUTPUT_PATH = (

    "outputs/"
    "wikitext_key_value_memory_test.pt"
)


torch.save(

    {

        "answers":
            ANSWERS,

        "train_per_class":
            train_per_class,

        "valid_per_class":
            valid_per_class,

        "test_per_class":
            test_per_class,

        "decoder_test_accuracy":
            test_result[
                "accuracy"
            ],

        "decoder_state_dict":
            decoder
            .state_dict(),

        "value_mean":
            value_mean,

        "value_std":
            value_std,

        "memory_results":
            all_results,
    },

    OUTPUT_PATH,
)


print(
    "Saved:",
    OUTPUT_PATH
)


# ============================================================
# DONE
# ============================================================

section(
    "EXPERIMENT COMPLETE"
)


print(
    "Used real WikiText-103 sentences."
)

print(
    "GPT-2 frozen."
)

print(
    "Address encoder frozen."
)

print(
    "Existing CandidateWriter not used."
)

print(
    "Existing reader not used."
)

print(
    "Scalar/vector gate not used."
)

print(
    "Only the diagnostic value decoder was trained."
)

print(
    "No models/ source files were modified."
)

print(
    "No original checkpoints overwritten."
)