# ============================================================
# STATISTICAL VERIFICATION OF GPT-2 LAYERS
# FOR MEMORY ADDRESS REPRESENTATION
#
# Research question:
#
#   Does GPT-2 Layer 1 preserve entity identity significantly
#   better than the other GPT-2 layers for memory addressing?
#
# IMPORTANT:
#   - NO TRAINING
#   - NO modification of models/
#   - NO checkpoint overwrite
#   - GPT-2 / memory model remains frozen
#
# Experimental design:
#
#   Layers:
#       0 = embedding output
#       1 = after GPT-2 block 1
#       ...
#       12 = final GPT-2 hidden state
#
#   Repetitions:
#       multiple random seeds
#
#   Entity distributions:
#       1. Sequential synthetic IDs
#       2. Random alphanumeric IDs
#       3. Pseudo-name entities
#       4. Project-style names
#
#   Cross-template retrieval:
#
#       WRITE FACT:
#           "Store the mapping Project-X to rabbit."
#
#       QUERY:
#           "What keyword belongs to Project-X?"
#
#   Representation:
#       ENTITY-SPAN hidden representation
#
#   Similarity:
#       cosine similarity
#
#   Retrieval target:
#       query entity i should retrieve fact entity i
#
#
# Statistical analysis:
#
#   - Mean Recall@1 ± SD
#   - 95% bootstrap CI
#   - Mean MRR ± SD
#   - Mean margin ± SD
#
#   Layer 1 vs every other layer:
#
#       Paired Wilcoxon signed-rank:
#           run-level Recall@1
#
#       Exact McNemar/binomial test:
#           per-query retrieval success
#
#       Paired Cohen's dz:
#           run-level effect size
#
#       Holm-Bonferroni correction:
#           controls multiple comparisons
#
#
# Run:
#
# python statistical_layer_address_verification.py \
#   2>&1 | tee statistical_layer_address_verification.log
#
# ============================================================

import csv
import math
import os
import random
import statistics
import string
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

from models.gpt2_memory import (
    MemoryAugmentedGPT2LMHeadModel,
    MemoryGPT2Config,
)


# ============================================================
# SCIPY
# ============================================================

try:

    from scipy.stats import (
        wilcoxon,
        binomtest,
    )

except ImportError:

    raise ImportError(
        "\nSciPy is required for the statistical tests.\n"
        "Install with:\n\n"
        "pip install scipy\n"
    )


# ============================================================
# CONFIGURATION
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

# ------------------------------------------------------------
# STATISTICAL REPETITIONS
# ------------------------------------------------------------

SEEDS = [
    11,
    22,
    33,
    44,
    55,
]

# Number of entities PER style PER seed.
#
# 300 x 4 styles x 5 seeds
# = 6000 paired examples.
#
NUM_ENTITIES_PER_STYLE = 300

BATCH_SIZE = 16

# ------------------------------------------------------------
# Layer proposed as best addressing layer.
# ------------------------------------------------------------

REFERENCE_LAYER = 1

# ------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------

BOOTSTRAP_SAMPLES = 5000

BOOTSTRAP_SEED = 12345

# ------------------------------------------------------------
# Significance level
# ------------------------------------------------------------

ALPHA = 0.05


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_DIR = "outputs/layer_statistical_verification"

RUN_RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "layer_run_results.csv",
)

STATISTICS_CSV = os.path.join(
    OUTPUT_DIR,
    "layer_statistical_tests.csv",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


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


# ============================================================
# WRITE TEMPLATES
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


# ============================================================
# QUERY TEMPLATES
# ============================================================

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
# DISPLAY HELPERS
# ============================================================

def section(title):

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def subsection(title):

    print()
    print("-" * 120)
    print(title)
    print("-" * 120)


def fmt(value):

    return f"{float(value):.6f}"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# MEMORY MODEL CONFIG
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
# LOAD MODEL
# ============================================================

section("LOAD FROZEN MODEL")

print(
    "Device:",
    DEVICE,
)

print(
    "Base model:",
    MODEL_NAME,
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

skipped = []

for name, value in checkpoint_state.items():

    if (
        name in current_state
        and current_state[name].shape
        == value.shape
    ):

        compatible[name] = value

    else:

        skipped.append(name)


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

model.to(DEVICE)

model.eval()

for parameter in model.parameters():

    parameter.requires_grad = False

print(
    "Model completely frozen."
)


# ============================================================
# ENTITY GENERATORS
# ============================================================

SYLLABLES = [

    "al",
    "bar",
    "cor",
    "den",
    "el",
    "far",
    "gal",
    "hel",
    "ion",
    "jor",
    "kal",
    "lor",
    "mer",
    "nor",
    "or",
    "pra",
    "quil",
    "ran",
    "sel",
    "tor",
    "ul",
    "ven",
    "wil",
    "xer",
    "yor",
    "zen",
]


PROJECT_WORDS = [

    "Atlas",
    "Cedar",
    "Nimbus",
    "Orchid",
    "Falcon",
    "Helix",
    "Quartz",
    "Aurora",
    "Vector",
    "Nova",
    "Summit",
    "Harbor",
    "Vertex",
    "Echo",
    "Delta",
    "Pioneer",
]


def random_alphanumeric(
    rng,
    length=8,
):

    alphabet = (
        string.ascii_uppercase
        + string.digits
    )

    return "".join(
        rng.choice(alphabet)
        for _ in range(length)
    )


def pseudo_name(
    rng,
):

    parts = rng.randint(
        2,
        4,
    )

    result = "".join(
        rng.choice(
            SYLLABLES
        )
        for _ in range(parts)
    )

    return (
        result[0].upper()
        + result[1:]
    )


def generate_entities(
    style,
    count,
    seed,
):

    rng = random.Random(
        seed
    )

    entities = []

    used = set()

    i = 0

    while len(entities) < count:

        if style == "sequential":

            entity = (
                f"Entity-{seed}-{i:05d}"
            )

        elif style == "alphanumeric":

            entity = (
                "ID-"
                + random_alphanumeric(
                    rng,
                    length=8,
                )
            )

        elif style == "pseudoname":

            entity = (
                pseudo_name(rng)
                + "-"
                + random_alphanumeric(
                    rng,
                    length=3,
                )
            )

        elif style == "project":

            entity = (
                f"Project "
                f"{rng.choice(PROJECT_WORDS)} "
                f"{rng.randint(1000, 9999)}"
            )

        else:

            raise ValueError(
                f"Unknown entity style: "
                f"{style}"
            )

        if entity not in used:

            used.add(entity)

            entities.append(
                entity
            )

        i += 1

    return entities


ENTITY_STYLES = [

    "sequential",

    "alphanumeric",

    "pseudoname",

    "project",
]


# ============================================================
# DATASET CREATION
# ============================================================

def build_examples(
    style,
    seed,
):

    rng = random.Random(
        seed + 10000
    )

    entities = generate_entities(
        style=style,
        count=NUM_ENTITIES_PER_STYLE,
        seed=seed,
    )

    examples = []

    for i, entity in enumerate(
        entities
    ):

        answer = ANSWERS[
            i % len(ANSWERS)
        ]

        write_idx = rng.randrange(
            len(WRITE_TEMPLATES)
        )

        query_idx = rng.randrange(
            len(QUERY_TEMPLATES)
        )

        write_text = (
            WRITE_TEMPLATES[
                write_idx
            ].format(
                entity=entity,
                answer=answer,
            )
        )

        query_text = (
            QUERY_TEMPLATES[
                query_idx
            ].format(
                entity=entity,
            )
        )

        examples.append(
            {
                "entity":
                    entity,

                "answer":
                    answer,

                "write":
                    write_text,

                "query":
                    query_text,

                "write_template":
                    write_idx,

                "query_template":
                    query_idx,
            }
        )

    return examples


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

    entity_mask = torch.zeros(
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

        if start < 0:

            raise RuntimeError(
                f"Could not find entity "
                f"{entity!r} inside text:\n"
                f"{text}"
            )

        end = (
            start
            + len(entity)
        )

        count = 0

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
                and token_end > start
            )

            if overlaps:

                entity_mask[
                    batch_idx,
                    token_idx,
                ] = True

                count += 1

        if count == 0:

            raise RuntimeError(
                f"No entity tokens found "
                f"for {entity!r}"
            )

    return entity_mask


# ============================================================
# ENTITY-SPAN AVERAGE
# ============================================================

def entity_average(
    hidden_states,
    entity_mask,
):

    weights = (
        entity_mask
        .unsqueeze(-1)
        .to(
            hidden_states.dtype
        )
    )

    numerator = (
        hidden_states
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
# EXTRACT ALL LAYERS
# ============================================================

@torch.no_grad()
def extract_all_layer_entity_vectors(
    texts,
    entities,
):

    all_layers = defaultdict(
        list
    )

    for start in range(
        0,
        len(texts),
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            len(texts),
        )

        batch_text = texts[
            start:end
        ]

        batch_entities = entities[
            start:end
        ]

        encoded = tokenizer(
            batch_text,
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
                texts=batch_text,
                entities=batch_entities,
                offset_mapping=offsets,
                attention_mask=(
                    attention_mask_cpu
                ),
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

        for layer_idx, hidden in enumerate(
            output.hidden_states
        ):

            representation = (
                entity_average(
                    hidden,
                    entity_mask,
                )
            )

            all_layers[
                layer_idx
            ].append(
                representation
                .detach()
                .cpu()
            )

    final = {}

    for layer_idx in all_layers:

        final[
            layer_idx
        ] = torch.cat(
            all_layers[
                layer_idx
            ],
            dim=0,
        )

    return final


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def calculate_retrieval_metrics(
    write_vectors,
    query_vectors,
):

    W = F.normalize(
        write_vectors.float(),
        p=2,
        dim=-1,
        eps=1e-8,
    )

    Q = F.normalize(
        query_vectors.float(),
        p=2,
        dim=-1,
        eps=1e-8,
    )

    similarity = (
        Q @ W.T
    )

    n = similarity.size(0)

    correct_scores = (
        similarity.diag()
    )

    ranks = (
        similarity
        .gt(
            correct_scores.unsqueeze(1)
        )
        .sum(dim=1)
        + 1
    )

    recall1_vector = (
        ranks.eq(1)
    )

    recall1 = float(
        recall1_vector
        .float()
        .mean()
        .item()
    )

    recall5 = float(
        ranks.le(5)
        .float()
        .mean()
        .item()
    )

    mrr_vector = (
        1.0
        / ranks.float()
    )

    mrr = float(
        mrr_vector.mean().item()
    )

    negative_matrix = (
        similarity.clone()
    )

    mask = torch.eye(
        n,
        dtype=torch.bool,
    )

    negative_matrix[
        mask
    ] = -float("inf")

    hardest_negative = (
        negative_matrix
        .max(dim=1)
        .values
    )

    margins = (
        correct_scores
        - hardest_negative
    )

    return {

        "recall1":
            recall1,

        "recall5":
            recall5,

        "mrr":
            mrr,

        "correct_cos":
            float(
                correct_scores
                .mean()
                .item()
            ),

        "hard_negative_cos":
            float(
                hardest_negative
                .mean()
                .item()
            ),

        "margin":
            float(
                margins
                .mean()
                .item()
            ),

        # Keep per-example data for
        # paired statistical analysis.

        "success_vector":
            recall1_vector
            .cpu()
            .numpy()
            .astype(np.int8),

        "rank_vector":
            ranks
            .cpu()
            .numpy(),

        "mrr_vector":
            mrr_vector
            .cpu()
            .numpy(),

        "margin_vector":
            margins
            .cpu()
            .numpy(),
    }


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def bootstrap_ci(
    values,
    samples=BOOTSTRAP_SAMPLES,
    seed=BOOTSTRAP_SEED,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(values)

    means = np.empty(
        samples,
        dtype=np.float64,
    )

    for i in range(
        samples
    ):

        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        means[i] = (
            sample.mean()
        )

    lower = np.percentile(
        means,
        2.5,
    )

    upper = np.percentile(
        means,
        97.5,
    )

    return (
        float(lower),
        float(upper),
    )


# ============================================================
# EFFECT SIZE: PAIRED COHEN'S DZ
# ============================================================

def paired_cohens_dz(
    reference_values,
    comparison_values,
):

    reference_values = np.asarray(
        reference_values,
        dtype=np.float64,
    )

    comparison_values = np.asarray(
        comparison_values,
        dtype=np.float64,
    )

    difference = (
        reference_values
        - comparison_values
    )

    if len(difference) < 2:

        return float("nan")

    sd = difference.std(
        ddof=1
    )

    if sd < 1e-12:

        if abs(
            difference.mean()
        ) < 1e-12:

            return 0.0

        return (
            float("inf")
            if difference.mean() > 0
            else -float("inf")
        )

    return float(
        difference.mean()
        / sd
    )


# ============================================================
# EXACT MCNEMAR TEST
#
# Uses exact binomial test on discordant pairs.
#
# b:
#   Layer 1 correct,
#   comparison wrong
#
# c:
#   Layer 1 wrong,
#   comparison correct
# ============================================================

def exact_mcnemar(
    reference_success,
    comparison_success,
):

    reference_success = np.asarray(
        reference_success,
        dtype=np.int8,
    )

    comparison_success = np.asarray(
        comparison_success,
        dtype=np.int8,
    )

    b = int(
        np.sum(
            (reference_success == 1)
            &
            (comparison_success == 0)
        )
    )

    c = int(
        np.sum(
            (reference_success == 0)
            &
            (comparison_success == 1)
        )
    )

    discordant = (
        b + c
    )

    if discordant == 0:

        p_value = 1.0

    else:

        result = binomtest(
            min(b, c),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        )

        p_value = float(
            result.pvalue
        )

    return (
        b,
        c,
        p_value,
    )


# ============================================================
# HOLM-BONFERRONI
# ============================================================

def holm_bonferroni(
    p_values,
):

    p_values = np.asarray(
        p_values,
        dtype=np.float64,
    )

    m = len(
        p_values
    )

    order = np.argsort(
        p_values
    )

    adjusted = np.empty(
        m,
        dtype=np.float64,
    )

    running_max = 0.0

    for rank, idx in enumerate(
        order
    ):

        multiplier = (
            m - rank
        )

        adjusted_value = min(
            1.0,
            p_values[idx]
            * multiplier,
        )

        running_max = max(
            running_max,
            adjusted_value,
        )

        adjusted[
            idx
        ] = running_max

    return adjusted


# ============================================================
# MAIN EXPERIMENT
# ============================================================

section(
    "STATISTICAL LAYER VERIFICATION"
)

print(
    "Seeds:",
    SEEDS,
)

print(
    "Entity styles:",
    ENTITY_STYLES,
)

print(
    "Entities per style per seed:",
    NUM_ENTITIES_PER_STYLE,
)

print(
    "Total independent condition runs:",
    len(SEEDS)
    * len(ENTITY_STYLES),
)

print(
    "Total query examples:",
    len(SEEDS)
    * len(ENTITY_STYLES)
    * NUM_ENTITIES_PER_STYLE,
)


# ============================================================
# STORAGE
#
# run_metrics[layer] -> list of run dictionaries
#
# Per-query success/mrr/margin are also accumulated for
# McNemar and supplementary analyses.
# ============================================================

run_metrics = defaultdict(
    list
)

all_success = defaultdict(
    list
)

all_mrr_examples = defaultdict(
    list
)

all_margin_examples = defaultdict(
    list
)

csv_rows = []

num_layers = None


# ============================================================
# RUN ALL CONDITIONS
# ============================================================

condition_number = 0

total_conditions = (
    len(SEEDS)
    * len(ENTITY_STYLES)
)

for seed in SEEDS:

    for style in ENTITY_STYLES:

        condition_number += 1

        section(
            f"CONDITION "
            f"{condition_number}/{total_conditions} "
            f"| SEED={seed} "
            f"| STYLE={style}"
        )

        set_seed(
            seed
        )

        examples = build_examples(
            style=style,
            seed=seed,
        )

        entities = [
            item["entity"]
            for item in examples
        ]

        write_texts = [
            item["write"]
            for item in examples
        ]

        query_texts = [
            item["query"]
            for item in examples
        ]

        print(
            "Example entity:",
            entities[0],
        )

        print(
            "WRITE:",
            write_texts[0],
        )

        print(
            "QUERY:",
            query_texts[0],
        )

        print()
        print(
            "Extracting write representations..."
        )

        write_layers = (
            extract_all_layer_entity_vectors(
                write_texts,
                entities,
            )
        )

        print(
            "Extracting query representations..."
        )

        query_layers = (
            extract_all_layer_entity_vectors(
                query_texts,
                entities,
            )
        )

        if num_layers is None:

            num_layers = len(
                write_layers
            )

            print(
                "Number of hidden-state levels:",
                num_layers,
            )

        print()
        print(
            f"{'Layer':<8}"
            f"{'R@1':>10}"
            f"{'R@5':>10}"
            f"{'MRR':>12}"
            f"{'Correct':>12}"
            f"{'HardNeg':>12}"
            f"{'Margin':>12}"
        )

        for layer in range(
            num_layers
        ):

            metrics = (
                calculate_retrieval_metrics(

                    write_layers[
                        layer
                    ],

                    query_layers[
                        layer
                    ],
                )
            )

            record = {

                "seed":
                    seed,

                "style":
                    style,

                "layer":
                    layer,

                "recall1":
                    metrics[
                        "recall1"
                    ],

                "recall5":
                    metrics[
                        "recall5"
                    ],

                "mrr":
                    metrics[
                        "mrr"
                    ],

                "correct_cos":
                    metrics[
                        "correct_cos"
                    ],

                "hard_negative_cos":
                    metrics[
                        "hard_negative_cos"
                    ],

                "margin":
                    metrics[
                        "margin"
                    ],
            }

            run_metrics[
                layer
            ].append(
                record
            )

            all_success[
                layer
            ].append(
                metrics[
                    "success_vector"
                ]
            )

            all_mrr_examples[
                layer
            ].append(
                metrics[
                    "mrr_vector"
                ]
            )

            all_margin_examples[
                layer
            ].append(
                metrics[
                    "margin_vector"
                ]
            )

            csv_rows.append(
                record
            )

            marker = (
                "  <-- REFERENCE"
                if layer
                == REFERENCE_LAYER
                else ""
            )

            print(
                f"{layer:<8}"
                f"{metrics['recall1'] * 100:>9.2f}%"
                f"{metrics['recall5'] * 100:>9.2f}%"
                f"{metrics['mrr']:>12.4f}"
                f"{metrics['correct_cos']:>12.5f}"
                f"{metrics['hard_negative_cos']:>12.5f}"
                f"{metrics['margin']:>12.5f}"
                f"{marker}"
            )

        del write_layers
        del query_layers

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# ============================================================
# SAVE RUN-LEVEL RESULTS
# ============================================================

with open(
    RUN_RESULTS_CSV,
    "w",
    newline="",
) as file:

    writer = csv.DictWriter(

        file,

        fieldnames=[
            "seed",
            "style",
            "layer",
            "recall1",
            "recall5",
            "mrr",
            "correct_cos",
            "hard_negative_cos",
            "margin",
        ],
    )

    writer.writeheader()

    writer.writerows(
        csv_rows
    )


# ============================================================
# AGGREGATE SUMMARY
# ============================================================

section(
    "AGGREGATE RESULTS ACROSS ALL RUNS"
)

summary_by_layer = {}

print(
    f"{'Layer':<8}"
    f"{'R@1 mean':>14}"
    f"{'R@1 SD':>12}"
    f"{'95% CI':>23}"
    f"{'MRR mean':>12}"
    f"{'MRR SD':>10}"
    f"{'Margin':>12}"
)

for layer in range(
    num_layers
):

    records = (
        run_metrics[
            layer
        ]
    )

    recall_values = np.array(
        [
            item["recall1"]
            for item in records
        ],
        dtype=np.float64,
    )

    mrr_values = np.array(
        [
            item["mrr"]
            for item in records
        ],
        dtype=np.float64,
    )

    margin_values = np.array(
        [
            item["margin"]
            for item in records
        ],
        dtype=np.float64,
    )

    recall_mean = float(
        recall_values.mean()
    )

    recall_std = float(
        recall_values.std(
            ddof=1
        )
    )

    mrr_mean = float(
        mrr_values.mean()
    )

    mrr_std = float(
        mrr_values.std(
            ddof=1
        )
    )

    margin_mean = float(
        margin_values.mean()
    )

    margin_std = float(
        margin_values.std(
            ddof=1
        )
    )

    ci_low, ci_high = (
        bootstrap_ci(
            recall_values,
            seed=(
                BOOTSTRAP_SEED
                + layer
            ),
        )
    )

    summary_by_layer[
        layer
    ] = {

        "recall_values":
            recall_values,

        "mrr_values":
            mrr_values,

        "margin_values":
            margin_values,

        "recall_mean":
            recall_mean,

        "recall_std":
            recall_std,

        "ci_low":
            ci_low,

        "ci_high":
            ci_high,

        "mrr_mean":
            mrr_mean,

        "mrr_std":
            mrr_std,

        "margin_mean":
            margin_mean,

        "margin_std":
            margin_std,
    }

    marker = (
        "  <-- Layer 1"
        if layer
        == REFERENCE_LAYER
        else ""
    )

    ci_string = (
        f"[{ci_low * 100:.2f}, "
        f"{ci_high * 100:.2f}]"
    )

    print(
        f"{layer:<8}"
        f"{recall_mean * 100:>13.2f}%"
        f"{recall_std * 100:>11.2f}%"
        f"{ci_string:>23}"
        f"{mrr_mean:>12.4f}"
        f"{mrr_std:>10.4f}"
        f"{margin_mean:>12.5f}"
        f"{marker}"
    )


# ============================================================
# IDENTIFY EMPIRICAL BEST LAYER
# ============================================================

section(
    "EMPIRICAL BEST LAYER"
)

best_layer = max(
    range(num_layers),
    key=lambda layer: (
        summary_by_layer[
            layer
        ][
            "recall_mean"
        ],

        summary_by_layer[
            layer
        ][
            "mrr_mean"
        ],
    ),
)

print(
    "Best layer by mean Recall@1:",
    best_layer,
)

print(
    "Mean Recall@1:",
    f"{summary_by_layer[best_layer]['recall_mean'] * 100:.2f}%"
)

print(
    "Standard deviation:",
    f"{summary_by_layer[best_layer]['recall_std'] * 100:.2f}%"
)

print(
    "95% bootstrap CI:",
    f"["
    f"{summary_by_layer[best_layer]['ci_low'] * 100:.2f}%, "
    f"{summary_by_layer[best_layer]['ci_high'] * 100:.2f}%"
    f"]"
)


# ============================================================
# PREPARE PAIRED STATISTICAL TESTS
# ============================================================

section(
    "STATISTICAL COMPARISON: LAYER 1 VS OTHER LAYERS"
)

reference_run_recall = (
    summary_by_layer[
        REFERENCE_LAYER
    ][
        "recall_values"
    ]
)

reference_run_mrr = (
    summary_by_layer[
        REFERENCE_LAYER
    ][
        "mrr_values"
    ]
)

reference_success = np.concatenate(
    all_success[
        REFERENCE_LAYER
    ]
)

statistical_rows = []


# ============================================================
# RUN TESTS
# ============================================================

for layer in range(
    num_layers
):

    if layer == REFERENCE_LAYER:

        continue

    comparison_run_recall = (
        summary_by_layer[
            layer
        ][
            "recall_values"
        ]
    )

    comparison_run_mrr = (
        summary_by_layer[
            layer
        ][
            "mrr_values"
        ]
    )

    # --------------------------------------------------------
    # Paired Wilcoxon on RUN-LEVEL Recall@1
    # --------------------------------------------------------

    recall_difference = (
        reference_run_recall
        - comparison_run_recall
    )

    if np.allclose(
        recall_difference,
        0.0,
    ):

        wilcoxon_recall_stat = 0.0
        wilcoxon_recall_p = 1.0

    else:

        result = wilcoxon(
            reference_run_recall,
            comparison_run_recall,
            alternative="two-sided",
            zero_method="wilcox",
        )

        wilcoxon_recall_stat = float(
            result.statistic
        )

        wilcoxon_recall_p = float(
            result.pvalue
        )

    # --------------------------------------------------------
    # Wilcoxon on RUN-LEVEL MRR
    # --------------------------------------------------------

    mrr_difference = (
        reference_run_mrr
        - comparison_run_mrr
    )

    if np.allclose(
        mrr_difference,
        0.0,
    ):

        wilcoxon_mrr_stat = 0.0
        wilcoxon_mrr_p = 1.0

    else:

        result = wilcoxon(
            reference_run_mrr,
            comparison_run_mrr,
            alternative="two-sided",
            zero_method="wilcox",
        )

        wilcoxon_mrr_stat = float(
            result.statistic
        )

        wilcoxon_mrr_p = float(
            result.pvalue
        )

    # --------------------------------------------------------
    # Effect size
    # --------------------------------------------------------

    effect_dz = (
        paired_cohens_dz(
            reference_run_recall,
            comparison_run_recall,
        )
    )

    # --------------------------------------------------------
    # McNemar / exact binomial test
    # --------------------------------------------------------

    comparison_success = (
        np.concatenate(
            all_success[
                layer
            ]
        )
    )

    (
        mcnemar_b,
        mcnemar_c,
        mcnemar_p,
    ) = exact_mcnemar(
        reference_success,
        comparison_success,
    )

    row = {

        "comparison_layer":
            layer,

        "layer1_mean_recall1":
            float(
                reference_run_recall.mean()
            ),

        "other_mean_recall1":
            float(
                comparison_run_recall.mean()
            ),

        "mean_difference":
            float(
                recall_difference.mean()
            ),

        "wilcoxon_recall_stat":
            wilcoxon_recall_stat,

        "wilcoxon_recall_p":
            wilcoxon_recall_p,

        "wilcoxon_mrr_stat":
            wilcoxon_mrr_stat,

        "wilcoxon_mrr_p":
            wilcoxon_mrr_p,

        "cohens_dz":
            effect_dz,

        "mcnemar_b":
            mcnemar_b,

        "mcnemar_c":
            mcnemar_c,

        "mcnemar_p":
            mcnemar_p,
    }

    statistical_rows.append(
        row
    )


# ============================================================
# MULTIPLE COMPARISON CORRECTION
# ============================================================

recall_p_values = [
    row[
        "wilcoxon_recall_p"
    ]
    for row in statistical_rows
]

mrr_p_values = [
    row[
        "wilcoxon_mrr_p"
    ]
    for row in statistical_rows
]

mcnemar_p_values = [
    row[
        "mcnemar_p"
    ]
    for row in statistical_rows
]


recall_adjusted = (
    holm_bonferroni(
        recall_p_values
    )
)

mrr_adjusted = (
    holm_bonferroni(
        mrr_p_values
    )
)

mcnemar_adjusted = (
    holm_bonferroni(
        mcnemar_p_values
    )
)


for index, row in enumerate(
    statistical_rows
):

    row[
        "wilcoxon_recall_p_holm"
    ] = float(
        recall_adjusted[
            index
        ]
    )

    row[
        "wilcoxon_mrr_p_holm"
    ] = float(
        mrr_adjusted[
            index
        ]
    )

    row[
        "mcnemar_p_holm"
    ] = float(
        mcnemar_adjusted[
            index
        ]
    )


# ============================================================
# PRINT STATISTICAL TABLE
# ============================================================

print()
print(
    f"{'Compare':<14}"
    f"{'L1 R@1':>10}"
    f"{'Other':>10}"
    f"{'Δ':>10}"
    f"{'Wilcox p*':>14}"
    f"{'McNemar p*':>14}"
    f"{'Cohen dz':>12}"
)

for row in statistical_rows:

    layer = row[
        "comparison_layer"
    ]

    print(
        f"L1 vs L{layer:<7}"
        f"{row['layer1_mean_recall1'] * 100:>9.2f}%"
        f"{row['other_mean_recall1'] * 100:>9.2f}%"
        f"{row['mean_difference'] * 100:>9.2f}"
        f"{row['wilcoxon_recall_p_holm']:>14.6g}"
        f"{row['mcnemar_p_holm']:>14.6g}"
        f"{row['cohens_dz']:>12.3f}"
    )

print()
print(
    "* p-values are Holm-Bonferroni corrected."
)


# ============================================================
# SAVE STATISTICAL TEST CSV
# ============================================================

with open(
    STATISTICS_CSV,
    "w",
    newline="",
) as file:

    fieldnames = list(
        statistical_rows[
            0
        ].keys()
    )

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        statistical_rows
    )


# ============================================================
# PER-STYLE SUMMARY
# ============================================================

section(
    "PER-ENTITY-STYLE RESULTS"
)

print(
    f"{'Style':<18}"
    f"{'Layer':>8}"
    f"{'Mean R@1':>14}"
    f"{'SD':>12}"
)

for style in ENTITY_STYLES:

    for layer in range(
        num_layers
    ):

        values = [

            record[
                "recall1"
            ]

            for record
            in run_metrics[
                layer
            ]

            if record[
                "style"
            ] == style
        ]

        mean_value = float(
            np.mean(values)
        )

        std_value = float(
            np.std(
                values,
                ddof=1,
            )
        )

        marker = (
            " <--"
            if layer
            == REFERENCE_LAYER
            else ""
        )

        print(
            f"{style:<18}"
            f"{layer:>8}"
            f"{mean_value * 100:>13.2f}%"
            f"{std_value * 100:>11.2f}%"
            f"{marker}"
        )

    print()


# ============================================================
# CHECK WHETHER LAYER 1 IS BEST IN EACH RUN
# ============================================================

section(
    "CONSISTENCY: HOW OFTEN IS LAYER 1 BEST?"
)

condition_keys = []

for seed in SEEDS:

    for style in ENTITY_STYLES:

        condition_keys.append(
            (
                seed,
                style,
            )
        )


layer1_best_count = 0

strict_layer1_best_count = 0


for seed, style in condition_keys:

    values = {}

    for layer in range(
        num_layers
    ):

        matching = [

            record

            for record
            in run_metrics[
                layer
            ]

            if (
                record["seed"]
                == seed
                and record["style"]
                == style
            )
        ]

        if len(matching) != 1:

            raise RuntimeError(
                "Unexpected duplicate/missing "
                "condition result."
            )

        values[
            layer
        ] = matching[
            0
        ][
            "recall1"
        ]

    best_value = max(
        values.values()
    )

    best_layers = [

        layer
        for layer, value
        in values.items()

        if math.isclose(
            value,
            best_value,
            abs_tol=1e-12,
        )
    ]

    if REFERENCE_LAYER in best_layers:

        layer1_best_count += 1

    if (
        len(best_layers) == 1
        and best_layers[0]
        == REFERENCE_LAYER
    ):

        strict_layer1_best_count += 1

    print(
        f"Seed={seed:<4} "
        f"Style={style:<15} "
        f"Best layer(s)="
        f"{best_layers} "
        f"Best R@1="
        f"{best_value * 100:.2f}%"
    )


total_runs = len(
    condition_keys
)

print()
print(
    "Layer 1 tied for/best in:",
    f"{layer1_best_count}/"
    f"{total_runs}",
    f"("
    f"{100 * layer1_best_count / total_runs:.2f}%"
    f")",
)

print(
    "Layer 1 uniquely best in:",
    f"{strict_layer1_best_count}/"
    f"{total_runs}",
    f"("
    f"{100 * strict_layer1_best_count / total_runs:.2f}%"
    f")",
)


# ============================================================
# RESEARCH INTERPRETATION
# ============================================================

section(
    "RESEARCH-GRADE INTERPRETATION"
)

layer1 = summary_by_layer[
    REFERENCE_LAYER
]

best = summary_by_layer[
    best_layer
]

print(
    "Reference Layer:",
    REFERENCE_LAYER,
)

print(
    "Empirical best layer:",
    best_layer,
)

print()

print(
    "Layer 1 Recall@1:",
    f"{layer1['recall_mean'] * 100:.2f}% "
    f"± "
    f"{layer1['recall_std'] * 100:.2f}%"
)

print(
    "Layer 1 95% CI:",
    f"["
    f"{layer1['ci_low'] * 100:.2f}%, "
    f"{layer1['ci_high'] * 100:.2f}%"
    f"]"
)

print()

print(
    "Layer 1 MRR:",
    f"{layer1['mrr_mean']:.4f} "
    f"± "
    f"{layer1['mrr_std']:.4f}"
)

print()

print(
    "Layer 1 mean retrieval margin:",
    f"{layer1['margin_mean']:.6f} "
    f"± "
    f"{layer1['margin_std']:.6f}"
)

print()


if best_layer == REFERENCE_LAYER:

    print(
        "PRIMARY RESULT:"
    )

    print(
        "Layer 1 has the highest mean "
        "cross-template Recall@1 across "
        "the tested seeds and entity "
        "distributions."
    )

else:

    print(
        "IMPORTANT:"
    )

    print(
        f"Layer {best_layer} achieved a "
        f"higher mean Recall@1 than "
        f"Layer {REFERENCE_LAYER}."
    )

    print(
        "Therefore Layer 1 should NOT be "
        "claimed as the best addressing "
        "layer based on this experiment."
    )


# ============================================================
# SIGNIFICANCE INTERPRETATION
# ============================================================

significant_better = []

not_significant = []

significant_worse = []


for row in statistical_rows:

    layer = row[
        "comparison_layer"
    ]

    difference = row[
        "mean_difference"
    ]

    p = row[
        "wilcoxon_recall_p_holm"
    ]

    if (
        p < ALPHA
        and difference > 0
    ):

        significant_better.append(
            layer
        )

    elif (
        p < ALPHA
        and difference < 0
    ):

        significant_worse.append(
            layer
        )

    else:

        not_significant.append(
            layer
        )


print()

print(
    "Layers that Layer 1 "
    "significantly outperforms:"
)

print(
    significant_better
)

print()

print(
    "Layers with no statistically "
    "significant Recall@1 difference "
    "from Layer 1:"
)

print(
    not_significant
)

print()

print(
    "Layers that significantly "
    "outperform Layer 1:"
)

print(
    significant_worse
)


# ============================================================
# CLAIM SAFETY
# ============================================================

section(
    "WHAT YOU CAN CLAIM TO YOUR GUIDE"
)

if (
    best_layer == REFERENCE_LAYER
    and len(
        significant_worse
    ) == 0
):

    print(
        "SUPPORTED CLAIM:"
    )

    print()

    print(
        "Across repeated cross-template "
        "retrieval experiments using "
        f"{len(SEEDS)} random seeds and "
        f"{len(ENTITY_STYLES)} entity-name "
        "distributions, GPT-2 Layer 1 "
        "produced the highest mean "
        "entity-address retrieval accuracy."
    )

    print()

    if len(
        significant_better
    ) > 0:

        print(
            "Its improvement over several "
            "deeper layers was statistically "
            "significant after "
            "Holm-Bonferroni correction."
        )

    if len(
        not_significant
    ) > 0:

        print()

        print(
            "IMPORTANT LIMITATION:"
        )

        print(
            "Layer 1 was not statistically "
            "distinguishable from layers:"
        )

        print(
            not_significant
        )

        print(
            "So do NOT claim it is "
            "significantly better than "
            "every single layer."
        )

else:

    print(
        "Layer 1 is NOT statistically "
        "supported as the universal best "
        "layer under this experiment."
    )

    print()

    print(
        "Use the measured results rather "
        "than assuming Layer 1 is optimal."
    )


# ============================================================
# OUTPUT PATHS
# ============================================================

section(
    "OUTPUT FILES"
)

print(
    "Run-level metrics:"
)

print(
    RUN_RESULTS_CSV
)

print()

print(
    "Statistical comparisons:"
)

print(
    STATISTICS_CSV
)


# ============================================================
# COMPLETE
# ============================================================

section(
    "EXPERIMENT COMPLETE"
)

print(
    "No parameters were trained."
)

print(
    "No model files were modified."
)

print(
    "No checkpoint was overwritten."
)

print()

print(
    "Send me the COMPLETE terminal output "
    "after this finishes."
)