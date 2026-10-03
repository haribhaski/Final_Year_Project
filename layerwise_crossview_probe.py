# ============================================================
# LAYER-WISE CROSS-VIEW IDENTITY PROBE
#
# Goal:
#   Determine whether GPT-2 already contains a representation
#   capable of associating:
#
#       WRITE FACT(entity)  <->  READ QUERY(entity)
#
#   across completely unseen entities.
#
# NO TRAINING
# NO CHANGES TO models/
# NO CHECKPOINT OVERWRITE
#
# Tests every GPT-2 hidden layer using:
#   1. ENTITY_SPAN representation
#   2. LAST_TOKEN representation
#   3. MASKED_MEAN representation
#
# Metrics:
#   - Recall@1
#   - Recall@5
#   - MRR
#   - Mean correct cosine
#   - Mean hardest-negative cosine
#   - Mean margin
#   - Median margin
#   - Positive-margin fraction
#   - Effective rank
#
# Run:
#
# python layerwise_crossview_probe.py \
#   2>&1 | tee layerwise_crossview_probe.log
#
# ============================================================

import math
import random
from collections import defaultdict

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

from models.gpt2_memory import (
    MemoryAugmentedGPT2LMHeadModel,
    MemoryGPT2Config,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gpt2"

CHECKPOINT = (
    "outputs/retrieval_gradient_test/"
    "checkpoint_best.pt"
)

SEED = 42

NUM_ENTITIES = 500

BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# We deliberately use entities NOT seen in the previous
# Project-A / Project-B diagnostics.
ENTITY_PREFIX = "ProbeEntity"

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
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def section(title):

    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def subsection(title):

    print()
    print("-" * 110)
    print(title)
    print("-" * 110)


def fmt(x):

    return f"{float(x):.6f}"


# ============================================================
# SYNTHETIC TEXT
# ============================================================

def fact_text(entity, answer):

    return (
        f"The assigned keyword for {entity} is {answer}. "
        f"Remember that the keyword associated with "
        f"{entity} is {answer}."
    )


def query_text(entity):

    return (
        f"The assigned keyword for {entity} is"
    )


# ============================================================
# MODEL CONFIG
#
# Memory configuration itself is not important for this
# experiment, because we only inspect GPT-2 hidden states.
#
# We nevertheless reproduce the current architecture so the
# checkpoint loads exactly as expected.
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
# LOAD TOKENIZER
# ============================================================

section("TOKENIZER")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer:", tokenizer.__class__.__name__)
print("Vocabulary:", len(tokenizer))


# ============================================================
# LOAD MODEL
# ============================================================

section("LOAD MODEL")

print("Device:", DEVICE)
print("Base model:", MODEL_NAME)
print("Checkpoint:", CHECKPOINT)

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
    checkpoint_state = checkpoint["model_state_dict"]

elif "state_dict" in checkpoint:
    checkpoint_state = checkpoint["state_dict"]

else:
    checkpoint_state = checkpoint


current_state = model.state_dict()

compatible = {}

skipped = []

for name, tensor in checkpoint_state.items():

    if (
        name in current_state
        and current_state[name].shape == tensor.shape
    ):

        compatible[name] = tensor

    else:

        skipped.append(name)


result = model.load_state_dict(
    compatible,
    strict=False,
)

print("Compatible tensors:", len(compatible))
print("Missing tensors:", len(result.missing_keys))
print("Skipped tensors:", len(skipped))

model.to(DEVICE)
model.eval()

print("Model loaded.")


# ============================================================
# CREATE DATASET
# ============================================================

section("BUILD UNSEEN ENTITY DATASET")

entities = [
    f"{ENTITY_PREFIX}-{i:04d}"
    for i in range(NUM_ENTITIES)
]

examples = []

for i, entity in enumerate(entities):

    # Answer assignment is irrelevant for identity retrieval,
    # but we vary answers so all facts are realistic.
    answer = ANSWERS[
        i % len(ANSWERS)
    ]

    examples.append(
        {
            "entity": entity,
            "answer": answer,
            "fact": fact_text(
                entity,
                answer,
            ),
            "query": query_text(
                entity,
            ),
        }
    )

print("Number of unseen entities:", len(examples))

print()
print("Example:")
print("Entity:", examples[0]["entity"])
print("Fact:  ", examples[0]["fact"])
print("Query: ", examples[0]["query"])


# ============================================================
# ENTITY TOKEN MASK
#
# Uses character offsets from the FAST tokenizer.
#
# We use ONLY THE FIRST occurrence of the entity in the fact,
# because the query contains one entity occurrence and this
# makes the write/read comparison cleaner.
# ============================================================

def build_entity_mask(
    texts,
    entities,
    offset_mapping,
    attention_mask,
):

    batch_size = len(texts)

    seq_len = offset_mapping.size(1)

    entity_mask = torch.zeros(
        batch_size,
        seq_len,
        dtype=torch.bool,
    )

    for b in range(batch_size):

        text = texts[b]
        entity = entities[b]

        start = text.find(entity)

        if start == -1:

            raise RuntimeError(
                f"Could not find entity "
                f"{entity!r} in text:\n{text}"
            )

        end = start + len(entity)

        found = 0

        for t in range(seq_len):

            if not bool(
                attention_mask[b, t]
            ):

                continue

            token_start = int(
                offset_mapping[
                    b,
                    t,
                    0,
                ].item()
            )

            token_end = int(
                offset_mapping[
                    b,
                    t,
                    1,
                ].item()
            )

            # Ignore special/padding offsets.
            if token_end <= token_start:
                continue

            # Token overlaps entity character span.
            overlaps = (
                token_start < end
                and token_end > start
            )

            if overlaps:

                entity_mask[
                    b,
                    t,
                ] = True

                found += 1

        if found == 0:

            print()
            print("FAILED ENTITY TOKENIZATION")
            print("Text:", text)
            print("Entity:", entity)
            print("Entity character span:", start, end)

            raise RuntimeError(
                "Entity mask contains zero tokens."
            )

    return entity_mask


# ============================================================
# REPRESENTATION FUNCTIONS
# ============================================================

def masked_average(
    hidden,
    mask,
):

    weights = (
        mask
        .unsqueeze(-1)
        .to(hidden.dtype)
    )

    numerator = (
        hidden * weights
    ).sum(dim=1)

    denominator = (
        weights
        .sum(dim=1)
        .clamp_min(1.0)
    )

    return (
        numerator / denominator
    )


def last_valid_token(
    hidden,
    attention_mask,
):

    indices = (
        attention_mask
        .long()
        .sum(dim=1)
        .sub(1)
        .clamp_min(0)
    )

    batch_index = torch.arange(
        hidden.size(0),
        device=hidden.device,
    )

    return hidden[
        batch_index,
        indices,
    ]


# ============================================================
# EXTRACT ALL LAYERS
#
# Returns:
#
# {
#   "entity": [layer0_tensor, ..., layer12_tensor],
#   "last":   [...],
#   "mean":   [...]
# }
#
# Each tensor:
#   [N, hidden_dim]
# ============================================================

@torch.no_grad()
def extract_representations(
    texts,
    entities,
    label,
):

    section(
        f"EXTRACT REPRESENTATIONS: {label}"
    )

    storage = {
        "ENTITY_SPAN": defaultdict(list),
        "LAST_TOKEN": defaultdict(list),
        "MASKED_MEAN": defaultdict(list),
    }

    total_batches = math.ceil(
        len(texts) / BATCH_SIZE
    )

    num_hidden_layers = None

    for batch_start in range(
        0,
        len(texts),
        BATCH_SIZE,
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            len(texts),
        )

        text_batch = texts[
            batch_start:batch_end
        ]

        entity_batch = entities[
            batch_start:batch_end
        ]

        encoded = tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mapping = encoded.pop(
            "offset_mapping"
        )

        attention_mask_cpu = (
            encoded["attention_mask"]
            .clone()
        )

        entity_mask_cpu = (
            build_entity_mask(
                texts=text_batch,
                entities=entity_batch,
                offset_mapping=offset_mapping,
                attention_mask=attention_mask_cpu,
            )
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value
            in encoded.items()
        }

        entity_mask = (
            entity_mask_cpu.to(
                DEVICE
            )
        )

        outputs = (
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

        hidden_states = (
            outputs.hidden_states
        )

        if num_hidden_layers is None:

            num_hidden_layers = len(
                hidden_states
            )

            print(
                "Hidden-state outputs:",
                num_hidden_layers,
            )

            print(
                "(Layer 0 = embedding output; "
                "final index = final GPT-2 state)"
            )

        for layer_idx, hidden in enumerate(
            hidden_states
        ):

            entity_repr = masked_average(
                hidden,
                entity_mask,
            )

            last_repr = last_valid_token(
                hidden,
                encoded[
                    "attention_mask"
                ],
            )

            mean_repr = masked_average(
                hidden,
                encoded[
                    "attention_mask"
                ].bool(),
            )

            storage[
                "ENTITY_SPAN"
            ][
                layer_idx
            ].append(
                entity_repr.detach().cpu()
            )

            storage[
                "LAST_TOKEN"
            ][
                layer_idx
            ].append(
                last_repr.detach().cpu()
            )

            storage[
                "MASKED_MEAN"
            ][
                layer_idx
            ].append(
                mean_repr.detach().cpu()
            )

        current_batch = (
            batch_start // BATCH_SIZE
        ) + 1

        if (
            current_batch == 1
            or current_batch % 10 == 0
            or current_batch == total_batches
        ):

            print(
                f"Batch "
                f"{current_batch:>3}/"
                f"{total_batches}"
            )

    final_storage = {}

    for method in storage:

        final_storage[method] = {}

        for layer_idx in storage[
            method
        ]:

            final_storage[
                method
            ][
                layer_idx
            ] = torch.cat(
                storage[
                    method
                ][
                    layer_idx
                ],
                dim=0,
            )

    return final_storage


# ============================================================
# BUILD TEXT ARRAYS
# ============================================================

fact_texts = [
    item["fact"]
    for item in examples
]

query_texts = [
    item["query"]
    for item in examples
]

entity_names = [
    item["entity"]
    for item in examples
]


# ============================================================
# EXTRACT
# ============================================================

fact_repr = extract_representations(
    texts=fact_texts,
    entities=entity_names,
    label="FACT / WRITE VIEW",
)

query_repr = extract_representations(
    texts=query_texts,
    entities=entity_names,
    label="QUERY / READ VIEW",
)


# ============================================================
# EFFECTIVE RANK
# ============================================================

def effective_rank(x):

    x = x.float()

    # Center representations.
    x = (
        x
        - x.mean(
            dim=0,
            keepdim=True,
        )
    )

    singular_values = (
        torch.linalg.svdvals(x)
    )

    total = (
        singular_values.sum()
        .clamp_min(1e-12)
    )

    p = (
        singular_values
        / total
    )

    entropy = -(
        p
        * p.clamp_min(
            1e-12
        ).log()
    ).sum()

    return float(
        torch.exp(entropy).item()
    )


# ============================================================
# RETRIEVAL METRICS
#
# Query i should retrieve Fact i.
# ============================================================

def retrieval_metrics(
    write_vectors,
    query_vectors,
):

    write_vectors = (
        F.normalize(
            write_vectors.float(),
            p=2,
            dim=-1,
            eps=1e-8,
        )
    )

    query_vectors = (
        F.normalize(
            query_vectors.float(),
            p=2,
            dim=-1,
            eps=1e-8,
        )
    )

    # [N, N]
    #
    # Rows = queries
    # Columns = stored fact representations
    similarity = (
        query_vectors
        @ write_vectors.T
    )

    n = similarity.size(0)

    diagonal = (
        similarity.diag()
    )

    # --------------------------------------------------------
    # Hardest negative
    # --------------------------------------------------------

    negative_matrix = (
        similarity.clone()
    )

    identity = torch.eye(
        n,
        dtype=torch.bool,
    )

    negative_matrix[
        identity
    ] = -float("inf")

    hardest_negative = (
        negative_matrix
        .max(dim=1)
        .values
    )

    margin = (
        diagonal
        - hardest_negative
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    correct_scores = (
        diagonal.unsqueeze(1)
    )

    ranks = (
        similarity
        .gt(correct_scores)
        .sum(dim=1)
        + 1
    )

    recall_1 = (
        ranks.eq(1)
        .float()
        .mean()
    )

    recall_5 = (
        ranks.le(5)
        .float()
        .mean()
    )

    recall_10 = (
        ranks.le(10)
        .float()
        .mean()
    )

    mrr = (
        1.0
        / ranks.float()
    ).mean()

    positive_margin_fraction = (
        margin.gt(0)
        .float()
        .mean()
    )

    return {

        "recall1":
            float(
                recall_1.item()
            ),

        "recall5":
            float(
                recall_5.item()
            ),

        "recall10":
            float(
                recall_10.item()
            ),

        "mrr":
            float(
                mrr.item()
            ),

        "correct_cos":
            float(
                diagonal.mean().item()
            ),

        "negative_cos":
            float(
                hardest_negative.mean().item()
            ),

        "margin_mean":
            float(
                margin.mean().item()
            ),

        "margin_median":
            float(
                margin.median().item()
            ),

        "positive_margin":
            float(
                positive_margin_fraction.item()
            ),

        "mean_rank":
            float(
                ranks.float()
                .mean()
                .item()
            ),

        "median_rank":
            float(
                ranks.float()
                .median()
                .item()
            ),

        "ranks":
            ranks,

        "margin":
            margin,

        "similarity":
            similarity,
    }


# ============================================================
# RUN ALL METHODS/LAYERS
# ============================================================

section(
    "LAYER-WISE CROSS-VIEW RETRIEVAL"
)

all_results = {}

methods = [
    "ENTITY_SPAN",
    "LAST_TOKEN",
    "MASKED_MEAN",
]

num_layers = len(
    fact_repr["ENTITY_SPAN"]
)

chance = (
    1.0 / NUM_ENTITIES
)

print(
    f"Random Recall@1 chance: "
    f"{chance * 100:.4f}%"
)

print()

for method in methods:

    subsection(method)

    print(
        f"{'Layer':<8}"
        f"{'R@1':>10}"
        f"{'R@5':>10}"
        f"{'MRR':>10}"
        f"{'Correct':>12}"
        f"{'HardNeg':>12}"
        f"{'Margin':>12}"
        f"{'PosMargin':>12}"
        f"{'EffRank W':>12}"
        f"{'EffRank Q':>12}"
    )

    method_results = {}

    for layer_idx in range(
        num_layers
    ):

        W = (
            fact_repr[
                method
            ][
                layer_idx
            ]
        )

        Q = (
            query_repr[
                method
            ][
                layer_idx
            ]
        )

        metrics = (
            retrieval_metrics(
                W,
                Q,
            )
        )

        metrics[
            "write_effective_rank"
        ] = effective_rank(W)

        metrics[
            "query_effective_rank"
        ] = effective_rank(Q)

        method_results[
            layer_idx
        ] = metrics

        print(
            f"{layer_idx:<8}"
            f"{metrics['recall1'] * 100:>9.2f}%"
            f"{metrics['recall5'] * 100:>9.2f}%"
            f"{metrics['mrr']:>10.4f}"
            f"{metrics['correct_cos']:>12.5f}"
            f"{metrics['negative_cos']:>12.5f}"
            f"{metrics['margin_mean']:>12.5f}"
            f"{metrics['positive_margin'] * 100:>11.2f}%"
            f"{metrics['write_effective_rank']:>12.2f}"
            f"{metrics['query_effective_rank']:>12.2f}"
        )

    all_results[
        method
    ] = method_results


# ============================================================
# BEST LAYER PER REPRESENTATION TYPE
# ============================================================

section(
    "BEST LAYER PER REPRESENTATION"
)

best_configs = {}

for method in methods:

    results = all_results[
        method
    ]

    best_layer = max(
        results.keys(),
        key=lambda layer: (
            results[layer][
                "recall1"
            ],
            results[layer][
                "mrr"
            ],
            results[layer][
                "margin_mean"
            ],
        ),
    )

    best = results[
        best_layer
    ]

    best_configs[
        method
    ] = (
        best_layer,
        best,
    )

    print()
    print(method)

    print(
        "Best layer:",
        best_layer,
    )

    print(
        "Recall@1:",
        f"{best['recall1'] * 100:.2f}%"
    )

    print(
        "Recall@5:",
        f"{best['recall5'] * 100:.2f}%"
    )

    print(
        "MRR:",
        fmt(
            best["mrr"]
        ),
    )

    print(
        "Mean correct cosine:",
        fmt(
            best["correct_cos"]
        ),
    )

    print(
        "Mean hardest negative:",
        fmt(
            best["negative_cos"]
        ),
    )

    print(
        "Mean margin:",
        fmt(
            best["margin_mean"]
        ),
    )

    print(
        "Positive margins:",
        f"{best['positive_margin'] * 100:.2f}%"
    )


# ============================================================
# GLOBAL BEST CONFIGURATION
# ============================================================

section(
    "GLOBAL BEST CROSS-VIEW REPRESENTATION"
)

global_candidates = []

for method in methods:

    for layer_idx, result in (
        all_results[method].items()
    ):

        global_candidates.append(
            (
                result["recall1"],
                result["mrr"],
                result["margin_mean"],
                method,
                layer_idx,
                result,
            )
        )

global_candidates.sort(
    reverse=True,
    key=lambda x: (
        x[0],
        x[1],
        x[2],
    ),
)

best = global_candidates[0]

best_method = best[3]
best_layer = best[4]
best_result = best[5]

print(
    "Representation:",
    best_method,
)

print(
    "Layer:",
    best_layer,
)

print(
    "Recall@1:",
    f"{best_result['recall1'] * 100:.2f}%"
)

print(
    "Recall@5:",
    f"{best_result['recall5'] * 100:.2f}%"
)

print(
    "Recall@10:",
    f"{best_result['recall10'] * 100:.2f}%"
)

print(
    "MRR:",
    fmt(
        best_result["mrr"]
    ),
)

print(
    "Mean rank:",
    fmt(
        best_result[
            "mean_rank"
        ]
    ),
)

print(
    "Mean correct cosine:",
    fmt(
        best_result[
            "correct_cos"
        ]
    ),
)

print(
    "Mean hardest-negative cosine:",
    fmt(
        best_result[
            "negative_cos"
        ]
    ),
)

print(
    "Mean margin:",
    fmt(
        best_result[
            "margin_mean"
        ]
    ),
)

print(
    "Positive margin fraction:",
    f"{best_result['positive_margin'] * 100:.2f}%"
)


# ============================================================
# TOP 10 CONFIGURATIONS
# ============================================================

section(
    "TOP 10 LAYER / REPRESENTATION COMBINATIONS"
)

print(
    f"{'Rank':<6}"
    f"{'Method':<16}"
    f"{'Layer':>8}"
    f"{'R@1':>10}"
    f"{'MRR':>10}"
    f"{'Margin':>12}"
)

for rank_idx, item in enumerate(
    global_candidates[:10],
    start=1,
):

    _, _, _, method, layer_idx, result = item

    print(
        f"{rank_idx:<6}"
        f"{method:<16}"
        f"{layer_idx:>8}"
        f"{result['recall1'] * 100:>9.2f}%"
        f"{result['mrr']:>10.4f}"
        f"{result['margin_mean']:>12.5f}"
    )


# ============================================================
# SAMPLE RETRIEVAL ERRORS FOR BEST CONFIG
# ============================================================

section(
    "BEST-CONFIG RETRIEVAL EXAMPLES"
)

similarity = (
    best_result[
        "similarity"
    ]
)

ranks = (
    best_result[
        "ranks"
    ]
)

margins = (
    best_result[
        "margin"
    ]
)

# Show 10 examples:
# first 5 successful + worst 5 failures.

correct_indices = (
    ranks.eq(1)
    .nonzero(
        as_tuple=False
    )
    .flatten()
)

wrong_indices = (
    ranks.gt(1)
    .nonzero(
        as_tuple=False
    )
    .flatten()
)

display_indices = []

if len(correct_indices) > 0:

    display_indices.extend(
        correct_indices[:5].tolist()
    )

if len(wrong_indices) > 0:

    wrong_sorted = sorted(
        wrong_indices.tolist(),
        key=lambda i:
            float(
                margins[i].item()
            ),
    )

    display_indices.extend(
        wrong_sorted[:5]
    )


for idx in display_indices:

    sims = similarity[
        idx
    ]

    top_values, top_indices = (
        torch.topk(
            sims,
            k=min(
                5,
                NUM_ENTITIES,
            ),
        )
    )

    print()
    print(
        "QUERY ENTITY:",
        entities[idx],
    )

    print(
        "Correct fact:",
        entities[idx],
    )

    print(
        "Correct rank:",
        int(
            ranks[idx].item()
        ),
    )

    print(
        "Margin:",
        fmt(
            margins[idx].item()
        ),
    )

    print("Top matches:")

    for position in range(
        len(top_indices)
    ):

        candidate_idx = int(
            top_indices[
                position
            ].item()
        )

        score = float(
            top_values[
                position
            ].item()
        )

        marker = (
            "<-- CORRECT"
            if candidate_idx == idx
            else ""
        )

        print(
            f"  {position + 1}. "
            f"{entities[candidate_idx]:<22} "
            f"cos={score:.6f} "
            f"{marker}"
        )


# ============================================================
# DIAGNOSTIC INTERPRETATION
# ============================================================

section(
    "AUTOMATIC INTERPRETATION"
)

r1 = best_result[
    "recall1"
]

margin = best_result[
    "margin_mean"
]

positive_margin = (
    best_result[
        "positive_margin"
    ]
)

print(
    "Best method:",
    best_method,
)

print(
    "Best layer:",
    best_layer,
)

print(
    "Recall@1:",
    f"{r1 * 100:.2f}%"
)

print(
    "Chance Recall@1:",
    f"{chance * 100:.4f}%"
)

print(
    "Mean margin:",
    fmt(margin),
)

print(
    "Positive margins:",
    f"{positive_margin * 100:.2f}%"
)

print()

if (
    r1 >= 0.90
    and positive_margin >= 0.90
):

    print(
        "RESULT: STRONG CROSS-VIEW IDENTITY SIGNAL"
    )

    print()

    print(
        "GPT-2 already contains a highly usable "
        "representation for matching a fact entity "
        "to the same entity at query time."
    )

    print()

    print(
        "NEXT EXPERIMENT:"
    )

    print(
        "Store the best-layer representation as "
        "an external key beside whichever memory "
        "slot receives the fact, then retrieve "
        "the slot using cosine similarity."
    )

elif (
    r1 >= 0.50
    and r1 > chance * 10
):

    print(
        "RESULT: MODERATE CROSS-VIEW IDENTITY SIGNAL"
    )

    print()

    print(
        "GPT-2 contains substantial entity identity "
        "information, but the raw representation is "
        "not sufficiently separated for robust "
        "memory addressing."
    )

    print()

    print(
        "NEXT EXPERIMENT:"
    )

    print(
        "Use this layer as the input to a small "
        "contrastive key projection trained with "
        "same-entity positives and different-entity "
        "negatives."
    )

elif r1 > chance * 5:

    print(
        "RESULT: WEAK BUT ABOVE-CHANCE SIGNAL"
    )

    print()

    print(
        "Entity identity exists in the hidden "
        "representation, but raw cosine geometry "
        "is poorly aligned for retrieval."
    )

    print()

    print(
        "NEXT EXPERIMENT:"
    )

    print(
        "A contrastive addressing objective is "
        "probably required."
    )

else:

    print(
        "RESULT: RAW GPT-2 CROSS-VIEW RETRIEVAL "
        "IS NEAR CHANCE"
    )

    print()

    print(
        "The existing hidden states do not provide "
        "a sufficiently usable associative key."
    )

    print()

    print(
        "NEXT EXPERIMENT:"
    )

    print(
        "Explicitly learn an entity/binding key "
        "representation rather than relying on "
        "raw GPT-2 geometry."
    )


# ============================================================
# IMPORTANT FINAL COMPARISON
# ============================================================

section(
    "FINAL COMPARISON: FINAL LAYER VS BEST LAYER"
)

final_layer = (
    num_layers - 1
)

for method in methods:

    final_result = (
        all_results[
            method
        ][
            final_layer
        ]
    )

    method_best_layer, method_best = (
        best_configs[
            method
        ]
    )

    print()
    print(method)

    print(
        f"Final layer {final_layer}: "
        f"R@1="
        f"{final_result['recall1'] * 100:.2f}% "
        f"MRR="
        f"{final_result['mrr']:.4f} "
        f"margin="
        f"{final_result['margin_mean']:.5f}"
    )

    print(
        f"Best layer {method_best_layer}: "
        f"R@1="
        f"{method_best['recall1'] * 100:.2f}% "
        f"MRR="
        f"{method_best['mrr']:.4f} "
        f"margin="
        f"{method_best['margin_mean']:.5f}"
    )


section(
    "PROBE COMPLETE"
)

print(
    "No parameters were trained."
)

print(
    "No checkpoint was modified."
)

print(
    "No model source file was modified."
)

print()
print(
    "Send the full output back for analysis."
)
