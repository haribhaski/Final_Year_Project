# ============================================================
# CROSS-TEMPLATE ENTITY PROBE
#
# Goal:
#   Test whether the SAME entity can still be matched across
#   DIFFERENT write/query sentence templates.
#
# This is harder and more meaningful than the previous probe
# because fact/query no longer share the same local wording.
#
# NO TRAINING
# NO CHANGES TO models/
# NO CHECKPOINT OVERWRITE
#
# Tests every GPT-2 hidden layer using:
#   1. ENTITY_SPAN
#   2. LAST_TOKEN
#   3. MASKED_MEAN
#
# Metrics:
#   - Recall@1
#   - Recall@5
#   - Recall@10
#   - MRR
#   - correct cosine
#   - hardest-negative cosine
#   - margin
#   - positive-margin fraction
#
# Run:
#
# python cross_template_entity_probe.py \
#   2>&1 | tee cross_template_entity_probe.log
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
# CONFIG
# ============================================================

MODEL_NAME = "gpt2"

CHECKPOINT = (
    "outputs/retrieval_gradient_test/"
    "checkpoint_best.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

SEED = 42

NUM_ENTITIES = 500

BATCH_SIZE = 16

ENTITY_PREFIX = "CrossEntity"

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
]


# ============================================================
# SEED
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
# CONFIG
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
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer:", tokenizer.__class__.__name__)
print("Vocabulary:", len(tokenizer))


# ============================================================
# LOAD MODEL
# ============================================================

section("LOAD MODEL")

print("Device:", DEVICE)
print("Base:", MODEL_NAME)
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
        and current_state[name].shape == value.shape
    ):

        compatible[name] = value

    else:

        skipped.append(name)


result = model.load_state_dict(
    compatible,
    strict=False,
)

print("Compatible tensors:", len(compatible))
print("Missing:", len(result.missing_keys))
print("Skipped:", len(skipped))

model.to(DEVICE)
model.eval()

print("Model loaded.")


# ============================================================
# BUILD DATASET
#
# IMPORTANT:
# Write template and query template are deliberately different.
# ============================================================

section("BUILD CROSS-TEMPLATE DATASET")

entities = [
    f"{ENTITY_PREFIX}-{i:04d}"
    for i in range(NUM_ENTITIES)
]

examples = []

rng = random.Random(SEED)

for i, entity in enumerate(entities):

    answer = ANSWERS[
        i % len(ANSWERS)
    ]

    write_template_idx = rng.randrange(
        len(WRITE_TEMPLATES)
    )

    query_template_idx = rng.randrange(
        len(QUERY_TEMPLATES)
    )

    write_template = WRITE_TEMPLATES[
        write_template_idx
    ]

    query_template = QUERY_TEMPLATES[
        query_template_idx
    ]

    fact = write_template.format(
        entity=entity,
        answer=answer,
    )

    query = query_template.format(
        entity=entity,
    )

    examples.append(
        {
            "entity": entity,
            "answer": answer,
            "fact": fact,
            "query": query,
            "write_template":
                write_template_idx,
            "query_template":
                query_template_idx,
        }
    )


print("Entities:", len(examples))
print("Write templates:", len(WRITE_TEMPLATES))
print("Query templates:", len(QUERY_TEMPLATES))

print()
print("Examples:")

for item in examples[:5]:

    print()
    print("Entity:", item["entity"])
    print("Fact: ", item["fact"])
    print("Query:", item["query"])


# ============================================================
# ENTITY TOKEN MASK
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
            seq_len
        ):

            if not bool(
                attention_mask[
                    b,
                    token_idx,
                ]
            ):

                continue

            token_start = int(
                offset_mapping[
                    b,
                    token_idx,
                    0,
                ].item()
            )

            token_end = int(
                offset_mapping[
                    b,
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
                    b,
                    token_idx,
                ] = True

                found += 1

        if found == 0:

            print()
            print("FAILED ENTITY MASK")
            print("Text:", text)
            print("Entity:", entity)

            raise RuntimeError(
                "No entity tokens found."
            )

    return entity_mask


# ============================================================
# REPRESENTATION HELPERS
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
        weights.sum(dim=1)
        .clamp_min(1.0)
    )

    return (
        numerator
        / denominator
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

    batch_indices = torch.arange(
        hidden.size(0),
        device=hidden.device,
    )

    return hidden[
        batch_indices,
        indices,
    ]


# ============================================================
# EXTRACT REPRESENTATIONS
# ============================================================

@torch.no_grad()
def extract_all_layers(
    texts,
    entity_names,
    label,
):

    section(
        f"EXTRACT: {label}"
    )

    storage = {

        "ENTITY_SPAN":
            defaultdict(list),

        "LAST_TOKEN":
            defaultdict(list),

        "MASKED_MEAN":
            defaultdict(list),
    }

    total_batches = math.ceil(
        len(texts)
        / BATCH_SIZE
    )

    num_layers = None

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

        batch_entities = entity_names[
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
                batch_text,
                batch_entities,
                offsets,
                attention_mask_cpu,
            )
        )

        encoded = {
            k: v.to(DEVICE)
            for k, v in encoded.items()
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

        hidden_states = (
            output.hidden_states
        )

        if num_layers is None:

            num_layers = len(
                hidden_states
            )

            print(
                "Hidden states:",
                num_layers,
            )

        for layer_idx, hidden in enumerate(
            hidden_states
        ):

            entity_repr = masked_average(
                hidden,
                entity_mask,
            )

            last_repr = (
                last_valid_token(
                    hidden,
                    encoded[
                        "attention_mask"
                    ],
                )
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
                entity_repr
                .detach()
                .cpu()
            )

            storage[
                "LAST_TOKEN"
            ][
                layer_idx
            ].append(
                last_repr
                .detach()
                .cpu()
            )

            storage[
                "MASKED_MEAN"
            ][
                layer_idx
            ].append(
                mean_repr
                .detach()
                .cpu()
            )

        batch_num = (
            start // BATCH_SIZE
        ) + 1

        if (
            batch_num == 1
            or batch_num % 10 == 0
            or batch_num == total_batches
        ):

            print(
                f"Batch "
                f"{batch_num}/"
                f"{total_batches}"
            )

    output_storage = {}

    for method in storage:

        output_storage[
            method
        ] = {}

        for layer_idx in storage[
            method
        ]:

            output_storage[
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

    return output_storage


# ============================================================
# INPUT ARRAYS
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
# EXTRACT FACT AND QUERY REPRESENTATIONS
# ============================================================

fact_repr = extract_all_layers(
    fact_texts,
    entity_names,
    "FACT / WRITE VIEW",
)

query_repr = extract_all_layers(
    query_texts,
    entity_names,
    "QUERY / READ VIEW",
)


# ============================================================
# EFFECTIVE RANK
# ============================================================

def effective_rank(x):

    x = x.float()

    x = (
        x
        - x.mean(
            dim=0,
            keepdim=True,
        )
    )

    singular_values = (
        torch.linalg.svdvals(
            x
        )
    )

    total = (
        singular_values
        .sum()
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
        torch.exp(
            entropy
        ).item()
    )


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def retrieval_metrics(
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

    # Rows = query
    # Columns = fact
    similarity = (
        Q @ W.T
    )

    n = similarity.size(0)

    correct = similarity.diag()

    identity = torch.eye(
        n,
        dtype=torch.bool,
    )

    negative_matrix = (
        similarity.clone()
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
        correct
        - hardest_negative
    )

    correct_expanded = (
        correct.unsqueeze(1)
    )

    ranks = (
        similarity
        .gt(
            correct_expanded
        )
        .sum(dim=1)
        + 1
    )

    recall1 = (
        ranks.eq(1)
        .float()
        .mean()
    )

    recall5 = (
        ranks.le(5)
        .float()
        .mean()
    )

    recall10 = (
        ranks.le(10)
        .float()
        .mean()
    )

    mrr = (
        1.0
        / ranks.float()
    ).mean()

    positive_margin = (
        margin.gt(0)
        .float()
        .mean()
    )

    return {

        "recall1":
            float(
                recall1.item()
            ),

        "recall5":
            float(
                recall5.item()
            ),

        "recall10":
            float(
                recall10.item()
            ),

        "mrr":
            float(
                mrr.item()
            ),

        "correct":
            float(
                correct.mean().item()
            ),

        "hard_negative":
            float(
                hardest_negative
                .mean()
                .item()
            ),

        "margin":
            float(
                margin.mean().item()
            ),

        "median_margin":
            float(
                margin.median().item()
            ),

        "positive_margin":
            float(
                positive_margin.item()
            ),

        "mean_rank":
            float(
                ranks.float()
                .mean()
                .item()
            ),

        "similarity":
            similarity,

        "ranks":
            ranks,

        "margins":
            margin,
    }


# ============================================================
# GLOBAL LAYER EVALUATION
# ============================================================

section(
    "CROSS-TEMPLATE LAYER-WISE RETRIEVAL"
)

methods = [
    "ENTITY_SPAN",
    "LAST_TOKEN",
    "MASKED_MEAN",
]

num_layers = len(
    fact_repr[
        "ENTITY_SPAN"
    ]
)

chance = (
    1.0
    / NUM_ENTITIES
)

print(
    "Random Recall@1 chance:",
    f"{chance * 100:.4f}%"
)

all_results = {}

for method in methods:

    subsection(method)

    print(
        f"{'Layer':<8}"
        f"{'R@1':>10}"
        f"{'R@5':>10}"
        f"{'R@10':>10}"
        f"{'MRR':>10}"
        f"{'Correct':>12}"
        f"{'HardNeg':>12}"
        f"{'Margin':>12}"
        f"{'PosMargin':>12}"
        f"{'ERank W':>12}"
        f"{'ERank Q':>12}"
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

        metrics = retrieval_metrics(
            W,
            Q,
        )

        metrics[
            "erank_write"
        ] = effective_rank(W)

        metrics[
            "erank_query"
        ] = effective_rank(Q)

        method_results[
            layer_idx
        ] = metrics

        print(
            f"{layer_idx:<8}"
            f"{metrics['recall1'] * 100:>9.2f}%"
            f"{metrics['recall5'] * 100:>9.2f}%"
            f"{metrics['recall10'] * 100:>9.2f}%"
            f"{metrics['mrr']:>10.4f}"
            f"{metrics['correct']:>12.5f}"
            f"{metrics['hard_negative']:>12.5f}"
            f"{metrics['margin']:>12.5f}"
            f"{metrics['positive_margin'] * 100:>11.2f}%"
            f"{metrics['erank_write']:>12.2f}"
            f"{metrics['erank_query']:>12.2f}"
        )

    all_results[
        method
    ] = method_results


# ============================================================
# BEST LAYER PER METHOD
# ============================================================

section(
    "BEST LAYER PER REPRESENTATION"
)

best_per_method = {}

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
                "margin"
            ],
        ),
    )

    best = results[
        best_layer
    ]

    best_per_method[
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
        "Correct cosine:",
        fmt(
            best["correct"]
        ),
    )

    print(
        "Hardest negative:",
        fmt(
            best[
                "hard_negative"
            ]
        ),
    )

    print(
        "Mean margin:",
        fmt(
            best["margin"]
        ),
    )

    print(
        "Positive margins:",
        f"{best['positive_margin'] * 100:.2f}%"
    )


# ============================================================
# GLOBAL BEST
# ============================================================

section(
    "GLOBAL BEST CONFIGURATION"
)

candidates = []

for method in methods:

    for layer_idx, result in (
        all_results[
            method
        ].items()
    ):

        candidates.append(
            (
                result[
                    "recall1"
                ],

                result[
                    "mrr"
                ],

                result[
                    "margin"
                ],

                method,

                layer_idx,

                result,
            )
        )


candidates.sort(
    reverse=True,
    key=lambda x: (
        x[0],
        x[1],
        x[2],
    ),
)

best = candidates[0]

best_method = best[3]
best_layer = best[4]
best_metrics = best[5]

print(
    "Method:",
    best_method,
)

print(
    "Layer:",
    best_layer,
)

print(
    "Recall@1:",
    f"{best_metrics['recall1'] * 100:.2f}%"
)

print(
    "Recall@5:",
    f"{best_metrics['recall5'] * 100:.2f}%"
)

print(
    "Recall@10:",
    f"{best_metrics['recall10'] * 100:.2f}%"
)

print(
    "MRR:",
    fmt(
        best_metrics["mrr"]
    ),
)

print(
    "Correct cosine:",
    fmt(
        best_metrics[
            "correct"
        ]
    ),
)

print(
    "Hardest negative:",
    fmt(
        best_metrics[
            "hard_negative"
        ]
    ),
)

print(
    "Mean margin:",
    fmt(
        best_metrics[
            "margin"
        ]
    ),
)

print(
    "Positive margin:",
    f"{best_metrics['positive_margin'] * 100:.2f}%"
)


# ============================================================
# TOP 15
# ============================================================

section(
    "TOP 15 CONFIGURATIONS"
)

print(
    f"{'Rank':<6}"
    f"{'Method':<16}"
    f"{'Layer':>8}"
    f"{'R@1':>10}"
    f"{'MRR':>10}"
    f"{'Margin':>12}"
)

for idx, item in enumerate(
    candidates[:15],
    start=1,
):

    _, _, _, method, layer_idx, metrics = item

    print(
        f"{idx:<6}"
        f"{method:<16}"
        f"{layer_idx:>8}"
        f"{metrics['recall1'] * 100:>9.2f}%"
        f"{metrics['mrr']:>10.4f}"
        f"{metrics['margin']:>12.5f}"
    )


# ============================================================
# TEMPLATE-PAIR ANALYSIS
#
# This tells us whether some wording combinations fail badly.
# ============================================================

section(
    "TEMPLATE-PAIR BREAKDOWN FOR BEST CONFIG"
)

similarity = (
    best_metrics[
        "similarity"
    ]
)

ranks = (
    best_metrics[
        "ranks"
    ]
)

pair_stats = defaultdict(
    lambda: {
        "count": 0,
        "correct": 0,
        "rank_sum": 0.0,
    }
)

for i, item in enumerate(
    examples
):

    pair = (
        item[
            "write_template"
        ],
        item[
            "query_template"
        ],
    )

    rank = int(
        ranks[i].item()
    )

    pair_stats[
        pair
    ][
        "count"
    ] += 1

    pair_stats[
        pair
    ][
        "rank_sum"
    ] += rank

    if rank == 1:

        pair_stats[
            pair
        ][
            "correct"
        ] += 1


print(
    f"{'WriteT':>8}"
    f"{'QueryT':>8}"
    f"{'N':>8}"
    f"{'R@1':>12}"
    f"{'MeanRank':>14}"
)

for pair in sorted(
    pair_stats.keys()
):

    stats = pair_stats[
        pair
    ]

    count = stats[
        "count"
    ]

    accuracy = (
        stats[
            "correct"
        ]
        / count
    )

    mean_rank = (
        stats[
            "rank_sum"
        ]
        / count
    )

    print(
        f"{pair[0]:>8}"
        f"{pair[1]:>8}"
        f"{count:>8}"
        f"{accuracy * 100:>11.2f}%"
        f"{mean_rank:>14.2f}"
    )


# ============================================================
# RETRIEVAL EXAMPLES
# ============================================================

section(
    "BEST-CONFIG RETRIEVAL EXAMPLES"
)

similarity = (
    best_metrics[
        "similarity"
    ]
)

ranks = (
    best_metrics[
        "ranks"
    ]
)

margins = (
    best_metrics[
        "margins"
    ]
)

wrong_indices = (
    ranks.gt(1)
    .nonzero(
        as_tuple=False
    )
    .flatten()
)

correct_indices = (
    ranks.eq(1)
    .nonzero(
        as_tuple=False
    )
    .flatten()
)

display = []

if len(correct_indices) > 0:

    display.extend(
        correct_indices[:5]
        .tolist()
    )

if len(wrong_indices) > 0:

    worst = sorted(
        wrong_indices.tolist(),
        key=lambda i:
            float(
                margins[
                    i
                ].item()
            ),
    )

    display.extend(
        worst[:10]
    )


for idx in display:

    item = examples[
        idx
    ]

    sims = similarity[
        idx
    ]

    top_values, top_indices = (
        torch.topk(
            sims,
            k=5,
        )
    )

    print()
    print(
        "ENTITY:",
        item["entity"],
    )

    print(
        "FACT:",
        item["fact"],
    )

    print(
        "QUERY:",
        item["query"],
    )

    print(
        "Correct rank:",
        int(
            ranks[
                idx
            ].item()
        ),
    )

    print(
        "Margin:",
        fmt(
            margins[
                idx
            ].item()
        ),
    )

    print(
        "Top matches:"
    )

    for pos in range(
        len(top_indices)
    ):

        candidate_idx = int(
            top_indices[
                pos
            ].item()
        )

        score = float(
            top_values[
                pos
            ].item()
        )

        marker = (
            "<-- CORRECT"
            if candidate_idx == idx
            else ""
        )

        print(
            f"  {pos + 1}. "
            f"{entities[candidate_idx]:<22} "
            f"cos={score:.6f} "
            f"{marker}"
        )


# ============================================================
# FINAL-LAYER VS BEST-LAYER
# ============================================================

section(
    "FINAL LAYER VS BEST LAYER"
)

final_layer = (
    num_layers - 1
)

for method in methods:

    best_layer_method, best_result = (
        best_per_method[
            method
        ]
    )

    final_result = (
        all_results[
            method
        ][
            final_layer
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
        f"Margin="
        f"{final_result['margin']:.6f}"
    )

    print(
        f"Best layer {best_layer_method}: "
        f"R@1="
        f"{best_result['recall1'] * 100:.2f}% "
        f"MRR="
        f"{best_result['mrr']:.4f} "
        f"Margin="
        f"{best_result['margin']:.6f}"
    )


# ============================================================
# AUTOMATIC INTERPRETATION
# ============================================================

section(
    "AUTOMATIC INTERPRETATION"
)

r1 = (
    best_metrics[
        "recall1"
    ]
)

mrr = (
    best_metrics[
        "mrr"
    ]
)

margin = (
    best_metrics[
        "margin"
    ]
)

positive = (
    best_metrics[
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
    "Chance:",
    f"{chance * 100:.4f}%"
)

print(
    "MRR:",
    fmt(mrr),
)

print(
    "Mean margin:",
    fmt(margin),
)

print(
    "Positive margins:",
    f"{positive * 100:.2f}%"
)

print()

if (
    r1 >= 0.90
    and positive >= 0.90
):

    print(
        "RESULT: CROSS-TEMPLATE ENTITY "
        "ADDRESSING IS STRONG."
    )

    print()

    print(
        "The entity representation remains "
        "stable even when fact and query "
        "wording differ."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Build a separate external key-memory "
        "test where this representation is "
        "stored beside the occupancy-selected "
        "memory slot."
    )

elif (
    r1 >= 0.50
):

    print(
        "RESULT: MODERATE CROSS-TEMPLATE SIGNAL."
    )

    print()

    print(
        "The representation contains useful "
        "identity information but is not yet "
        "reliable enough for direct addressing."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Test a small contrastive projection "
        "using this best layer."
    )

elif (
    r1 >= 0.10
):

    print(
        "RESULT: WEAK CROSS-TEMPLATE SIGNAL."
    )

    print()

    print(
        "The previous 100% result was heavily "
        "helped by identical local context."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Use explicit contrastive key learning "
        "rather than raw cosine retrieval."
    )

else:

    print(
        "RESULT: CROSS-TEMPLATE RETRIEVAL FAILS."
    )

    print()

    print(
        "Raw entity-span hidden states are not "
        "context-invariant enough for memory "
        "addressing."
    )

    print()

    print(
        "The previous same-template 100% result "
        "was mostly an identical-context effect."
    )


section(
    "PROBE COMPLETE"
)

print(
    "No training performed."
)

print(
    "No checkpoint modified."
)

print(
    "No model source files modified."
)