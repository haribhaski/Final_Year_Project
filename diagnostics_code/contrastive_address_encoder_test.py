# ============================================================
# CONTRASTIVE ADDRESS ENCODER TEST
#
# Goal:
#   Learn a SMALL projection that makes the same entity match
#   across different fact/query templates.
#
#   IMPORTANT:
#   - GPT-2 is FROZEN
#   - models/ is NOT modified
#   - original checkpoints are NOT overwritten
#   - only the small address encoder is trained
#
# Pipeline:
#
#   FACT
#     ↓
#   GPT-2 layer 1 entity-span representation
#     ↓
#   Address Encoder
#     ↓
#   write key
#
#   QUERY
#     ↓
#   GPT-2 layer 1 entity-span representation
#     ↓
#   SAME Address Encoder
#     ↓
#   query key
#
# Train using symmetric InfoNCE.
#
# Evaluation:
#   unseen entities
#   unseen fact/query template pairs
#
# Run:
#
# python contrastive_address_encoder_test.py \
#   2>&1 | tee contrastive_address_encoder_test.log
#
# ============================================================

import math
import random
from collections import defaultdict

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

CHECKPOINT = (
    "outputs/retrieval_gradient_test/"
    "checkpoint_best.pt"
)

OUTPUT_CHECKPOINT = (
    "outputs/contrastive_address_encoder.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

SEED = 42

# ------------------------------------------------------------
# IMPORTANT:
# hidden_states[0] = embedding output
# hidden_states[1] = after GPT-2 block 1
# ------------------------------------------------------------

ADDRESS_LAYER = 1

# ------------------------------------------------------------
# DATASET SIZE
# ------------------------------------------------------------

NUM_TRAIN_ENTITIES = 4000
NUM_VALID_ENTITIES = 500
NUM_TEST_ENTITIES = 1000

# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

BATCH_SIZE = 64
EPOCHS = 12

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

TEMPERATURE = 0.07

# ------------------------------------------------------------
# ADDRESS ENCODER
# ------------------------------------------------------------

HIDDEN_DIM = 512
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


# ============================================================
# TEMPLATES
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
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# PRINT HELPERS
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

section("LOAD FROZEN GPT-2 MEMORY MODEL")

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

# ------------------------------------------------------------
# FREEZE EVERYTHING
# ------------------------------------------------------------

for parameter in model.parameters():

    parameter.requires_grad = False

print("GPT-2 + memory model completely FROZEN.")


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

    def forward(self, x):

        x = self.net(x)

        x = F.normalize(
            x,
            p=2,
            dim=-1,
            eps=1e-8,
        )

        return x


D_MODEL = (
    model.backbone.config.n_embd
)

encoder = AddressEncoder(
    input_dim=D_MODEL,
    hidden_dim=HIDDEN_DIM,
    output_dim=ADDRESS_DIM,
    dropout=DROPOUT,
).to(DEVICE)

trainable_parameters = sum(
    p.numel()
    for p in encoder.parameters()
    if p.requires_grad
)

print()
print(
    "Address encoder parameters:",
    trainable_parameters,
)


# ============================================================
# ENTITY DATA
# ============================================================

def make_entities(
    prefix,
    count,
):

    return [
        f"{prefix}-{i:05d}"
        for i in range(count)
    ]


TRAIN_ENTITIES = make_entities(
    "TrainEntity",
    NUM_TRAIN_ENTITIES,
)

VALID_ENTITIES = make_entities(
    "ValidEntity",
    NUM_VALID_ENTITIES,
)

TEST_ENTITIES = make_entities(
    "TestEntity",
    NUM_TEST_ENTITIES,
)


# ============================================================
# BUILD EXAMPLES
# ============================================================

def create_examples(
    entities,
    seed,
):

    rng = random.Random(
        seed
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


train_examples = create_examples(
    TRAIN_ENTITIES,
    SEED + 1,
)

valid_examples = create_examples(
    VALID_ENTITIES,
    SEED + 2,
)

test_examples = create_examples(
    TEST_ENTITIES,
    SEED + 3,
)


section("DATASET")

print(
    "Train entities:",
    len(train_examples),
)

print(
    "Validation entities:",
    len(valid_examples),
)

print(
    "Test entities:",
    len(test_examples),
)

print()

print("Example training pair:")

print(
    "WRITE:",
    train_examples[0][
        "write"
    ],
)

print(
    "QUERY:",
    train_examples[0][
        "query"
    ],
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

    seq_len = (
        offset_mapping.size(1)
    )

    entity_mask = torch.zeros(
        batch_size,
        seq_len,
        dtype=torch.bool,
    )

    for b in range(
        batch_size
    ):

        text = texts[b]
        entity = entities[b]

        start = text.find(
            entity
        )

        if start == -1:

            raise RuntimeError(
                f"Entity {entity!r} "
                f"not found in:\n{text}"
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

            if (
                token_end
                <= token_start
            ):

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

            raise RuntimeError(
                f"No entity tokens found "
                f"for {entity!r}"
            )

    return entity_mask


# ============================================================
# MASKED ENTITY AVERAGE
# ============================================================

def entity_average(
    hidden,
    mask,
):

    weights = (
        mask
        .unsqueeze(-1)
        .to(hidden.dtype)
    )

    numerator = (
        hidden
        * weights
    ).sum(
        dim=1
    )

    denominator = (
        weights.sum(
            dim=1
        )
        .clamp_min(
            1.0
        )
    )

    return (
        numerator
        / denominator
    )


# ============================================================
# EXTRACT LAYER-1 ENTITY REPRESENTATION
#
# IMPORTANT:
# no gradients through GPT-2
# ============================================================

@torch.no_grad()
def extract_entity_representations(
    texts,
    entities,
):

    encoded = tokenizer(
        texts,
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
            texts=texts,
            entities=entities,
            offset_mapping=offsets,
            attention_mask=(
                attention_mask_cpu
            ),
        )
    )

    encoded = {
        key: value.to(
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
            ADDRESS_LAYER
        ]
    )

    representation = (
        entity_average(
            hidden,
            entity_mask,
        )
    )

    return representation


# ============================================================
# BATCH CREATION
# ============================================================

def batches(
    examples,
    batch_size,
    shuffle,
):

    indices = list(
        range(
            len(examples)
        )
    )

    if shuffle:

        random.shuffle(
            indices
        )

    for start in range(
        0,
        len(indices),
        batch_size,
    ):

        selected = indices[
            start:
            start + batch_size
        ]

        yield [
            examples[i]
            for i in selected
        ]


# ============================================================
# SYMMETRIC CONTRASTIVE LOSS
# ============================================================

def contrastive_loss(
    write_keys,
    query_keys,
):

    logits = (
        query_keys
        @ write_keys.T
    )

    logits = (
        logits
        / TEMPERATURE
    )

    targets = torch.arange(
        logits.size(0),
        device=logits.device,
    )

    query_to_write = (
        F.cross_entropy(
            logits,
            targets,
        )
    )

    write_to_query = (
        F.cross_entropy(
            logits.T,
            targets,
        )
    )

    loss = (
        query_to_write
        + write_to_query
    ) / 2.0

    with torch.no_grad():

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        accuracy = (
            predictions
            .eq(targets)
            .float()
            .mean()
        )

    return (
        loss,
        accuracy,
        logits,
    )


# ============================================================
# GLOBAL RETRIEVAL EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    encoder,
    examples,
    label,
):

    encoder.eval()

    all_write_keys = []
    all_query_keys = []

    for batch in batches(
        examples,
        BATCH_SIZE,
        shuffle=False,
    ):

        entities = [
            item[
                "entity"
            ]
            for item in batch
        ]

        write_texts = [
            item[
                "write"
            ]
            for item in batch
        ]

        query_texts = [
            item[
                "query"
            ]
            for item in batch
        ]

        write_repr = (
            extract_entity_representations(
                write_texts,
                entities,
            )
        )

        query_repr = (
            extract_entity_representations(
                query_texts,
                entities,
            )
        )

        write_keys = (
            encoder(
                write_repr
            )
        )

        query_keys = (
            encoder(
                query_repr
            )
        )

        all_write_keys.append(
            write_keys.cpu()
        )

        all_query_keys.append(
            query_keys.cpu()
        )

    W = torch.cat(
        all_write_keys,
        dim=0,
    )

    Q = torch.cat(
        all_query_keys,
        dim=0,
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

    recall1 = float(
        ranks.eq(1)
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

    recall10 = float(
        ranks.le(10)
        .float()
        .mean()
        .item()
    )

    mrr = float(
        (
            1.0
            / ranks.float()
        )
        .mean()
        .item()
    )

    negative = (
        similarity.clone()
    )

    eye = torch.eye(
        n,
        dtype=torch.bool,
    )

    negative[
        eye
    ] = -float("inf")

    hardest_negative = (
        negative
        .max(dim=1)
        .values
    )

    margins = (
        correct_scores
        - hardest_negative
    )

    positive_margin = float(
        margins.gt(0)
        .float()
        .mean()
        .item()
    )

    mean_margin = float(
        margins.mean()
        .item()
    )

    mean_correct = float(
        correct_scores.mean()
        .item()
    )

    mean_hard_negative = float(
        hardest_negative.mean()
        .item()
    )

    print()
    print(
        f"{label}"
    )

    print(
        "Recall@1:",
        f"{recall1 * 100:.2f}%"
    )

    print(
        "Recall@5:",
        f"{recall5 * 100:.2f}%"
    )

    print(
        "Recall@10:",
        f"{recall10 * 100:.2f}%"
    )

    print(
        "MRR:",
        fmt(mrr),
    )

    print(
        "Correct cosine:",
        fmt(
            mean_correct
        ),
    )

    print(
        "Hard-negative cosine:",
        fmt(
            mean_hard_negative
        ),
    )

    print(
        "Mean margin:",
        fmt(
            mean_margin
        ),
    )

    print(
        "Positive margins:",
        f"{positive_margin * 100:.2f}%"
    )

    return {

        "recall1":
            recall1,

        "recall5":
            recall5,

        "recall10":
            recall10,

        "mrr":
            mrr,

        "margin":
            mean_margin,

        "positive_margin":
            positive_margin,

        "similarity":
            similarity,

        "ranks":
            ranks,

        "margins":
            margins,

        "write_keys":
            W,

        "query_keys":
            Q,
    }


# ============================================================
# RAW BASELINE BEFORE TRAINING
# ============================================================

section(
    "RAW LAYER-1 BASELINE"
)

raw_encoder = nn.Identity()

# ------------------------------------------------------------
# Identity does not normalize, so evaluate manually
# ------------------------------------------------------------

@torch.no_grad()
def evaluate_raw(
    examples,
):

    all_write = []
    all_query = []

    for batch in batches(
        examples,
        BATCH_SIZE,
        shuffle=False,
    ):

        entities = [
            item["entity"]
            for item in batch
        ]

        write_text = [
            item["write"]
            for item in batch
        ]

        query_text = [
            item["query"]
            for item in batch
        ]

        W = (
            extract_entity_representations(
                write_text,
                entities,
            )
        )

        Q = (
            extract_entity_representations(
                query_text,
                entities,
            )
        )

        W = F.normalize(
            W,
            p=2,
            dim=-1,
        )

        Q = F.normalize(
            Q,
            p=2,
            dim=-1,
        )

        all_write.append(
            W.cpu()
        )

        all_query.append(
            Q.cpu()
        )

    W = torch.cat(
        all_write
    )

    Q = torch.cat(
        all_query
    )

    similarity = (
        Q @ W.T
    )

    correct = (
        similarity.diag()
    )

    ranks = (
        similarity
        .gt(
            correct.unsqueeze(1)
        )
        .sum(dim=1)
        + 1
    )

    r1 = float(
        ranks.eq(1)
        .float()
        .mean()
        .item()
    )

    r5 = float(
        ranks.le(5)
        .float()
        .mean()
        .item()
    )

    mrr = float(
        (
            1.0
            / ranks.float()
        )
        .mean()
        .item()
    )

    print(
        "Raw test Recall@1:",
        f"{r1 * 100:.2f}%"
    )

    print(
        "Raw test Recall@5:",
        f"{r5 * 100:.2f}%"
    )

    print(
        "Raw test MRR:",
        fmt(mrr),
    )

    return r1


raw_test_r1 = evaluate_raw(
    test_examples
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    encoder.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# TRAINING
# ============================================================

section(
    "TRAIN CONTRASTIVE ADDRESS ENCODER"
)

best_validation = -1.0

best_state = None

for epoch in range(
    1,
    EPOCHS + 1,
):

    encoder.train()

    epoch_loss = 0.0
    epoch_acc = 0.0
    step_count = 0

    for step, batch in enumerate(
        batches(
            train_examples,
            BATCH_SIZE,
            shuffle=True,
        ),
        start=1,
    ):

        entities = [
            item[
                "entity"
            ]
            for item in batch
        ]

        write_texts = [
            item[
                "write"
            ]
            for item in batch
        ]

        query_texts = [
            item[
                "query"
            ]
            for item in batch
        ]

        # ----------------------------------------------------
        # Frozen GPT-2 representation
        # ----------------------------------------------------

        with torch.no_grad():

            write_repr = (
                extract_entity_representations(
                    write_texts,
                    entities,
                )
            )

            query_repr = (
                extract_entity_representations(
                    query_texts,
                    entities,
                )
            )

        # ----------------------------------------------------
        # Trainable address projection
        # ----------------------------------------------------

        write_keys = (
            encoder(
                write_repr
            )
        )

        query_keys = (
            encoder(
                query_repr
            )
        )

        loss, batch_acc, _ = (
            contrastive_loss(
                write_keys,
                query_keys,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            encoder.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        epoch_loss += float(
            loss.item()
        )

        epoch_acc += float(
            batch_acc.item()
        )

        step_count += 1

        if step % 20 == 0:

            print(
                f"Epoch "
                f"{epoch:02d}/{EPOCHS} "
                f"| Step "
                f"{step:03d} "
                f"| Loss "
                f"{loss.item():.4f} "
                f"| Batch R@1 "
                f"{batch_acc.item() * 100:.2f}%"
            )

    epoch_loss /= max(
        step_count,
        1,
    )

    epoch_acc /= max(
        step_count,
        1,
    )

    print()
    print(
        f"Epoch {epoch:02d} complete "
        f"| mean loss="
        f"{epoch_loss:.4f} "
        f"| mean batch R@1="
        f"{epoch_acc * 100:.2f}%"
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    validation = evaluate(
        encoder,
        valid_examples,
        label=(
            f"VALIDATION AFTER "
            f"EPOCH {epoch}"
        ),
    )

    if (
        validation[
            "recall1"
        ]
        > best_validation
    ):

        best_validation = (
            validation[
                "recall1"
            ]
        )

        best_state = {
            key:
                value.detach()
                .cpu()
                .clone()

            for key, value
            in encoder
            .state_dict()
            .items()
        }

        print()
        print(
            "NEW BEST VALIDATION MODEL:"
        )

        print(
            f"Recall@1 = "
            f"{best_validation * 100:.2f}%"
        )


# ============================================================
# RESTORE BEST MODEL
# ============================================================

section(
    "RESTORE BEST VALIDATION MODEL"
)

if best_state is None:

    raise RuntimeError(
        "No best encoder state was saved."
    )

encoder.load_state_dict(
    best_state
)

encoder.to(DEVICE)
encoder.eval()

print(
    "Best validation Recall@1:",
    f"{best_validation * 100:.2f}%"
)


# ============================================================
# FINAL UNSEEN TEST
# ============================================================

section(
    "FINAL UNSEEN-ENTITY TEST"
)

test_results = evaluate(
    encoder,
    test_examples,
    label="UNSEEN TEST",
)


# ============================================================
# IMPROVEMENT
# ============================================================

section(
    "RAW VS CONTRASTIVE"
)

print(
    "Raw layer-1 test R@1:",
    f"{raw_test_r1 * 100:.2f}%"
)

print(
    "Contrastive test R@1:",
    f"{test_results['recall1'] * 100:.2f}%"
)

print(
    "Absolute improvement:",
    f"{(test_results['recall1'] - raw_test_r1) * 100:.2f} points"
)


# ============================================================
# TEST TEMPLATE PAIRS
# ============================================================

section(
    "TEST TEMPLATE-PAIR BREAKDOWN"
)

pair_stats = defaultdict(
    lambda: {
        "count": 0,
        "correct": 0,
        "rank_sum": 0.0,
    }
)

ranks = (
    test_results[
        "ranks"
    ]
)

for i, item in enumerate(
    test_examples
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

    count = (
        stats[
            "count"
        ]
    )

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
# WORST FAILURES
# ============================================================

section(
    "WORST UNSEEN RETRIEVAL FAILURES"
)

similarity = (
    test_results[
        "similarity"
    ]
)

margins = (
    test_results[
        "margins"
    ]
)

ranks = (
    test_results[
        "ranks"
    ]
)

wrong_indices = (
    ranks.gt(1)
    .nonzero(
        as_tuple=False
    )
    .flatten()
    .tolist()
)

wrong_indices = sorted(
    wrong_indices,
    key=lambda idx:
        float(
            margins[idx]
        ),
)

for idx in wrong_indices[:10]:

    example = (
        test_examples[
            idx
        ]
    )

    sims = (
        similarity[
            idx
        ]
    )

    values, indices = (
        torch.topk(
            sims,
            k=5,
        )
    )

    print()
    print(
        "ENTITY:",
        example[
            "entity"
        ],
    )

    print(
        "WRITE:",
        example[
            "write"
        ],
    )

    print(
        "QUERY:",
        example[
            "query"
        ],
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
        len(indices)
    ):

        candidate_idx = int(
            indices[
                pos
            ].item()
        )

        score = float(
            values[
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
            f"{TEST_ENTITIES[candidate_idx]:<25} "
            f"cos={score:.6f} "
            f"{marker}"
        )


# ============================================================
# SAVE ONLY EXPERIMENTAL ENCODER
# ============================================================

section(
    "SAVE EXPERIMENTAL ENCODER"
)

torch.save(
    {
        "encoder_state_dict":
            encoder.state_dict(),

        "address_layer":
            ADDRESS_LAYER,

        "input_dim":
            D_MODEL,

        "hidden_dim":
            HIDDEN_DIM,

        "address_dim":
            ADDRESS_DIM,

        "temperature":
            TEMPERATURE,

        "best_validation_recall1":
            best_validation,

        "test_recall1":
            test_results[
                "recall1"
            ],

        "test_mrr":
            test_results[
                "mrr"
            ],

        "test_margin":
            test_results[
                "margin"
            ],
    },

    OUTPUT_CHECKPOINT,
)

print(
    "Saved experimental encoder to:",
    OUTPUT_CHECKPOINT,
)


# ============================================================
# FINAL DECISION
# ============================================================

section(
    "FINAL DECISION"
)

r1 = (
    test_results[
        "recall1"
    ]
)

mrr = (
    test_results[
        "mrr"
    ]
)

positive = (
    test_results[
        "positive_margin"
    ]
)

print(
    "Unseen test Recall@1:",
    f"{r1 * 100:.2f}%"
)

print(
    "Unseen test MRR:",
    fmt(mrr),
)

print(
    "Positive margins:",
    f"{positive * 100:.2f}%"
)

print()

if r1 >= 0.90:

    print(
        "RESULT: STRONG."
    )

    print()

    print(
        "The contrastive address encoder "
        "generalizes well enough to move to "
        "the next isolated experiment:"
    )

    print()

    print(
        "STORE THESE KEYS BESIDE "
        "OCCUPANCY-ALLOCATED MEMORY SLOTS "
        "AND TEST 2/4/8-FACT SLOT RETRIEVAL."
    )

elif r1 >= 0.75:

    print(
        "RESULT: PROMISING BUT NOT YET "
        "ROBUST."
    )

    print()

    print(
        "Do not integrate yet."
    )

    print(
        "Inspect template-specific failures "
        "and improve contrastive training."
    )

elif r1 > raw_test_r1:

    print(
        "RESULT: CONTRASTIVE TRAINING HELPS, "
        "BUT GENERALIZATION IS STILL WEAK."
    )

    print()

    print(
        "Do not integrate."
    )

    print(
        "We need to improve the address "
        "representation/training objective."
    )

else:

    print(
        "RESULT: CONTRASTIVE PROJECTION "
        "DID NOT SOLVE THE PROBLEM."
    )

    print()

    print(
        "Do not integrate this mechanism."
    )


section(
    "DONE"
)

print(
    "GPT-2 was frozen."
)

print(
    "Original memory architecture unchanged."
)

print(
    "Original checkpoint unchanged."
)