import random
from dataclasses import dataclass
from typing import List

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
CHECKPOINT = "outputs/retrieval_gradient_test/checkpoint_best.pt"

SEED = 42

TRAIN_EPISODES = 500
STEPS_PER_EPISODE = 1

TEST_EPISODES = 100

LR = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRAIN_MEMORY_SIZES = [2, 4, 8]
TEST_MEMORY_SIZES = [2, 4, 8]

# Completely separate names for train/test.
TRAIN_ENTITIES = [
    f"TrainEntity-{i:04d}"
    for i in range(1000)
]

TEST_ENTITIES = [
    f"UnseenEntity-{i:04d}"
    for i in range(1000)
]

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


@dataclass
class Fact:
    entity: str
    answer: str


# ============================================================
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# TEXT
# ============================================================

def fact_text(fact):
    return (
        f"The assigned keyword for {fact.entity} "
        f"is {fact.answer}. "
        f"Remember that the keyword associated with "
        f"{fact.entity} is {fact.answer}."
    )


def query_text(fact):
    return (
        f"The assigned keyword for "
        f"{fact.entity} is"
    )


# ============================================================
# MODEL CONFIG
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
        reader_top_k=None,
        reader_temperature=0.8,

        candidate_diversity_weight=0.0,
        update_orthogonality_weight=0.0,
        router_balance_weight=0.0,
        reader_balance_weight=0.0,
        memory_collapse_weight=0.0,

        detach_memory_between_steps=False,
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading base:", MODEL_NAME)

    model = (
        MemoryAugmentedGPT2LMHeadModel
        .from_pretrained(
            MODEL_NAME,
            memory_config=build_config(),
        )
    )

    print("Loading checkpoint:", CHECKPOINT)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]

    else:
        state = checkpoint

    current = model.state_dict()

    compatible = {}

    for name, value in state.items():

        if (
            name in current
            and current[name].shape == value.shape
        ):
            compatible[name] = value

    result = model.load_state_dict(
        compatible,
        strict=False,
    )

    print(
        "Loaded compatible tensors:",
        len(compatible),
    )

    if result.missing_keys:

        print("Missing keys:")

        for key in result.missing_keys:
            print(" ", key)

    model.to(DEVICE)

    # Freeze original architecture.
    for p in model.parameters():
        p.requires_grad = False

    model.eval()

    return model


# ============================================================
# EXPERIMENTAL BINDER
# ============================================================

class QueryAddressBinder(nn.Module):

    def __init__(self, d_model):

        super().__init__()

        self.query_norm = nn.LayerNorm(
            d_model
        )

        self.query_projection = nn.Sequential(

            nn.Linear(
                d_model,
                d_model,
            ),

            nn.GELU(),

            nn.Linear(
                d_model,
                d_model,
                bias=False,
            ),
        )

    def forward(self, query):

        query = self.query_norm(query)

        query = self.query_projection(
            query
        )

        return F.normalize(
            query,
            p=2,
            dim=-1,
            eps=1e-8,
        )


# ============================================================
# ENCODE
# ============================================================

def encode(text):

    batch = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    )

    return {
        k: v.to(DEVICE)
        for k, v in batch.items()
    }


# ============================================================
# QUERY REPRESENTATION
# ============================================================

@torch.no_grad()
def get_query_representation(
    model,
    fact,
):

    batch = encode(
        query_text(fact)
    )

    outputs = model.backbone.transformer(

        input_ids=batch["input_ids"],

        attention_mask=batch[
            "attention_mask"
        ],

        use_cache=False,

        return_dict=True,
    )

    hidden = outputs.last_hidden_state

    lengths = (
        batch["attention_mask"]
        .sum(dim=1)
        - 1
    )

    idx = torch.arange(
        hidden.size(0),
        device=DEVICE,
    )

    return hidden[
        idx,
        lengths,
    ]


# ============================================================
# WRITE FACT
# ============================================================

@torch.no_grad()
def write_fact(
    model,
    memory,
    fact,
):

    batch = encode(
        fact_text(fact)
    )

    output = model(

        input_ids=batch["input_ids"],

        attention_mask=batch[
            "attention_mask"
        ],

        memory_state=memory,

        update_memory=True,

        return_diagnostics=True,
    )

    if output.routing_output is None:

        raise RuntimeError(
            "Router output missing."
        )

    route = (
        output.routing_output
        .weights[0]
    )

    slot = int(
        route.argmax().item()
    )

    return (
        output.memory_state,
        slot,
    )


# ============================================================
# SLOT ADDRESSES
# ============================================================

@torch.no_grad()
def get_slot_addresses(model):

    addresses = (
        model.router.slot_embeddings
    )

    addresses = (
        model.router.slot_projection(
            addresses
        )
    )

    return F.normalize(
        addresses,
        p=2,
        dim=-1,
        eps=1e-8,
    )


# ============================================================
# CREATE RANDOM EPISODE
# ============================================================

def make_episode(
    entity_pool,
    size,
):

    entities = random.sample(
        entity_pool,
        size,
    )

    answers = random.sample(
        ANSWERS,
        size,
    )

    return [
        Fact(entity, answer)
        for entity, answer
        in zip(
            entities,
            answers,
        )
    ]


# ============================================================
# BUILD MEMORY EPISODE
# ============================================================

@torch.no_grad()
def build_memory_episode(
    model,
    facts,
):

    memory = model.initialize_memory(
        batch_size=1,
        device=DEVICE,
        dtype=next(
            model.parameters()
        ).dtype,
    )

    slots = []

    for fact in facts:

        memory, slot = write_fact(
            model,
            memory,
            fact,
        )

        slots.append(slot)

    return memory, slots


# ============================================================
# ADDRESS LOGITS
# ============================================================

def address_logits(
    binder,
    queries,
    slot_addresses,
    written_mask,
    temperature=0.1,
):

    projected = binder(
        queries
    )

    logits = torch.matmul(
        projected,
        slot_addresses.T,
    )

    logits = (
        logits / temperature
    )

    logits = logits.masked_fill(
        ~written_mask.unsqueeze(0),
        torch.finfo(
            logits.dtype
        ).min,
    )

    return logits


# ============================================================
# TRAIN ONE EPISODE
# ============================================================

def train_episode(
    model,
    binder,
    optimizer,
    slot_addresses,
    size,
):

    facts = make_episode(
        TRAIN_ENTITIES,
        size,
    )

    memory, slots = (
        build_memory_episode(
            model,
            facts,
        )
    )

    # Occupancy routing should allocate
    # different slots.
    if len(set(slots)) != len(slots):

        raise RuntimeError(
            f"WRITE COLLISION: {slots}"
        )

    written_mask = (
        memory.write_count[0] > 0
    )

    queries = torch.cat(
        [
            get_query_representation(
                model,
                fact,
            )
            for fact in facts
        ],
        dim=0,
    )

    targets = torch.tensor(
        slots,
        dtype=torch.long,
        device=DEVICE,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = address_logits(
        binder,
        queries,
        slot_addresses,
        written_mask,
    )

    loss = F.cross_entropy(
        logits,
        targets,
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        binder.parameters(),
        1.0,
    )

    optimizer.step()

    predictions = (
        logits.argmax(dim=-1)
    )

    correct = (
        predictions == targets
    ).sum().item()

    return (
        loss.item(),
        correct,
        len(facts),
    )


# ============================================================
# TRAIN
# ============================================================

def train(
    model,
    binder,
    slot_addresses,
):

    optimizer = torch.optim.AdamW(
        binder.parameters(),
        lr=LR,
    )

    print()
    print("=" * 90)
    print("TRAINING SCALED ADDRESS BINDER")
    print("=" * 90)

    rolling_loss = 0.0
    rolling_correct = 0
    rolling_total = 0

    for episode in range(
        1,
        TRAIN_EPISODES + 1,
    ):

        # Curriculum:
        #
        # First learn small memories,
        # then progressively harder ones.

        if episode <= 100:
            size = 2

        elif episode <= 250:
            size = random.choice(
                [2, 4]
            )

        else:
            size = random.choice(
                TRAIN_MEMORY_SIZES
            )

        loss, correct, total = (
            train_episode(
                model,
                binder,
                optimizer,
                slot_addresses,
                size,
            )
        )

        rolling_loss += loss
        rolling_correct += correct
        rolling_total += total

        if episode % 25 == 0:

            accuracy = (
                100.0
                * rolling_correct
                / rolling_total
            )

            print(
                f"Episode "
                f"{episode:04d}/{TRAIN_EPISODES} "
                f"| size={size} "
                f"| loss="
                f"{rolling_loss / 25:.4f} "
                f"| address acc="
                f"{accuracy:.2f}%"
            )

            rolling_loss = 0.0
            rolling_correct = 0
            rolling_total = 0


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    binder,
    slot_addresses,
    memory_size,
    episodes,
):

    binder.eval()

    total = 0
    correct = 0

    reciprocal_rank = 0.0

    collisions = 0

    per_slot_correct = [0] * 8
    per_slot_total = [0] * 8

    for _ in range(episodes):

        facts = make_episode(
            TEST_ENTITIES,
            memory_size,
        )

        memory, slots = (
            build_memory_episode(
                model,
                facts,
            )
        )

        if len(set(slots)) != len(slots):
            collisions += 1
            continue

        written_mask = (
            memory.write_count[0] > 0
        )

        queries = torch.cat(
            [
                get_query_representation(
                    model,
                    fact,
                )
                for fact in facts
            ],
            dim=0,
        )

        logits = address_logits(
            binder,
            queries,
            slot_addresses,
            written_mask,
        )

        predictions = (
            logits.argmax(dim=-1)
        )

        for i, target in enumerate(
            slots
        ):

            pred = int(
                predictions[i].item()
            )

            total += 1

            per_slot_total[
                target
            ] += 1

            if pred == target:

                correct += 1

                per_slot_correct[
                    target
                ] += 1

            # Rank of correct address
            order = torch.argsort(
                logits[i],
                descending=True,
            )

            position = (
                order == target
            ).nonzero(
                as_tuple=False
            )

            rank = int(
                position[0].item()
            ) + 1

            reciprocal_rank += (
                1.0 / rank
            )

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    mrr = (
        reciprocal_rank / total
        if total
        else 0.0
    )

    return {
        "size": memory_size,
        "accuracy": accuracy,
        "mrr": mrr,
        "correct": correct,
        "total": total,
        "collisions": collisions,
        "per_slot_correct":
            per_slot_correct,
        "per_slot_total":
            per_slot_total,
    }


# ============================================================
# SHOW EXAMPLE
# ============================================================

@torch.no_grad()
def show_example(
    model,
    binder,
    slot_addresses,
    memory_size=8,
):

    facts = make_episode(
        TEST_ENTITIES,
        memory_size,
    )

    memory, slots = (
        build_memory_episode(
            model,
            facts,
        )
    )

    written_mask = (
        memory.write_count[0] > 0
    )

    queries = torch.cat(
        [
            get_query_representation(
                model,
                fact,
            )
            for fact in facts
        ],
        dim=0,
    )

    logits = address_logits(
        binder,
        queries,
        slot_addresses,
        written_mask,
    )

    probs = F.softmax(
        logits,
        dim=-1,
    )

    predictions = (
        probs.argmax(dim=-1)
    )

    print()
    print("=" * 90)
    print(
        f"EXAMPLE UNSEEN "
        f"{memory_size}-FACT MEMORY"
    )
    print("=" * 90)

    for i, fact in enumerate(facts):

        print()
        print(
            f"{fact.entity} "
            f"-> {fact.answer}"
        )

        print(
            f"WRITE SLOT: "
            f"{slots[i]}"
        )

        print(
            f"PRED READ SLOT: "
            f"{int(predictions[i])}"
        )

        print(
            "ADDRESS PROBS:",
            [
                round(x, 3)
                for x
                in probs[i].tolist()
            ],
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print(
        "SCALED UNSEEN-ENTITY "
        "ADDRESS-BINDING TEST"
    )
    print("=" * 90)

    print("Device:", DEVICE)

    print(
        "Original model files: UNCHANGED"
    )

    print(
        "Original model parameters: FROZEN"
    )

    print(
        "Train entities:",
        len(TRAIN_ENTITIES),
    )

    print(
        "Unseen test entities:",
        len(TEST_ENTITIES),
    )

    model = load_model()

    slot_addresses = (
        get_slot_addresses(model)
    )

    d_model = (
        slot_addresses.size(-1)
    )

    binder = QueryAddressBinder(
        d_model
    ).to(DEVICE)

    trainable = sum(
        p.numel()
        for p in binder.parameters()
        if p.requires_grad
    )

    print(
        "Binder parameters:",
        f"{trainable:,}",
    )

    # ========================================================
    # TRAIN
    # ========================================================

    train(
        model,
        binder,
        slot_addresses,
    )

    # ========================================================
    # UNSEEN EVALUATION
    # ========================================================

    print()
    print("=" * 90)
    print(
        "UNSEEN ENTITY EVALUATION"
    )
    print("=" * 90)

    results = []

    for size in TEST_MEMORY_SIZES:

        result = evaluate(
            model,
            binder,
            slot_addresses,
            memory_size=size,
            episodes=TEST_EPISODES,
        )

        results.append(result)

        chance = 100.0 / size

        print()
        print(
            f"{size}-FACT MEMORY"
        )

        print(
            f"Accuracy: "
            f"{result['accuracy'] * 100:.2f}% "
            f"({result['correct']}/"
            f"{result['total']})"
        )

        print(
            f"Chance: {chance:.2f}%"
        )

        print(
            f"MRR: "
            f"{result['mrr']:.4f}"
        )

        print(
            f"Write collisions: "
            f"{result['collisions']}"
        )

    # ========================================================
    # PER SLOT
    # ========================================================

    print()
    print("=" * 90)
    print(
        "PER-SLOT ACCURACY "
        "(8-FACT TEST)"
    )
    print("=" * 90)

    result8 = results[-1]

    for slot in range(8):

        total = (
            result8[
                "per_slot_total"
            ][slot]
        )

        correct = (
            result8[
                "per_slot_correct"
            ][slot]
        )

        accuracy = (
            100.0 * correct / total
            if total
            else 0.0
        )

        print(
            f"Slot {slot}: "
            f"{accuracy:.2f}% "
            f"({correct}/{total})"
        )

    # ========================================================
    # EXAMPLE
    # ========================================================

    show_example(
        model,
        binder,
        slot_addresses,
        memory_size=8,
    )

    # ========================================================
    # DECISION
    # ========================================================

    acc2 = results[0][
        "accuracy"
    ]

    acc4 = results[1][
        "accuracy"
    ]

    acc8 = results[2][
        "accuracy"
    ]

    print()
    print("=" * 90)
    print("FINAL DIAGNOSTIC")
    print("=" * 90)

    print(
        f"2 facts: "
        f"{acc2 * 100:.2f}%"
    )

    print(
        f"4 facts: "
        f"{acc4 * 100:.2f}%"
    )

    print(
        f"8 facts: "
        f"{acc8 * 100:.2f}%"
    )

    print()

    # Deliberately strict threshold.
    if (
        acc2 >= 0.90
        and acc4 >= 0.80
        and acc8 >= 0.65
    ):

        print(
            "GENERALIZATION TEST: PASSED"
        )

        print()
        print(
            "Address binder generalizes "
            "to unseen entity names."
        )

        print(
            "Next step: test actual "
            "answer retrieval before "
            "integrating into main model."
        )

    else:

        print(
            "GENERALIZATION TEST: "
            "NOT PASSED"
        )

        print()
        print(
            "Do NOT integrate this binder "
            "into the original architecture."
        )

        print(
            "The two-fact result was likely "
            "memorization or insufficiently "
            "general addressing."
        )

    print("=" * 90)


if __name__ == "__main__":
    main()