from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from models.gpt2_memory import (
    MemoryAugmentedGPT2LMHeadModel,
    MemoryGPT2Config,
)


# ============================================================
# SYNTHETIC DATA
# ============================================================

ANSWER_POOL = [
    "amber",
    "frost",
    "harbor",
    "maple",
    "olive",
    "pearl",
    "rose",
    "silver",
    "stone",
    "violet",
    "cloud",
    "river",
    "ocean",
]

TRAIN_FACT_TEMPLATES = [
    "{entity}'s secret code is {answer}.",
    "The code assigned to {entity} is {answer}.",
    "For {entity}, remember the code {answer}.",
    "Record this: {entity} has code {answer}.",
]

TRAIN_QUERY_TEMPLATES = [
    "What is {entity}'s secret code?",
    "Which code belongs to {entity}?",
    "Give the secret code for {entity}.",
    "What code was assigned to {entity}?",
]

EVAL_FACT_TEMPLATES = [
    "The secret code associated with {entity} is {answer}.",
    "{entity} is linked to the secret code {answer}.",
]

EVAL_QUERY_TEMPLATES = [
    "What is the secret code associated with {entity}?",
    "Which secret code is assigned to {entity}?",
]


# ============================================================
# CONFIG
# ============================================================

def build_memory_config():

    return MemoryGPT2Config(
        num_slots=8,

        gate_type="vector",
        gate_mode="sigmoid",
        gate_init_bias=-2.0,

        router_enabled=True,
        router_mode="softmax",
        router_top_k=2,
        router_temperature=0.7,

        writer_mode="attention",
        writer_attention_heads=8,

        orthogonal_mode="other_slots",
        orthogonal_strength=0.5,
        preserve_update_norm=True,
        learned_basis_rank=4,

        reader_mode="hybrid",
        reader_fusion="gated",
        reader_heads=8,
        reader_top_k=3,
        reader_temperature=0.8,
        reader_residual_scale=0.1,

        memory_normalization="layernorm",
        memory_max_slot_norm=None,
        trainable_initial_memory=True,

        summary_mode="masked_mean",
        read_before_write=True,
        detach_memory_between_steps=False,

        candidate_diversity_weight=0.0,
        update_orthogonality_weight=0.0,
        router_balance_weight=0.0,
        reader_balance_weight=0.0,
        head_diversity_weight=0.0,
        memory_collapse_weight=0.0,
        gate_sparsity_weight=0.0,
    )


def set_seed(seed):

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TOKENIZER HELPERS
# ============================================================

def get_single_token_answers(tokenizer):

    usable = []

    for word in ANSWER_POOL:

        ids = tokenizer(
            " " + word,
            add_special_tokens=False,
        )["input_ids"]

        if len(ids) == 1:
            usable.append(word)

    if len(usable) < 8:
        raise RuntimeError(
            "Need at least 8 usable single-token answer words."
        )

    return usable


def tokenize(
    tokenizer,
    texts,
    device,
):

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        add_special_tokens=False,
    )

    return (
        enc["input_ids"].to(device),
        enc["attention_mask"].to(device),
    )


# ============================================================
# LOAD ORIGINAL CHECKPOINT
# ============================================================

def load_original(
    checkpoint_path,
    model_name,
    device,
):

    print("Loading original checkpoint...")

    model = (
        MemoryAugmentedGPT2LMHeadModel
        .from_pretrained(
            model_name,
            memory_config=build_memory_config(),
        )
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model


# ============================================================
# EPISODES
#
# Each episode contains several facts.
#
# IMPORTANT:
# target slot is based on WRITE POSITION:
#
# fact 0 -> slot 0
# fact 1 -> slot 1
# ...
#
# This does NOT claim this is the final routing strategy.
# It is only a controlled allocation test:
#
# "Can this router architecture learn to distribute writes?"
#
# We explicitly supply write-position embedding below because
# otherwise the exact same router receives only semantic summary
# and has no information telling it which free slot is next.
# ============================================================

def build_episodes(
    n_episodes,
    facts_per_episode,
    answers,
    seed,
    start_id,
    split,
):

    rng = random.Random(seed)

    if split == "train":

        fact_templates = TRAIN_FACT_TEMPLATES
        query_templates = TRAIN_QUERY_TEMPLATES

    else:

        fact_templates = EVAL_FACT_TEMPLATES
        query_templates = EVAL_QUERY_TEMPLATES

    episodes = []

    counter = start_id

    for _ in range(n_episodes):

        # distinct answers within episode
        chosen_answers = rng.sample(
            answers,
            facts_per_episode,
        )

        facts = []

        for position in range(
            facts_per_episode
        ):

            entity = (
                f"person_{counter}"
            )

            counter += 1

            answer = (
                chosen_answers[
                    position
                ]
            )

            fact = rng.choice(
                fact_templates
            ).format(
                entity=entity,
                answer=answer,
            )

            query = rng.choice(
                query_templates
            ).format(
                entity=entity,
            )

            facts.append(
                {
                    "entity": entity,
                    "fact": fact,
                    "query": query,
                    "answer": answer,
                    "target_slot": position,
                }
            )

        episodes.append(facts)

    return episodes


# ============================================================
# FLATTEN FOR PRECOMPUTATION
# ============================================================

def flatten_episodes(
    episodes,
):

    flat = []

    for episode_id, episode in enumerate(
        episodes
    ):

        for fact_position, item in enumerate(
            episode
        ):

            x = dict(item)

            x["episode_id"] = (
                episode_id
            )

            x["fact_position"] = (
                fact_position
            )

            flat.append(x)

    return flat


# ============================================================
# PRECOMPUTE FROZEN GPT-2 REPRESENTATIONS
# ============================================================

@torch.no_grad()
def precompute(
    original,
    tokenizer,
    flat_examples,
    answer_to_class,
    device,
    batch_size,
):

    summaries = []
    query_vectors = []
    labels = []
    target_slots = []
    episode_ids = []
    positions = []

    for start in range(
        0,
        len(flat_examples),
        batch_size,
    ):

        batch = flat_examples[
            start:start + batch_size
        ]

        fact_text = [
            x["fact"]
            for x in batch
        ]

        query_text = [
            x["query"]
            for x in batch
        ]

        # ====================================================
        # FACT SUMMARY
        # ====================================================

        ids, mask = tokenize(
            tokenizer,
            fact_text,
            device,
        )

        output = (
            original
            .backbone
            .transformer(
                input_ids=ids,
                attention_mask=mask,
                return_dict=True,
            )
        )

        hidden = (
            output.last_hidden_state
        )

        weights = (
            mask
            .unsqueeze(-1)
            .to(hidden.dtype)
        )

        summary = (
            (hidden * weights)
            .sum(dim=1)
            /
            weights.sum(dim=1)
            .clamp_min(1.0)
        )

        # ====================================================
        # QUERY VECTOR
        # ====================================================

        q_ids, q_mask = tokenize(
            tokenizer,
            query_text,
            device,
        )

        q_output = (
            original
            .backbone
            .transformer(
                input_ids=q_ids,
                attention_mask=q_mask,
                return_dict=True,
            )
        )

        q_hidden = (
            q_output.last_hidden_state
        )

        last = (
            q_mask.sum(dim=1)
            .long()
            .sub(1)
            .clamp_min(0)
        )

        rows = torch.arange(
            q_hidden.size(0),
            device=device,
        )

        query = q_hidden[
            rows,
            last,
            :
        ]

        summaries.append(
            summary.detach().cpu()
        )

        query_vectors.append(
            query.detach().cpu()
        )

        labels.append(
            torch.tensor(
                [
                    answer_to_class[
                        x["answer"]
                    ]
                    for x in batch
                ],
                dtype=torch.long,
            )
        )

        target_slots.append(
            torch.tensor(
                [
                    x["target_slot"]
                    for x in batch
                ],
                dtype=torch.long,
            )
        )

        episode_ids.append(
            torch.tensor(
                [
                    x["episode_id"]
                    for x in batch
                ],
                dtype=torch.long,
            )
        )

        positions.append(
            torch.tensor(
                [
                    x["fact_position"]
                    for x in batch
                ],
                dtype=torch.long,
            )
        )

    return {
        "summary": torch.cat(
            summaries,
            dim=0,
        ),

        "query": torch.cat(
            query_vectors,
            dim=0,
        ),

        "labels": torch.cat(
            labels,
            dim=0,
        ),

        "target_slot": torch.cat(
            target_slots,
            dim=0,
        ),

        "episode_id": torch.cat(
            episode_ids,
            dim=0,
        ),

        "position": torch.cat(
            positions,
            dim=0,
        ),
    }


# ============================================================
# DATASET
# ============================================================

class FlatDataset(Dataset):

    def __init__(
        self,
        data,
    ):

        self.data = data

    def __len__(self):

        return (
            self.data["labels"]
            .size(0)
        )

    def __getitem__(
        self,
        i,
    ):

        return (
            self.data["summary"][i],
            self.data["query"][i],
            self.data["labels"][i],
            self.data["target_slot"][i],
            self.data["position"][i],
        )


# ============================================================
# VALUE ENCODER
#
# Known-good direct-summary value pathway.
# ============================================================

class ValueEncoder(nn.Module):

    def __init__(
        self,
        d_model,
    ):

        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(
                d_model
            ),
            nn.Linear(
                d_model,
                d_model,
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.net(x)


# ============================================================
# SIMPLE RETRIEVAL READER
#
# Used first so router test is not confounded by
# original MemoryReader.
# ============================================================

class SimpleReader(nn.Module):

    def __init__(
        self,
        d_model,
        num_classes,
    ):

        super().__init__()

        self.q_norm = nn.LayerNorm(
            d_model
        )

        self.net = nn.Sequential(
            nn.Linear(
                3 * d_model,
                d_model,
            ),
            nn.GELU(),
            nn.Linear(
                d_model,
                num_classes,
            ),
        )

    def forward(
        self,
        query,
        value,
    ):

        q = self.q_norm(
            query
        )

        x = torch.cat(
            [
                q,
                value,
                q * value,
            ],
            dim=-1,
        )

        return self.net(x)


# ============================================================
# WRITE POSITION CONDITIONER
#
# Critical methodological point:
#
# Original router only gets semantic summary.
# To perform deterministic "next free slot" allocation,
# the router needs some occupancy / position signal.
#
# We therefore test TWO things:
#
#   raw original router(summary)
#
# and
#
#   position-conditioned router(
#       summary + position_embedding
#   )
#
# We are NOT changing models/slot_router.py.
# ============================================================

class PositionConditioner(nn.Module):

    def __init__(
        self,
        num_slots,
        d_model,
    ):

        super().__init__()

        self.embedding = (
            nn.Embedding(
                num_slots,
                d_model,
            )
        )

        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02,
        )

    def forward(
        self,
        summary,
        position,
    ):

        return (
            summary
            +
            self.embedding(
                position
            )
        )


# ============================================================
# ROUTER CLASSIFICATION TEST
# ============================================================

class RouterClassifier(
    nn.Module
):

    def __init__(
        self,
        original,
        train_router,
        use_position,
    ):

        super().__init__()

        self.router = copy.deepcopy(
            original.router
        )

        if self.router is None:

            raise RuntimeError(
                "Original model has no router."
            )

        for p in (
            self.router
            .parameters()
        ):

            p.requires_grad = (
                train_router
            )

        self.use_position = (
            use_position
        )

        self.position_conditioner = (
            PositionConditioner(
                original.num_slots,
                original.d_model,
            )
        )

        for p in (
            self.position_conditioner
            .parameters()
        ):

            p.requires_grad = (
                train_router
                and
                use_position
            )

    def forward(
        self,
        summary,
        position,
    ):

        if self.use_position:

            query = (
                self.position_conditioner(
                    summary,
                    position,
                )
            )

        else:

            query = summary

        output = self.router(
            query=query,
            memory_slots=None,
            slot_mask=None,
        )

        return output


# ============================================================
# ROUTER EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_router(
    model,
    data,
    device,
    batch_size,
):

    model.eval()

    loader = DataLoader(
        FlatDataset(
            data
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    total = 0
    top1_correct = 0
    topk_correct = 0

    selections = []

    for (
        summary,
        query,
        labels,
        target,
        position,
    ) in loader:

        del query
        del labels

        summary = summary.to(
            device
        )

        target = target.to(
            device
        )

        position = position.to(
            device
        )

        output = model(
            summary,
            position,
        )

        pred = (
            output.weights
            .argmax(dim=-1)
        )

        top1_correct += (
            pred.eq(target)
            .sum()
            .item()
        )

        if (
            output.selected_indices
            is not None
        ):

            hit = (
                output
                .selected_indices
                .eq(
                    target
                    .unsqueeze(-1)
                )
                .any(dim=-1)
            )

        else:

            hit = (
                pred.eq(target)
            )

        topk_correct += (
            hit.sum().item()
        )

        total += (
            target.size(0)
        )

        selections.append(
            pred.detach().cpu()
        )

    selections = torch.cat(
        selections,
        dim=0,
    )

    counts = torch.bincount(
        selections,
        minlength=(
            model.router.num_slots
        ),
    ).float()

    distribution = (
        counts
        /
        counts.sum()
        .clamp_min(1)
    )

    unused_fraction = (
        (counts == 0)
        .float()
        .mean()
        .item()
    )

    maximum_share = (
        distribution.max()
        .item()
    )

    return {
        "top1_accuracy": (
            100
            * top1_correct
            / total
        ),

        "topk_accuracy": (
            100
            * topk_correct
            / total
        ),

        "unused_slot_fraction": (
            unused_fraction
        ),

        "maximum_slot_share": (
            maximum_share
        ),

        "slot_distribution": (
            distribution.tolist()
        ),
    }


# ============================================================
# TRAIN ROUTER
#
# We use raw logits for supervised slot allocation.
# ============================================================

def train_router(
    model,
    train_data,
    validation_data,
    device,
    args,
):

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    if not trainable:

        return None

    optimizer = (
        torch.optim.AdamW(
            trainable,
            lr=(
                args.router_learning_rate
            ),
            weight_decay=1e-4,
        )
    )

    loader = DataLoader(
        FlatDataset(
            train_data
        ),
        batch_size=(
            args.batch_size
        ),
        shuffle=True,
    )

    best_acc = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(
        1,
        args.router_epochs + 1,
    ):

        model.train()

        total_loss = 0.0
        total = 0

        for (
            summary,
            query,
            labels,
            target,
            position,
        ) in loader:

            del query
            del labels

            summary = summary.to(
                device
            )

            target = target.to(
                device
            )

            position = position.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                summary,
                position,
            )

            loss = (
                F.cross_entropy(
                    output.logits,
                    target,
                )
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable,
                5.0,
            )

            optimizer.step()

            n = target.size(0)

            total += n

            total_loss += (
                float(
                    loss.item()
                )
                * n
            )

        val = evaluate_router(
            model,
            validation_data,
            device,
            args.batch_size,
        )

        print(
            f"EPOCH {epoch:02d} | "
            f"loss="
            f"{total_loss / total:.4f} | "
            f"top1="
            f"{val['top1_accuracy']:.2f}% | "
            f"topk="
            f"{val['topk_accuracy']:.2f}% | "
            f"unused="
            f"{val['unused_slot_fraction']:.3f} | "
            f"max_share="
            f"{val['maximum_slot_share']:.3f}"
        )

        if (
            val["top1_accuracy"]
            > best_acc
        ):

            best_acc = (
                val[
                    "top1_accuracy"
                ]
            )

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            print(
                "Saved best router."
            )

    model.load_state_dict(
        best_state
    )

    return {
        "best_epoch": best_epoch,
        "best_validation": (
            best_acc
        ),
    }


# ============================================================
# TRAIN VALUE ENCODER + SIMPLE READER
#
# Oracle single-value control first.
# ============================================================

class ValueControl(nn.Module):

    def __init__(
        self,
        d_model,
        num_classes,
    ):

        super().__init__()

        self.value_encoder = (
            ValueEncoder(
                d_model
            )
        )

        self.reader = (
            SimpleReader(
                d_model,
                num_classes,
            )
        )

    def forward(
        self,
        summary,
        query,
    ):

        value = (
            self.value_encoder(
                summary
            )
        )

        logits = (
            self.reader(
                query,
                value,
            )
        )

        return (
            logits,
            value,
        )


def train_value_control(
    model,
    train_data,
    validation_data,
    device,
    args,
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=(
            args.value_learning_rate
        ),
        weight_decay=1e-4,
    )

    loader = DataLoader(
        FlatDataset(
            train_data
        ),
        batch_size=(
            args.batch_size
        ),
        shuffle=True,
    )

    best_acc = -1.0
    best_state = None

    for epoch in range(
        1,
        args.value_epochs + 1,
    ):

        model.train()

        for (
            summary,
            query,
            labels,
            target,
            position,
        ) in loader:

            del target
            del position

            summary = summary.to(
                device
            )

            query = query.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits, value = model(
                summary,
                query,
            )

            # mismatched value
            perm = torch.randperm(
                value.size(0),
                device=device,
            )

            wrong_logits = (
                model.reader(
                    query,
                    value[perm],
                )
            )

            matched_loss = (
                F.cross_entropy(
                    logits,
                    labels,
                    reduction="none",
                )
            )

            wrong_loss = (
                F.cross_entropy(
                    wrong_logits,
                    labels,
                    reduction="none",
                )
            )

            rank = F.relu(
                0.5
                + matched_loss
                - wrong_loss
            ).mean()

            loss = (
                matched_loss.mean()
                +
                0.5 * rank
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

        val = evaluate_value_control(
            model,
            validation_data,
            device,
            args.batch_size,
        )

        print(
            f"EPOCH {epoch:02d} | "
            f"match="
            f"{val['accuracy']:.2f}%"
        )

        if (
            val["accuracy"]
            > best_acc
        ):

            best_acc = (
                val["accuracy"]
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

    model.load_state_dict(
        best_state
    )


@torch.no_grad()
def evaluate_value_control(
    model,
    data,
    device,
    batch_size,
):

    model.eval()

    loader = DataLoader(
        FlatDataset(
            data
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    correct = 0
    total = 0

    for (
        summary,
        query,
        labels,
        target,
        position,
    ) in loader:

        del target
        del position

        summary = summary.to(
            device
        )

        query = query.to(
            device
        )

        labels = labels.to(
            device
        )

        logits, _ = model(
            summary,
            query,
        )

        correct += (
            logits.argmax(dim=-1)
            .eq(labels)
            .sum()
            .item()
        )

        total += (
            labels.size(0)
        )

    return {
        "accuracy": (
            100
            * correct
            / total
        )
    }


# ============================================================
# BUILD EPISODE INDEX
# ============================================================

def episode_index(
    data,
):

    result = {}

    for i in range(
        data["labels"].size(0)
    ):

        ep = int(
            data[
                "episode_id"
            ][i]
        )

        result.setdefault(
            ep,
            []
        ).append(i)

    for ep in result:

        result[ep].sort(
            key=lambda idx:
            int(
                data[
                    "position"
                ][idx]
            )
        )

    return result


# ============================================================
# MULTI-SLOT EVALUATION
#
# allocation:
#   oracle
#   frozen_router
#   trained_router
#
# Each episode:
#   1. create values for all facts
#   2. allocate each fact to one memory slot
#   3. query each fact
#   4. simple reader receives oracle selected stored slot
#
# IMPORTANT:
# This isolates WRITE ALLOCATION.
#
# We are NOT testing MemoryReader addressing here yet.
# ============================================================

@torch.no_grad()
def evaluate_multislot_allocation(
    allocation,
    router_model,
    value_model,
    data,
    original,
    facts_per_episode,
    device,
):

    value_model.eval()

    if router_model is not None:
        router_model.eval()

    by_episode = (
        episode_index(
            data
        )
    )

    answer_correct = 0
    answer_total = 0

    routing_correct = 0
    routing_total = 0

    collision_count = 0
    total_writes = 0

    unique_slots_per_episode = []

    for ep, indices in (
        by_episode.items()
    ):

        summaries = (
            data["summary"][
                indices
            ]
            .to(device)
        )

        queries = (
            data["query"][
                indices
            ]
            .to(device)
        )

        labels = (
            data["labels"][
                indices
            ]
            .to(device)
        )

        targets = (
            data["target_slot"][
                indices
            ]
            .to(device)
        )

        positions = (
            data["position"][
                indices
            ]
            .to(device)
        )

        values = (
            value_model
            .value_encoder(
                summaries
            )
        )

        selected_slots = []

        if allocation == "oracle":

            selected_slots = (
                targets.clone()
            )

        else:

            output = (
                router_model(
                    summaries,
                    positions,
                )
            )

            selected_slots = (
                output.weights
                .argmax(dim=-1)
            )

        routing_correct += (
            selected_slots
            .eq(targets)
            .sum()
            .item()
        )

        routing_total += (
            targets.size(0)
        )

        # ====================================================
        # WRITE TO MEMORY
        # ====================================================

        state = (
            original
            .memory_bank
            .initialize(
                batch_size=1,
                device=device,
                dtype=values.dtype,
            )
        )

        used = set()

        for j in range(
            len(indices)
        ):

            slot = int(
                selected_slots[j]
                .item()
            )

            if slot in used:
                collision_count += 1

            used.add(slot)

            candidate = (
                state.slots.clone()
            )

            candidate[
                0,
                slot,
                :
            ] = values[
                j
            ]

            gate = torch.zeros(
                1,
                original.num_slots,
                1,
                device=device,
                dtype=values.dtype,
            )

            gate[
                0,
                slot,
                0
            ] = 1.0

            mask = (
                gate.clone()
            )

            state = (
                original
                .memory_bank(
                    state=state,
                    candidate=(
                        candidate
                    ),
                    write_gate=(
                        gate
                    ),
                    write_mask=(
                        mask
                    ),
                    confidence=None,
                )
            )

            total_writes += 1

        unique_slots_per_episode.append(
            len(used)
        )

        # ====================================================
        # ORACLE READ OF EXPECTED SLOT
        #
        # This tells us whether write allocation succeeded.
        # ====================================================

        for j in range(
            len(indices)
        ):

            expected_slot = int(
                targets[j]
                .item()
            )

            stored = (
                state.slots[
                    0,
                    expected_slot,
                    :
                ]
                .unsqueeze(0)
            )

            logits = (
                value_model.reader(
                    queries[
                        j
                    ].unsqueeze(0),
                    stored,
                )
            )

            prediction = (
                logits.argmax(
                    dim=-1
                )
            )

            answer_correct += (
                prediction
                .eq(
                    labels[
                        j
                    ].view(1)
                )
                .sum()
                .item()
            )

            answer_total += 1

    return {
        "routing_top1": (
            100
            * routing_correct
            / routing_total
        ),

        "answer_accuracy": (
            100
            * answer_correct
            / answer_total
        ),

        "collision_rate": (
            100
            * collision_count
            / max(
                total_writes,
                1,
            )
        ),

        "mean_unique_slots": (
            sum(
                unique_slots_per_episode
            )
            /
            len(
                unique_slots_per_episode
            )
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/"
            "retrieval_gradient_test/"
            "checkpoint_best.pt"
        ),
    )

    parser.add_argument(
        "--model-name",
        default="gpt2",
    )

    parser.add_argument(
        "--train-episodes",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=125,
    )

    parser.add_argument(
        "--test-episodes",
        type=int,
        default=125,
    )

    parser.add_argument(
        "--facts-per-episode",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--value-epochs",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--router-epochs",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--value-learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--router-learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2090,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/"
            "router_multislot_all_tests"
        ),
    )

    args = parser.parse_args()

    if (
        args.facts_per_episode
        > 8
    ):

        raise ValueError(
            "facts-per-episode must be <= 8."
        )

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 100)
    print(
        "ROUTER / MULTI-SLOT COMPLETE ISOLATION"
    )
    print("=" * 100)

    print(
        "Device:",
        device,
    )

    print(
        "Checkpoint:",
        args.checkpoint,
    )

    print()
    print(
        "This tests WRITE-SIDE routing."
    )

    print(
        "The MemoryReader is NOT being tested "
        "for addressing in this script."
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.model_name
        )
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = (
        "right"
    )

    answers = (
        get_single_token_answers(
            tokenizer
        )
    )

    answer_to_class = {
        answer: i
        for i, answer
        in enumerate(answers)
    }

    print(
        "Classes:",
        len(answers),
    )

    print(
        "Answer chance:",
        f"{100 / len(answers):.2f}%"
    )

    print(
        "Slots:",
        8,
    )

    print(
        "Router random top-1 chance:",
        "12.50%",
    )

    # ========================================================
    # ORIGINAL
    # ========================================================

    original = load_original(
        args.checkpoint,
        args.model_name,
        device,
    )

    # ========================================================
    # BUILD DATA
    # ========================================================

    train_episodes = (
        build_episodes(
            args.train_episodes,
            args.facts_per_episode,
            answers,
            args.seed,
            0,
            "train",
        )
    )

    val_episodes = (
        build_episodes(
            args.validation_episodes,
            args.facts_per_episode,
            answers,
            args.seed + 1000,
            100000,
            "eval",
        )
    )

    test_episodes = (
        build_episodes(
            args.test_episodes,
            args.facts_per_episode,
            answers,
            args.seed + 2000,
            200000,
            "eval",
        )
    )

    train_flat = flatten_episodes(
        train_episodes
    )

    val_flat = flatten_episodes(
        val_episodes
    )

    test_flat = flatten_episodes(
        test_episodes
    )

    print()
    print(
        "Precomputing train..."
    )

    train_data = precompute(
        original,
        tokenizer,
        train_flat,
        answer_to_class,
        device,
        args.batch_size,
    )

    print(
        "Precomputing validation..."
    )

    val_data = precompute(
        original,
        tokenizer,
        val_flat,
        answer_to_class,
        device,
        args.batch_size,
    )

    print(
        "Precomputing test..."
    )

    test_data = precompute(
        original,
        tokenizer,
        test_flat,
        answer_to_class,
        device,
        args.batch_size,
    )

    # ========================================================
    # 1. TRAIN KNOWN-GOOD VALUE PATH
    # ========================================================

    print()
    print("=" * 100)
    print(
        "STAGE 1 — KNOWN-GOOD VALUE PATH"
    )
    print("=" * 100)

    value_model = (
        ValueControl(
            original.d_model,
            len(answers),
        )
        .to(device)
    )

    train_value_control(
        value_model,
        train_data,
        val_data,
        device,
        args,
    )

    value_test = (
        evaluate_value_control(
            value_model,
            test_data,
            device,
            args.batch_size,
        )
    )

    print(
        f"VALUE CONTROL TEST: "
        f"{value_test['accuracy']:.2f}%"
    )

    for p in (
        value_model.parameters()
    ):

        p.requires_grad = False

    # ========================================================
    # 2. FROZEN ORIGINAL ROUTER — RAW
    # ========================================================

    print()
    print("=" * 100)
    print(
        "STAGE 2 — FROZEN ORIGINAL ROUTER"
    )
    print("=" * 100)

    frozen_router = (
        RouterClassifier(
            original=original,
            train_router=False,
            use_position=False,
        )
        .to(device)
    )

    frozen_router_test = (
        evaluate_router(
            frozen_router,
            test_data,
            device,
            args.batch_size,
        )
    )

    print(
        json.dumps(
            frozen_router_test,
            indent=2,
        )
    )

    # ========================================================
    # 3. TRAINABLE ORIGINAL ROUTER — RAW SUMMARY ONLY
    # ========================================================

    print()
    print("=" * 100)
    print(
        "STAGE 3 — TRAINABLE ORIGINAL ROUTER "
        "(SUMMARY ONLY)"
    )
    print("=" * 100)

    raw_trainable_router = (
        RouterClassifier(
            original=original,
            train_router=True,
            use_position=False,
        )
        .to(device)
    )

    train_router(
        raw_trainable_router,
        train_data,
        val_data,
        device,
        args,
    )

    raw_router_test = (
        evaluate_router(
            raw_trainable_router,
            test_data,
            device,
            args.batch_size,
        )
    )

    print(
        json.dumps(
            raw_router_test,
            indent=2,
        )
    )

    # ========================================================
    # 4. TRAINABLE ORIGINAL ROUTER + WRITE POSITION
    # ========================================================

    print()
    print("=" * 100)
    print(
        "STAGE 4 — TRAINABLE ROUTER "
        "+ WRITE-POSITION SIGNAL"
    )
    print("=" * 100)

    conditioned_router = (
        RouterClassifier(
            original=original,
            train_router=True,
            use_position=True,
        )
        .to(device)
    )

    train_router(
        conditioned_router,
        train_data,
        val_data,
        device,
        args,
    )

    conditioned_test = (
        evaluate_router(
            conditioned_router,
            test_data,
            device,
            args.batch_size,
        )
    )

    print(
        json.dumps(
            conditioned_test,
            indent=2,
        )
    )

    # ========================================================
    # 5. MULTI-SLOT ORACLE ALLOCATION
    # ========================================================

    print()
    print("=" * 100)
    print(
        "STAGE 5 — ORACLE MULTI-SLOT ALLOCATION"
    )
    print("=" * 100)

    oracle_multislot = (
        evaluate_multislot_allocation(
            allocation="oracle",
            router_model=None,
            value_model=value_model,
            data=test_data,
            original=original,
            facts_per_episode=(
                args.facts_per_episode
            ),
            device=device,
        )
    )

    print(
        json.dumps(
            oracle_multislot,
            indent=2,
        )
    )

    # ========================================================
    # 6. FROZEN ROUTER MULTI-SLOT
    # ========================================================

    print()
    print("=" * 100)
    print(
        "STAGE 6 — FROZEN ROUTER MULTI-SLOT"
    )
    print("=" * 100)

    frozen_multislot = (
        evaluate_multislot_allocation(
            allocation=(
                "frozen_router"
            ),
            router_model=(
                frozen_router
            ),
            value_model=(
                value_model
            ),
            data=test_data,
            original=original,
            facts_per_episode=(
                args.facts_per_episode
            ),
            device=device,
        )
    )

    print(
        json.dumps(
            frozen_multislot,
            indent=2,
        )
    )

    # ========================================================
    # 7. TRAINED CONDITIONED ROUTER MULTI-SLOT
    # ========================================================

    print()
    print("=" * 100)
    print(
        "STAGE 7 — TRAINED CONDITIONED ROUTER MULTI-SLOT"
    )
    print("=" * 100)

    trained_multislot = (
        evaluate_multislot_allocation(
            allocation=(
                "trained_router"
            ),
            router_model=(
                conditioned_router
            ),
            value_model=(
                value_model
            ),
            data=test_data,
            original=original,
            facts_per_episode=(
                args.facts_per_episode
            ),
            device=device,
        )
    )

    print(
        json.dumps(
            trained_multislot,
            indent=2,
        )
    )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()
    print("=" * 110)
    print(
        "FINAL ROUTER / MULTI-SLOT COMPARISON"
    )
    print("=" * 110)

    print(
        f"{'TEST':<34}"
        f"{'ROUTE TOP1':>14}"
        f"{'ANSWER':>14}"
        f"{'COLLISION':>14}"
        f"{'UNIQUE SLOTS':>16}"
    )

    print("-" * 95)

    def row(
        name,
        x,
    ):

        print(
            f"{name:<34}"
            f"{x['routing_top1']:>13.2f}%"
            f"{x['answer_accuracy']:>13.2f}%"
            f"{x['collision_rate']:>13.2f}%"
            f"{x['mean_unique_slots']:>16.2f}"
        )

    row(
        "oracle allocation",
        oracle_multislot,
    )

    row(
        "frozen original router",
        frozen_multislot,
    )

    row(
        "trained conditioned router",
        trained_multislot,
    )

    print()
    print(
        "RAW ROUTER CLASSIFICATION"
    )

    print(
        f"Frozen original router: "
        f"{frozen_router_test['top1_accuracy']:.2f}%"
    )

    print(
        f"Trainable summary-only router: "
        f"{raw_router_test['top1_accuracy']:.2f}%"
    )

    print(
        f"Trainable + position router: "
        f"{conditioned_test['top1_accuracy']:.2f}%"
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print()
    print("=" * 110)
    print(
        "INTERPRETATION"
    )
    print("=" * 110)

    oracle_acc = (
        oracle_multislot[
            "answer_accuracy"
        ]
    )

    frozen_route = (
        frozen_router_test[
            "top1_accuracy"
        ]
    )

    raw_route = (
        raw_router_test[
            "top1_accuracy"
        ]
    )

    cond_route = (
        conditioned_test[
            "top1_accuracy"
        ]
    )

    if (
        oracle_acc < 70
    ):

        print(
            "ORACLE MULTI-SLOT CONTROL FAILED."
        )

        print(
            "Do not diagnose router yet; "
            "multi-slot storage itself needs inspection."
        )

    else:

        print(
            "Oracle multi-slot storage works."
        )

        if (
            frozen_route < 30
        ):

            print(
                "Original checkpoint router weights "
                "do not implement reliable controlled "
                "slot allocation."
            )

        if (
            raw_route < 50
            and
            cond_route >= 80
        ):

            print(
                "IMPORTANT:"
            )

            print(
                "The router architecture becomes effective "
                "when it is given explicit write-state / "
                "position information."
            )

            print(
                "Summary semantics alone are insufficient "
                "for deterministic collision-free allocation."
            )

        elif (
            raw_route >= 80
        ):

            print(
                "The original SlotRouter architecture can "
                "learn allocation directly from the summary."
            )

        elif (
            cond_route < 60
        ):

            print(
                "Even position-conditioned routing is weak."
            )

            print(
                "Router formulation should be investigated "
                "before final integration."
            )

    # ========================================================
    # SAVE
    # ========================================================

    results = {
        "value_control": (
            value_test
        ),

        "frozen_router": (
            frozen_router_test
        ),

        "trainable_summary_router": (
            raw_router_test
        ),

        "trainable_position_router": (
            conditioned_test
        ),

        "oracle_multislot": (
            oracle_multislot
        ),

        "frozen_multislot": (
            frozen_multislot
        ),

        "trained_multislot": (
            trained_multislot
        ),
    }

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir
        / "router_multislot_results.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    torch.save(
        {
            "conditioned_router": (
                conditioned_router
                .state_dict()
            ),

            "raw_trainable_router": (
                raw_trainable_router
                .state_dict()
            ),

            "value_model": (
                value_model
                .state_dict()
            ),

            "arguments": (
                vars(args)
            ),
        },
        output_dir
        / "router_multislot_checkpoint.pt",
    )

    print()
    print(
        "Saved:",
        output_dir
        / "router_multislot_results.json",
    )


if __name__ == "__main__":
    main()