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
# DATA
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
# VALUE ENCODER
#
# Must match the encoder from read_side_all_tests.py
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
# SIMPLE READER
# ============================================================

class SimpleReader(nn.Module):

    def __init__(
        self,
        d_model,
        num_classes,
    ):

        super().__init__()

        self.query_norm = nn.LayerNorm(
            d_model
        )

        self.net = nn.Sequential(
            nn.Linear(
                d_model * 3,
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

        q = self.query_norm(
            query
        )

        features = torch.cat(
            [
                q,
                value,
                q * value,
            ],
            dim=-1,
        )

        return self.net(
            features
        )


# ============================================================
# TOKENIZER HELPERS
# ============================================================

def get_single_token_answers(
    tokenizer,
):

    usable = []

    for word in ANSWER_POOL:

        ids = tokenizer(
            " " + word,
            add_special_tokens=False,
        )["input_ids"]

        if len(ids) == 1:
            usable.append(word)

    if len(usable) < 4:

        raise RuntimeError(
            "Need at least four usable "
            "single-token answers."
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
# LOAD ORIGINAL MODEL
# ============================================================

def load_original(
    checkpoint_path,
    model_name,
    device,
):

    print(
        "Loading original model..."
    )

    model = (
        MemoryAugmentedGPT2LMHeadModel
        .from_pretrained(
            model_name,
            memory_config=(
                build_memory_config()
            ),
        )
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.to(device)
    model.eval()

    for p in model.parameters():

        p.requires_grad = False

    return model


# ============================================================
# LOAD KNOWN-GOOD VALUE ENCODER
#
# From:
# outputs/read_side_all_tests/
# known_good_value_encoder.pt
# ============================================================

def load_value_encoder(
    checkpoint_path,
    d_model,
    device,
):

    path = Path(
        checkpoint_path
    )

    if not path.exists():

        raise FileNotFoundError(
            "\nKnown-good VALUE encoder not found:\n"
            f"{path}\n\n"
            "Run read_side_all_tests.py first."
        )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    encoder = ValueEncoder(
        d_model
    )

    encoder.load_state_dict(
        checkpoint[
            "value_encoder"
        ],
        strict=True,
    )

    encoder.to(device)
    encoder.eval()

    for p in encoder.parameters():

        p.requires_grad = False

    return encoder


# ============================================================
# BUILD FLAT EXAMPLES
# ============================================================

def build_examples(
    n,
    answers,
    seed,
    start_id,
    split,
):

    rng = random.Random(seed)

    if split == "train":

        fact_templates = (
            TRAIN_FACT_TEMPLATES
        )

        query_templates = (
            TRAIN_QUERY_TEMPLATES
        )

    else:

        fact_templates = (
            EVAL_FACT_TEMPLATES
        )

        query_templates = (
            EVAL_QUERY_TEMPLATES
        )

    examples = []

    for i in range(n):

        entity = (
            f"person_{start_id + i}"
        )

        answer = rng.choice(
            answers
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

        examples.append(
            {
                "fact": fact,
                "query": query,
                "answer": answer,
            }
        )

    return examples


# ============================================================
# BUILD FOUR-FACT EPISODES
# ============================================================

def build_episodes(
    n,
    answers,
    seed,
    start_id,
    facts_per_episode,
):

    rng = random.Random(seed)

    episodes = []

    counter = start_id

    for _ in range(n):

        # Distinct answers inside each episode.
        chosen = rng.sample(
            answers,
            facts_per_episode,
        )

        episode = []

        for j in range(
            facts_per_episode
        ):

            entity = (
                f"person_{counter}"
            )

            counter += 1

            answer = chosen[j]

            fact = rng.choice(
                EVAL_FACT_TEMPLATES
            ).format(
                entity=entity,
                answer=answer,
            )

            query = rng.choice(
                EVAL_QUERY_TEMPLATES
            ).format(
                entity=entity,
            )

            episode.append(
                {
                    "fact": fact,
                    "query": query,
                    "answer": answer,
                    "slot": j,
                }
            )

        episodes.append(
            episode
        )

    return episodes


# ============================================================
# PRECOMPUTE FLAT DATA
# ============================================================

@torch.no_grad()
def precompute_flat(
    original,
    tokenizer,
    examples,
    answer_to_class,
    device,
    batch_size,
):

    summaries = []
    queries = []
    labels = []

    for start in range(
        0,
        len(examples),
        batch_size,
    ):

        batch = examples[
            start:start + batch_size
        ]

        facts = [
            x["fact"]
            for x in batch
        ]

        questions = [
            x["query"]
            for x in batch
        ]

        # FACT
        ids, mask = tokenize(
            tokenizer,
            facts,
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

        # QUERY
        q_ids, q_mask = tokenize(
            tokenizer,
            questions,
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

        queries.append(
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

    return {
        "summary": torch.cat(
            summaries,
            dim=0,
        ),

        "query": torch.cat(
            queries,
            dim=0,
        ),

        "labels": torch.cat(
            labels,
            dim=0,
        ),
    }


# ============================================================
# PRECOMPUTE EPISODES
# ============================================================

@torch.no_grad()
def precompute_episodes(
    original,
    tokenizer,
    episodes,
    answer_to_class,
    value_encoder,
    device,
):

    cached = []

    for episode in episodes:

        facts = [
            x["fact"]
            for x in episode
        ]

        queries_text = [
            x["query"]
            for x in episode
        ]

        # FACT
        ids, mask = tokenize(
            tokenizer,
            facts,
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

        values = value_encoder(
            summary
        )

        # QUERY
        q_ids, q_mask = tokenize(
            tokenizer,
            queries_text,
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

        labels = torch.tensor(
            [
                answer_to_class[
                    x["answer"]
                ]
                for x in episode
            ],
            dtype=torch.long,
            device=device,
        )

        cached.append(
            {
                "values": (
                    values.detach().cpu()
                ),

                "queries": (
                    query.detach().cpu()
                ),

                "labels": (
                    labels.detach().cpu()
                ),
            }
        )

    return cached


# ============================================================
# FLAT DATASET
# ============================================================

class FlatDataset(Dataset):

    def __init__(
        self,
        data,
    ):

        self.data = data

    def __len__(self):

        return (
            self.data[
                "labels"
            ].size(0)
        )

    def __getitem__(
        self,
        i,
    ):

        return (
            self.data[
                "summary"
            ][i],

            self.data[
                "query"
            ][i],

            self.data[
                "labels"
            ][i],
        )


# ============================================================
# WRITE EXACT VALUE TO ONE SLOT
# ============================================================

def write_slot(
    memory_bank,
    state,
    value,
    slot,
):

    candidate = (
        state.slots.clone()
    )

    candidate[
        :,
        slot,
        :
    ] = value

    gate = torch.zeros(
        value.size(0),
        memory_bank.num_slots,
        1,
        dtype=value.dtype,
        device=value.device,
    )

    gate[
        :,
        slot,
        0
    ] = 1.0

    mask = gate.clone()

    new_state = memory_bank(
        state=state,
        candidate=candidate,
        write_gate=gate,
        write_mask=mask,
        confidence=None,
    )

    return new_state


# ============================================================
# ONE-SHOT MULTI-SLOT WRITE
# ============================================================

def write_all_slots_once(
    memory_bank,
    values,
):

    # values: [K,D]

    k = values.size(0)

    state = memory_bank.initialize(
        batch_size=1,
        device=values.device,
        dtype=values.dtype,
    )

    candidate = (
        state.slots.clone()
    )

    candidate[
        0,
        :k,
        :
    ] = values

    gate = torch.zeros(
        1,
        memory_bank.num_slots,
        1,
        device=values.device,
        dtype=values.dtype,
    )

    gate[
        0,
        :k,
        0
    ] = 1.0

    mask = gate.clone()

    state = memory_bank(
        state=state,
        candidate=candidate,
        write_gate=gate,
        write_mask=mask,
        confidence=None,
    )

    return state


# ============================================================
# MISMATCH
# ============================================================

def random_mismatch(
    labels,
):

    result = []

    for i in range(
        labels.size(0)
    ):

        valid = torch.nonzero(
            labels != labels[i],
            as_tuple=False,
        ).flatten()

        if valid.numel() == 0:

            result.append(
                (i + 1)
                % labels.size(0)
            )

        else:

            choice = valid[
                torch.randint(
                    0,
                    valid.numel(),
                    (1,),
                    device=labels.device,
                )
            ]

            result.append(
                int(
                    choice.item()
                )
            )

    return torch.tensor(
        result,
        dtype=torch.long,
        device=labels.device,
    )


# ============================================================
# TRAIN SIMPLE READER
#
# Frozen known-good ValueEncoder.
# Single-slot MemoryBank storage.
# ============================================================

def train_reader(
    reader,
    value_encoder,
    memory_bank,
    train_data,
    validation_data,
    device,
    args,
):

    optimizer = torch.optim.AdamW(
        reader.parameters(),
        lr=(
            args.reader_learning_rate
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
    best_epoch = -1

    for epoch in range(
        1,
        args.reader_epochs + 1,
    ):

        reader.train()

        for (
            summary,
            query,
            labels,
        ) in loader:

            summary = summary.to(
                device
            )

            query = query.to(
                device
            )

            labels = labels.to(
                device
            )

            with torch.no_grad():

                value = value_encoder(
                    summary
                )

                state = (
                    memory_bank.initialize(
                        batch_size=(
                            value.size(0)
                        ),
                        device=device,
                        dtype=value.dtype,
                    )
                )

                state = write_slot(
                    memory_bank,
                    state,
                    value,
                    0,
                )

                stored = (
                    state.slots[
                        :,
                        0,
                        :
                    ]
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = reader(
                query,
                stored,
            )

            matched_loss = (
                F.cross_entropy(
                    logits,
                    labels,
                    reduction="none",
                )
            )

            wrong_idx = (
                random_mismatch(
                    labels
                )
            )

            wrong_logits = reader(
                query,
                stored[
                    wrong_idx
                ],
            )

            wrong_loss = (
                F.cross_entropy(
                    wrong_logits,
                    labels,
                    reduction="none",
                )
            )

            rank_loss = F.relu(
                0.5
                + matched_loss
                - wrong_loss
            ).mean()

            loss = (
                matched_loss.mean()
                +
                0.5
                * rank_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                reader.parameters(),
                5.0,
            )

            optimizer.step()

        val = evaluate_single_slot(
            reader,
            value_encoder,
            memory_bank,
            validation_data,
            device,
            args.batch_size,
        )

        print(
            f"EPOCH {epoch:02d} | "
            f"match="
            f"{val['matched']:.2f}% | "
            f"mismatch="
            f"{val['mismatched']:.2f}% | "
            f"gap="
            f"{val['nll_gap']:+.4f}"
        )

        if (
            val["matched"]
            > best_acc
        ):

            best_acc = (
                val["matched"]
            )

            best_epoch = epoch

            best_state = copy.deepcopy(
                reader.state_dict()
            )

            print(
                "Saved best reader."
            )

    reader.load_state_dict(
        best_state
    )

    return {
        "best_epoch": (
            best_epoch
        ),

        "best_validation": (
            best_acc
        ),
    }


# ============================================================
# SINGLE SLOT EVAL
# ============================================================

@torch.no_grad()
def evaluate_single_slot(
    reader,
    value_encoder,
    memory_bank,
    data,
    device,
    batch_size,
):

    reader.eval()

    loader = DataLoader(
        FlatDataset(data),
        batch_size=batch_size,
        shuffle=False,
    )

    all_logits = []
    all_wrong = []
    all_labels = []

    for (
        summary,
        query,
        labels,
    ) in loader:

        summary = summary.to(
            device
        )

        query = query.to(
            device
        )

        labels = labels.to(
            device
        )

        value = value_encoder(
            summary
        )

        state = (
            memory_bank.initialize(
                batch_size=(
                    value.size(0)
                ),
                device=device,
                dtype=value.dtype,
            )
        )

        state = write_slot(
            memory_bank,
            state,
            value,
            0,
        )

        stored = (
            state.slots[
                :,
                0,
                :
            ]
        )

        logits = reader(
            query,
            stored,
        )

        wrong_idx = (
            random_mismatch(
                labels
            )
        )

        wrong_logits = reader(
            query,
            stored[
                wrong_idx
            ],
        )

        all_logits.append(
            logits
        )

        all_wrong.append(
            wrong_logits
        )

        all_labels.append(
            labels
        )

    logits = torch.cat(
        all_logits,
        dim=0,
    )

    wrong_logits = torch.cat(
        all_wrong,
        dim=0,
    )

    labels = torch.cat(
        all_labels,
        dim=0,
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

    return {
        "matched": float(
            logits.argmax(dim=-1)
            .eq(labels)
            .float()
            .mean()
            .item()
            * 100
        ),

        "mismatched": float(
            wrong_logits.argmax(dim=-1)
            .eq(labels)
            .float()
            .mean()
            .item()
            * 100
        ),

        "nll_gap": float(
            (
                wrong_loss
                - matched_loss
            )
            .mean()
            .item()
        ),
    }


# ============================================================
# REPRESENTATION DRIFT
# ============================================================

def representation_difference(
    a,
    b,
):

    a = a.float()
    b = b.float()

    l2 = (
        (a - b)
        .norm(dim=-1)
    )

    relative = (
        l2
        /
        a.norm(
            dim=-1
        ).clamp_min(
            1e-8
        )
    )

    cosine = (
        F.cosine_similarity(
            a,
            b,
            dim=-1,
        )
    )

    return (
        float(
            l2.mean().item()
        ),
        float(
            relative.mean().item()
        ),
        float(
            cosine.mean().item()
        ),
    )


# ============================================================
# MULTI-SLOT EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_multislot(
    reader,
    memory_bank,
    episodes,
    device,
):

    reader.eval()

    one_shot_correct = 0
    sequential_correct = 0
    total = 0

    # Compare final slot representation
    # between:
    #
    # one-shot storage
    # sequential storage
    #
    # Also track drift of an early slot
    # after later writes.

    one_vs_seq_l2 = []
    one_vs_seq_rel = []
    one_vs_seq_cos = []

    early_drift_l2 = []
    early_drift_rel = []
    early_drift_cos = []

    per_position_correct = {
        i: 0
        for i in range(
            episodes[0][
                "values"
            ].size(0)
        )
    }

    per_position_total = {
        i: 0
        for i in per_position_correct
    }

    for episode in episodes:

        values = (
            episode[
                "values"
            ].to(device)
        )

        queries = (
            episode[
                "queries"
            ].to(device)
        )

        labels = (
            episode[
                "labels"
            ].to(device)
        )

        k = values.size(0)

        # ====================================================
        # ONE-SHOT WRITE
        # ====================================================

        one_shot_state = (
            write_all_slots_once(
                memory_bank,
                values,
            )
        )

        # ====================================================
        # SEQUENTIAL WRITE
        # ====================================================

        sequential_state = (
            memory_bank.initialize(
                batch_size=1,
                device=device,
                dtype=values.dtype,
            )
        )

        slot0_immediate = None

        for j in range(k):

            sequential_state = (
                write_slot(
                    memory_bank,
                    sequential_state,
                    values[
                        j
                    ].unsqueeze(0),
                    j,
                )
            )

            if j == 0:

                slot0_immediate = (
                    sequential_state
                    .slots[
                        0,
                        0,
                        :
                    ]
                    .clone()
                )

        # ====================================================
        # SLOT DRIFT
        # ====================================================

        l2, rel, cos = (
            representation_difference(
                one_shot_state
                .slots[
                    0,
                    :k,
                    :
                ],
                sequential_state
                .slots[
                    0,
                    :k,
                    :
                ],
            )
        )

        one_vs_seq_l2.append(
            l2
        )

        one_vs_seq_rel.append(
            rel
        )

        one_vs_seq_cos.append(
            cos
        )

        l2, rel, cos = (
            representation_difference(
                slot0_immediate
                .unsqueeze(0),
                sequential_state
                .slots[
                    0,
                    0,
                    :
                ]
                .unsqueeze(0),
            )
        )

        early_drift_l2.append(
            l2
        )

        early_drift_rel.append(
            rel
        )

        early_drift_cos.append(
            cos
        )

        # ====================================================
        # EXACT SLOT READ
        # ====================================================

        for j in range(k):

            q = (
                queries[
                    j
                ].unsqueeze(0)
            )

            target = (
                labels[
                    j
                ].view(1)
            )

            one_value = (
                one_shot_state
                .slots[
                    0,
                    j,
                    :
                ]
                .unsqueeze(0)
            )

            seq_value = (
                sequential_state
                .slots[
                    0,
                    j,
                    :
                ]
                .unsqueeze(0)
            )

            one_logits = reader(
                q,
                one_value,
            )

            seq_logits = reader(
                q,
                seq_value,
            )

            one_prediction = (
                one_logits.argmax(
                    dim=-1
                )
            )

            seq_prediction = (
                seq_logits.argmax(
                    dim=-1
                )
            )

            one_shot_correct += (
                one_prediction
                .eq(target)
                .sum()
                .item()
            )

            sequential_correct += (
                seq_prediction
                .eq(target)
                .sum()
                .item()
            )

            per_position_correct[
                j
            ] += (
                seq_prediction
                .eq(target)
                .sum()
                .item()
            )

            per_position_total[
                j
            ] += 1

            total += 1

    return {
        "one_shot_accuracy": float(
            100
            * one_shot_correct
            / total
        ),

        "sequential_accuracy": float(
            100
            * sequential_correct
            / total
        ),

        "one_shot_vs_sequential": {
            "mean_l2": float(
                sum(
                    one_vs_seq_l2
                )
                /
                len(
                    one_vs_seq_l2
                )
            ),

            "relative_l2": float(
                sum(
                    one_vs_seq_rel
                )
                /
                len(
                    one_vs_seq_rel
                )
            ),

            "cosine": float(
                sum(
                    one_vs_seq_cos
                )
                /
                len(
                    one_vs_seq_cos
                )
            ),
        },

        "slot0_after_later_writes": {
            "mean_l2": float(
                sum(
                    early_drift_l2
                )
                /
                len(
                    early_drift_l2
                )
            ),

            "relative_l2": float(
                sum(
                    early_drift_rel
                )
                /
                len(
                    early_drift_rel
                )
            ),

            "cosine": float(
                sum(
                    early_drift_cos
                )
                /
                len(
                    early_drift_cos
                )
            ),
        },

        "sequential_position_accuracy": {
            str(position): float(
                100
                * per_position_correct[
                    position
                ]
                /
                per_position_total[
                    position
                ]
            )
            for position
            in per_position_correct
        },
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
        "--value-checkpoint",
        default=(
            "outputs/"
            "read_side_all_tests/"
            "known_good_value_encoder.pt"
        ),
    )

    parser.add_argument(
        "--model-name",
        default="gpt2",
    )

    parser.add_argument(
        "--train-examples",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--validation-examples",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--test-examples",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--test-episodes",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--facts-per-episode",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--reader-epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--reader-learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
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
            "oracle_multislot_storage_test"
        ),
    )

    args = parser.parse_args()

    if (
        args.facts_per_episode
        > 8
    ):

        raise ValueError(
            "facts-per-episode "
            "cannot exceed 8."
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
        "ORACLE MULTI-SLOT STORAGE TEST"
    )

    print("=" * 100)

    print(
        "Device:",
        device,
    )

    print()
    print(
        "Disabled:"
    )

    print(
        "  Router"
    )

    print(
        "  CandidateWriter"
    )

    print(
        "  VectorGate"
    )

    print(
        "  OrthogonalUpdate"
    )

    print(
        "  Original MemoryReader"
    )

    print()
    print(
        "Testing:"
    )

    print(
        "  Single-slot storage"
    )

    print(
        "  One-shot 4-slot storage"
    )

    print(
        "  Sequential 4-slot storage"
    )

    print(
        "  Drift of earlier slots"
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
        answer: index
        for index, answer
        in enumerate(
            answers
        )
    }

    print()
    print(
        "Classes:",
        len(answers),
    )

    print(
        "Chance:",
        f"{100 / len(answers):.2f}%"
    )

    # ========================================================
    # ORIGINAL MODEL
    # ========================================================

    original = load_original(
        args.checkpoint,
        args.model_name,
        device,
    )

    memory_bank = copy.deepcopy(
        original.memory_bank
    ).to(device)

    memory_bank.eval()

    for p in memory_bank.parameters():

        p.requires_grad = False

    # ========================================================
    # KNOWN GOOD VALUE ENCODER
    # ========================================================

    value_encoder = (
        load_value_encoder(
            args.value_checkpoint,
            original.d_model,
            device,
        )
    )

    print(
        "Loaded known-good VALUE encoder:"
    )

    print(
        args.value_checkpoint
    )

    # ========================================================
    # FLAT DATA
    # ========================================================

    train_examples = (
        build_examples(
            args.train_examples,
            answers,
            args.seed,
            0,
            "train",
        )
    )

    val_examples = (
        build_examples(
            args.validation_examples,
            answers,
            args.seed + 1000,
            100000,
            "eval",
        )
    )

    test_examples = (
        build_examples(
            args.test_examples,
            answers,
            args.seed + 2000,
            200000,
            "eval",
        )
    )

    print()
    print(
        "Precomputing flat train..."
    )

    train_data = precompute_flat(
        original,
        tokenizer,
        train_examples,
        answer_to_class,
        device,
        args.batch_size,
    )

    print(
        "Precomputing flat validation..."
    )

    val_data = precompute_flat(
        original,
        tokenizer,
        val_examples,
        answer_to_class,
        device,
        args.batch_size,
    )

    print(
        "Precomputing flat test..."
    )

    test_data = precompute_flat(
        original,
        tokenizer,
        test_examples,
        answer_to_class,
        device,
        args.batch_size,
    )

    # ========================================================
    # TRAIN READER ON SINGLE SLOT
    # ========================================================

    print()
    print("=" * 100)
    print(
        "STAGE 1 — SINGLE-SLOT CONTROL"
    )
    print("=" * 100)

    reader = SimpleReader(
        original.d_model,
        len(answers),
    ).to(device)

    reader_info = train_reader(
        reader,
        value_encoder,
        memory_bank,
        train_data,
        val_data,
        device,
        args,
    )

    single_test = (
        evaluate_single_slot(
            reader,
            value_encoder,
            memory_bank,
            test_data,
            device,
            args.batch_size,
        )
    )

    print()
    print(
        "SINGLE SLOT TEST"
    )

    print(
        f"MATCHED:    "
        f"{single_test['matched']:.2f}%"
    )

    print(
        f"MISMATCHED: "
        f"{single_test['mismatched']:.2f}%"
    )

    print(
        f"NLL GAP:    "
        f"{single_test['nll_gap']:+.6f}"
    )

    # ========================================================
    # MULTI-SLOT EPISODES
    # ========================================================

    episodes = build_episodes(
        args.test_episodes,
        answers,
        args.seed + 3000,
        300000,
        args.facts_per_episode,
    )

    print()
    print(
        "Precomputing multi-slot episodes..."
    )

    cached_episodes = (
        precompute_episodes(
            original,
            tokenizer,
            episodes,
            answer_to_class,
            value_encoder,
            device,
        )
    )

    # ========================================================
    # MULTI-SLOT EVAL
    # ========================================================

    print()
    print("=" * 100)
    print(
        "STAGE 2 — ORACLE MULTI-SLOT STORAGE"
    )
    print("=" * 100)

    multi = evaluate_multislot(
        reader,
        memory_bank,
        cached_episodes,
        device,
    )

    print()
    print(
        "ONE-SHOT 4-SLOT ACCURACY:"
    )

    print(
        f"{multi['one_shot_accuracy']:.2f}%"
    )

    print()

    print(
        "SEQUENTIAL 4-SLOT ACCURACY:"
    )

    print(
        f"{multi['sequential_accuracy']:.2f}%"
    )

    print()

    print(
        "ONE-SHOT vs SEQUENTIAL:"
    )

    print(
        json.dumps(
            multi[
                "one_shot_vs_sequential"
            ],
            indent=2,
        )
    )

    print()

    print(
        "SLOT 0 DRIFT AFTER "
        "WRITING SLOTS 1,2,3:"
    )

    print(
        json.dumps(
            multi[
                "slot0_after_later_writes"
            ],
            indent=2,
        )
    )

    print()

    print(
        "SEQUENTIAL ACCURACY "
        "BY SLOT POSITION:"
    )

    print(
        json.dumps(
            multi[
                "sequential_position_accuracy"
            ],
            indent=2,
        )
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 100)
    print(
        "FINAL COMPARISON"
    )
    print("=" * 100)

    print(
        f"{'TEST':<32}"
        f"{'ACCURACY':>14}"
    )

    print("-" * 48)

    print(
        f"{'single_slot':<32}"
        f"{single_test['matched']:>13.2f}%"
    )

    print(
        f"{'one_shot_4_slot':<32}"
        f"{multi['one_shot_accuracy']:>13.2f}%"
    )

    print(
        f"{'sequential_4_slot':<32}"
        f"{multi['sequential_accuracy']:>13.2f}%"
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    single = (
        single_test[
            "matched"
        ]
    )

    one = (
        multi[
            "one_shot_accuracy"
        ]
    )

    seq = (
        multi[
            "sequential_accuracy"
        ]
    )

    drift = (
        multi[
            "slot0_after_later_writes"
        ][
            "relative_l2"
        ]
    )

    print()
    print("=" * 100)
    print(
        "INTERPRETATION"
    )
    print("=" * 100)

    if single < 75:

        print(
            "STOP:"
        )

        print(
            "Single-slot known-good baseline "
            "is still too weak."
        )

        print(
            "Do not diagnose multi-slot "
            "storage from this run."
        )

    elif (
        one >= single - 5
        and
        seq >= single - 5
    ):

        print(
            "PASS:"
        )

        print(
            "MemoryBank preserves multiple "
            "independent VALUEs."
        )

        print(
            "Multi-slot storage is not "
            "the cause of the previous "
            "~65% ceiling."
        )

        if drift < 0.01:

            print(
                "Earlier slots show "
                "negligible sequential-write drift."
            )

    elif (
        one >= single - 5
        and
        seq < one - 10
    ):

        print(
            "SEQUENTIAL WRITE PROBLEM:"
        )

        print(
            "Writing four values simultaneously works, "
            "but later MemoryBank updates damage "
            "earlier slots."
        )

        print(
            "Inspect normalization / repeated "
            "MemoryBank updates."
        )

    elif (
        one < single - 10
    ):

        print(
            "MULTI-SLOT STORAGE PROBLEM:"
        )

        print(
            "Even oracle one-shot storage "
            "degrades strongly compared "
            "with the single-slot control."
        )

    else:

        print(
            "PARTIAL:"
        )

        print(
            "There is some multi-slot degradation."
        )

        print(
            "Use the one-shot/sequential drift "
            "statistics to locate it."
        )

    # ========================================================
    # SAVE
    # ========================================================

    results = {
        "reader_training": (
            reader_info
        ),

        "single_slot": (
            single_test
        ),

        "multi_slot": (
            multi
        ),

        "arguments": (
            vars(args)
        ),
    }

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "oracle_multislot_storage_results.json"
    )

    with open(
        output_path,
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
            "reader": (
                reader.state_dict()
            ),

            "arguments": (
                vars(args)
            ),
        },
        output_dir
        / "oracle_multislot_storage_checkpoint.pt",
    )

    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":

    main()