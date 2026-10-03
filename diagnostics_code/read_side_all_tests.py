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
# ORIGINAL MODEL CONFIG
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


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# ANSWERS
# ============================================================

def get_single_token_answers(tokenizer):

    usable = {}

    for word in ANSWER_POOL:

        ids = tokenizer(
            " " + word,
            add_special_tokens=False,
        )["input_ids"]

        if len(ids) == 1:
            usable[word] = ids[0]

    if len(usable) < 4:
        raise RuntimeError(
            "Not enough single-token answers."
        )

    return usable


# ============================================================
# EXAMPLES
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
# TOKENIZE
# ============================================================

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

    print(
        "Loading original checkpoint..."
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
# PRECOMPUTE FROZEN GPT-2
#
# We need:
#   fact masked mean
#   complete query hidden sequence
#   query attention mask
#   last query hidden
# ============================================================

@torch.no_grad()
def precompute(
    original,
    tokenizer,
    examples,
    answer_to_class,
    device,
    batch_size,
):

    summaries = []
    labels = []

    query_states = []
    query_masks = []

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

        queries = [
            x["query"]
            for x in batch
        ]

        # ====================================================
        # FACT
        # ====================================================

        fact_ids, fact_mask = tokenize(
            tokenizer,
            facts,
            device,
        )

        fact_out = (
            original
            .backbone
            .transformer(
                input_ids=fact_ids,
                attention_mask=fact_mask,
                return_dict=True,
            )
        )

        fact_hidden = (
            fact_out.last_hidden_state
        )

        weights = (
            fact_mask
            .unsqueeze(-1)
            .to(fact_hidden.dtype)
        )

        summary = (
            (fact_hidden * weights)
            .sum(dim=1)
            /
            weights.sum(dim=1)
            .clamp_min(1.0)
        )

        summaries.append(
            summary.detach().cpu()
        )

        # ====================================================
        # QUERY
        # ====================================================

        q_ids, q_mask = tokenize(
            tokenizer,
            queries,
            device,
        )

        q_out = (
            original
            .backbone
            .transformer(
                input_ids=q_ids,
                attention_mask=q_mask,
                return_dict=True,
            )
        )

        q_hidden = (
            q_out.last_hidden_state
        )

        for j in range(
            q_hidden.size(0)
        ):

            length = int(
                q_mask[j]
                .sum()
                .item()
            )

            query_states.append(
                q_hidden[
                    j,
                    :length,
                    :
                ]
                .detach()
                .cpu()
            )

            query_masks.append(
                q_mask[
                    j,
                    :length
                ]
                .detach()
                .cpu()
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

        done = min(
            start + batch_size,
            len(examples),
        )

        if (
            done % 500 == 0
            or done == len(examples)
        ):

            print(
                f"  {done}/{len(examples)}"
            )

    return {
        "summary": torch.cat(
            summaries,
            dim=0,
        ),

        "labels": torch.cat(
            labels,
            dim=0,
        ),

        "query_states": (
            query_states
        ),

        "query_masks": (
            query_masks
        ),
    }


# ============================================================
# DATASET
# ============================================================

class RepDataset(Dataset):

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
            i,
            self.data[
                "summary"
            ][i],
            self.data[
                "labels"
            ][i],
        )


# ============================================================
# QUERY BATCH
# ============================================================

def build_query_batch(
    data,
    indices,
    device,
):

    states = [
        data[
            "query_states"
        ][int(i)]
        for i in indices
    ]

    masks = [
        data[
            "query_masks"
        ][int(i)]
        for i in indices
    ]

    max_len = max(
        x.size(0)
        for x in states
    )

    d_model = (
        states[0]
        .size(-1)
    )

    batch_states = torch.zeros(
        len(states),
        max_len,
        d_model,
        dtype=states[0].dtype,
        device=device,
    )

    batch_mask = torch.zeros(
        len(states),
        max_len,
        dtype=torch.long,
        device=device,
    )

    for j, state in enumerate(
        states
    ):

        length = (
            state.size(0)
        )

        batch_states[
            j,
            :length,
            :
        ] = state.to(device)

        batch_mask[
            j,
            :length
        ] = masks[j].to(device)

    return (
        batch_states,
        batch_mask,
    )


# ============================================================
# LAST VALID TOKEN
# ============================================================

def last_valid(
    hidden,
    mask,
):

    lengths = (
        mask.sum(dim=1)
        .long()
        .sub(1)
        .clamp_min(0)
    )

    rows = torch.arange(
        hidden.size(0),
        device=hidden.device,
    )

    return hidden[
        rows,
        lengths,
        :
    ]


# ============================================================
# VALUE ENCODER
#
# This is our known-good write-side control.
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
        summary,
    ):

        return self.net(
            summary
        )


# ============================================================
# SIMPLE CONTROL READER
# ============================================================

class SimpleControlReader(nn.Module):

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
        query_last,
        stored_value,
    ):

        q = self.query_norm(
            query_last
        )

        features = torch.cat(
            [
                q,
                stored_value,
                q * stored_value,
            ],
            dim=-1,
        )

        return self.net(
            features
        )


# ============================================================
# CLASSIFIER FOR ORIGINAL READER OUTPUT
# ============================================================

class ReadClassifier(nn.Module):

    def __init__(
        self,
        d_model,
        num_classes,
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
            nn.GELU(),
            nn.Linear(
                d_model,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.net(x)


# ============================================================
# KNOWN-GOOD MEMORY BUILDER
#
# Only slot 0 contains VALUE.
# Gate = 1.
# Other slots remain initial memory.
#
# Reader memory_mask exposes ONLY slot 0.
# Therefore this test does NOT test addressing.
# ============================================================

def build_memory_slots(
    memory_bank,
    value,
    forced_slot,
):

    batch_size = (
        value.size(0)
    )

    state = (
        memory_bank.initialize(
            batch_size=(
                batch_size
            ),
            device=value.device,
            dtype=value.dtype,
        )
    )

    candidate = (
        state.slots.clone()
    )

    candidate[
        :,
        forced_slot,
        :
    ] = value

    write_gate = torch.zeros(
        batch_size,
        memory_bank.num_slots,
        1,
        device=value.device,
        dtype=value.dtype,
    )

    write_gate[
        :,
        forced_slot,
        0
    ] = 1.0

    write_mask = torch.zeros(
        batch_size,
        memory_bank.num_slots,
        1,
        device=value.device,
        dtype=value.dtype,
    )

    write_mask[
        :,
        forced_slot,
        0
    ] = 1.0

    new_state = memory_bank(
        state=state,
        candidate=candidate,
        write_gate=write_gate,
        write_mask=write_mask,
        confidence=None,
    )

    reader_mask = torch.zeros(
        batch_size,
        memory_bank.num_slots,
        dtype=torch.bool,
        device=value.device,
    )

    reader_mask[
        :,
        forced_slot
    ] = True

    return (
        new_state.slots,
        reader_mask,
    )


# ============================================================
# MISMATCH
# ============================================================

def random_mismatch(
    labels,
):

    result = torch.empty(
        labels.size(0),
        dtype=torch.long,
        device=labels.device,
    )

    for i in range(
        labels.size(0)
    ):

        choices = torch.nonzero(
            labels != labels[i],
            as_tuple=False,
        ).flatten()

        if choices.numel() == 0:

            result[i] = (
                (i + 1)
                % labels.size(0)
            )

        else:

            result[i] = choices[
                torch.randint(
                    0,
                    choices.numel(),
                    (1,),
                    device=labels.device,
                )
            ]

    return result


def deterministic_mismatch(
    labels,
):

    cpu = labels.cpu()

    result = []

    for i in range(
        labels.size(0)
    ):

        chosen = None

        for offset in range(
            1,
            labels.size(0),
        ):

            j = (
                i + offset
            ) % labels.size(0)

            if (
                cpu[j]
                != cpu[i]
            ):

                chosen = j
                break

        if chosen is None:

            raise RuntimeError(
                "Could not build mismatch."
            )

        result.append(
            chosen
        )

    return torch.tensor(
        result,
        dtype=torch.long,
        device=labels.device,
    )


# ============================================================
# STAGE 1:
# TRAIN KNOWN-GOOD VALUE ENCODER
#
# masked mean -> ValueEncoder -> MemoryBank gate1
#                           -> simple control reader
# ============================================================

class ValueControlModel(nn.Module):

    def __init__(
        self,
        original,
        num_classes,
        forced_slot,
    ):

        super().__init__()

        self.d_model = (
            original.d_model
        )

        self.forced_slot = (
            forced_slot
        )

        self.memory_bank = (
            copy.deepcopy(
                original.memory_bank
            )
        )

        for p in (
            self.memory_bank
            .parameters()
        ):

            p.requires_grad = False

        self.value_encoder = (
            ValueEncoder(
                self.d_model
            )
        )

        self.reader = (
            SimpleControlReader(
                self.d_model,
                num_classes,
            )
        )

    def forward(
        self,
        summary,
        query_hidden,
        query_mask,
    ):

        value = (
            self.value_encoder(
                summary
            )
        )

        (
            memory_slots,
            memory_mask,
        ) = build_memory_slots(
            self.memory_bank,
            value,
            self.forced_slot,
        )

        stored = (
            memory_slots[
                :,
                self.forced_slot,
                :
            ]
        )

        query_last = (
            last_valid(
                query_hidden,
                query_mask,
            )
        )

        logits = (
            self.reader(
                query_last,
                stored,
            )
        )

        return {
            "logits": logits,
            "value": value,
            "stored": stored,
            "memory_slots": (
                memory_slots
            ),
            "memory_mask": (
                memory_mask
            ),
            "query_last": (
                query_last
            ),
        }


# ============================================================
# READER TEST MODEL
#
# variant:
#   frozen_context
#   frozen_fused
#   trainable_context
#   trainable_fused
# ============================================================

class OriginalReaderTestModel(
    nn.Module
):

    VALID_VARIANTS = {
        "frozen_context",
        "frozen_fused",
        "trainable_context",
        "trainable_fused",
    }

    def __init__(
        self,
        original,
        value_encoder,
        num_classes,
        forced_slot,
        variant,
    ):

        super().__init__()

        if (
            variant
            not in self.VALID_VARIANTS
        ):

            raise ValueError(
                f"Unknown variant: "
                f"{variant}"
            )

        self.variant = (
            variant
        )

        self.d_model = (
            original.d_model
        )

        self.forced_slot = (
            forced_slot
        )

        # ====================================================
        # KNOWN-GOOD VALUE ENCODER - ALWAYS FROZEN
        # ====================================================

        self.value_encoder = (
            copy.deepcopy(
                value_encoder
            )
        )

        for p in (
            self.value_encoder
            .parameters()
        ):

            p.requires_grad = False

        # ====================================================
        # MEMORY BANK - FROZEN
        # ====================================================

        self.memory_bank = (
            copy.deepcopy(
                original.memory_bank
            )
        )

        for p in (
            self.memory_bank
            .parameters()
        ):

            p.requires_grad = False

        # ====================================================
        # ORIGINAL MEMORY READER
        # ====================================================

        self.reader = (
            copy.deepcopy(
                original.reader
            )
        )

        reader_trainable = (
            variant.startswith(
                "trainable_"
            )
        )

        for p in (
            self.reader
            .parameters()
        ):

            p.requires_grad = (
                reader_trainable
            )

        # ====================================================
        # DIAGNOSTIC CLASSIFIER
        # ====================================================

        self.classifier = (
            ReadClassifier(
                self.d_model,
                num_classes,
            )
        )

    def forward(
        self,
        summary,
        query_hidden,
        query_mask,
        memory_override=None,
    ):

        # ====================================================
        # STORED VALUE
        # ====================================================

        with torch.no_grad():

            value = (
                self.value_encoder(
                    summary
                )
            )

            (
                memory_slots,
                memory_mask,
            ) = build_memory_slots(
                self.memory_bank,
                value,
                self.forced_slot,
            )

        # Allows matched/mismatched/zero memory.
        if memory_override is not None:

            memory_slots = (
                memory_override
            )

        # ====================================================
        # ORIGINAL READER
        # ====================================================

        output = (
            self.reader(
                hidden_states=(
                    query_hidden
                ),
                memory_slots=(
                    memory_slots
                ),
                attention_mask=(
                    query_mask
                ),
                memory_mask=(
                    memory_mask
                ),
                routing_prior=None,
                memory_confidence=None,
                return_attention=True,
            )
        )

        # ====================================================
        # CONTEXT OR FUSED
        # ====================================================

        if (
            self.variant.endswith(
                "_context"
            )
        ):

            representation = (
                last_valid(
                    output.context,
                    query_mask,
                )
            )

        else:

            representation = (
                last_valid(
                    output.fused_hidden,
                    query_mask,
                )
            )

        logits = (
            self.classifier(
                representation
            )
        )

        return {
            "logits": logits,
            "memory_slots": (
                memory_slots
            ),
            "memory_mask": (
                memory_mask
            ),
            "representation": (
                representation
            ),
            "attention_weights": (
                output.attention_weights
            ),
            "slot_usage": (
                output.slot_usage
            ),
            "read_confidence": (
                output.read_confidence
            ),
        }


# ============================================================
# TRAIN VALUE CONTROL
# ============================================================

def train_value_control_epoch(
    model,
    loader,
    data,
    optimizer,
    device,
    mismatch_weight,
    mismatch_margin,
):

    model.train()

    model.memory_bank.eval()

    total_loss = 0.0
    total = 0

    for (
        indices,
        summary,
        labels,
    ) in loader:

        summary = (
            summary.to(device)
        )

        labels = (
            labels.to(device)
        )

        (
            query_hidden,
            query_mask,
        ) = build_query_batch(
            data,
            indices,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            summary,
            query_hidden,
            query_mask,
        )

        logits = (
            output["logits"]
        )

        stored = (
            output["stored"]
        )

        query_last = (
            output["query_last"]
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

        wrong_logits = (
            model.reader(
                query_last,
                stored[
                    wrong_idx
                ],
            )
        )

        wrong_loss = (
            F.cross_entropy(
                wrong_logits,
                labels,
                reduction="none",
            )
        )

        rank_loss = F.relu(
            mismatch_margin
            + matched_loss
            - wrong_loss
        ).mean()

        loss = (
            matched_loss.mean()
            +
            mismatch_weight
            * rank_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [
                p
                for p in model.parameters()
                if p.requires_grad
            ],
            5.0,
        )

        optimizer.step()

        n = (
            labels.size(0)
        )

        total += n

        total_loss += (
            float(
                loss.item()
            )
            * n
        )

    return (
        total_loss
        / total
    )


# ============================================================
# EVALUATE VALUE CONTROL
# ============================================================

@torch.no_grad()
def evaluate_value_control(
    model,
    data,
    device,
    batch_size,
):

    model.eval()

    loader = DataLoader(
        RepDataset(data),
        batch_size=batch_size,
        shuffle=False,
    )

    all_logits = []
    all_stored = []
    all_query = []
    all_labels = []

    for (
        indices,
        summary,
        labels,
    ) in loader:

        summary = (
            summary.to(device)
        )

        labels = (
            labels.to(device)
        )

        (
            query_hidden,
            query_mask,
        ) = build_query_batch(
            data,
            indices,
            device,
        )

        output = model(
            summary,
            query_hidden,
            query_mask,
        )

        all_logits.append(
            output["logits"]
        )

        all_stored.append(
            output["stored"]
        )

        all_query.append(
            output["query_last"]
        )

        all_labels.append(
            labels
        )

    logits = torch.cat(
        all_logits,
        dim=0,
    )

    stored = torch.cat(
        all_stored,
        dim=0,
    )

    query = torch.cat(
        all_query,
        dim=0,
    )

    labels = torch.cat(
        all_labels,
        dim=0,
    )

    matched_losses = (
        F.cross_entropy(
            logits,
            labels,
            reduction="none",
        )
    )

    matched = (
        logits.argmax(dim=-1)
        .eq(labels)
        .float()
        .mean()
        .item()
        * 100
    )

    wrong_idx = (
        deterministic_mismatch(
            labels
        )
    )

    wrong_logits = (
        model.reader(
            query,
            stored[
                wrong_idx
            ],
        )
    )

    wrong_losses = (
        F.cross_entropy(
            wrong_logits,
            labels,
            reduction="none",
        )
    )

    mismatched = (
        wrong_logits
        .argmax(dim=-1)
        .eq(labels)
        .float()
        .mean()
        .item()
        * 100
    )

    zero_logits = (
        model.reader(
            query,
            torch.zeros_like(
                stored
            ),
        )
    )

    query_only = (
        zero_logits
        .argmax(dim=-1)
        .eq(labels)
        .float()
        .mean()
        .item()
        * 100
    )

    gap = (
        wrong_losses
        - matched_losses
    )

    return {
        "matched": float(
            matched
        ),
        "mismatched": float(
            mismatched
        ),
        "query_only": float(
            query_only
        ),
        "nll_gap": float(
            gap.mean().item()
        ),
        "positive_gap": float(
            (gap > 0)
            .float()
            .mean()
            .item()
            * 100
        ),
    }


# ============================================================
# HELPER:
# GENERATE MEMORY FOR ENTIRE BATCH
# ============================================================

@torch.no_grad()
def make_memory(
    model,
    summary,
):

    value = (
        model.value_encoder(
            summary
        )
    )

    return build_memory_slots(
        model.memory_bank,
        value,
        model.forced_slot,
    )


# ============================================================
# TRAIN ORIGINAL READER VARIANT
# ============================================================

def train_reader_epoch(
    model,
    loader,
    data,
    optimizer,
    device,
    mismatch_weight,
    mismatch_margin,
):

    model.train()

    model.value_encoder.eval()
    model.memory_bank.eval()

    if not any(
        p.requires_grad
        for p in (
            model.reader
            .parameters()
        )
    ):

        model.reader.eval()

    total_loss = 0.0
    total = 0

    for (
        indices,
        summary,
        labels,
    ) in loader:

        summary = (
            summary.to(device)
        )

        labels = (
            labels.to(device)
        )

        (
            query_hidden,
            query_mask,
        ) = build_query_batch(
            data,
            indices,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ====================================================
        # MATCHED
        # ====================================================

        output = model(
            summary,
            query_hidden,
            query_mask,
        )

        logits = (
            output["logits"]
        )

        matched_loss = (
            F.cross_entropy(
                logits,
                labels,
                reduction="none",
            )
        )

        # ====================================================
        # BUILD WRONG MEMORY
        # ====================================================

        with torch.no_grad():

            (
                memory_slots,
                memory_mask,
            ) = make_memory(
                model,
                summary,
            )

        wrong_idx = (
            random_mismatch(
                labels
            )
        )

        wrong_memory = (
            memory_slots[
                wrong_idx
            ]
        )

        # ====================================================
        # MISMATCHED
        # ====================================================

        wrong_output = (
            model.reader(
                hidden_states=(
                    query_hidden
                ),
                memory_slots=(
                    wrong_memory
                ),
                attention_mask=(
                    query_mask
                ),
                memory_mask=(
                    memory_mask
                ),
                routing_prior=None,
                memory_confidence=None,
                return_attention=False,
            )
        )

        if (
            model.variant.endswith(
                "_context"
            )
        ):

            wrong_rep = (
                last_valid(
                    wrong_output.context,
                    query_mask,
                )
            )

        else:

            wrong_rep = (
                last_valid(
                    wrong_output.fused_hidden,
                    query_mask,
                )
            )

        wrong_logits = (
            model.classifier(
                wrong_rep
            )
        )

        wrong_loss = (
            F.cross_entropy(
                wrong_logits,
                labels,
                reduction="none",
            )
        )

        rank_loss = F.relu(
            mismatch_margin
            + matched_loss
            - wrong_loss
        ).mean()

        loss = (
            matched_loss.mean()
            +
            mismatch_weight
            * rank_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [
                p
                for p in model.parameters()
                if p.requires_grad
            ],
            5.0,
        )

        optimizer.step()

        n = (
            labels.size(0)
        )

        total += n

        total_loss += (
            float(
                loss.item()
            )
            * n
        )

    return (
        total_loss
        / total
    )


# ============================================================
# EVALUATE ORIGINAL READER VARIANT
# ============================================================

@torch.no_grad()
def evaluate_reader(
    model,
    data,
    device,
    batch_size,
):

    model.eval()

    loader = DataLoader(
        RepDataset(data),
        batch_size=batch_size,
        shuffle=False,
    )

    matched_logits_all = []
    wrong_logits_all = []
    zero_logits_all = []
    labels_all = []

    attention_active = []
    confidence_all = []

    for (
        indices,
        summary,
        labels,
    ) in loader:

        summary = (
            summary.to(device)
        )

        labels = (
            labels.to(device)
        )

        (
            query_hidden,
            query_mask,
        ) = build_query_batch(
            data,
            indices,
            device,
        )

        # ====================================================
        # MATCHED
        # ====================================================

        matched_out = model(
            summary,
            query_hidden,
            query_mask,
        )

        matched_logits = (
            matched_out["logits"]
        )

        # ====================================================
        # MEMORY
        # ====================================================

        (
            memory_slots,
            memory_mask,
        ) = make_memory(
            model,
            summary,
        )

        # ====================================================
        # WRONG MEMORY
        # ====================================================

        wrong_idx = (
            deterministic_mismatch(
                labels
            )
        )

        wrong_memory = (
            memory_slots[
                wrong_idx
            ]
        )

        wrong_read = (
            model.reader(
                hidden_states=(
                    query_hidden
                ),
                memory_slots=(
                    wrong_memory
                ),
                attention_mask=(
                    query_mask
                ),
                memory_mask=(
                    memory_mask
                ),
                routing_prior=None,
                memory_confidence=None,
                return_attention=False,
            )
        )

        if (
            model.variant.endswith(
                "_context"
            )
        ):

            wrong_rep = (
                last_valid(
                    wrong_read.context,
                    query_mask,
                )
            )

        else:

            wrong_rep = (
                last_valid(
                    wrong_read.fused_hidden,
                    query_mask,
                )
            )

        wrong_logits = (
            model.classifier(
                wrong_rep
            )
        )

        # ====================================================
        # QUERY ONLY / ZERO MEMORY
        # ====================================================

        zero_memory = (
            torch.zeros_like(
                memory_slots
            )
        )

        zero_read = (
            model.reader(
                hidden_states=(
                    query_hidden
                ),
                memory_slots=(
                    zero_memory
                ),
                attention_mask=(
                    query_mask
                ),
                memory_mask=(
                    memory_mask
                ),
                routing_prior=None,
                memory_confidence=None,
                return_attention=False,
            )
        )

        if (
            model.variant.endswith(
                "_context"
            )
        ):

            zero_rep = (
                last_valid(
                    zero_read.context,
                    query_mask,
                )
            )

        else:

            zero_rep = (
                last_valid(
                    zero_read.fused_hidden,
                    query_mask,
                )
            )

        zero_logits = (
            model.classifier(
                zero_rep
            )
        )

        # ====================================================
        # DIAGNOSTICS
        # ====================================================

        attn = (
            matched_out[
                "attention_weights"
            ]
        )

        if attn is not None:

            # slot 0 is forced available.
            active_weight = (
                attn[
                    ...,
                    model.forced_slot
                ]
                .mean()
                .item()
            )

            attention_active.append(
                active_weight
            )

        confidence_all.append(
            matched_out[
                "read_confidence"
            ]
            .mean()
            .item()
        )

        matched_logits_all.append(
            matched_logits
        )

        wrong_logits_all.append(
            wrong_logits
        )

        zero_logits_all.append(
            zero_logits
        )

        labels_all.append(
            labels
        )

    matched_logits = (
        torch.cat(
            matched_logits_all,
            dim=0,
        )
    )

    wrong_logits = (
        torch.cat(
            wrong_logits_all,
            dim=0,
        )
    )

    zero_logits = (
        torch.cat(
            zero_logits_all,
            dim=0,
        )
    )

    labels = torch.cat(
        labels_all,
        dim=0,
    )

    matched_losses = (
        F.cross_entropy(
            matched_logits,
            labels,
            reduction="none",
        )
    )

    wrong_losses = (
        F.cross_entropy(
            wrong_logits,
            labels,
            reduction="none",
        )
    )

    matched_accuracy = (
        matched_logits
        .argmax(dim=-1)
        .eq(labels)
        .float()
        .mean()
        .item()
        * 100
    )

    wrong_accuracy = (
        wrong_logits
        .argmax(dim=-1)
        .eq(labels)
        .float()
        .mean()
        .item()
        * 100
    )

    query_accuracy = (
        zero_logits
        .argmax(dim=-1)
        .eq(labels)
        .float()
        .mean()
        .item()
        * 100
    )

    gap = (
        wrong_losses
        - matched_losses
    )

    return {
        "matched": float(
            matched_accuracy
        ),
        "mismatched": float(
            wrong_accuracy
        ),
        "query_only": float(
            query_accuracy
        ),
        "nll_gap": float(
            gap.mean().item()
        ),
        "positive_gap": float(
            (gap > 0)
            .float()
            .mean()
            .item()
            * 100
        ),
        "active_slot_attention": float(
            sum(attention_active)
            /
            max(
                len(
                    attention_active
                ),
                1,
            )
        ),
        "read_confidence": float(
            sum(confidence_all)
            /
            max(
                len(
                    confidence_all
                ),
                1,
            )
        ),
    }


# ============================================================
# TRAIN STAGE 1
# ============================================================

def train_known_good_value(
    original,
    train_data,
    validation_data,
    num_classes,
    device,
    args,
):

    print()
    print("=" * 100)
    print(
        "STAGE 1 — BUILD KNOWN-GOOD STORED VALUE"
    )
    print("=" * 100)

    set_seed(
        args.seed
    )

    model = ValueControlModel(
        original=original,
        num_classes=num_classes,
        forced_slot=(
            args.forced_slot
        ),
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    model.value_encoder
                    .parameters()
                ),
                "lr": (
                    args.value_learning_rate
                ),
            },
            {
                "params": (
                    model.reader
                    .parameters()
                ),
                "lr": (
                    args.classifier_learning_rate
                ),
            },
        ],
        weight_decay=1e-4,
    )

    loader = DataLoader(
        RepDataset(
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
        args.value_epochs + 1,
    ):

        loss = (
            train_value_control_epoch(
                model=model,
                loader=loader,
                data=train_data,
                optimizer=optimizer,
                device=device,
                mismatch_weight=(
                    args.mismatch_weight
                ),
                mismatch_margin=(
                    args.mismatch_margin
                ),
            )
        )

        val = (
            evaluate_value_control(
                model,
                validation_data,
                device,
                args.batch_size,
            )
        )

        print(
            f"EPOCH {epoch:02d} | "
            f"loss={loss:.4f} | "
            f"match={val['matched']:.2f}% | "
            f"mismatch={val['mismatched']:.2f}% | "
            f"query={val['query_only']:.2f}% | "
            f"gap={val['nll_gap']:+.4f}"
        )

        if (
            val["matched"]
            > best_acc
        ):

            best_acc = (
                val["matched"]
            )

            best_epoch = (
                epoch
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

    model.load_state_dict(
        best_state
    )

    print()
    print(
        f"Best VALUE control epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation matched: "
        f"{best_acc:.2f}%"
    )

    if (
        best_acc < 70.0
    ):

        print()
        print(
            "WARNING:"
        )

        print(
            "Known-good VALUE stage did not "
            "reach a strong accuracy."
        )

        print(
            "Reader conclusions should not "
            "be trusted until this stage passes."
        )

    return model


# ============================================================
# TRAIN ONE READER VARIANT
# ============================================================

def train_reader_variant(
    variant,
    original,
    value_encoder,
    train_data,
    validation_data,
    test_data,
    num_classes,
    device,
    args,
):

    print()
    print("#" * 100)
    print(
        f"READER TEST: "
        f"{variant.upper()}"
    )
    print("#" * 100)

    set_seed(
        args.seed
    )

    model = (
        OriginalReaderTestModel(
            original=original,
            value_encoder=(
                value_encoder
            ),
            num_classes=(
                num_classes
            ),
            forced_slot=(
                args.forced_slot
            ),
            variant=variant,
        )
        .to(device)
    )

    classifier_params = list(
        model.classifier
        .parameters()
    )

    reader_params = [
        p
        for p in (
            model.reader
            .parameters()
        )
        if p.requires_grad
    ]

    groups = [
        {
            "params": (
                classifier_params
            ),
            "lr": (
                args.classifier_learning_rate
            ),
        }
    ]

    if reader_params:

        groups.append(
            {
                "params": (
                    reader_params
                ),
                "lr": (
                    args.reader_learning_rate
                ),
            }
        )

    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=1e-4,
    )

    print(
        "Reader trainable:",
        bool(
            reader_params
        ),
    )

    print(
        "Reader trainable params:",
        f"{sum(p.numel() for p in reader_params):,}"
    )

    print(
        "Classifier params:",
        f"{sum(p.numel() for p in classifier_params):,}"
    )

    loader = DataLoader(
        RepDataset(
            train_data
        ),
        batch_size=(
            args.batch_size
        ),
        shuffle=True,
    )

    best_acc = -1.0
    best_gap = -float(
        "inf"
    )
    best_state = None
    best_epoch = -1

    history = []

    for epoch in range(
        1,
        args.reader_epochs + 1,
    ):

        loss = (
            train_reader_epoch(
                model=model,
                loader=loader,
                data=train_data,
                optimizer=optimizer,
                device=device,
                mismatch_weight=(
                    args.mismatch_weight
                ),
                mismatch_margin=(
                    args.mismatch_margin
                ),
            )
        )

        val = (
            evaluate_reader(
                model,
                validation_data,
                device,
                args.batch_size,
            )
        )

        print(
            f"EPOCH {epoch:02d} | "
            f"loss={loss:.4f} | "
            f"match={val['matched']:.2f}% | "
            f"mismatch={val['mismatched']:.2f}% | "
            f"query={val['query_only']:.2f}% | "
            f"gap={val['nll_gap']:+.4f} | "
            f"slot_attn="
            f"{val['active_slot_attention']:.4f} | "
            f"confidence="
            f"{val['read_confidence']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "validation": val,
            }
        )

        better = False

        if (
            val["matched"]
            > best_acc
        ):

            better = True

        elif (
            val["matched"]
            == best_acc
            and
            val["nll_gap"]
            > best_gap
        ):

            better = True

        if better:

            best_acc = (
                val["matched"]
            )

            best_gap = (
                val["nll_gap"]
            )

            best_epoch = (
                epoch
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            print(
                "Saved best checkpoint."
            )

    model.load_state_dict(
        best_state
    )

    test = evaluate_reader(
        model,
        test_data,
        device,
        args.batch_size,
    )

    print()
    print("=" * 90)
    print(
        f"{variant.upper()} "
        f"FINAL TEST "
        f"— BEST EPOCH {best_epoch}"
    )
    print("=" * 90)

    print(
        f"MATCHED:       "
        f"{test['matched']:.2f}%"
    )

    print(
        f"MISMATCHED:    "
        f"{test['mismatched']:.2f}%"
    )

    print(
        f"QUERY ONLY:    "
        f"{test['query_only']:.2f}%"
    )

    print(
        f"NLL GAP:       "
        f"{test['nll_gap']:+.6f}"
    )

    print(
        f"POSITIVE GAP:  "
        f"{test['positive_gap']:.2f}%"
    )

    print(
        f"ACTIVE SLOT ATTENTION: "
        f"{test['active_slot_attention']:.6f}"
    )

    print(
        f"READ CONFIDENCE: "
        f"{test['read_confidence']:.6f}"
    )

    return {
        "variant": variant,
        "best_epoch": best_epoch,
        "best_validation_accuracy": (
            best_acc
        ),
        "test": test,
        "history": history,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

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
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--value-epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--reader-epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--value-learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--reader-learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--classifier-learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--mismatch-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--mismatch-margin",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--forced-slot",
        type=int,
        default=0,
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
            "read_side_all_tests"
        ),
    )

    args = parser.parse_args()

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
        "READ SIDE — COMPLETE ISOLATION TEST"
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
        "Tests:"
    )

    print(
        "A. SIMPLE_CONTROL"
    )

    print(
        "B. FROZEN_READER_CONTEXT"
    )

    print(
        "C. FROZEN_READER_FUSED"
    )

    print(
        "D. TRAINABLE_READER_CONTEXT"
    )

    print(
        "E. TRAINABLE_READER_FUSED"
    )

    print()
    print(
        "Router: DISABLED"
    )

    print(
        "CandidateWriter: DISABLED"
    )

    print(
        "VectorGate: DISABLED"
    )

    print(
        "Correct slot: FORCED"
    )

    print(
        "Reader memory mask: ONLY slot 0 visible"
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

    if (
        tokenizer.pad_token
        is None
    ):

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = (
        "right"
    )

    answer_map = (
        get_single_token_answers(
            tokenizer
        )
    )

    answers = list(
        answer_map.keys()
    )

    answer_to_class = {
        word: i
        for i, word
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
        "Chance accuracy:",
        f"{100 / len(answers):.2f}%"
    )

    print(
        "Chance CE:",
        f"{math.log(len(answers)):.4f}"
    )

    # ========================================================
    # ORIGINAL MODEL
    # ========================================================

    original = load_original(
        args.checkpoint,
        args.model_name,
        device,
    )

    # ========================================================
    # DATA
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

    validation_examples = (
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
        "Precomputing train..."
    )

    train_data = precompute(
        original,
        tokenizer,
        train_examples,
        answer_to_class,
        device,
        args.batch_size,
    )

    print()
    print(
        "Precomputing validation..."
    )

    validation_data = (
        precompute(
            original,
            tokenizer,
            validation_examples,
            answer_to_class,
            device,
            args.batch_size,
        )
    )

    print()
    print(
        "Precomputing test..."
    )

    test_data = precompute(
        original,
        tokenizer,
        test_examples,
        answer_to_class,
        device,
        args.batch_size,
    )

    # ========================================================
    # STAGE 1:
    # KNOWN-GOOD VALUE
    # ========================================================

    value_control = (
        train_known_good_value(
            original=original,
            train_data=train_data,
            validation_data=(
                validation_data
            ),
            num_classes=(
                len(answers)
            ),
            device=device,
            args=args,
        )
    )

    value_test = (
        evaluate_value_control(
            value_control,
            test_data,
            device,
            args.batch_size,
        )
    )

    print()
    print("=" * 100)
    print(
        "A. SIMPLE_CONTROL FINAL TEST"
    )
    print("=" * 100)

    print(
        f"MATCHED:       "
        f"{value_test['matched']:.2f}%"
    )

    print(
        f"MISMATCHED:    "
        f"{value_test['mismatched']:.2f}%"
    )

    print(
        f"QUERY ONLY:    "
        f"{value_test['query_only']:.2f}%"
    )

    print(
        f"NLL GAP:       "
        f"{value_test['nll_gap']:+.6f}"
    )

    print(
        f"POSITIVE GAP:  "
        f"{value_test['positive_gap']:.2f}%"
    )

    # ========================================================
    # FREEZE KNOWN-GOOD VALUE ENCODER
    # ========================================================

    value_encoder = (
        value_control
        .value_encoder
    )

    for p in (
        value_encoder
        .parameters()
    ):

        p.requires_grad = False

    # ========================================================
    # READER VARIANTS
    # ========================================================

    variants = [
        "frozen_context",
        "frozen_fused",
        "trainable_context",
        "trainable_fused",
    ]

    results = {
        "simple_control": (
            value_test
        )
    }

    for variant in variants:

        result = (
            train_reader_variant(
                variant=variant,
                original=original,
                value_encoder=(
                    value_encoder
                ),
                train_data=(
                    train_data
                ),
                validation_data=(
                    validation_data
                ),
                test_data=(
                    test_data
                ),
                num_classes=(
                    len(answers)
                ),
                device=device,
                args=args,
            )
        )

        results[
            variant
        ] = result

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print("=" * 110)
    print(
        "FINAL READ-SIDE COMPARISON"
    )
    print("=" * 110)

    print(
        f"{'TEST':<28}"
        f"{'MATCH':>12}"
        f"{'MISMATCH':>14}"
        f"{'QUERY':>12}"
        f"{'NLL GAP':>14}"
    )

    print("-" * 82)

    print(
        f"{'simple_control':<28}"
        f"{value_test['matched']:>11.2f}%"
        f"{value_test['mismatched']:>13.2f}%"
        f"{value_test['query_only']:>11.2f}%"
        f"{value_test['nll_gap']:>14.4f}"
    )

    for variant in variants:

        t = (
            results[
                variant
            ]["test"]
        )

        print(
            f"{variant:<28}"
            f"{t['matched']:>11.2f}%"
            f"{t['mismatched']:>13.2f}%"
            f"{t['query_only']:>11.2f}%"
            f"{t['nll_gap']:>14.4f}"
        )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    control_acc = (
        value_test[
            "matched"
        ]
    )

    frozen_context = (
        results[
            "frozen_context"
        ]["test"][
            "matched"
        ]
    )

    frozen_fused = (
        results[
            "frozen_fused"
        ]["test"][
            "matched"
        ]
    )

    train_context = (
        results[
            "trainable_context"
        ]["test"][
            "matched"
        ]
    )

    train_fused = (
        results[
            "trainable_fused"
        ]["test"][
            "matched"
        ]
    )

    print()
    print("=" * 110)
    print(
        "INTERPRETATION"
    )
    print("=" * 110)

    print(
        f"Simple control:          "
        f"{control_acc:.2f}%"
    )

    print(
        f"Frozen reader context:   "
        f"{frozen_context:.2f}%"
    )

    print(
        f"Frozen reader fused:     "
        f"{frozen_fused:.2f}%"
    )

    print(
        f"Trainable reader context:"
        f" {train_context:.2f}%"
    )

    print(
        f"Trainable reader fused:  "
        f"{train_fused:.2f}%"
    )

    print()

    if (
        control_acc < 70
    ):

        print(
            "STOP:"
        )

        print(
            "Known-good VALUE control is weak."
        )

        print(
            "Do not diagnose the MemoryReader "
            "from this run."
        )

    elif (
        frozen_fused >= 70
    ):

        print(
            "STRONG RESULT:"
        )

        print(
            "The ORIGINAL CHECKPOINT MemoryReader "
            "can use a good stored VALUE."
        )

        print(
            "Reader is unlikely to be the main "
            "remaining bottleneck."
        )

    elif (
        frozen_fused < 40
        and
        train_fused >= 70
    ):

        print(
            "CHECKPOINT READER FAILURE:"
        )

        print(
            "The reader architecture works when trained, "
            "but the original checkpoint reader weights "
            "were poorly learned."
        )

    elif (
        train_context >= 70
        and
        train_fused < 50
    ):

        print(
            "FUSION BOTTLENECK:"
        )

        print(
            "The reader can recover memory context, "
            "but the gated fusion into GPT-2 hidden "
            "states destroys or suppresses it."
        )

    elif (
        train_context < 50
        and
        train_fused < 50
    ):

        print(
            "READER ARCHITECTURE WARNING:"
        )

        print(
            "Even when trainable, the original reader "
            "cannot reliably use the known-good VALUE."
        )

        print(
            "The query/key/value read formulation "
            "should be inspected next."
        )

    else:

        print(
            "PARTIAL RESULT:"
        )

        print(
            "Reader is learning, but there is still "
            "substantial degradation."
        )

        print(
            "Compare context vs fused results to "
            "separate retrieval from fusion."
        )

    # ========================================================
    # SAVE
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "read_side_all_results.json"
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
            "value_encoder": (
                value_encoder
                .state_dict()
            ),
            "arguments": (
                vars(args)
            ),
        },
        output_dir
        / "known_good_value_encoder.pt",
    )

    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()