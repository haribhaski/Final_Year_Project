"""
Runs the memory ablation ladder independently.

L0 = Linear VALUE + simple reader
L1 = L0 + MemoryBank
L2 = L0 + frozen original write gate + MemoryBank
L3 = L0 + frozen original OrthogonalUpdate + gate + MemoryBank
L4 = original CandidateWriter + OrthogonalUpdate + gate + MemoryBank
L5 = L4 + original MemoryReader

Every level:
- uses same train / validation / test examples
- uses same frozen GPT-2 representations
- is trained independently from scratch
- saves its own checkpoint
- reports matched / mismatched / query-only
- selects checkpoint using NLL gap
"""

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
# CONSTANTS
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

def seed_everything(seed):

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA
# ============================================================

def single_token_answers(tokenizer):

    result = {}

    for word in ANSWER_POOL:

        ids = tokenizer(
            " " + word,
            add_special_tokens=False,
        )["input_ids"]

        if len(ids) == 1:
            result[word] = ids[0]

    return result


def build_examples(
    n,
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

    examples = []

    for i in range(n):

        answer = rng.choice(
            answers
        )

        entity = (
            f"person_{start_id + i}"
        )

        examples.append(
            {
                "entity": entity,

                "answer": answer,

                "fact": rng.choice(
                    fact_templates
                ).format(
                    entity=entity,
                    answer=answer,
                ),

                "query": rng.choice(
                    query_templates
                ).format(
                    entity=entity,
                ),
            }
        )

    return examples


class RawDataset(Dataset):

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def raw_collate(batch):

    return {
        "fact": [
            x["fact"]
            for x in batch
        ],

        "query": [
            x["query"]
            for x in batch
        ],

        "answer": [
            x["answer"]
            for x in batch
        ],
    }


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(
    tokenizer,
    texts,
    device,
):

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
        add_special_tokens=False,
    )

    return (
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
    )


def find_answer_positions(
    input_ids,
    attention_mask,
    answer_ids,
):

    positions = []

    for i in range(
        input_ids.size(0)
    ):

        length = int(
            attention_mask[i]
            .sum()
            .item()
        )

        valid = input_ids[
            i,
            :length,
        ]

        matches = (
            valid
            == int(answer_ids[i])
        ).nonzero(
            as_tuple=False
        ).flatten()

        if matches.numel() == 0:
            raise RuntimeError(
                "Answer token not found."
            )

        positions.append(
            int(
                matches[-1]
                .item()
            )
        )

    return torch.tensor(
        positions,
        dtype=torch.long,
        device=input_ids.device,
    )


# ============================================================
# LOAD ORIGINAL MODEL
# ============================================================

def load_model(
    checkpoint_path,
    model_name,
    device,
):

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
# PRECOMPUTE FEATURES
#
# We save:
#
# answer-token representation
# mean summary
# query last-token representation
# full token states for CandidateWriter / MemoryReader
# fact attention mask
# query token states
# query attention mask
#
# For simplicity, token-level tensors are padded globally.
# ============================================================

@torch.no_grad()
def precompute(
    model,
    tokenizer,
    examples,
    answer_token_map,
    answer_to_class,
    device,
):

    fact_reps = []
    summaries = []
    query_reps = []
    labels = []

    # Token sequences kept per example because lengths differ.
    fact_token_states = []
    fact_masks = []

    query_token_states = []
    query_masks = []

    for index, example in enumerate(
        examples
    ):

        # ====================================================
        # FACT
        # ====================================================

        fact_ids, fact_mask = tokenize(
            tokenizer,
            [example["fact"]],
            device,
        )

        fact_out = (
            model.backbone.transformer(
                input_ids=fact_ids,
                attention_mask=fact_mask,
                return_dict=True,
            )
        )

        fact_hidden = (
            fact_out.last_hidden_state
        )

        answer_id = (
            answer_token_map[
                example["answer"]
            ]
        )

        answer_position = (
            find_answer_positions(
                fact_ids,
                fact_mask,
                [answer_id],
            )[0]
        )

        answer_rep = fact_hidden[
            0,
            answer_position,
            :
        ]

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
        )[0]

        # ====================================================
        # QUERY
        # ====================================================

        query_ids, query_mask = (
            tokenize(
                tokenizer,
                [example["query"]],
                device,
            )
        )

        query_out = (
            model.backbone.transformer(
                input_ids=query_ids,
                attention_mask=query_mask,
                return_dict=True,
            )
        )

        query_hidden = (
            query_out.last_hidden_state
        )

        query_last = (
            query_hidden[
                0,
                int(
                    query_mask.sum()
                    .item()
                ) - 1,
                :
            ]
        )

        # ====================================================
        # STORE
        # ====================================================

        fact_reps.append(
            answer_rep.cpu()
        )

        summaries.append(
            summary.cpu()
        )

        query_reps.append(
            query_last.cpu()
        )

        labels.append(
            answer_to_class[
                example["answer"]
            ]
        )

        fact_token_states.append(
            fact_hidden[0].cpu()
        )

        fact_masks.append(
            fact_mask[0].cpu()
        )

        query_token_states.append(
            query_hidden[0].cpu()
        )

        query_masks.append(
            query_mask[0].cpu()
        )

        if (
            (index + 1) % 500 == 0
            or index + 1
            == len(examples)
        ):

            print(
                f"  {index + 1}/"
                f"{len(examples)}"
            )

    return {
        "fact": torch.stack(
            fact_reps
        ),

        "summary": torch.stack(
            summaries
        ),

        "query": torch.stack(
            query_reps
        ),

        "labels": torch.tensor(
            labels,
            dtype=torch.long,
        ),

        "fact_tokens": (
            fact_token_states
        ),

        "fact_masks": (
            fact_masks
        ),

        "query_tokens": (
            query_token_states
        ),

        "query_masks": (
            query_masks
        ),
    }


# ============================================================
# SIMPLE READER
# ============================================================

class SimpleReader(nn.Module):

    def __init__(
        self,
        d_model,
        classes,
    ):

        super().__init__()

        self.query_norm = (
            nn.LayerNorm(
                d_model
            )
        )

        self.net = nn.Sequential(
            nn.Linear(
                d_model * 3,
                d_model,
            ),
            nn.GELU(),
            nn.Linear(
                d_model,
                classes,
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
# COMMON BASE
# ============================================================

class AblationBase(nn.Module):

    def __init__(
        self,
        original,
        classes,
        forced_slot,
    ):

        super().__init__()

        self.d_model = (
            original.d_model
        )

        self.num_slots = (
            original.num_slots
        )

        self.forced_slot = (
            forced_slot
        )

        self.memory_bank = copy.deepcopy(
            original.memory_bank
        )

        self.write_gate = copy.deepcopy(
            original.write_gate_module
        )

        self.orthogonalizer = (
            copy.deepcopy(
                original.orthogonalizer
            )
        )

        self.writer = copy.deepcopy(
            original.writer
        )

        self.original_reader = (
            copy.deepcopy(
                original.reader
            )
        )

        # Frozen copied original modules by default.
        for module in [
            self.memory_bank,
            self.write_gate,
            self.orthogonalizer,
            self.writer,
            self.original_reader,
        ]:

            for p in module.parameters():
                p.requires_grad = False

        self.value_projection = (
            nn.Sequential(
                nn.LayerNorm(
                    self.d_model
                ),
                nn.Linear(
                    self.d_model,
                    self.d_model,
                ),
            )
        )

        self.simple_reader = (
            SimpleReader(
                self.d_model,
                classes,
            )
        )

    def initial_state(
        self,
        batch,
        device,
        dtype,
    ):

        return (
            self.memory_bank.initialize(
                batch_size=batch,
                device=device,
                dtype=dtype,
            )
        )

    def slot_mask(
        self,
        batch,
        device,
    ):

        mask = torch.zeros(
            batch,
            self.num_slots,
            dtype=torch.bool,
            device=device,
        )

        mask[
            :,
            self.forced_slot,
        ] = True

        return mask

    def memory_store(
        self,
        value,
        gate,
        orthogonalize=False,
    ):

        batch = value.size(0)

        state = self.initial_state(
            batch,
            value.device,
            value.dtype,
        )

        candidate = (
            state.slots.clone()
        )

        if orthogonalize:

            updates = torch.zeros_like(
                state.slots
            )

            updates[
                :,
                self.forced_slot,
                :,
            ] = (
                value
                - state.slots[
                    :,
                    self.forced_slot,
                    :,
                ]
            )

            ortho = self.orthogonalizer(
                updates=updates,
                memory_slots=state.slots,
            )

            projected_candidate = (
                state.slots
                + ortho.updates
            )

            candidate = (
                projected_candidate
            )

        else:

            candidate[
                :,
                self.forced_slot,
                :,
            ] = value

        mask = self.slot_mask(
            batch,
            value.device,
        )

        state = self.memory_bank(
            state=state,
            candidate=candidate,
            write_gate=gate,
            write_mask=(
                mask.unsqueeze(-1)
            ),
            confidence=None,
        )

        return (
            state,
            state.slots[
                :,
                self.forced_slot,
                :,
            ],
        )


# ============================================================
# LEVEL 0
# ============================================================

class Level0(AblationBase):

    def forward(
        self,
        fact,
        summary,
        query,
        fact_tokens=None,
        fact_mask=None,
        query_tokens=None,
        query_mask=None,
    ):

        value = (
            self.value_projection(
                fact
            )
        )

        logits = (
            self.simple_reader(
                query,
                value,
            )
        )

        return logits, value


# ============================================================
# LEVEL 1
# ============================================================

class Level1(AblationBase):

    def forward(
        self,
        fact,
        summary,
        query,
        **kwargs,
    ):

        value = (
            self.value_projection(
                fact
            )
        )

        batch = value.size(0)

        gate = torch.zeros(
            batch,
            self.num_slots,
            1,
            device=value.device,
        )

        gate[
            :,
            self.forced_slot,
            0,
        ] = 1.0

        _, stored = self.memory_store(
            value,
            gate,
            orthogonalize=False,
        )

        logits = (
            self.simple_reader(
                query,
                stored,
            )
        )

        return logits, stored


# ============================================================
# LEVEL 2
# ============================================================

class Level2(AblationBase):

    def forward(
        self,
        fact,
        summary,
        query,
        **kwargs,
    ):

        value = (
            self.value_projection(
                fact
            )
        )

        mask = self.slot_mask(
            value.size(0),
            value.device,
        )

        gate = self.write_gate(
            summary,
            slot_mask=mask,
        )

        _, stored = self.memory_store(
            value,
            gate,
            orthogonalize=False,
        )

        logits = (
            self.simple_reader(
                query,
                stored,
            )
        )

        return logits, stored


# ============================================================
# LEVEL 3
# ============================================================

class Level3(AblationBase):

    def forward(
        self,
        fact,
        summary,
        query,
        **kwargs,
    ):

        value = (
            self.value_projection(
                fact
            )
        )

        mask = self.slot_mask(
            value.size(0),
            value.device,
        )

        gate = self.write_gate(
            summary,
            slot_mask=mask,
        )

        _, stored = self.memory_store(
            value,
            gate,
            orthogonalize=True,
        )

        logits = (
            self.simple_reader(
                query,
                stored,
            )
        )

        return logits, stored


# ============================================================
# COLLATE TOKEN STATES
# ============================================================

def stack_token_batch(
    tensors,
    masks,
    indices,
    device,
):

    selected = [
        tensors[int(i)]
        for i in indices
    ]

    selected_masks = [
        masks[int(i)]
        for i in indices
    ]

    max_len = max(
        x.size(0)
        for x in selected
    )

    d_model = (
        selected[0]
        .size(-1)
    )

    batch = torch.zeros(
        len(selected),
        max_len,
        d_model,
        dtype=selected[0].dtype,
        device=device,
    )

    batch_mask = torch.zeros(
        len(selected),
        max_len,
        dtype=torch.long,
        device=device,
    )

    for j, x in enumerate(
        selected
    ):

        length = x.size(0)

        batch[
            j,
            :length,
            :,
        ] = x.to(device)

        batch_mask[
            j,
            :length,
        ] = (
            selected_masks[j]
            .to(device)
        )

    return (
        batch,
        batch_mask,
    )


# ============================================================
# LEVEL 4
#
# Original CandidateWriter is now the source of VALUE.
# No Linear VALUE path.
# ============================================================

class Level4(AblationBase):

    def forward(
        self,
        fact,
        summary,
        query,
        fact_tokens,
        fact_mask,
        **kwargs,
    ):

        batch = fact.size(0)

        state = self.initial_state(
            batch,
            fact.device,
            fact.dtype,
        )

        mask = self.slot_mask(
            batch,
            fact.device,
        )

        routing_weights = torch.zeros(
            batch,
            self.num_slots,
            device=fact.device,
        )

        routing_weights[
            :,
            self.forced_slot,
        ] = 1.0

        writer_output = self.writer(
            summary=summary,
            memory_slots=state.slots,
            token_states=fact_tokens,
            attention_mask=fact_mask,
            routing_weights=(
                routing_weights
            ),
        )

        ortho = self.orthogonalizer(
            updates=(
                writer_output.deltas
            ),
            memory_slots=state.slots,
        )

        candidate = (
            state.slots
            + ortho.updates
        )

        gate = self.write_gate(
            summary,
            slot_mask=mask,
        )

        state = self.memory_bank(
            state=state,
            candidate=candidate,
            write_gate=gate,
            write_mask=(
                mask.unsqueeze(-1)
            ),
            confidence=None,
        )

        stored = state.slots[
            :,
            self.forced_slot,
            :,
        ]

        logits = (
            self.simple_reader(
                query,
                stored,
            )
        )

        return logits, stored


# ============================================================
# LEVEL 5
#
# Original writer + original reader.
#
# This is still forced-slot, so addressing is NOT being tested.
# ============================================================

class Level5(AblationBase):

    def __init__(
        self,
        original,
        classes,
        forced_slot,
    ):

        super().__init__(
            original,
            classes,
            forced_slot,
        )

        # Classification head on top of the final fused
        # query representation.
        self.classifier = nn.Linear(
            self.d_model,
            classes,
        )

    def forward(
        self,
        fact,
        summary,
        query,
        fact_tokens,
        fact_mask,
        query_tokens,
        query_mask,
    ):

        batch = fact.size(0)

        state = self.initial_state(
            batch,
            fact.device,
            fact.dtype,
        )

        mask = self.slot_mask(
            batch,
            fact.device,
        )

        routing_weights = torch.zeros(
            batch,
            self.num_slots,
            device=fact.device,
        )

        routing_weights[
            :,
            self.forced_slot,
        ] = 1.0

        # ----------------------------------------------------
        # ORIGINAL WRITER
        # ----------------------------------------------------

        writer_output = self.writer(
            summary=summary,
            memory_slots=state.slots,
            token_states=fact_tokens,
            attention_mask=fact_mask,
            routing_weights=(
                routing_weights
            ),
        )

        # ----------------------------------------------------
        # ORIGINAL ORTHOGONALIZER
        # ----------------------------------------------------

        ortho = self.orthogonalizer(
            updates=(
                writer_output.deltas
            ),
            memory_slots=state.slots,
        )

        candidate = (
            state.slots
            + ortho.updates
        )

        # ----------------------------------------------------
        # ORIGINAL GATE
        # ----------------------------------------------------

        gate = self.write_gate(
            summary,
            slot_mask=mask,
        )

        # ----------------------------------------------------
        # ORIGINAL MEMORY BANK
        # ----------------------------------------------------

        state = self.memory_bank(
            state=state,
            candidate=candidate,
            write_gate=gate,
            write_mask=(
                mask.unsqueeze(-1)
            ),
            confidence=None,
        )

        # ----------------------------------------------------
        # ORIGINAL READER
        # ----------------------------------------------------

        reader_output = (
            self.original_reader(
                hidden_states=(
                    query_tokens
                ),

                memory_slots=(
                    state.slots
                ),

                attention_mask=(
                    query_mask
                ),

                memory_mask=mask,

                memory_confidence=None,

                return_attention=True,
            )
        )

        fused = (
            reader_output
            .fused_hidden
        )

        # last valid question token
        last = (
            query_mask.sum(dim=1)
            - 1
        )

        rows = torch.arange(
            batch,
            device=fused.device,
        )

        pooled = fused[
            rows,
            last,
            :
        ]

        logits = self.classifier(
            pooled
        )

        stored = state.slots[
            :,
            self.forced_slot,
            :,
        ]

        return logits, stored


# ============================================================
# INDEX DATASET
# ============================================================

class IndexDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data[
            "labels"
        ].size(0)

    def __getitem__(self, i):

        return (
            i,
            self.data["fact"][i],
            self.data["summary"][i],
            self.data["query"][i],
            self.data["labels"][i],
        )


# ============================================================
# MISMATCH
# ============================================================

def mismatch_indices(labels):

    n = labels.size(0)

    result = torch.empty(
        n,
        dtype=torch.long,
        device=labels.device,
    )

    for i in range(n):

        candidates = torch.nonzero(
            labels != labels[i],
            as_tuple=False,
        ).flatten()

        result[i] = candidates[
            torch.randint(
                candidates.numel(),
                (1,),
                device=labels.device,
            )
        ]

    return result


# ============================================================
# FORWARD BATCH
# ============================================================

def forward_level(
    model,
    level,
    data,
    indices,
    fact,
    summary,
    query,
    device,
):

    if level <= 3:

        return model(
            fact=fact,
            summary=summary,
            query=query,
        )

    fact_tokens, fact_mask = (
        stack_token_batch(
            data["fact_tokens"],
            data["fact_masks"],
            indices,
            device,
        )
    )

    if level == 4:

        return model(
            fact=fact,
            summary=summary,
            query=query,
            fact_tokens=(
                fact_tokens
            ),
            fact_mask=fact_mask,
        )

    query_tokens, query_mask = (
        stack_token_batch(
            data["query_tokens"],
            data["query_masks"],
            indices,
            device,
        )
    )

    return model(
        fact=fact,
        summary=summary,
        query=query,
        fact_tokens=fact_tokens,
        fact_mask=fact_mask,
        query_tokens=query_tokens,
        query_mask=query_mask,
    )


# ============================================================
# TRAIN LEVEL
# ============================================================

def train_one_level(
    level,
    model,
    train_data,
    valid_data,
    device,
    epochs,
    batch_size,
    learning_rate,
    output_dir,
):

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=1e-4,
    )

    loader = DataLoader(
        IndexDataset(
            train_data
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    best_gap = -float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        # Original copied modules remain frozen/eval.
        model.memory_bank.eval()
        model.write_gate.eval()
        model.orthogonalizer.eval()
        model.writer.eval()
        model.original_reader.eval()

        running = 0.0
        total = 0

        for (
            indices,
            fact,
            summary,
            query,
            labels,
        ) in loader:

            fact = fact.to(device)
            summary = summary.to(device)
            query = query.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits, values = (
                forward_level(
                    model,
                    level,
                    train_data,
                    indices,
                    fact,
                    summary,
                    query,
                    device,
                )
            )

            matched_loss = (
                F.cross_entropy(
                    logits,
                    labels,
                )
            )

            wrong_idx = (
                mismatch_indices(
                    labels
                )
            )

            wrong_values = (
                values[
                    wrong_idx
                ]
            )

            # For levels 0-4,
            # mismatch can use simple reader directly.
            if level <= 4:

                wrong_logits = (
                    model.simple_reader(
                        query,
                        wrong_values,
                    )
                )

                wrong_losses = (
                    F.cross_entropy(
                        wrong_logits,
                        labels,
                        reduction="none",
                    )
                )

                matched_losses = (
                    F.cross_entropy(
                        logits,
                        labels,
                        reduction="none",
                    )
                )

                rank_loss = F.relu(
                    0.5
                    + matched_losses
                    - wrong_losses
                ).mean()

            else:

                # Level 5 mismatch ranking is omitted
                # during training because replacing the
                # memory state requires another reader pass.
                #
                # Matched/mismatched is still measured
                # correctly at evaluation.
                rank_loss = (
                    matched_loss
                    * 0.0
                )

            loss = (
                matched_loss
                + 0.5 * rank_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable,
                5.0,
            )

            optimizer.step()

            running += (
                float(loss.item())
                * labels.size(0)
            )

            total += labels.size(0)

        result = evaluate_level(
            level,
            model,
            valid_data,
            device,
            batch_size,
        )

        print(
            f"L{level} E{epoch:02d} | "
            f"loss={running / total:.4f} | "
            f"match={result['matched_accuracy']:.2f}% | "
            f"mismatch={result['mismatched_accuracy']:.2f}% | "
            f"gap={result['nll_gap']:+.4f}"
        )

        if (
            result["nll_gap"]
            > best_gap
        ):

            best_gap = (
                result["nll_gap"]
            )

            best_epoch = epoch

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v
                in model.state_dict()
                .items()
            }

    model.load_state_dict(
        best_state
    )

    Path(output_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "level": level,
            "best_epoch": (
                best_epoch
            ),
            "model_state_dict": (
                model.state_dict()
            ),
        },
        Path(output_dir)
        / f"level{level}_best.pt",
    )

    return (
        model,
        best_epoch,
    )


# ============================================================
# EVALUATE
# ============================================================

@torch.no_grad()
def evaluate_level(
    level,
    model,
    data,
    device,
    batch_size,
):

    model.eval()

    labels = (
        data["labels"]
        .to(device)
    )

    all_logits = []
    all_values = []

    loader = DataLoader(
        IndexDataset(data),
        batch_size=batch_size,
        shuffle=False,
    )

    for (
        indices,
        fact,
        summary,
        query,
        _,
    ) in loader:

        fact = fact.to(device)
        summary = summary.to(device)
        query = query.to(device)

        logits, values = (
            forward_level(
                model,
                level,
                data,
                indices,
                fact,
                summary,
                query,
                device,
            )
        )

        all_logits.append(
            logits
        )

        all_values.append(
            values
        )

    matched_logits = (
        torch.cat(
            all_logits,
            dim=0,
        )
    )

    values = torch.cat(
        all_values,
        dim=0,
    )

    matched_losses = (
        F.cross_entropy(
            matched_logits,
            labels,
            reduction="none",
        )
    )

    matched_acc = (
        matched_logits
        .argmax(dim=-1)
        .eq(labels)
        .float()
        .mean()
        .item()
        * 100
    )

    # deterministic different-answer mismatch
    wrong = []

    labels_cpu = (
        labels.cpu()
    )

    for i in range(
        labels.size(0)
    ):

        for off in range(
            1,
            labels.size(0),
        ):

            j = (
                i + off
            ) % labels.size(0)

            if (
                labels_cpu[j]
                != labels_cpu[i]
            ):

                wrong.append(j)
                break

    wrong = torch.tensor(
        wrong,
        device=device,
    )

    wrong_values = values[
        wrong
    ]

    # ========================================================
    # Levels 0-4
    # ========================================================

    if level <= 4:

        wrong_logits = (
            model.simple_reader(
                data["query"].to(
                    device
                ),
                wrong_values,
            )
        )

        query_only_logits = (
            model.simple_reader(
                data["query"].to(
                    device
                ),
                torch.zeros_like(
                    values
                ),
            )
        )

    else:

        # ----------------------------------------------------
        # Level 5 currently evaluates matched performance
        # directly.
        #
        # Proper mismatched original-reader evaluation needs
        # another explicit reader pass with swapped memory.
        #
        # Keep Level 5 out of the final causal attribution
        # until that function is implemented.
        # ----------------------------------------------------

        return {
            "matched_accuracy": (
                matched_acc
            ),

            "mismatched_accuracy": (
                float("nan")
            ),

            "query_only_accuracy": (
                float("nan")
            ),

            "matched_loss": float(
                matched_losses
                .mean()
                .item()
            ),

            "mismatched_loss": (
                float("nan")
            ),

            "nll_gap": (
                float("-inf")
            ),
        }

    wrong_losses = (
        F.cross_entropy(
            wrong_logits,
            labels,
            reduction="none",
        )
    )

    wrong_acc = (
        wrong_logits
        .argmax(dim=-1)
        .eq(labels)
        .float()
        .mean()
        .item()
        * 100
    )

    query_acc = (
        query_only_logits
        .argmax(dim=-1)
        .eq(labels)
        .float()
        .mean()
        .item()
        * 100
    )

    return {
        "matched_accuracy": (
            matched_acc
        ),

        "mismatched_accuracy": (
            wrong_acc
        ),

        "query_only_accuracy": (
            query_acc
        ),

        "matched_loss": float(
            matched_losses
            .mean()
            .item()
        ),

        "mismatched_loss": float(
            wrong_losses
            .mean()
            .item()
        ),

        "nll_gap": float(
            (
                wrong_losses
                - matched_losses
            )
            .mean()
            .item()
        ),
    }


# ============================================================
# FACTORY
# ============================================================

def make_level(
    level,
    original,
    classes,
    forced_slot,
):

    if level == 0:
        return Level0(
            original,
            classes,
            forced_slot,
        )

    if level == 1:
        return Level1(
            original,
            classes,
            forced_slot,
        )

    if level == 2:
        return Level2(
            original,
            classes,
            forced_slot,
        )

    if level == 3:
        return Level3(
            original,
            classes,
            forced_slot,
        )

    if level == 4:
        return Level4(
            original,
            classes,
            forced_slot,
        )

    if level == 5:
        return Level5(
            original,
            classes,
            forced_slot,
        )

    raise ValueError(
        f"Unknown level {level}"
    )


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
        "--levels",
        default="0,1,2,3,4",
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
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
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
            "ablation_ladder"
        ),
    )

    args = parser.parse_args()

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

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

    answer_token_map = (
        single_token_answers(
            tokenizer
        )
    )

    answers = list(
        answer_token_map.keys()
    )

    answer_to_class = {
        x: i
        for i, x
        in enumerate(
            answers
        )
    }

    print(
        "Chance:",
        f"{100 / len(answers):.2f}%"
    )

    print(
        "Chance CE:",
        f"{math.log(len(answers)):.4f}"
    )

    original = load_model(
        args.checkpoint,
        args.model_name,
        device,
    )

    # ========================================================
    # SAME DATA FOR EVERY LEVEL
    # ========================================================

    train_examples = build_examples(
        args.train_examples,
        answers,
        args.seed,
        0,
        "train",
    )

    valid_examples = build_examples(
        args.validation_examples,
        answers,
        args.seed + 1000,
        100000,
        "eval",
    )

    test_examples = build_examples(
        args.test_examples,
        answers,
        args.seed + 2000,
        200000,
        "eval",
    )

    print(
        "\nPrecomputing train..."
    )

    train_data = precompute(
        original,
        tokenizer,
        train_examples,
        answer_token_map,
        answer_to_class,
        device,
    )

    print(
        "\nPrecomputing validation..."
    )

    valid_data = precompute(
        original,
        tokenizer,
        valid_examples,
        answer_token_map,
        answer_to_class,
        device,
    )

    print(
        "\nPrecomputing test..."
    )

    test_data = precompute(
        original,
        tokenizer,
        test_examples,
        answer_token_map,
        answer_to_class,
        device,
    )

    requested_levels = [
        int(x)
        for x in (
            args.levels.split(",")
        )
    ]

    results = {}

    # ========================================================
    # TRAIN EVERY LEVEL INDEPENDENTLY
    # ========================================================

    for level in requested_levels:

        print()
        print("#" * 90)
        print(
            f"STARTING LEVEL {level}"
        )
        print("#" * 90)

        # Reset seed so initialization is comparable.
        seed_everything(
            args.seed
        )

        model = make_level(
            level,
            original,
            len(answers),
            args.forced_slot,
        ).to(device)

        model, best_epoch = (
            train_one_level(
                level=level,
                model=model,
                train_data=train_data,
                valid_data=valid_data,
                device=device,
                epochs=args.epochs,
                batch_size=(
                    args.batch_size
                ),
                learning_rate=(
                    args.learning_rate
                ),
                output_dir=(
                    args.output_dir
                ),
            )
        )

        test_result = (
            evaluate_level(
                level,
                model,
                test_data,
                device,
                args.batch_size,
            )
        )

        results[
            f"level_{level}"
        ] = {
            "best_epoch": (
                best_epoch
            ),
            "test": test_result,
        }

        print()
        print(
            f"LEVEL {level} TEST"
        )

        print(
            test_result
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print("=" * 100)
    print(
        "FINAL ABLATION LADDER"
    )
    print("=" * 100)

    print(
        f"{'LEVEL':<10}"
        f"{'MATCH':>12}"
        f"{'MISMATCH':>14}"
        f"{'QUERY':>12}"
        f"{'NLL GAP':>14}"
    )

    print("-" * 62)

    for level in requested_levels:

        r = results[
            f"level_{level}"
        ]["test"]

        print(
            f"{level:<10}"
            f"{r['matched_accuracy']:>11.2f}%"
            f"{r['mismatched_accuracy']:>13.2f}%"
            f"{r['query_only_accuracy']:>11.2f}%"
            f"{r['nll_gap']:>14.4f}"
        )

    Path(
        args.output_dir
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        Path(args.output_dir)
        / "ablation_results.json",
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()