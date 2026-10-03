from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, List

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
# SYNTHETIC ASSOCIATIVE MEMORY DATA
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
# ORIGINAL PROJECT CONFIG
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
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA GENERATION
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

    episodes = []

    counter = start_id

    for _ in range(n_episodes):

        chosen_answers = rng.sample(
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

            answer = (
                chosen_answers[j]
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

            episode.append(
                {
                    "fact": fact,
                    "query": query,
                    "answer": answer,
                }
            )

        episodes.append(
            episode
        )

    return episodes


# ============================================================
# TOKEN HELPERS
# ============================================================

def get_single_token_answers(
    tokenizer,
):

    answers = {}

    for word in ANSWER_POOL:

        ids = tokenizer(
            " " + word,
            add_special_tokens=False,
        )["input_ids"]

        if len(ids) == 1:

            answers[word] = (
                ids[0]
            )

    if len(answers) < 8:

        raise RuntimeError(
            "Too few single-token answers."
        )

    return answers


def tokenize(
    tokenizer,
    texts,
    device,
):

    result = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
        add_special_tokens=False,
    )

    return (
        result["input_ids"].to(device),
        result["attention_mask"].to(device),
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

    load_result = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )

    print("Missing keys:")
    for key in load_result.missing_keys:
        print("  ", key)

    print("Unexpected keys:")
    for key in load_result.unexpected_keys:
        print("  ", key)

    model.to(device)
    model.eval()

    allowed_missing = {
        "address_encoder.0.weight",
        "address_encoder.0.bias",
        "address_encoder.1.weight",
    }

    unexpected_missing = (
        set(load_result.missing_keys)
        - allowed_missing
    )

    if unexpected_missing:
        raise RuntimeError(
            "Unexpected missing checkpoint keys: "
            f"{sorted(unexpected_missing)}"
        )

    if load_result.unexpected_keys:
        raise RuntimeError(
            "Unexpected checkpoint keys: "
            f"{load_result.unexpected_keys}"
        )

    for p in model.parameters():

        p.requires_grad = False

    return model


# ============================================================
# PRECOMPUTE FROZEN GPT-2 REPRESENTATIONS
# ============================================================

@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    texts,
    device,
    batch_size,
):

    result = []

    for start in range(
        0,
        len(texts),
        batch_size,
    ):

        batch = texts[
            start:start + batch_size
        ]

        ids, mask = tokenize(
            tokenizer,
            batch,
            device,
        )

        output = (
            model
            .backbone
            .transformer(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True,
                return_dict=True,
            )
        )

        final_hidden = output.last_hidden_state
        address_hidden = output.hidden_states[
            model.memory_config.address_hidden_index
        ]

        for i in range(
            final_hidden.size(0)
        ):

            length = int(
                mask[i]
                .sum()
                .item()
            )

            final_seq = (
                final_hidden[
                    i,
                    :length,
                    :
                ]
                .detach()
                .cpu()
                .half()
            )

            address_seq = (
                address_hidden[
                    i,
                    :length,
                    :
                ]
                .detach()
                .cpu()
                .half()
            )

            result.append(
                {
                    "final": final_seq,
                    "address": address_seq,
                }
            )

    return result


@torch.no_grad()
def precompute_episodes(
    model,
    tokenizer,
    episodes,
    answer_to_token,
    answer_to_class,
    device,
    batch_size,
):

    flat_facts = []
    flat_queries = []
    metadata = []

    for episode_id, episode in enumerate(
        episodes
    ):

        for position, item in enumerate(
            episode
        ):

            flat_facts.append(
                item["fact"]
            )

            flat_queries.append(
                item["query"]
            )

            metadata.append(
                {
                    "episode": (
                        episode_id
                    ),

                    "position": (
                        position
                    ),

                    "target_token": (
                        answer_to_token[
                            item["answer"]
                        ]
                    ),

                    "target_class": (
                        answer_to_class[
                            item["answer"]
                        ]
                    ),
                }
            )

    fact_hidden = encode_texts(
        model,
        tokenizer,
        flat_facts,
        device,
        batch_size,
    )

    query_hidden = encode_texts(
        model,
        tokenizer,
        flat_queries,
        device,
        batch_size,
    )

    cached = [
        []
        for _ in episodes
    ]

    for i, meta in enumerate(
        metadata
    ):

        fact_final = (
            fact_hidden[i]["final"]
            .float()
        )

        fact_address = (
            fact_hidden[i]["address"]
            .float()
        )

        summary = (
            fact_final.mean(
                dim=0
            )
        )

        address_summary = (
            fact_address.mean(
                dim=0
            )
        )

        cached[
            meta["episode"]
        ].append(
            {
                "fact_hidden": (
                    fact_hidden[i]["final"]
                ),

                "fact_address_hidden": (
                    fact_hidden[i]["address"]
                ),

                "summary": (
                    summary.half()
                ),

                "address_summary": (
                    address_summary.half()
                ),

                "query_hidden": (
                    query_hidden[i]["final"]
                ),

                "query_address_hidden": (
                    query_hidden[i]["address"]
                ),

                "target_token": (
                    meta[
                        "target_token"
                    ]
                ),

                "target_class": (
                    meta[
                        "target_class"
                    ]
                ),
            }
        )

    return cached


# ============================================================
# EPISODE DATASET
# ============================================================

class EpisodeDataset(Dataset):

    def __init__(
        self,
        cached,
    ):

        self.cached = cached

    def __len__(self):

        return len(
            self.cached
        )

    def __getitem__(
        self,
        index,
    ):

        return index


def collate_indices(
    batch,
):

    return list(batch)


# ============================================================
# PAD PRECOMPUTED HIDDEN STATES
# ============================================================

def get_hidden_batch(
    cached,
    indices,
    position,
    key,
    device,
    dtype=torch.float32,
):

    sequences = [
        cached[
            int(index)
        ][position][key]
        for index in indices
    ]

    max_length = max(
        x.size(0)
        for x in sequences
    )

    d_model = (
        sequences[0]
        .size(-1)
    )

    hidden = torch.zeros(
        len(sequences),
        max_length,
        d_model,
        dtype=dtype,
        device=device,
    )

    mask = torch.zeros(
        len(sequences),
        max_length,
        dtype=torch.long,
        device=device,
    )

    for i, sequence in enumerate(
        sequences
    ):

        sequence = (
            sequence
            .to(
                device=device,
                dtype=dtype,
            )
        )

        length = (
            sequence.size(0)
        )

        hidden[
            i,
            :length,
            :
        ] = sequence

        mask[
            i,
            :length
        ] = 1

    return (
        hidden,
        mask,
    )


def get_summary_batch(
    cached,
    indices,
    position,
    device,
):

    return torch.stack(
        [
            cached[
                int(index)
            ][position][
                "summary"
            ]
            for index in indices
        ],
        dim=0,
    ).to(
        device=device,
        dtype=torch.float32,
    )


def get_address_summary_batch(
    cached,
    indices,
    position,
    device,
):

    return torch.stack(
        [
            cached[
                int(index)
            ][position][
                "address_summary"
            ]
            for index in indices
        ],
        dim=0,
    ).to(
        device=device,
        dtype=torch.float32,
    )


def get_targets(
    cached,
    indices,
    position,
    device,
):

    target_tokens = torch.tensor(
        [
            cached[
                int(index)
            ][position][
                "target_token"
            ]
            for index in indices
        ],
        dtype=torch.long,
        device=device,
    )

    target_classes = torch.tensor(
        [
            cached[
                int(index)
            ][position][
                "target_class"
            ]
            for index in indices
        ],
        dtype=torch.long,
        device=device,
    )

    return (
        target_tokens,
        target_classes,
    )


# ============================================================
# STRAIGHT-THROUGH OCCUPANCY-AWARE ROUTER
#
# Uses original router MLP weights.
#
# NEW STATE SIGNAL:
#
#   free_mask = write_count == 0
#
# Occupied slots receive -inf.
#
# Forward:
#   one slot is selected.
#
# Backward:
#   gradient flows through soft probabilities.
#
# ============================================================

class OccupancyAwareRouter(
    nn.Module
):

    def __init__(
        self,
        original_router,
        temperature=0.7,
    ):

        super().__init__()

        if original_router is None:

            raise RuntimeError(
                "Original model does not "
                "contain a router."
            )

        self.base = copy.deepcopy(
            original_router
        )

        self.temperature = (
            temperature
        )

        for p in self.base.parameters():

            p.requires_grad = True

    def forward(
        self,
        summary,
        free_mask,
    ):

        # Original router MLP.
        logits = (
            self.base.router(
                self.base.query_norm(
                    summary
                )
            )
        )

        minimum = (
            torch.finfo(
                logits.dtype
            ).min
        )

        masked_logits = (
            logits.masked_fill(
                ~free_mask,
                minimum,
            )
        )

        soft = torch.softmax(
            masked_logits
            /
            self.temperature,
            dim=-1,
        )

        selected = (
            soft.argmax(
                dim=-1
            )
        )

        hard = F.one_hot(
            selected,
            num_classes=(
                soft.size(-1)
            ),
        ).to(
            soft.dtype
        )

        # Straight-through estimator.
        weights = (
            hard
            +
            soft
            -
            soft.detach()
        )

        return {
            "logits": logits,
            "masked_logits": (
                masked_logits
            ),
            "soft": soft,
            "hard": hard,
            "weights": weights,
            "selected": (
                selected
            ),
        }


# ============================================================
# REPAIRED INTEGRATED MEMORY MODEL
# ============================================================

class RepairedMemorySystem(
    nn.Module
):

    def __init__(
        self,
        original,
        router_temperature=0.7,
    ):

        super().__init__()

        self.d_model = (
            original.d_model
        )

        self.num_slots = (
            original.num_slots
        )

        # ====================================================
        # TRAINABLE MEMORY COMPONENTS
        # ====================================================

        self.writer = copy.deepcopy(
            original.writer
        )

        self.gate = copy.deepcopy(
            original.write_gate_module
        )

        self.reader = copy.deepcopy(
            original.reader
        )

        self.address_encoder = copy.deepcopy(
            original.address_encoder
        )

        # Re-enable training for repaired memory modules.
        for p in self.writer.parameters():
            p.requires_grad = True

        for p in self.gate.parameters():
            p.requires_grad = True

        for p in self.reader.parameters():
            p.requires_grad = True

        for p in self.address_encoder.parameters():
            p.requires_grad = True

        self.router = (
            OccupancyAwareRouter(
                original.router,
                temperature=(
                    router_temperature
                ),
            )
        )

        # ====================================================
        # FROZEN COMPONENTS
        # ====================================================

        self.memory_bank = (
            copy.deepcopy(
                original.memory_bank
            )
        )

        self.orthogonalizer = (
            copy.deepcopy(
                original.orthogonalizer
            )
        )

        self.lm_head = (
            copy.deepcopy(
                original
                .backbone
                .lm_head
            )
        )

        for module in (
            self.memory_bank,
            self.orthogonalizer,
            self.lm_head,
        ):

            for p in module.parameters():

                p.requires_grad = False

        # During training use dense reader attention.
        # This prevents top-k from blocking gradients
        # before addressing has been learned.
        self.reader.top_k = None

    def initialize_memory(
        self,
        batch_size,
        device,
    ):

        return (
            self.memory_bank
            .initialize(
                batch_size=(
                    batch_size
                ),
                device=device,
                dtype=torch.float32,
            )
        )

    # ========================================================
    # WRITE
    # ========================================================

    def write_fact(
        self,
        state,
        summary,
        address_summary,
        token_states,
        attention_mask,
    ):

        # --------------------------------------------
        # OCCUPANCY SIGNAL
        # --------------------------------------------

        free_mask = (
            state.write_count
            == 0
        )

        if (
            (~free_mask)
            .all(dim=-1)
            .any()
        ):

            raise RuntimeError(
                "No free memory slots remain."
            )

        # --------------------------------------------
        # ROUTER
        # --------------------------------------------

        route = self.router(
            summary,
            free_mask,
        )

        routing_weights = (
            route["weights"]
        )

        # --------------------------------------------
        # CANDIDATE WRITER
        # --------------------------------------------

        writer_output = (
            self.writer(
                summary=summary,
                memory_slots=(
                    state.slots
                ),
                token_states=(
                    token_states
                ),
                attention_mask=(
                    attention_mask
                ),
                routing_weights=(
                    routing_weights
                ),
            )
        )

        # --------------------------------------------
        # ORTHOGONAL UPDATE
        # --------------------------------------------

        orthogonal_output = (
            self.orthogonalizer(
                updates=(
                    writer_output
                    .deltas
                ),
                memory_slots=(
                    state.slots
                ),
            )
        )

        candidate = (
            state.slots
            +
            orthogonal_output
            .updates
        )

        # --------------------------------------------
        # VECTOR GATE
        # --------------------------------------------

        base_gate = self.gate(
            summary,
            slot_mask=(
                free_mask
            ),
        )

        # Only routed slot can be written.
        write_gate = (
            base_gate
            *
            routing_weights
            .unsqueeze(-1)
        )

        # Hard mask in forward pass.
        write_mask = (
            route["hard"]
            .unsqueeze(-1)
        )

        # --------------------------------------------
        # ADDRESS KEY
        # --------------------------------------------

        address_vector = (
            self.address_encoder(
                address_summary
            )
        )

        address_candidate = (
            address_vector
            .unsqueeze(1)
            .expand(
                -1,
                self.num_slots,
                -1,
            )
        )

        # --------------------------------------------
        # MEMORY BANK
        # --------------------------------------------

        new_state = (
            self.memory_bank(
                state=state,
                candidate=candidate,
                write_gate=(
                    write_gate
                ),
                write_mask=(
                    write_mask
                ),
                confidence=None,
                address_candidate=(
                    address_candidate
                ),
            )
        )

        # Gate value at selected slot.
        active_gate = (
            base_gate
            .squeeze(-1)
            .gather(
                1,
                route[
                    "selected"
                ]
                .unsqueeze(-1),
            )
            .squeeze(-1)
        )

        return (
            new_state,
            {
                "route": route,
                "writer": (
                    writer_output
                ),
                "orthogonal": (
                    orthogonal_output
                ),
                "base_gate": (
                    base_gate
                ),
                "write_gate": (
                    write_gate
                ),
                "active_gate": (
                    active_gate
                ),
            },
        )

    # ========================================================
    # READ
    # ========================================================

    def read_query(
        self,
        state,
        query_hidden,
        query_address_hidden,
        query_mask,
    ):

        memory_mask = (
            state.write_count
            > 0
        )

        output = (
            self.reader(
                hidden_states=(
                    query_hidden
                ),
                memory_slots=(
                    state.slots
                ),
                address_slots=(
                    state.address_slots
                ),
                address_hidden_states=(
                    query_address_hidden
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

        lengths = (
            query_mask
            .sum(dim=-1)
            .long()
            .sub(1)
            .clamp_min(0)
        )

        rows = torch.arange(
            query_hidden.size(0),
            device=(
                query_hidden.device
            ),
        )

        fused_last = (
            output
            .fused_hidden[
                rows,
                lengths,
                :
            ]
        )

        logits = (
            self.lm_head(
                fused_last
            )
        )

        return (
            logits,
            output,
            lengths,
        )


# ============================================================
# MISMATCHED EPISODE INDICES
# ============================================================

def build_wrong_indices(
    targets,
):

    batch_size = (
        targets.size(0)
    )

    result = []

    cpu_targets = (
        targets
        .detach()
        .cpu()
    )

    for i in range(
        batch_size
    ):

        chosen = None

        for offset in range(
            1,
            batch_size,
        ):

            j = (
                i + offset
            ) % batch_size

            if (
                cpu_targets[j]
                != cpu_targets[i]
            ):

                chosen = j
                break

        if chosen is None:

            chosen = (
                i + 1
            ) % batch_size

        result.append(
            chosen
        )

    return torch.tensor(
        result,
        dtype=torch.long,
        device=targets.device,
    )


# ============================================================
# READ ATTENTION SUPERVISION
# ============================================================

def reader_address_loss(
    read_output,
    query_mask,
    correct_slot,
):

    attention = (
        read_output
        .attention_weights
    )

    if attention is None:

        raise RuntimeError(
            "Reader attention was not returned."
        )

    # attention:
    # [B, H, T, N]

    last = (
        query_mask
        .sum(dim=-1)
        .long()
        .sub(1)
        .clamp_min(0)
    )

    rows = torch.arange(
        attention.size(0),
        device=attention.device,
    )

    # [B,H,N]
    final_attention = (
        attention[
            rows,
            :,
            last,
            :
        ]
    )

    # Average heads.
    final_attention = (
        final_attention
        .mean(dim=1)
    )

    target_probability = (
        final_attention
        .gather(
            1,
            correct_slot
            .unsqueeze(-1),
        )
        .squeeze(-1)
    )

    loss = (
        -torch.log(
            target_probability
            .clamp_min(
                1e-8
            )
        )
        .mean()
    )

    mean_target_attention = (
        target_probability
        .mean()
    )

    return (
        loss,
        mean_target_attention,
    )


# ============================================================
# ROUTER BALANCE LOSS
# ============================================================

def router_balance_loss(
    soft_weights,
):

    mean_usage = (
        soft_weights
        .mean(dim=0)
    )

    mean_usage = (
        mean_usage
        /
        mean_usage
        .sum()
        .clamp_min(
            1e-8
        )
    )

    target = torch.full_like(
        mean_usage,
        1.0
        /
        mean_usage.numel(),
    )

    return F.mse_loss(
        mean_usage,
        target,
    )


# ============================================================
# CANDIDATE-RESTRICTED ACCURACY
# ============================================================

def candidate_predictions(
    logits,
    answer_token_ids,
):

    candidate_logits = (
        logits[
            :,
            answer_token_ids
        ]
    )

    return (
        candidate_logits
        .argmax(dim=-1)
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch(
    model,
    loader,
    cached,
    optimizer,
    answer_token_ids,
    facts_per_episode,
    device,
    args,
):

    model.train()

    # Frozen modules should remain eval.
    model.memory_bank.eval()
    model.orthogonalizer.eval()
    model.lm_head.eval()

    total_loss = 0.0
    total_answer = 0.0
    total_address = 0.0
    total_mismatch = 0.0
    total_gate = 0.0
    total_balance = 0.0

    steps = 0

    for indices in loader:

        batch_size = (
            len(indices)
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        state = (
            model
            .initialize_memory(
                batch_size,
                device,
            )
        )

        assigned_slots = []

        gate_losses = []
        balance_losses = []
        gate_values = []

        # ====================================================
        # WRITE ALL FACTS
        # ====================================================

        for position in range(
            facts_per_episode
        ):

            fact_hidden, fact_mask = (
                get_hidden_batch(
                    cached,
                    indices,
                    position,
                    "fact_hidden",
                    device,
                )
            )

            summary = (
                get_summary_batch(
                    cached,
                    indices,
                    position,
                    device,
                )
            )

            address_summary = (
                get_address_summary_batch(
                    cached,
                    indices,
                    position,
                    device,
                )
            )

            state, info = (
                model.write_fact(
                    state=state,
                    summary=summary,
                    address_summary=address_summary,
                    token_states=(
                        fact_hidden
                    ),
                    attention_mask=(
                        fact_mask
                    ),
                )
            )

            assigned_slots.append(
                info[
                    "route"
                ][
                    "selected"
                ]
            )

            active_gate = (
                info[
                    "active_gate"
                ]
            )

            gate_values.append(
                active_gate.mean()
            )

            gate_losses.append(
                F.relu(
                    args.minimum_gate
                    -
                    active_gate
                ).mean()
            )

            balance_losses.append(
                router_balance_loss(
                    info[
                        "route"
                    ][
                        "soft"
                    ]
                )
            )

        # ====================================================
        # READ ALL FACTS
        # ====================================================

        answer_losses = []
        address_losses = []
        mismatch_losses = []
        target_attentions = []

        for position in range(
            facts_per_episode
        ):

            query_hidden, query_mask = (
                get_hidden_batch(
                    cached,
                    indices,
                    position,
                    "query_hidden",
                    device,
                )
            )

            query_address_hidden, _ = (
                get_hidden_batch(
                    cached,
                    indices,
                    position,
                    "query_address_hidden",
                    device,
                )
            )

            (
                target_tokens,
                target_classes,
            ) = get_targets(
                cached,
                indices,
                position,
                device,
            )

            logits, read_output, _ = (
                model.read_query(
                    state,
                    query_hidden,
                    query_address_hidden,
                    query_mask,
                )
            )

            matched_nll = (
                F.cross_entropy(
                    logits,
                    target_tokens,
                    reduction="none",
                )
            )

            answer_losses.append(
                matched_nll.mean()
            )

            # --------------------------------------------
            # Reader must attend to actual write slot.
            # --------------------------------------------

            address_loss, target_attention = (
                reader_address_loss(
                    read_output,
                    query_mask,
                    assigned_slots[
                        position
                    ],
                )
            )

            address_losses.append(
                address_loss
            )

            target_attentions.append(
                target_attention
            )

            # --------------------------------------------
            # Explicit memory-dependence supervision.
            #
            # Same query, wrong episode memory.
            # Correct memory should produce lower NLL.
            # --------------------------------------------

            wrong_indices = (
                build_wrong_indices(
                    target_tokens
                )
            )

            wrong_state_slots = (
                state.slots[
                    wrong_indices
                ]
            )

            wrong_address_slots = (
                state.address_slots[
                    wrong_indices
                ]
            )

            wrong_write_count = (
                state.write_count[
                    wrong_indices
                ]
            )

            wrong_memory_mask = (
                wrong_write_count
                > 0
            )

            wrong_output = (
                model.reader(
                    hidden_states=(
                        query_hidden
                    ),
                    memory_slots=(
                        wrong_state_slots
                    ),
                    address_slots=(
                        wrong_address_slots
                    ),
                    address_hidden_states=(
                        query_address_hidden
                    ),
                    attention_mask=(
                        query_mask
                    ),
                    memory_mask=(
                        wrong_memory_mask
                    ),
                    routing_prior=None,
                    memory_confidence=None,
                    return_attention=False,
                )
            )

            lengths = (
                query_mask
                .sum(dim=-1)
                .long()
                .sub(1)
                .clamp_min(0)
            )

            rows = torch.arange(
                batch_size,
                device=device,
            )

            wrong_last = (
                wrong_output
                .fused_hidden[
                    rows,
                    lengths,
                    :
                ]
            )

            wrong_logits = (
                model.lm_head(
                    wrong_last
                )
            )

            wrong_nll = (
                F.cross_entropy(
                    wrong_logits,
                    target_tokens,
                    reduction="none",
                )
            )

            mismatch_loss = F.relu(
                args.mismatch_margin
                +
                matched_nll
                -
                wrong_nll
            ).mean()

            mismatch_losses.append(
                mismatch_loss
            )

        # ====================================================
        # COMBINE LOSSES
        # ====================================================

        answer_loss = torch.stack(
            answer_losses
        ).mean()

        address_loss = torch.stack(
            address_losses
        ).mean()

        mismatch_loss = torch.stack(
            mismatch_losses
        ).mean()

        gate_loss = torch.stack(
            gate_losses
        ).mean()

        balance_loss = torch.stack(
            balance_losses
        ).mean()

        loss = (
            answer_loss

            +
            args.address_loss_weight
            * address_loss

            +
            args.mismatch_loss_weight
            * mismatch_loss

            +
            args.gate_floor_weight
            * gate_loss

            +
            args.router_balance_weight
            * balance_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [
                p
                for p in model.parameters()
                if p.requires_grad
            ],
            args.gradient_clip,
        )

        optimizer.step()

        total_loss += float(
            loss.item()
        )

        total_answer += float(
            answer_loss.item()
        )

        total_address += float(
            address_loss.item()
        )

        total_mismatch += float(
            mismatch_loss.item()
        )

        total_gate += float(
            torch.stack(
                gate_values
            ).mean().item()
        )

        total_balance += float(
            balance_loss.item()
        )

        steps += 1

    return {
        "loss": (
            total_loss
            / steps
        ),

        "answer_loss": (
            total_answer
            / steps
        ),

        "address_loss": (
            total_address
            / steps
        ),

        "mismatch_loss": (
            total_mismatch
            / steps
        ),

        "gate_mean": (
            total_gate
            / steps
        ),

        "router_balance_loss": (
            total_balance
            / steps
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    cached,
    answer_token_ids,
    facts_per_episode,
    device,
    batch_size,
    reader_top_k=None,
):

    model.eval()

    old_top_k = (
        model.reader.top_k
    )

    model.reader.top_k = (
        reader_top_k
    )

    loader = DataLoader(
        EpisodeDataset(
            cached
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=(
            collate_indices
        ),
    )

    total = 0

    full_correct = 0
    candidate_correct = 0

    mismatch_full_correct = 0
    mismatch_candidate_correct = 0

    query_only_full = 0
    query_only_candidate = 0

    matched_nll_total = 0.0
    wrong_nll_total = 0.0

    target_attention_total = 0.0

    gate_total = 0.0
    gate_count = 0

    slot_counts = torch.zeros(
        model.num_slots,
        device=device,
    )

    collision_count = 0

    for indices in loader:

        B = len(indices)

        state = (
            model
            .initialize_memory(
                B,
                device,
            )
        )

        assigned_slots = []

        used_slots = [
            set()
            for _ in range(B)
        ]

        # ====================================================
        # WRITE
        # ====================================================

        for position in range(
            facts_per_episode
        ):

            fact_hidden, fact_mask = (
                get_hidden_batch(
                    cached,
                    indices,
                    position,
                    "fact_hidden",
                    device,
                )
            )

            summary = (
                get_summary_batch(
                    cached,
                    indices,
                    position,
                    device,
                )
            )

            address_summary = (
                get_address_summary_batch(
                    cached,
                    indices,
                    position,
                    device,
                )
            )

            state, info = (
                model.write_fact(
                    state,
                    summary,
                    address_summary,
                    fact_hidden,
                    fact_mask,
                )
            )

            selected = (
                info[
                    "route"
                ][
                    "selected"
                ]
            )

            assigned_slots.append(
                selected
            )

            gate_total += float(
                info[
                    "active_gate"
                ]
                .sum()
                .item()
            )

            gate_count += (
                selected.size(0)
            )

            for i, slot in enumerate(
                selected.tolist()
            ):

                if (
                    slot
                    in used_slots[i]
                ):

                    collision_count += 1

                used_slots[i].add(
                    slot
                )

                slot_counts[
                    slot
                ] += 1

        # ====================================================
        # READ
        # ====================================================

        for position in range(
            facts_per_episode
        ):

            query_hidden, query_mask = (
                get_hidden_batch(
                    cached,
                    indices,
                    position,
                    "query_hidden",
                    device,
                )
            )

            query_address_hidden, _ = (
                get_hidden_batch(
                    cached,
                    indices,
                    position,
                    "query_address_hidden",
                    device,
                )
            )

            (
                target_tokens,
                target_classes,
            ) = get_targets(
                cached,
                indices,
                position,
                device,
            )

            logits, read_output, lengths = (
                model.read_query(
                    state,
                    query_hidden,
                    query_address_hidden,
                    query_mask,
                )
            )

            # --------------------------------------------
            # MATCHED
            # --------------------------------------------

            full_pred = (
                logits.argmax(
                    dim=-1
                )
            )

            candidate_pred = (
                candidate_predictions(
                    logits,
                    answer_token_ids,
                )
            )

            full_correct += (
                full_pred
                .eq(
                    target_tokens
                )
                .sum()
                .item()
            )

            candidate_correct += (
                candidate_pred
                .eq(
                    target_classes
                )
                .sum()
                .item()
            )

            matched_nll = (
                F.cross_entropy(
                    logits,
                    target_tokens,
                    reduction="none",
                )
            )

            matched_nll_total += float(
                matched_nll
                .sum()
                .item()
            )

            # --------------------------------------------
            # READER ATTENTION
            # --------------------------------------------

            _, target_attention = (
                reader_address_loss(
                    read_output,
                    query_mask,
                    assigned_slots[
                        position
                    ],
                )
            )

            target_attention_total += float(
                target_attention
                .item()
                *
                B
            )

            # --------------------------------------------
            # WRONG MEMORY
            # --------------------------------------------

            wrong_indices = (
                build_wrong_indices(
                    target_tokens
                )
            )

            wrong_slots = (
                state.slots[
                    wrong_indices
                ]
            )

            wrong_address_slots = (
                state.address_slots[
                    wrong_indices
                ]
            )

            wrong_mask = (
                state.write_count[
                    wrong_indices
                ]
                > 0
            )

            wrong_output = (
                model.reader(
                    hidden_states=(
                        query_hidden
                    ),
                    memory_slots=(
                        wrong_slots
                    ),
                    address_slots=(
                        wrong_address_slots
                    ),
                    address_hidden_states=(
                        query_address_hidden
                    ),
                    attention_mask=(
                        query_mask
                    ),
                    memory_mask=(
                        wrong_mask
                    ),
                    routing_prior=None,
                    memory_confidence=None,
                    return_attention=False,
                )
            )

            rows = torch.arange(
                B,
                device=device,
            )

            wrong_last = (
                wrong_output
                .fused_hidden[
                    rows,
                    lengths,
                    :
                ]
            )

            wrong_logits = (
                model.lm_head(
                    wrong_last
                )
            )

            wrong_full_pred = (
                wrong_logits
                .argmax(dim=-1)
            )

            wrong_candidate_pred = (
                candidate_predictions(
                    wrong_logits,
                    answer_token_ids,
                )
            )

            mismatch_full_correct += (
                wrong_full_pred
                .eq(
                    target_tokens
                )
                .sum()
                .item()
            )

            mismatch_candidate_correct += (
                wrong_candidate_pred
                .eq(
                    target_classes
                )
                .sum()
                .item()
            )

            wrong_nll = (
                F.cross_entropy(
                    wrong_logits,
                    target_tokens,
                    reduction="none",
                )
            )

            wrong_nll_total += float(
                wrong_nll
                .sum()
                .item()
            )

            # --------------------------------------------
            # QUERY ONLY
            #
            # Frozen GPT-2 without memory.
            # --------------------------------------------

            rows = torch.arange(
                B,
                device=device,
            )

            query_last = (
                query_hidden[
                    rows,
                    lengths,
                    :
                ]
            )

            baseline_logits = (
                model.lm_head(
                    query_last
                )
            )

            baseline_full_pred = (
                baseline_logits
                .argmax(dim=-1)
            )

            baseline_candidate_pred = (
                candidate_predictions(
                    baseline_logits,
                    answer_token_ids,
                )
            )

            query_only_full += (
                baseline_full_pred
                .eq(
                    target_tokens
                )
                .sum()
                .item()
            )

            query_only_candidate += (
                baseline_candidate_pred
                .eq(
                    target_classes
                )
                .sum()
                .item()
            )

            total += B

    slot_distribution = (
        slot_counts
        /
        slot_counts.sum()
        .clamp_min(1.0)
    )

    model.reader.top_k = (
        old_top_k
    )

    return {
        "full_vocab_accuracy": (
            100
            * full_correct
            / total
        ),

        "candidate_accuracy": (
            100
            * candidate_correct
            / total
        ),

        "mismatched_full_accuracy": (
            100
            * mismatch_full_correct
            / total
        ),

        "mismatched_candidate_accuracy": (
            100
            * mismatch_candidate_correct
            / total
        ),

        "query_only_full_accuracy": (
            100
            * query_only_full
            / total
        ),

        "query_only_candidate_accuracy": (
            100
            * query_only_candidate
            / total
        ),

        "matched_nll": (
            matched_nll_total
            / total
        ),

        "mismatched_nll": (
            wrong_nll_total
            / total
        ),

        "nll_gap": (
            (
                wrong_nll_total
                -
                matched_nll_total
            )
            / total
        ),

        "target_slot_attention": (
            target_attention_total
            / total
        ),

        "mean_gate": (
            gate_total
            /
            max(
                gate_count,
                1,
            )
        ),

        "collision_rate": (
            100
            * collision_count
            /
            (
                len(cached)
                *
                facts_per_episode
            )
        ),

        "slot_distribution": (
            slot_distribution
            .detach()
            .cpu()
            .tolist()
        ),

        "unused_slot_fraction": float(
            (
                slot_counts
                == 0
            )
            .float()
            .mean()
            .item()
        ),
    }


# ============================================================
# PRINT EVALUATION
# ============================================================

def print_eval(
    name,
    metrics,
):

    print()
    print(
        "=" * 95
    )

    print(name)

    print(
        "=" * 95
    )

    print(
        f"FULL VOCAB ACC:       "
        f"{metrics['full_vocab_accuracy']:.2f}%"
    )

    print(
        f"CANDIDATE ACC:        "
        f"{metrics['candidate_accuracy']:.2f}%"
    )

    print(
        f"MISMATCH CANDIDATE:   "
        f"{metrics['mismatched_candidate_accuracy']:.2f}%"
    )

    print(
        f"QUERY-ONLY CANDIDATE: "
        f"{metrics['query_only_candidate_accuracy']:.2f}%"
    )

    print(
        f"MATCHED NLL:          "
        f"{metrics['matched_nll']:.4f}"
    )

    print(
        f"MISMATCHED NLL:       "
        f"{metrics['mismatched_nll']:.4f}"
    )

    print(
        f"NLL GAP:              "
        f"{metrics['nll_gap']:+.4f}"
    )

    print(
        f"TARGET SLOT ATTENTION:"
        f" {metrics['target_slot_attention']:.4f}"
    )

    print(
        f"MEAN ACTIVE GATE:     "
        f"{metrics['mean_gate']:.4f}"
    )

    print(
        f"COLLISION RATE:       "
        f"{metrics['collision_rate']:.2f}%"
    )

    print(
        f"UNUSED SLOT FRACTION: "
        f"{metrics['unused_slot_fraction']:.4f}"
    )

    print(
        "SLOT DISTRIBUTION:"
    )

    print(
        [
            round(x, 4)
            for x in metrics[
                "slot_distribution"
            ]
        ]
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
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--precompute-batch-size",
        type=int,
        default=128,
    )

    # --------------------------------------------
    # LEARNING RATES
    # --------------------------------------------

    parser.add_argument(
        "--writer-learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--gate-learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--reader-learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--router-learning-rate",
        type=float,
        default=1e-4,
    )

    # --------------------------------------------
    # EXPLICIT MEMORY SUPERVISION
    # --------------------------------------------

    parser.add_argument(
        "--address-loss-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--mismatch-loss-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--mismatch-margin",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--gate-floor-weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--minimum-gate",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--router-balance-weight",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--router-temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=5.0,
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
            "repaired_memory_integration"
        ),
    )

    args = parser.parse_args()

    if (
        args.facts_per_episode
        > 8
    ):

        raise ValueError(
            "facts-per-episode "
            "must be <= 8."
        )

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 100
    )

    print(
        "REPAIRED MEMORY — INTEGRATED TRAINING"
    )

    print(
        "=" * 100
    )

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
        "TRAINABLE:"
    )

    print(
        "  CandidateWriter"
    )

    print(
        "  VectorGate"
    )

    print(
        "  MemoryReader + fusion"
    )

    print(
        "  SlotRouter"
    )

    print(
        "  AddressEncoder"
    )

    print()

    print(
        "FROZEN:"
    )

    print(
        "  GPT-2"
    )

    print(
        "  LM head weights"
    )

    print(
        "  MemoryBank"
    )

    print(
        "  OrthogonalUpdate"
    )

    print()

    print(
        "REPAIR:"
    )

    print(
        "  write_count -> free-slot mask"
    )

    print(
        "  straight-through top-1 routing"
    )

    print(
        "  explicit answer-token loss"
    )

    print(
        "  reader slot-address supervision"
    )

    print(
        "  matched-vs-mismatched memory loss"
    )

    print(
        "  minimum gate-strength penalty"
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

    answer_to_token = (
        get_single_token_answers(
            tokenizer
        )
    )

    answers = list(
        answer_to_token.keys()
    )

    answer_to_class = {
        answer: index
        for index, answer
        in enumerate(
            answers
        )
    }

    answer_token_ids = torch.tensor(
        [
            answer_to_token[
                answer
            ]
            for answer in answers
        ],
        dtype=torch.long,
        device=device,
    )

    print()

    print(
        "Classes:",
        len(answers),
    )

    print(
        "Candidate chance:",
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

    # ========================================================
    # DATA
    # ========================================================

    train_episodes = build_episodes(
        args.train_episodes,
        args.facts_per_episode,
        answers,
        args.seed,
        0,
        "train",
    )

    validation_episodes = build_episodes(
        args.validation_episodes,
        args.facts_per_episode,
        answers,
        args.seed + 1000,
        100000,
        "eval",
    )

    test_episodes = build_episodes(
        args.test_episodes,
        args.facts_per_episode,
        answers,
        args.seed + 2000,
        200000,
        "eval",
    )

    print()

    print(
        "Precomputing train GPT-2 states..."
    )

    train_cache = (
        precompute_episodes(
            original,
            tokenizer,
            train_episodes,
            answer_to_token,
            answer_to_class,
            device,
            args.precompute_batch_size,
        )
    )

    print(
        "Precomputing validation GPT-2 states..."
    )

    validation_cache = (
        precompute_episodes(
            original,
            tokenizer,
            validation_episodes,
            answer_to_token,
            answer_to_class,
            device,
            args.precompute_batch_size,
        )
    )

    print(
        "Precomputing test GPT-2 states..."
    )

    test_cache = (
        precompute_episodes(
            original,
            tokenizer,
            test_episodes,
            answer_to_token,
            answer_to_class,
            device,
            args.precompute_batch_size,
        )
    )

    # GPT-2 itself no longer needed on GPU.
    del original.backbone.transformer

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    # ========================================================
    # MODEL
    # ========================================================

    model = (
        RepairedMemorySystem(
            original,
            router_temperature=(
                args.router_temperature
            ),
        )
        .to(device)
    )

    print()

    print(
        "Trainable parameter counts:"
    )

    print(
        "Writer:",
        f"{sum(p.numel() for p in model.writer.parameters() if p.requires_grad):,}"
    )

    print(
        "Gate:",
        f"{sum(p.numel() for p in model.gate.parameters() if p.requires_grad):,}"
    )

    print(
        "Reader:",
        f"{sum(p.numel() for p in model.reader.parameters() if p.requires_grad):,}"
    )

    print(
        "Router:",
        f"{sum(p.numel() for p in model.router.parameters() if p.requires_grad):,}"
    )

    print(
        "Address encoder:",
        f"{sum(p.numel() for p in model.address_encoder.parameters() if p.requires_grad):,}"
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    model.writer
                    .parameters()
                ),
                "lr": (
                    args.writer_learning_rate
                ),
            },

            {
                "params": (
                    model.gate
                    .parameters()
                ),
                "lr": (
                    args.gate_learning_rate
                ),
            },

            {
                "params": (
                    model.reader
                    .parameters()
                ),
                "lr": (
                    args.reader_learning_rate
                ),
            },

            {
                "params": (
                    model.router
                    .parameters()
                ),
                "lr": (
                    args.router_learning_rate
                ),
            },

            {
                "params": (
                    model.address_encoder
                    .parameters()
                ),
                "lr": (
                    args.reader_learning_rate
                ),
            },
        ],

        weight_decay=1e-4,
    )

    # ========================================================
    # LOADER
    # ========================================================

    train_loader = DataLoader(
        EpisodeDataset(
            train_cache
        ),
        batch_size=(
            args.batch_size
        ),
        shuffle=True,
        collate_fn=(
            collate_indices
        ),
    )

    # ========================================================
    # PRETRAIN EVALUATION
    # ========================================================

    print()

    pre = evaluate(
        model,
        validation_cache,
        answer_token_ids,
        args.facts_per_episode,
        device,
        args.batch_size,
        reader_top_k=None,
    )

    print_eval(
        "PRETRAIN VALIDATION",
        pre,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    best_candidate = -1.0
    best_full = -1.0
    best_state = None
    best_epoch = -1

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            cached=train_cache,
            optimizer=optimizer,
            answer_token_ids=(
                answer_token_ids
            ),
            facts_per_episode=(
                args.facts_per_episode
            ),
            device=device,
            args=args,
        )

        validation = evaluate(
            model,
            validation_cache,
            answer_token_ids,
            args.facts_per_episode,
            device,
            args.batch_size,
            reader_top_k=None,
        )

        print()

        print(
            f"EPOCH {epoch:02d}"
        )

        print(
            f"train loss="
            f"{train_metrics['loss']:.4f} | "
            f"answer="
            f"{train_metrics['answer_loss']:.4f} | "
            f"address="
            f"{train_metrics['address_loss']:.4f} | "
            f"mismatch="
            f"{train_metrics['mismatch_loss']:.4f} | "
            f"gate="
            f"{train_metrics['gate_mean']:.4f}"
        )

        print(
            f"VAL candidate="
            f"{validation['candidate_accuracy']:.2f}% | "
            f"full="
            f"{validation['full_vocab_accuracy']:.2f}% | "
            f"mismatch="
            f"{validation['mismatched_candidate_accuracy']:.2f}% | "
            f"query="
            f"{validation['query_only_candidate_accuracy']:.2f}% | "
            f"gap="
            f"{validation['nll_gap']:+.4f} | "
            f"read_attn="
            f"{validation['target_slot_attention']:.4f} | "
            f"gate="
            f"{validation['mean_gate']:.4f} | "
            f"collision="
            f"{validation['collision_rate']:.2f}%"
        )

        history.append(
            {
                "epoch": epoch,
                "train": (
                    train_metrics
                ),
                "validation": (
                    validation
                ),
            }
        )

        candidate = (
            validation[
                "candidate_accuracy"
            ]
        )

        full = (
            validation[
                "full_vocab_accuracy"
            ]
        )

        better = False

        if candidate > best_candidate:

            better = True

        elif (
            candidate
            == best_candidate
            and
            full
            > best_full
        ):

            better = True

        if better:

            best_candidate = candidate
            best_full = full
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            print(
                "Saved best integrated checkpoint."
            )

    # ========================================================
    # LOAD BEST
    # ========================================================

    model.load_state_dict(
        best_state
    )

    # ========================================================
    # TEST — DENSE READER
    # ========================================================

    dense_test = evaluate(
        model,
        test_cache,
        answer_token_ids,
        args.facts_per_episode,
        device,
        args.batch_size,
        reader_top_k=None,
    )

    print_eval(
        f"FINAL TEST — DENSE READER "
        f"(BEST EPOCH {best_epoch})",
        dense_test,
    )

    # ========================================================
    # TEST — ORIGINAL TOP-K=3
    # ========================================================

    sparse_test = evaluate(
        model,
        test_cache,
        answer_token_ids,
        args.facts_per_episode,
        device,
        args.batch_size,
        reader_top_k=3,
    )

    print_eval(
        "FINAL TEST — READER TOP-K=3",
        sparse_test,
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print()

    print(
        "=" * 100
    )

    print(
        "FINAL INTEGRATION INTERPRETATION"
    )

    print(
        "=" * 100
    )

    if (
        dense_test[
            "candidate_accuracy"
        ] >= 80

        and

        dense_test[
            "collision_rate"
        ] < 1

        and

        dense_test[
            "target_slot_attention"
        ] >= 0.6
    ):

        print(
            "INTEGRATION PASS."
        )

        print(
            "Writer + Gate + Router + MemoryBank + "
            "MemoryReader are functioning together."
        )

        print(
            "The next step is to transfer this exact "
            "training logic into the main model/train.py."
        )

    elif (
        dense_test[
            "collision_rate"
        ] >= 1
    ):

        print(
            "ROUTING / OCCUPANCY FAILURE."
        )

    elif (
        dense_test[
            "target_slot_attention"
        ] < 0.4
    ):

        print(
            "READER ADDRESSING REMAINS WEAK."
        )

    elif (
        dense_test[
            "candidate_accuracy"
        ] < 50
    ):

        print(
            "END-TO-END MEMORY USE IS STILL WEAK."
        )

        print(
            "Inspect writer/gate/read gradients before "
            "making another architecture change."
        )

    else:

        print(
            "PARTIAL INTEGRATION PASS."
        )

        print(
            "Inspect candidate accuracy, NLL gap, "
            "reader attention, and gate strength."
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

    checkpoint_path = (
        output_dir
        / "checkpoint_best.pt"
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),

            "best_epoch": (
                best_epoch
            ),

            "best_validation_candidate_accuracy": (
                best_candidate
            ),

            "best_validation_full_accuracy": (
                best_full
            ),

            "arguments": (
                vars(args)
            ),
        },
        checkpoint_path,
    )

    results = {
        "best_epoch": (
            best_epoch
        ),

        "dense_test": (
            dense_test
        ),

        "topk3_test": (
            sparse_test
        ),

        "history": (
            history
        ),

        "arguments": (
            vars(args)
        ),
    }

    with open(
        output_dir
        / "integration_results.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print()

    print(
        "Saved checkpoint:",
        checkpoint_path,
    )

    print(
        "Saved results:",
        output_dir
        / "integration_results.json",
    )


if __name__ == "__main__":

    main()