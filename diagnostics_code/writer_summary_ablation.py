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
# SINGLE TOKEN ANSWERS
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
            "Not enough single-token answer words."
        )

    return usable


# ============================================================
# SYNTHETIC EXAMPLES
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

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        add_special_tokens=False,
    )

    return (
        encoded["input_ids"].to(
            device
        ),

        encoded[
            "attention_mask"
        ].to(device),
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
# PRECOMPUTE FROZEN GPT-2 REPRESENTATIONS
#
# Save:
#   masked mean fact summary
#   all fact token states
#   query last-token hidden state
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
    queries = []
    labels = []

    token_states = []
    token_masks = []

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

        # ====================================================
        # FACT
        # ====================================================

        ids, mask = tokenize(
            tokenizer,
            facts,
            device,
        )

        out = (
            original
            .backbone
            .transformer(
                input_ids=ids,
                attention_mask=mask,
                return_dict=True,
            )
        )

        hidden = (
            out.last_hidden_state
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
            weights
            .sum(dim=1)
            .clamp_min(1.0)
        )

        # Store each fact token sequence
        # individually to avoid massive padding.

        for j in range(
            hidden.size(0)
        ):

            length = int(
                mask[j]
                .sum()
                .item()
            )

            token_states.append(
                hidden[
                    j,
                    :length,
                    :
                ]
                .detach()
                .cpu()
            )

            token_masks.append(
                mask[
                    j,
                    :length
                ]
                .detach()
                .cpu()
            )

        # ====================================================
        # QUERY
        # ====================================================

        q_ids, q_mask = tokenize(
            tokenizer,
            questions,
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

        last = (
            q_mask.sum(dim=1)
            - 1
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
            summary
            .detach()
            .cpu()
        )

        queries.append(
            query
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

        "query": torch.cat(
            queries,
            dim=0,
        ),

        "labels": torch.cat(
            labels,
            dim=0,
        ),

        "token_states": (
            token_states
        ),

        "token_masks": (
            token_masks
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
                "query"
            ][i],
            self.data[
                "labels"
            ][i],
        )


# ============================================================
# BUILD PADDED TOKEN BATCH
# ============================================================

def build_token_batch(
    data,
    indices,
    device,
):

    states = [
        data[
            "token_states"
        ][int(i)]
        for i in indices
    ]

    masks = [
        data[
            "token_masks"
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

        length = state.size(0)

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
# SIMPLE DIAGNOSTIC READER
# ============================================================

class SimpleReader(nn.Module):

    def __init__(
        self,
        d_model,
        num_classes,
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
# THREE-WAY WRITER ABLATION
#
# ORIGINAL_WRITER:
#
#   CandidateWriter
#       ↓
#   OrthogonalUpdate
#       ↓
#   Gate = 1
#
#
# DIRECT_SUMMARY:
#
#   masked mean
#       ↓
#   LN + Linear
#       ↓
#   OrthogonalUpdate
#       ↓
#   Gate = 1
#
#
# WRITER_PLUS_SUMMARY:
#
#   CandidateWriter delta
#         +
#   alpha * Linear(summary)
#         ↓
#   OrthogonalUpdate
#         ↓
#   Gate = 1
#
# ============================================================

class WriterAblationModel(
    nn.Module
):

    VALID_VARIANTS = {
        "original_writer",
        "direct_summary",
        "writer_plus_summary",
    }

    def __init__(
        self,
        original,
        num_classes,
        variant,
        forced_slot,
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

        self.variant = variant

        self.d_model = (
            original.d_model
        )

        self.num_slots = (
            original.num_slots
        )

        self.forced_slot = (
            forced_slot
        )

        # ====================================================
        # ORIGINAL COMPONENTS
        # ====================================================

        self.writer = copy.deepcopy(
            original.writer
        )

        self.orthogonalizer = (
            copy.deepcopy(
                original.orthogonalizer
            )
        )

        self.memory_bank = (
            copy.deepcopy(
                original.memory_bank
            )
        )

        # Frozen orthogonalizer
        for p in (
            self.orthogonalizer
            .parameters()
        ):

            p.requires_grad = False

        # Frozen MemoryBank
        for p in (
            self.memory_bank
            .parameters()
        ):

            p.requires_grad = False

        # ====================================================
        # DIRECT SUMMARY PROJECTION
        # ====================================================

        self.summary_projection = (
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

        # ====================================================
        # LEARNABLE RESIDUAL SCALE
        #
        # sigmoid(0) = 0.5 initially
        # ====================================================

        self.residual_logit = (
            nn.Parameter(
                torch.tensor(
                    0.0
                )
            )
        )

        # ====================================================
        # ENABLE ONLY WHAT EACH VARIANT NEEDS
        # ====================================================

        if (
            variant
            == "original_writer"
        ):

            for p in (
                self.writer
                .parameters()
            ):

                p.requires_grad = True

            for p in (
                self.summary_projection
                .parameters()
            ):

                p.requires_grad = False

            self.residual_logit.requires_grad = (
                False
            )

        elif (
            variant
            == "direct_summary"
        ):

            for p in (
                self.writer
                .parameters()
            ):

                p.requires_grad = False

            for p in (
                self.summary_projection
                .parameters()
            ):

                p.requires_grad = True

            self.residual_logit.requires_grad = (
                False
            )

        elif (
            variant
            == "writer_plus_summary"
        ):

            for p in (
                self.writer
                .parameters()
            ):

                p.requires_grad = True

            for p in (
                self.summary_projection
                .parameters()
            ):

                p.requires_grad = True

            self.residual_logit.requires_grad = (
                True
            )

        # ====================================================
        # SIMPLE READER
        # ====================================================

        self.reader = SimpleReader(
            self.d_model,
            num_classes,
        )

    # ========================================================
    # FORCE ONE SLOT
    # ========================================================

    def slot_mask(
        self,
        batch_size,
        device,
    ):

        mask = torch.zeros(
            batch_size,
            self.num_slots,
            dtype=torch.bool,
            device=device,
        )

        mask[
            :,
            self.forced_slot
        ] = True

        return mask

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        summary,
        query,
        token_states,
        attention_mask,
    ):

        batch_size = (
            summary.size(0)
        )

        # ====================================================
        # INITIAL MEMORY
        # ====================================================

        state = (
            self.memory_bank
            .initialize(
                batch_size=(
                    batch_size
                ),
                device=(
                    summary.device
                ),
                dtype=(
                    summary.dtype
                ),
            )
        )

        # ====================================================
        # FORCE ROUTING TO ONE SLOT
        # ====================================================

        routing_weights = (
            torch.zeros(
                batch_size,
                self.num_slots,
                device=(
                    summary.device
                ),
                dtype=(
                    summary.dtype
                ),
            )
        )

        routing_weights[
            :,
            self.forced_slot
        ] = 1.0

        slot_mask = (
            self.slot_mask(
                batch_size,
                summary.device,
            )
        )

        # ====================================================
        # SUMMARY RESIDUAL VALUE
        # ====================================================

        summary_value = (
            self.summary_projection(
                summary
            )
        )

        # ====================================================
        # ORIGINAL WRITER OUTPUT
        #
        # Only needed by writer variants.
        # ====================================================

        writer_output = None

        if self.variant in {
            "original_writer",
            "writer_plus_summary",
        }:

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

        # ====================================================
        # BUILD RAW UPDATE
        # ====================================================

        raw_updates = (
            torch.zeros_like(
                state.slots
            )
        )

        # ----------------------------------------------------
        # A) ORIGINAL WRITER
        # ----------------------------------------------------

        if (
            self.variant
            == "original_writer"
        ):

            raw_updates = (
                writer_output.deltas
            )

            residual_alpha = (
                torch.tensor(
                    0.0,
                    device=summary.device,
                    dtype=summary.dtype,
                )
            )

            source_candidate = (
                state.slots
                + raw_updates
            )

        # ----------------------------------------------------
        # B) DIRECT SUMMARY
        #
        # Treat summary projection as desired slot value.
        # Therefore:
        #
        # update = value - old_slot
        # ----------------------------------------------------

        elif (
            self.variant
            == "direct_summary"
        ):

            raw_updates[
                :,
                self.forced_slot,
                :
            ] = (
                summary_value
                -
                state.slots[
                    :,
                    self.forced_slot,
                    :
                ]
            )

            residual_alpha = (
                torch.tensor(
                    1.0,
                    device=summary.device,
                    dtype=summary.dtype,
                )
            )

            source_candidate = (
                state.slots
                + raw_updates
            )

        # ----------------------------------------------------
        # C) WRITER + DIRECT SUMMARY RESIDUAL
        #
        # update =
        #   writer_delta
        #   +
        #   alpha * summary_residual
        #
        # summary_residual is relative to old slot.
        # ----------------------------------------------------

        elif (
            self.variant
            == "writer_plus_summary"
        ):

            residual_alpha = (
                torch.sigmoid(
                    self.residual_logit
                )
            )

            summary_residual = (
                summary_value
                -
                state.slots[
                    :,
                    self.forced_slot,
                    :
                ]
            )

            raw_updates = (
                writer_output.deltas
                .clone()
            )

            raw_updates[
                :,
                self.forced_slot,
                :
            ] = (
                raw_updates[
                    :,
                    self.forced_slot,
                    :
                ]
                +
                residual_alpha
                * summary_residual
            )

            source_candidate = (
                state.slots
                + raw_updates
            )

        else:

            raise RuntimeError(
                "Invalid variant."
            )

        # ====================================================
        # ORIGINAL ORTHOGONAL UPDATE
        # ====================================================

        ortho_output = (
            self.orthogonalizer(
                updates=(
                    raw_updates
                ),
                memory_slots=(
                    state.slots
                ),
            )
        )

        candidate = (
            state.slots
            + ortho_output.updates
        )

        # ====================================================
        # GATE = 1 EXACTLY
        # ====================================================

        write_gate = torch.zeros(
            batch_size,
            self.num_slots,
            1,
            device=summary.device,
            dtype=summary.dtype,
        )

        write_gate[
            :,
            self.forced_slot,
            0
        ] = 1.0

        # ====================================================
        # ORIGINAL MEMORY BANK
        # ====================================================

        new_state = (
            self.memory_bank(
                state=state,
                candidate=candidate,
                write_gate=(
                    write_gate
                ),
                write_mask=(
                    slot_mask
                    .unsqueeze(-1)
                ),
                confidence=None,
            )
        )

        stored = (
            new_state.slots[
                :,
                self.forced_slot,
                :
            ]
        )

        # ====================================================
        # SIMPLE READER
        # ====================================================

        logits = (
            self.reader(
                query,
                stored,
            )
        )

        return {
            "logits": logits,

            "stored": stored,

            "summary_value": (
                summary_value
            ),

            "source_candidate": (
                source_candidate[
                    :,
                    self.forced_slot,
                    :
                ]
            ),

            "raw_update": (
                raw_updates[
                    :,
                    self.forced_slot,
                    :
                ]
            ),

            "post_ortho": (
                candidate[
                    :,
                    self.forced_slot,
                    :
                ]
            ),

            "alpha": (
                residual_alpha
            ),
        }


# ============================================================
# MISMATCH HELPERS
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

        choices = (
            torch.nonzero(
                labels != labels[i],
                as_tuple=False,
            )
            .flatten()
        )

        # Extremely unlikely with batch size 128,
        # but keep robust fallback.

        if choices.numel() == 0:

            result[i] = (
                (i + 1)
                % labels.size(0)
            )

        else:

            index = torch.randint(
                0,
                choices.numel(),
                (1,),
                device=labels.device,
            )

            result[i] = (
                choices[index]
            )

    return result


def deterministic_mismatch(
    labels,
):

    labels_cpu = (
        labels.cpu()
    )

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
                labels_cpu[j]
                != labels_cpu[i]
            ):

                chosen = j
                break

        if chosen is None:

            raise RuntimeError(
                "Could not create "
                "different-label mismatch."
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
# GEOMETRY
# ============================================================

@torch.no_grad()
def geometry(x):

    x = (
        x.detach()
        .float()
        .cpu()
    )

    if x.size(0) > 500:

        ids = torch.linspace(
            0,
            x.size(0) - 1,
            500,
        ).long()

        x = x[ids]

    mean_norm = (
        x.norm(
            dim=-1
        ).mean()
    )

    distances = torch.cdist(
        x,
        x,
    )

    normalized = F.normalize(
        x,
        dim=-1,
    )

    cosine = (
        normalized
        @ normalized.T
    )

    n = x.size(0)

    mask = torch.triu(
        torch.ones(
            n,
            n,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    mean_l2 = (
        distances[
            mask
        ].mean()
    )

    mean_cosine = (
        cosine[
            mask
        ].mean()
    )

    return {
        "mean_norm": float(
            mean_norm.item()
        ),

        "mean_pairwise_l2": float(
            mean_l2.item()
        ),

        "relative_l2": float(
            (
                mean_l2
                /
                (
                    mean_norm
                    + 1e-8
                )
            ).item()
        ),

        "mean_cosine": float(
            mean_cosine.item()
        ),
    }


# ============================================================
# COLLECT OUTPUTS
# ============================================================

@torch.no_grad()
def collect_outputs(
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

    results = {
        "logits": [],
        "stored": [],
        "summary_value": [],
        "source_candidate": [],
        "raw_update": [],
        "post_ortho": [],
        "query": [],
        "labels": [],
    }

    alpha_values = []

    for (
        indices,
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

        (
            token_states,
            attention_mask,
        ) = build_token_batch(
            data,
            indices,
            device,
        )

        output = model(
            summary=summary,
            query=query,
            token_states=(
                token_states
            ),
            attention_mask=(
                attention_mask
            ),
        )

        for key in [
            "logits",
            "stored",
            "summary_value",
            "source_candidate",
            "raw_update",
            "post_ortho",
        ]:

            results[key].append(
                output[key]
            )

        results[
            "query"
        ].append(
            query
        )

        results[
            "labels"
        ].append(
            labels
        )

        alpha_values.append(
            float(
                output["alpha"]
                .detach()
                .float()
                .mean()
                .item()
            )
        )

    for key in results:

        results[key] = (
            torch.cat(
                results[key],
                dim=0,
            )
        )

    results["alpha"] = (
        sum(alpha_values)
        /
        len(alpha_values)
    )

    return results


# ============================================================
# EVALUATE
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    data,
    device,
    batch_size,
):

    output = collect_outputs(
        model,
        data,
        device,
        batch_size,
    )

    logits = (
        output["logits"]
    )

    stored = (
        output["stored"]
    )

    query = (
        output["query"]
    )

    labels = (
        output["labels"]
    )

    # ========================================================
    # MATCHED
    # ========================================================

    matched_losses = (
        F.cross_entropy(
            logits,
            labels,
            reduction="none",
        )
    )

    matched_accuracy = (
        logits
        .argmax(dim=-1)
        .eq(labels)
        .float()
        .mean()
        .item()
        * 100
    )

    # ========================================================
    # MISMATCHED MEMORY
    # ========================================================

    mismatch_indices = (
        deterministic_mismatch(
            labels
        )
    )

    mismatched_stored = (
        stored[
            mismatch_indices
        ]
    )

    mismatched_logits = (
        model.reader(
            query,
            mismatched_stored,
        )
    )

    mismatched_losses = (
        F.cross_entropy(
            mismatched_logits,
            labels,
            reduction="none",
        )
    )

    mismatched_accuracy = (
        mismatched_logits
        .argmax(dim=-1)
        .eq(labels)
        .float()
        .mean()
        .item()
        * 100
    )

    # ========================================================
    # QUERY ONLY
    # ========================================================

    zero_memory = (
        torch.zeros_like(
            stored
        )
    )

    query_logits = (
        model.reader(
            query,
            zero_memory,
        )
    )

    query_accuracy = (
        query_logits
        .argmax(dim=-1)
        .eq(labels)
        .float()
        .mean()
        .item()
        * 100
    )

    # ========================================================
    # NLL GAP
    # ========================================================

    gap = (
        mismatched_losses
        - matched_losses
    )

    return {
        "matched_accuracy": float(
            matched_accuracy
        ),

        "mismatched_accuracy": float(
            mismatched_accuracy
        ),

        "query_only_accuracy": float(
            query_accuracy
        ),

        "matched_loss": float(
            matched_losses
            .mean()
            .item()
        ),

        "mismatched_loss": float(
            mismatched_losses
            .mean()
            .item()
        ),

        "nll_gap": float(
            gap.mean().item()
        ),

        "positive_gap_fraction": float(
            (gap > 0)
            .float()
            .mean()
            .item()
            * 100
        ),

        "alpha": float(
            output["alpha"]
        ),

        "summary_geometry": (
            geometry(
                output[
                    "summary_value"
                ]
            )
        ),

        "source_geometry": (
            geometry(
                output[
                    "source_candidate"
                ]
            )
        ),

        "update_geometry": (
            geometry(
                output[
                    "raw_update"
                ]
            )
        ),

        "post_ortho_geometry": (
            geometry(
                output[
                    "post_ortho"
                ]
            )
        ),

        "stored_geometry": (
            geometry(
                output[
                    "stored"
                ]
            )
        ),
    }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch(
    model,
    loader,
    train_data,
    optimizer,
    device,
    mismatch_margin,
    mismatch_weight,
):

    model.train()

    # Frozen modules stay deterministic.
    model.memory_bank.eval()
    model.orthogonalizer.eval()

    # Writer should be eval only when frozen.
    if not any(
        p.requires_grad
        for p in (
            model.writer
            .parameters()
        )
    ):

        model.writer.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_rank = 0.0
    total_examples = 0

    for (
        indices,
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

        (
            token_states,
            attention_mask,
        ) = build_token_batch(
            train_data,
            indices,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            summary=summary,
            query=query,
            token_states=(
                token_states
            ),
            attention_mask=(
                attention_mask
            ),
        )

        logits = (
            output["logits"]
        )

        stored = (
            output["stored"]
        )

        # ====================================================
        # MATCHED CE
        # ====================================================

        matched_losses = (
            F.cross_entropy(
                logits,
                labels,
                reduction="none",
            )
        )

        ce_loss = (
            matched_losses.mean()
        )

        # ====================================================
        # MISMATCHED MEMORY
        # ====================================================

        wrong_indices = (
            random_mismatch(
                labels
            )
        )

        wrong_stored = (
            stored[
                wrong_indices
            ]
        )

        wrong_logits = (
            model.reader(
                query,
                wrong_stored,
            )
        )

        wrong_losses = (
            F.cross_entropy(
                wrong_logits,
                labels,
                reduction="none",
            )
        )

        rank_loss = F.relu(
            mismatch_margin
            + matched_losses
            - wrong_losses
        ).mean()

        total = (
            ce_loss
            +
            mismatch_weight
            * rank_loss
        )

        total.backward()

        trainable_parameters = [
            p
            for p in (
                model.parameters()
            )
            if p.requires_grad
        ]

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=5.0,
        )

        optimizer.step()

        batch_size = (
            labels.size(0)
        )

        total_examples += (
            batch_size
        )

        total_loss += (
            float(
                total.item()
            )
            * batch_size
        )

        total_ce += (
            float(
                ce_loss.item()
            )
            * batch_size
        )

        total_rank += (
            float(
                rank_loss.item()
            )
            * batch_size
        )

    return {
        "loss": (
            total_loss
            / total_examples
        ),

        "ce": (
            total_ce
            / total_examples
        ),

        "rank": (
            total_rank
            / total_examples
        ),
    }


# ============================================================
# TRAIN ONE VARIANT
# ============================================================

def train_variant(
    variant,
    original,
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
        f"VARIANT: {variant.upper()}"
    )
    print("#" * 100)

    # Independent initialization / shuffle.
    set_seed(
        args.seed
    )

    model = WriterAblationModel(
        original=original,
        num_classes=num_classes,
        variant=variant,
        forced_slot=(
            args.forced_slot
        ),
    ).to(device)

    # ========================================================
    # PARAMETER GROUPS
    # ========================================================

    parameter_groups = []

    reader_params = [
        p
        for p in (
            model.reader
            .parameters()
        )
        if p.requires_grad
    ]

    if reader_params:

        parameter_groups.append(
            {
                "params": (
                    reader_params
                ),
                "lr": (
                    args.reader_learning_rate
                ),
            }
        )

    writer_params = [
        p
        for p in (
            model.writer
            .parameters()
        )
        if p.requires_grad
    ]

    if writer_params:

        parameter_groups.append(
            {
                "params": (
                    writer_params
                ),
                "lr": (
                    args.writer_learning_rate
                ),
            }
        )

    summary_params = [
        p
        for p in (
            model.summary_projection
            .parameters()
        )
        if p.requires_grad
    ]

    if summary_params:

        parameter_groups.append(
            {
                "params": (
                    summary_params
                ),
                "lr": (
                    args.summary_learning_rate
                ),
            }
        )

    if (
        model.residual_logit
        .requires_grad
    ):

        parameter_groups.append(
            {
                "params": [
                    model.residual_logit
                ],
                "lr": (
                    args.residual_learning_rate
                ),
            }
        )

    optimizer = (
        torch.optim.AdamW(
            parameter_groups,
            weight_decay=1e-4,
        )
    )

    print(
        "Writer trainable params:",
        f"{sum(p.numel() for p in writer_params):,}"
    )

    print(
        "Summary projection params:",
        f"{sum(p.numel() for p in summary_params):,}"
    )

    print(
        "Reader params:",
        f"{sum(p.numel() for p in reader_params):,}"
    )

    print(
        "Residual alpha trainable:",
        model.residual_logit
        .requires_grad,
    )

    # ========================================================
    # DATA
    # ========================================================

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        RepDataset(
            train_data
        ),
        batch_size=(
            args.batch_size
        ),
        shuffle=True,
        generator=generator,
    )

    # ========================================================
    # PRETRAIN
    # ========================================================

    pre = evaluate(
        model,
        validation_data,
        device,
        args.batch_size,
    )

    print()
    print(
        "PRETRAIN"
    )

    print(
        f"match={pre['matched_accuracy']:.2f}% | "
        f"mismatch={pre['mismatched_accuracy']:.2f}% | "
        f"query={pre['query_only_accuracy']:.2f}% | "
        f"gap={pre['nll_gap']:+.6f} | "
        f"source_relL2="
        f"{pre['source_geometry']['relative_l2']:.4f} | "
        f"stored_relL2="
        f"{pre['stored_geometry']['relative_l2']:.4f} | "
        f"alpha={pre['alpha']:.4f}"
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = (
        Path(
            args.output_dir
        )
        / variant
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_accuracy = (
        -float("inf")
    )

    best_gap = (
        -float("inf")
    )

    best_epoch = -1

    history = []

    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_metrics = (
            train_epoch(
                model=model,
                loader=train_loader,
                train_data=train_data,
                optimizer=optimizer,
                device=device,
                mismatch_margin=(
                    args.mismatch_margin
                ),
                mismatch_weight=(
                    args.mismatch_weight
                ),
            )
        )

        validation = evaluate(
            model,
            validation_data,
            device,
            args.batch_size,
        )

        print()
        print(
            f"EPOCH {epoch:02d} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"match={validation['matched_accuracy']:.2f}% | "
            f"mismatch={validation['mismatched_accuracy']:.2f}% | "
            f"query={validation['query_only_accuracy']:.2f}% | "
            f"gap={validation['nll_gap']:+.4f} | "
            f"source_relL2="
            f"{validation['source_geometry']['relative_l2']:.4f} | "
            f"stored_relL2="
            f"{validation['stored_geometry']['relative_l2']:.4f} | "
            f"alpha="
            f"{validation['alpha']:.4f}"
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

        # ====================================================
        # CHECKPOINT SELECTION
        #
        # Primary criterion:
        #   validation matched accuracy
        #
        # Tie break:
        #   NLL gap
        #
        # Better than selecting only the ever-growing
        # mismatch NLL gap.
        # ====================================================

        current_accuracy = (
            validation[
                "matched_accuracy"
            ]
        )

        current_gap = (
            validation[
                "nll_gap"
            ]
        )

        better = False

        if (
            current_accuracy
            > best_accuracy
        ):

            better = True

        elif (
            current_accuracy
            == best_accuracy
            and
            current_gap > best_gap
        ):

            better = True

        if better:

            best_accuracy = (
                current_accuracy
            )

            best_gap = (
                current_gap
            )

            best_epoch = epoch

            torch.save(
                {
                    "variant": (
                        variant
                    ),

                    "epoch": (
                        epoch
                    ),

                    "model_state_dict": (
                        model.state_dict()
                    ),

                    "validation": (
                        validation
                    ),

                    "arguments": (
                        vars(args)
                    ),
                },
                output_dir
                / "checkpoint_best.pt",
            )

            print(
                "Saved best checkpoint."
            )

    # ========================================================
    # LOAD BEST
    # ========================================================

    checkpoint = torch.load(
        output_dir
        / "checkpoint_best.pt",
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    test = evaluate(
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
        f"{test['matched_accuracy']:.2f}%"
    )

    print(
        f"MISMATCHED:    "
        f"{test['mismatched_accuracy']:.2f}%"
    )

    print(
        f"QUERY ONLY:    "
        f"{test['query_only_accuracy']:.2f}%"
    )

    print(
        f"NLL GAP:       "
        f"{test['nll_gap']:+.6f}"
    )

    print(
        f"POSITIVE GAP:  "
        f"{test['positive_gap_fraction']:.2f}%"
    )

    print(
        f"ALPHA:         "
        f"{test['alpha']:.6f}"
    )

    print()

    print(
        "SOURCE CANDIDATE GEOMETRY"
    )

    for key, value in (
        test[
            "source_geometry"
        ].items()
    ):

        print(
            f"{key:<24}"
            f"{value:.6f}"
        )

    print()

    print(
        "RAW UPDATE GEOMETRY"
    )

    for key, value in (
        test[
            "update_geometry"
        ].items()
    ):

        print(
            f"{key:<24}"
            f"{value:.6f}"
        )

    print()

    print(
        "POST ORTHOGONAL GEOMETRY"
    )

    for key, value in (
        test[
            "post_ortho_geometry"
        ].items()
    ):

        print(
            f"{key:<24}"
            f"{value:.6f}"
        )

    print()

    print(
        "FINAL STORED GEOMETRY"
    )

    for key, value in (
        test[
            "stored_geometry"
        ].items()
    ):

        print(
            f"{key:<24}"
            f"{value:.6f}"
        )

    result = {
        "variant": variant,

        "best_epoch": (
            best_epoch
        ),

        "best_validation_accuracy": (
            best_accuracy
        ),

        "best_validation_gap": (
            best_gap
        ),

        "pretrain": pre,

        "test": test,

        "history": history,
    }

    with open(
        output_dir
        / "results.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    return result


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
        "--writer-learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--summary-learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--reader-learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--residual-learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--mismatch-margin",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--mismatch-weight",
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
            "writer_summary_ablation"
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
        "WRITER ARCHITECTURE ABLATION"
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
        "A = ORIGINAL_WRITER"
    )

    print(
        "B = DIRECT_SUMMARY"
    )

    print(
        "C = WRITER_PLUS_SUMMARY"
    )

    print()

    print(
        "All variants use:"
    )

    print(
        "  Gate = 1.0"
    )

    print(
        "  frozen OrthogonalUpdate"
    )

    print(
        "  frozen MemoryBank"
    )

    print(
        "  forced slot 0"
    )

    print(
        "  simple diagnostic reader"
    )

    print(
        "  frozen GPT-2"
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
    # EXAMPLES
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

    # ========================================================
    # PRECOMPUTE ONCE
    # ========================================================

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

    validation_data = precompute(
        original,
        tokenizer,
        validation_examples,
        answer_to_class,
        device,
        args.batch_size,
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
    # RUN 3 INDEPENDENT EXPERIMENTS
    # ========================================================

    variants = [
        "original_writer",
        "direct_summary",
        "writer_plus_summary",
    ]

    all_results = {}

    for variant in variants:

        result = train_variant(
            variant=variant,
            original=original,
            train_data=train_data,
            validation_data=(
                validation_data
            ),
            test_data=test_data,
            num_classes=(
                len(answers)
            ),
            device=device,
            args=args,
        )

        all_results[
            variant
        ] = result

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()
    print("=" * 110)
    print(
        "FINAL WRITER ARCHITECTURE COMPARISON"
    )
    print("=" * 110)

    print(
        f"{'VARIANT':<24}"
        f"{'MATCH':>12}"
        f"{'MISMATCH':>14}"
        f"{'QUERY':>12}"
        f"{'NLL GAP':>14}"
        f"{'SOURCE L2':>14}"
        f"{'STORED L2':>14}"
        f"{'ALPHA':>10}"
    )

    print("-" * 114)

    for variant in variants:

        r = (
            all_results[
                variant
            ]["test"]
        )

        print(
            f"{variant:<24}"
            f"{r['matched_accuracy']:>11.2f}%"
            f"{r['mismatched_accuracy']:>13.2f}%"
            f"{r['query_only_accuracy']:>11.2f}%"
            f"{r['nll_gap']:>14.4f}"
            f"{r['source_geometry']['relative_l2']:>14.4f}"
            f"{r['stored_geometry']['relative_l2']:>14.4f}"
            f"{r['alpha']:>10.4f}"
        )

    # ========================================================
    # SIMPLE INTERPRETATION
    # ========================================================

    original_acc = (
        all_results[
            "original_writer"
        ]["test"][
            "matched_accuracy"
        ]
    )

    direct_acc = (
        all_results[
            "direct_summary"
        ]["test"][
            "matched_accuracy"
        ]
    )

    residual_acc = (
        all_results[
            "writer_plus_summary"
        ]["test"][
            "matched_accuracy"
        ]
    )

    print()
    print("=" * 110)
    print(
        "INTERPRETATION"
    )
    print("=" * 110)

    print(
        f"Original writer:        "
        f"{original_acc:.2f}%"
    )

    print(
        f"Direct summary:         "
        f"{direct_acc:.2f}%"
    )

    print(
        f"Writer + summary:       "
        f"{residual_acc:.2f}%"
    )

    print()

    if (
        residual_acc
        >= direct_acc - 5.0
        and
        residual_acc
        > original_acc + 10.0
    ):

        print(
            "RESULT: DIRECT SUMMARY RESIDUAL HELPS STRONGLY."
        )

        print(
            "The original CandidateWriter appears to lose useful "
            "summary information."
        )

        print(
            "A direct summary residual is a justified writer-side fix "
            "to investigate further."
        )

    elif (
        direct_acc
        > original_acc + 10.0
        and
        residual_acc
        <= original_acc + 10.0
    ):

        print(
            "RESULT: DIRECT SUMMARY WORKS, BUT COMBINING IT WITH "
            "THE WRITER DOES NOT."
        )

        print(
            "This suggests the writer contribution may interfere "
            "with the clean summary VALUE."
        )

    elif (
        original_acc
        >= direct_acc - 5.0
    ):

        print(
            "RESULT: ORIGINAL WRITER IS COMPETITIVE WITH DIRECT SUMMARY."
        )

        print(
            "The writer architecture is probably not the primary "
            "remaining limitation."
        )

    else:

        print(
            "RESULT: MIXED."
        )

        print(
            "Inspect the three held-out accuracies and representation "
            "geometry before changing the architecture."
        )

    # ========================================================
    # SAVE MASTER JSON
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir
        / "writer_summary_ablation_results.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
        )

    print()
    print(
        "Saved:",
        output_dir
        / "writer_summary_ablation_results.json",
    )


if __name__ == "__main__":
    main()