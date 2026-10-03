from __future__ import annotations

import argparse
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
# ANSWERS
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


# ============================================================
# TEMPLATES
# ============================================================

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
# CONFIG MUST MATCH ORIGINAL CHECKPOINT
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
# SINGLE-TOKEN ANSWERS
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
            "Need at least four single-token answer classes."
        )

    return usable


# ============================================================
# DATA
# ============================================================

def build_examples(
    n,
    answer_words,
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

        entity = f"person_{start_id + i}"
        answer = rng.choice(answer_words)

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
                "id": start_id + i,
                "entity": entity,
                "answer": answer,
                "fact": fact,
                "query": query,
            }
        )

    return examples


class ExampleDataset(Dataset):

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_examples(batch):

    return {
        "fact": [x["fact"] for x in batch],
        "query": [x["query"] for x in batch],
        "answer": [x["answer"] for x in batch],
    }


# ============================================================
# LOAD ORIGINAL BACKBONE + ORIGINAL MEMORY BANK
#
# IMPORTANT:
#
# We use the MemoryBank parameters from the checkpoint,
# not a newly initialized MemoryBank.
# ============================================================

def load_components(
    checkpoint_path,
    model_name,
    device,
):

    print("Loading original model/checkpoint...")

    full_model = (
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

    full_model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    full_model.to(device)
    full_model.eval()

    # Keep references to exactly these two components.
    backbone = full_model.backbone.transformer
    memory_bank = full_model.memory_bank

    backbone.eval()
    memory_bank.eval()

    for parameter in backbone.parameters():
        parameter.requires_grad = False

    for parameter in memory_bank.parameters():
        parameter.requires_grad = False

    hidden_size = int(
        full_model.backbone.config.n_embd
    )

    num_slots = int(
        full_model.num_slots
    )

    return (
        full_model,
        backbone,
        memory_bank,
        hidden_size,
        num_slots,
    )


# ============================================================
# TOKENIZATION
# ============================================================

def encode_texts(
    tokenizer,
    texts,
    device,
    max_length=128,
):

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    return (
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
    )


# ============================================================
# LOCATE ANSWER TOKEN
# ============================================================

def find_answer_positions(
    input_ids,
    attention_mask,
    answer_token_ids,
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

        tokens = input_ids[
            i,
            :length,
        ]

        target = int(
            answer_token_ids[i]
        )

        matches = (
            tokens == target
        ).nonzero(
            as_tuple=False
        ).flatten()

        if matches.numel() == 0:

            raise RuntimeError(
                f"Could not locate answer token "
                f"{target} in example {i}."
            )

        positions.append(
            int(matches[-1].item())
        )

    return torch.tensor(
        positions,
        dtype=torch.long,
        device=input_ids.device,
    )


# ============================================================
# FACT REPRESENTATION
#
# SAME REPRESENTATION THAT WON LEVEL 0:
# hidden state at the answer token.
# ============================================================

@torch.no_grad()
def encode_fact(
    backbone,
    tokenizer,
    facts,
    answers,
    answer_token_map,
    device,
):

    input_ids, attention_mask = (
        encode_texts(
            tokenizer,
            facts,
            device,
        )
    )

    output = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )

    hidden = output.last_hidden_state

    answer_ids = [
        answer_token_map[a]
        for a in answers
    ]

    positions = (
        find_answer_positions(
            input_ids,
            attention_mask,
            answer_ids,
        )
    )

    batch_indices = torch.arange(
        hidden.size(0),
        device=device,
    )

    fact_rep = hidden[
        batch_indices,
        positions,
        :
    ]

    return fact_rep.float()


# ============================================================
# QUERY REPRESENTATION
#
# last GPT-2 state of question
# ============================================================

@torch.no_grad()
def encode_query(
    backbone,
    tokenizer,
    queries,
    device,
):

    input_ids, attention_mask = (
        encode_texts(
            tokenizer,
            queries,
            device,
        )
    )

    output = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )

    hidden = output.last_hidden_state

    last_positions = (
        attention_mask.sum(dim=1)
        - 1
    )

    batch_indices = torch.arange(
        hidden.size(0),
        device=device,
    )

    query_rep = hidden[
        batch_indices,
        last_positions,
        :
    ]

    return query_rep.float()


# ============================================================
# PRECOMPUTE FROZEN GPT-2 REPRESENTATIONS
# ============================================================

@torch.no_grad()
def precompute(
    backbone,
    tokenizer,
    examples,
    answer_token_map,
    answer_to_class,
    device,
    batch_size,
):

    loader = DataLoader(
        ExampleDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_examples,
    )

    fact_all = []
    query_all = []
    labels_all = []

    total = 0

    for batch in loader:

        fact_rep = encode_fact(
            backbone=backbone,
            tokenizer=tokenizer,
            facts=batch["fact"],
            answers=batch["answer"],
            answer_token_map=(
                answer_token_map
            ),
            device=device,
        )

        query_rep = encode_query(
            backbone=backbone,
            tokenizer=tokenizer,
            queries=batch["query"],
            device=device,
        )

        labels = torch.tensor(
            [
                answer_to_class[x]
                for x in batch["answer"]
            ],
            dtype=torch.long,
        )

        fact_all.append(
            fact_rep.cpu()
        )

        query_all.append(
            query_rep.cpu()
        )

        labels_all.append(
            labels
        )

        total += len(
            batch["answer"]
        )

        if (
            total % 500 == 0
            or total == len(examples)
        ):
            print(
                f"  {total}/{len(examples)}"
            )

    return (
        torch.cat(
            fact_all,
            dim=0,
        ),
        torch.cat(
            query_all,
            dim=0,
        ),
        torch.cat(
            labels_all,
            dim=0,
        ),
    )


# ============================================================
# LEVEL 1 MODEL
#
# Difference from Level 0:
#
# fact representation
#       ↓
# linear VALUE
#       ↓
# REAL MemoryBank
#   gate = 1.0
#   slot = 0
#   normalization applied
#       ↓
# stored VALUE
#       ↓
# SAME simple reader idea
#       ↓
# 13 classes
# ============================================================

class Level1Memory(nn.Module):

    def __init__(
        self,
        hidden_size,
        number_classes,
        memory_bank,
        num_slots,
        forced_slot=0,
    ):

        super().__init__()

        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.forced_slot = forced_slot

        # ORIGINAL MEMORY BANK
        self.memory_bank = memory_bank

        # Keep memory bank frozen for this experiment.
        for parameter in (
            self.memory_bank.parameters()
        ):
            parameter.requires_grad = False

        # ----------------------------------------------------
        # SAME BASIC VALUE WRITER AS LEVEL 0
        # ----------------------------------------------------

        self.value_projection = nn.Sequential(
            nn.LayerNorm(
                hidden_size
            ),
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
        )

        # ----------------------------------------------------
        # QUERY
        # ----------------------------------------------------

        self.query_norm = nn.LayerNorm(
            hidden_size
        )

        # ----------------------------------------------------
        # SAME SIMPLE READER FAMILY
        # ----------------------------------------------------

        self.reader = nn.Sequential(
            nn.Linear(
                hidden_size * 3,
                hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_size,
                number_classes,
            ),
        )

    # ========================================================
    # MEMORY BANK STORAGE
    # ========================================================

    def store_value(
        self,
        raw_value,
    ):

        batch_size = (
            raw_value.size(0)
        )

        device = (
            raw_value.device
        )

        dtype = (
            raw_value.dtype
        )

        # ----------------------------------------------------
        # Initialize real MemoryBank state.
        # ----------------------------------------------------

        state = (
            self.memory_bank.initialize(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        )

        # ----------------------------------------------------
        # Candidate tensor must have:
        #
        # [B, N, D]
        #
        # For all unused slots simply leave their existing
        # values as candidates.
        # ----------------------------------------------------

        candidate = (
            state.slots.clone()
        )

        candidate[
            :,
            self.forced_slot,
            :,
        ] = raw_value

        # ----------------------------------------------------
        # Gate = 1 ONLY for slot 0.
        #
        # Therefore slot0 becomes candidate directly BEFORE
        # MemoryBank normalization:
        #
        # M' = (1 - 1)M + 1*C = C
        # ----------------------------------------------------

        write_gate = torch.zeros(
            batch_size,
            self.num_slots,
            1,
            dtype=dtype,
            device=device,
        )

        write_gate[
            :,
            self.forced_slot,
            0,
        ] = 1.0

        # ----------------------------------------------------
        # Explicit write mask.
        # ----------------------------------------------------

        write_mask = torch.zeros(
            batch_size,
            self.num_slots,
            1,
            dtype=dtype,
            device=device,
        )

        write_mask[
            :,
            self.forced_slot,
            0,
        ] = 1.0

        # ----------------------------------------------------
        # REAL MEMORY BANK UPDATE
        # ----------------------------------------------------

        new_state = self.memory_bank(
            state=state,
            candidate=candidate,
            write_gate=write_gate,
            write_mask=write_mask,
            confidence=None,
        )

        stored_value = (
            new_state.slots[
                :,
                self.forced_slot,
                :,
            ]
        )

        return stored_value

    # ========================================================
    # WRITE
    # ========================================================

    def write(
        self,
        fact_rep,
    ):

        raw_value = (
            self.value_projection(
                fact_rep
            )
        )

        stored_value = (
            self.store_value(
                raw_value
            )
        )

        return (
            stored_value,
            raw_value,
        )

    # ========================================================
    # READ
    # ========================================================

    def read(
        self,
        query_rep,
        stored_value,
    ):

        q = self.query_norm(
            query_rep
        )

        combined = torch.cat(
            [
                q,
                stored_value,
                q * stored_value,
            ],
            dim=-1,
        )

        return self.reader(
            combined
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        fact_rep,
        query_rep,
    ):

        (
            stored_value,
            raw_value,
        ) = self.write(
            fact_rep
        )

        logits = self.read(
            query_rep,
            stored_value,
        )

        return (
            logits,
            stored_value,
            raw_value,
        )


# ============================================================
# PRECOMPUTED DATASET
# ============================================================

class RepresentationDataset(Dataset):

    def __init__(
        self,
        fact,
        query,
        labels,
    ):

        self.fact = fact
        self.query = query
        self.labels = labels

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, i):

        return (
            self.fact[i],
            self.query[i],
            self.labels[i],
        )


# ============================================================
# BUILD MISMATCH INDICES
#
# IMPORTANT:
# Mismatched VALUE is guaranteed to come from a DIFFERENT
# ANSWER CLASS.
# ============================================================

def different_answer_indices(
    labels,
):

    labels = labels.detach()

    n = labels.size(0)

    indices = torch.empty(
        n,
        dtype=torch.long,
        device=labels.device,
    )

    for i in range(n):

        candidates = torch.nonzero(
            labels != labels[i],
            as_tuple=False,
        ).flatten()

        if candidates.numel() == 0:

            raise RuntimeError(
                "Batch has only one answer class; "
                "cannot create mismatched values."
            )

        j = candidates[
            torch.randint(
                0,
                candidates.numel(),
                (1,),
                device=labels.device,
            )
        ]

        indices[i] = j

    return indices


# ============================================================
# VALUE GEOMETRY
# ============================================================

@torch.no_grad()
def geometry(values):

    x = values.detach().float().cpu()

    if x.size(0) > 500:

        ids = torch.linspace(
            0,
            x.size(0) - 1,
            500,
        ).long()

        x = x[ids]

    norms = x.norm(
        dim=-1
    )

    normalized = F.normalize(
        x,
        p=2,
        dim=-1,
    )

    cosine = (
        normalized
        @ normalized.T
    )

    distances = torch.cdist(
        x,
        x,
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

    pair_cos = cosine[
        mask
    ]

    pair_dist = distances[
        mask
    ]

    mean_norm = float(
        norms.mean().item()
    )

    mean_distance = float(
        pair_dist.mean().item()
    )

    return {
        "mean_norm": (
            mean_norm
        ),

        "mean_pairwise_l2": (
            mean_distance
        ),

        "relative_l2": (
            mean_distance
            / (mean_norm + 1e-8)
        ),

        "mean_pairwise_cosine": float(
            pair_cos.mean().item()
        ),
    }


# ============================================================
# DETERMINISTIC EVALUATION MISMATCH
#
# For each sample choose the first later example having
# a different answer.
# ============================================================

def evaluation_mismatch_indices(
    labels,
):

    labels = labels.cpu()

    n = labels.size(0)

    result = []

    for i in range(n):

        found = None

        for offset in range(
            1,
            n,
        ):

            j = (
                i + offset
            ) % n

            if (
                labels[j]
                != labels[i]
            ):

                found = j
                break

        if found is None:

            raise RuntimeError(
                "Could not construct mismatch."
            )

        result.append(
            found
        )

    return torch.tensor(
        result,
        dtype=torch.long,
    )


# ============================================================
# EVALUATE
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    fact_x,
    query_x,
    labels,
    device,
):

    model.eval()

    fact_x = fact_x.to(
        device
    )

    query_x = query_x.to(
        device
    )

    labels = labels.to(
        device
    )

    # --------------------------------------------------------
    # WRITE THROUGH REAL MEMORY BANK
    # --------------------------------------------------------

    (
        matched_logits,
        stored_values,
        raw_values,
    ) = model(
        fact_x,
        query_x,
    )

    # --------------------------------------------------------
    # MATCHED
    # --------------------------------------------------------

    matched_losses = (
        F.cross_entropy(
            matched_logits,
            labels,
            reduction="none",
        )
    )

    matched_loss = float(
        matched_losses.mean().item()
    )

    matched_acc = float(
        (
            matched_logits.argmax(dim=-1)
            == labels
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    # --------------------------------------------------------
    # MISMATCHED
    #
    # GUARANTEED different answer.
    # --------------------------------------------------------

    mismatch_indices = (
        evaluation_mismatch_indices(
            labels
        ).to(device)
    )

    wrong_values = (
        stored_values[
            mismatch_indices
        ]
    )

    mismatched_logits = (
        model.read(
            query_x,
            wrong_values,
        )
    )

    mismatched_losses = (
        F.cross_entropy(
            mismatched_logits,
            labels,
            reduction="none",
        )
    )

    mismatched_loss = float(
        mismatched_losses.mean().item()
    )

    mismatched_acc = float(
        (
            mismatched_logits.argmax(dim=-1)
            == labels
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    # --------------------------------------------------------
    # QUERY ONLY
    # --------------------------------------------------------

    zero_values = torch.zeros_like(
        stored_values
    )

    query_only_logits = (
        model.read(
            query_x,
            zero_values,
        )
    )

    query_only_loss = float(
        F.cross_entropy(
            query_only_logits,
            labels,
        ).item()
    )

    query_only_acc = float(
        (
            query_only_logits.argmax(dim=-1)
            == labels
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    # --------------------------------------------------------
    # NLL GAP
    # --------------------------------------------------------

    per_example_gap = (
        mismatched_losses
        - matched_losses
    )

    nll_gap = float(
        per_example_gap.mean().item()
    )

    positive_gap = float(
        (
            per_example_gap > 0
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    return {
        "matched_accuracy": (
            matched_acc
        ),

        "mismatched_accuracy": (
            mismatched_acc
        ),

        "query_only_accuracy": (
            query_only_acc
        ),

        "matched_loss": (
            matched_loss
        ),

        "mismatched_loss": (
            mismatched_loss
        ),

        "query_only_loss": (
            query_only_loss
        ),

        "nll_gap": (
            nll_gap
        ),

        "positive_gap_fraction": (
            positive_gap
        ),

        # Compare before and after MemoryBank.
        "raw_value_geometry": (
            geometry(
                raw_values
            )
        ),

        "stored_value_geometry": (
            geometry(
                stored_values
            )
        ),
    }


# ============================================================
# PRINT
# ============================================================

def print_result(
    title,
    result,
):

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    print(
        f"{'Condition':<22}"
        f"{'Accuracy':>14}"
        f"{'Loss':>14}"
    )

    print("-" * 50)

    print(
        f"{'MATCHED':<22}"
        f"{result['matched_accuracy']:>13.2f}%"
        f"{result['matched_loss']:>14.4f}"
    )

    print(
        f"{'MISMATCHED':<22}"
        f"{result['mismatched_accuracy']:>13.2f}%"
        f"{result['mismatched_loss']:>14.4f}"
    )

    print(
        f"{'QUERY_ONLY':<22}"
        f"{result['query_only_accuracy']:>13.2f}%"
        f"{result['query_only_loss']:>14.4f}"
    )

    print()

    print(
        "Matched - mismatched accuracy: "
        f"{result['matched_accuracy'] - result['mismatched_accuracy']:+.2f} pp"
    )

    print(
        "Mismatched NLL - matched NLL:   "
        f"{result['nll_gap']:+.6f}"
    )

    print(
        "Positive-gap examples:          "
        f"{result['positive_gap_fraction']:.2f}%"
    )

    print()
    print("RAW VALUE — BEFORE MEMORY BANK")

    for key, value in (
        result[
            "raw_value_geometry"
        ].items()
    ):
        print(
            f"  {key:<24}"
            f"{value:.6f}"
        )

    print()
    print("STORED VALUE — AFTER MEMORY BANK")

    for key, value in (
        result[
            "stored_value_geometry"
        ].items()
    ):
        print(
            f"  {key:<24}"
            f"{value:.6f}"
        )


# ============================================================
# TRAIN
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    device,
    mismatch_margin,
    mismatch_weight,
):

    model.train()

    # MemoryBank is fixed in this Level.
    model.memory_bank.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_rank = 0.0
    total = 0

    for (
        fact_x,
        query_x,
        labels,
    ) in loader:

        fact_x = fact_x.to(
            device
        )

        query_x = query_x.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # MATCHED
        # ----------------------------------------------------

        (
            matched_logits,
            stored_values,
            _,
        ) = model(
            fact_x,
            query_x,
        )

        matched_losses = (
            F.cross_entropy(
                matched_logits,
                labels,
                reduction="none",
            )
        )

        matched_ce = (
            matched_losses.mean()
        )

        # ----------------------------------------------------
        # GUARANTEED DIFFERENT-ANSWER MEMORY
        # ----------------------------------------------------

        mismatch_indices = (
            different_answer_indices(
                labels
            )
        )

        wrong_values = (
            stored_values[
                mismatch_indices
            ]
        )

        mismatch_logits = (
            model.read(
                query_x,
                wrong_values,
            )
        )

        mismatch_losses = (
            F.cross_entropy(
                mismatch_logits,
                labels,
                reduction="none",
            )
        )

        # ----------------------------------------------------
        # Ranking:
        #
        # mismatch must have higher NLL than match.
        # ----------------------------------------------------

        rank_loss = F.relu(
            mismatch_margin
            + matched_losses
            - mismatch_losses
        ).mean()

        loss = (
            matched_ce
            + mismatch_weight
            * rank_loss
        )

        loss.backward()

        trainable_parameters = [
            p
            for p in model.parameters()
            if p.requires_grad
        ]

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=5.0,
        )

        optimizer.step()

        n = labels.size(0)

        total += n

        total_loss += (
            float(loss.item())
            * n
        )

        total_ce += (
            float(
                matched_ce.item()
            )
            * n
        )

        total_rank += (
            float(
                rank_loss.item()
            )
            * n
        )

    return {
        "loss": (
            total_loss / total
        ),

        "matched_ce": (
            total_ce / total
        ),

        "rank_loss": (
            total_rank / total
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "outputs/"
            "retrieval_gradient_test/"
            "checkpoint_best.pt"
        ),
    )

    parser.add_argument(
        "--model-name",
        type=str,
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
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
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
        type=str,
        default=(
            "outputs/"
            "level1_memory_bank"
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

    print("=" * 90)
    print("LEVEL-1 MEMORY BANK EXPERIMENT")
    print("=" * 90)

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
        "ADDED COMPONENT:"
    )
    print(
        "  + ORIGINAL MemoryBank"
    )
    print(
        "  + ORIGINAL MemoryBank normalization"
    )

    print()
    print(
        "STILL DISABLED:"
    )
    print(
        "  - CandidateWriter"
    )
    print(
        "  - learned write gate"
    )
    print(
        "  - orthogonalizer"
    )
    print(
        "  - router"
    )
    print(
        "  - original MemoryReader"
    )
    print(
        "  - E5"
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

    tokenizer.padding_side = "right"

    answer_token_map = (
        get_single_token_answers(
            tokenizer
        )
    )

    answer_words = list(
        answer_token_map.keys()
    )

    answer_to_class = {
        answer: i
        for i, answer
        in enumerate(
            answer_words
        )
    }

    number_classes = len(
        answer_words
    )

    print()
    print(
        "Answer classes:",
        answer_words,
    )

    print(
        "Chance accuracy:",
        f"{100 / number_classes:.2f}%"
    )

    print(
        "Chance CE:",
        f"{math.log(number_classes):.4f}"
    )

    # ========================================================
    # ORIGINAL COMPONENTS
    # ========================================================

    (
        full_model,
        backbone,
        memory_bank,
        hidden_size,
        num_slots,
    ) = load_components(
        checkpoint_path=(
            args.checkpoint
        ),
        model_name=args.model_name,
        device=device,
    )

    print()
    print(
        "GPT-2 hidden size:",
        hidden_size,
    )

    print(
        "Memory slots:",
        num_slots,
    )

    print(
        "Memory normalization:",
        memory_bank.normalization,
    )

    print(
        "Forced slot:",
        args.forced_slot,
    )

    # ========================================================
    # DATA
    # ========================================================

    train_examples = build_examples(
        n=args.train_examples,
        answer_words=answer_words,
        seed=args.seed,
        start_id=0,
        split="train",
    )

    valid_examples = build_examples(
        n=args.validation_examples,
        answer_words=answer_words,
        seed=args.seed + 1000,
        start_id=100000,
        split="eval",
    )

    test_examples = build_examples(
        n=args.test_examples,
        answer_words=answer_words,
        seed=args.seed + 2000,
        start_id=200000,
        split="eval",
    )

    print()
    print("=" * 90)
    print("PRECOMPUTING TRAIN")
    print("=" * 90)

    train_fact, train_query, train_y = (
        precompute(
            backbone=backbone,
            tokenizer=tokenizer,
            examples=train_examples,
            answer_token_map=(
                answer_token_map
            ),
            answer_to_class=(
                answer_to_class
            ),
            device=device,
            batch_size=args.batch_size,
        )
    )

    print()
    print("=" * 90)
    print("PRECOMPUTING VALIDATION")
    print("=" * 90)

    valid_fact, valid_query, valid_y = (
        precompute(
            backbone=backbone,
            tokenizer=tokenizer,
            examples=valid_examples,
            answer_token_map=(
                answer_token_map
            ),
            answer_to_class=(
                answer_to_class
            ),
            device=device,
            batch_size=args.batch_size,
        )
    )

    print()
    print("=" * 90)
    print("PRECOMPUTING TEST")
    print("=" * 90)

    test_fact, test_query, test_y = (
        precompute(
            backbone=backbone,
            tokenizer=tokenizer,
            examples=test_examples,
            answer_token_map=(
                answer_token_map
            ),
            answer_to_class=(
                answer_to_class
            ),
            device=device,
            batch_size=args.batch_size,
        )
    )

    # ========================================================
    # LEVEL 1
    # ========================================================

    model = Level1Memory(
        hidden_size=hidden_size,
        number_classes=(
            number_classes
        ),
        memory_bank=memory_bank,
        num_slots=num_slots,
        forced_slot=(
            args.forced_slot
        ),
    ).to(device)

    # Only our tiny Level-1 writer/reader train.
    trainable_parameters = [
        p
        for name, p
        in model.named_parameters()
        if p.requires_grad
    ]

    print()
    print(
        "Trainable parameters:",
        f"{sum(p.numel() for p in trainable_parameters):,}"
    )

    print(
        "MemoryBank trainable:",
        any(
            p.requires_grad
            for p in (
                model.memory_bank
                .parameters()
            )
        ),
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_loader = DataLoader(
        RepresentationDataset(
            train_fact,
            train_query,
            train_y,
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # ========================================================
    # PRETRAIN
    # ========================================================

    pre_result = evaluate(
        model=model,
        fact_x=valid_fact,
        query_x=valid_query,
        labels=valid_y,
        device=device,
    )

    print_result(
        "PRE-TRAIN VALIDATION",
        pre_result,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_gap = -float("inf")
    best_epoch = -1
    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        print()
        print("=" * 90)
        print(
            f"EPOCH {epoch}/{args.epochs}"
        )
        print("=" * 90)

        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            mismatch_margin=(
                args.mismatch_margin
            ),
            mismatch_weight=(
                args.mismatch_weight
            ),
        )

        print(
            "Train total loss:",
            f"{train_metrics['loss']:.4f}"
        )

        print(
            "Train matched CE:",
            f"{train_metrics['matched_ce']:.4f}"
        )

        print(
            "Train rank loss:",
            f"{train_metrics['rank_loss']:.4f}"
        )

        validation = evaluate(
            model=model,
            fact_x=valid_fact,
            query_x=valid_query,
            labels=valid_y,
            device=device,
        )

        print_result(
            f"VALIDATION EPOCH {epoch}",
            validation,
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation,
            }
        )

        # Primary checkpoint criterion:
        # mismatched NLL - matched NLL
        gap = validation[
            "nll_gap"
        ]

        if gap > best_gap:

            best_gap = gap
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "validation": (
                        validation
                    ),
                    "arguments": vars(args),
                    "experiment": (
                        "level1_memory_bank"
                    ),
                },
                output_dir
                / "checkpoint_best.pt",
            )

            print(
                "Saved new best checkpoint."
            )

    # ========================================================
    # BEST CHECKPOINT
    # ========================================================

    print()
    print("=" * 90)
    print(
        f"LOADING BEST EPOCH {best_epoch}"
    )
    print("=" * 90)

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

    model.eval()

    # ========================================================
    # TEST
    # ========================================================

    test_result = evaluate(
        model=model,
        fact_x=test_fact,
        query_x=test_query,
        labels=test_y,
        device=device,
    )

    print_result(
        "FINAL HELD-OUT TEST",
        test_result,
    )

    # ========================================================
    # VERDICT
    # ========================================================

    matched = (
        test_result[
            "matched_accuracy"
        ]
    )

    mismatch = (
        test_result[
            "mismatched_accuracy"
        ]
    )

    query_only = (
        test_result[
            "query_only_accuracy"
        ]
    )

    gap = (
        test_result[
            "nll_gap"
        ]
    )

    print()
    print("=" * 90)
    print("LEVEL-1 VERDICT")
    print("=" * 90)

    print(
        f"MATCHED:       "
        f"{matched:.2f}%"
    )

    print(
        f"MISMATCHED:    "
        f"{mismatch:.2f}%"
    )

    print(
        f"QUERY ONLY:    "
        f"{query_only:.2f}%"
    )

    print(
        f"NLL GAP:       "
        f"{gap:+.6f}"
    )

    print()

    if (
        matched >= 90.0
        and (
            matched
            - mismatch
        ) >= 70.0
        and gap > 1.0
    ):

        verdict = (
            "PASS: MemoryBank storage and normalization "
            "preserve the useful latent VALUE. "
            "Proceed to Level 2: add the learned write gate."
        )

    elif (
        matched >= 60.0
        and (
            matched
            - mismatch
        ) >= 20.0
        and gap > 0
    ):

        verdict = (
            "PARTIAL PASS: MemoryBank preserves useful "
            "information, but performance is lower than "
            "Level 0. Inspect normalization geometry before "
            "adding the gate."
        )

    else:

        verdict = (
            "FAIL: adding only MemoryBank storage/"
            "normalization destroys the Level-0 result. "
            "Stop here and diagnose MemoryBank normalization."
        )

    print(verdict)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    result_path = (
        output_dir
        / "level1_results.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "level": 1,
                "added_component": (
                    "MemoryBank with forced gate=1"
                ),
                "best_epoch": (
                    best_epoch
                ),
                "best_validation_gap": (
                    best_gap
                ),
                "pretrain": (
                    pre_result
                ),
                "test": (
                    test_result
                ),
                "history": (
                    history
                ),
            },
            f,
            indent=2,
        )

    print()
    print(
        "Saved:",
        result_path,
    )

    print(
        "Best checkpoint:",
        output_dir
        / "checkpoint_best.pt",
    )


if __name__ == "__main__":
    main()