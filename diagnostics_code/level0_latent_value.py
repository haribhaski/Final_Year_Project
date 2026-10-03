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
# MEMORY CONFIG
#
# Only needed so the old checkpoint loads correctly.
# We DO NOT use the memory architecture in this experiment.
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

        read_before_write=True,
        detach_memory_between_steps=False,

        candidate_diversity_weight=0.01,
        update_orthogonality_weight=0.01,
        router_balance_weight=0.01,
        reader_balance_weight=0.01,
        memory_collapse_weight=0.01,
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
# TOKENIZER ANSWERS
# ============================================================

def get_single_token_answers(tokenizer):

    result = {}

    for word in ANSWER_POOL:

        ids = tokenizer(
            " " + word,
            add_special_tokens=False,
        )["input_ids"]

        if len(ids) == 1:
            result[word] = ids[0]

    if len(result) < 4:
        raise RuntimeError(
            "Need at least four single-token answers."
        )

    return result


# ============================================================
# DATASET
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

        answer = rng.choice(
            answer_words
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


def collate_batch(batch):

    return {
        "id": [
            x["id"]
            for x in batch
        ],

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
# LOAD FROZEN GPT-2 BACKBONE
# ============================================================

def load_backbone(
    checkpoint_path,
    model_name,
    device,
):

    print("Loading original checkpoint...")

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

    backbone = (
        full_model.backbone.transformer
    )

    backbone.to(device)
    backbone.eval()

    for parameter in backbone.parameters():
        parameter.requires_grad = False

    hidden_size = int(
        full_model.backbone.config.n_embd
    )

    # We no longer need the whole memory model.
    del full_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return backbone, hidden_size


# ============================================================
# ENCODE TEXT
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
# FIND ANSWER POSITION
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

        valid_length = int(
            attention_mask[
                i
            ].sum().item()
        )

        tokens = input_ids[
            i,
            :valid_length
        ]

        target = int(
            answer_token_ids[i]
        )

        found = (
            tokens == target
        ).nonzero(
            as_tuple=False
        ).flatten()

        if found.numel() == 0:

            raise RuntimeError(
                f"Could not find answer token "
                f"{target} in example {i}."
            )

        positions.append(
            int(
                found[-1].item()
            )
        )

    return torch.tensor(
        positions,
        dtype=torch.long,
        device=input_ids.device,
    )


# ============================================================
# FACT REPRESENTATION
#
# Default:
# answer-token hidden state
#
# Can also run:
# --fact-rep mean
# ============================================================

@torch.no_grad()
def encode_fact(
    backbone,
    tokenizer,
    facts,
    answers,
    answer_token_map,
    device,
    fact_rep,
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

    hidden = (
        output.last_hidden_state
    )

    if fact_rep == "mean":

        weights = (
            attention_mask
            .unsqueeze(-1)
            .to(hidden.dtype)
        )

        representation = (
            (hidden * weights).sum(dim=1)
            /
            weights.sum(dim=1)
            .clamp_min(1.0)
        )

        return representation.float()

    if fact_rep == "answer_token":

        answer_token_ids = [
            answer_token_map[a]
            for a in answers
        ]

        positions = (
            find_answer_positions(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
                answer_token_ids=(
                    answer_token_ids
                ),
            )
        )

        batch_indices = torch.arange(
            hidden.size(0),
            device=device,
        )

        representation = hidden[
            batch_indices,
            positions,
            :
        ]

        return representation.float()

    raise ValueError(
        f"Unknown fact representation: {fact_rep}"
    )


# ============================================================
# QUERY REPRESENTATION
#
# Use last query token.
# The query does NOT contain the answer.
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

    hidden = (
        output.last_hidden_state
    )

    last_indices = (
        attention_mask.sum(dim=1)
        - 1
    )

    batch_indices = torch.arange(
        hidden.size(0),
        device=device,
    )

    representation = hidden[
        batch_indices,
        last_indices,
        :
    ]

    return representation.float()


# ============================================================
# LEVEL-0 MODEL
#
# fact representation
#       ↓
# Linear projection
#       ↓
# latent VALUE (768)
#
# question representation
#       +
# latent VALUE
#       ↓
# tiny decoder
#       ↓
# 13 answer classes
# ============================================================

class Level0Memory(nn.Module):

    def __init__(
        self,
        hidden_size,
        number_classes,
    ):

        super().__init__()

        # ----------------------------------------------------
        # VALUE WRITER
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
        # QUERY NORMALIZATION
        # ----------------------------------------------------

        self.query_norm = nn.LayerNorm(
            hidden_size
        )

        # ----------------------------------------------------
        # READER
        #
        # Concatenate:
        #
        # q
        # v
        # q * v
        #
        # This is intentionally simple.
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

    def write(
        self,
        fact_rep,
    ):

        return self.value_projection(
            fact_rep
        )

    def read(
        self,
        query_rep,
        value,
    ):

        q = self.query_norm(
            query_rep
        )

        combined = torch.cat(
            [
                q,
                value,
                q * value,
            ],
            dim=-1,
        )

        return self.reader(
            combined
        )

    def forward(
        self,
        fact_rep,
        query_rep,
    ):

        value = self.write(
            fact_rep
        )

        logits = self.read(
            query_rep,
            value,
        )

        return logits, value


# ============================================================
# PRECOMPUTE FROZEN GPT-2 REPRESENTATIONS
#
# This makes training very fast.
# ============================================================

@torch.no_grad()
def precompute(
    backbone,
    tokenizer,
    examples,
    answer_token_map,
    answer_to_class,
    device,
    fact_rep,
    batch_size=128,
):

    dataset = ExampleDataset(
        examples
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    fact_representations = []
    query_representations = []
    labels = []

    total = 0

    for batch in loader:

        fact_vectors = encode_fact(
            backbone=backbone,
            tokenizer=tokenizer,
            facts=batch["fact"],
            answers=batch["answer"],
            answer_token_map=(
                answer_token_map
            ),
            device=device,
            fact_rep=fact_rep,
        )

        query_vectors = encode_query(
            backbone=backbone,
            tokenizer=tokenizer,
            queries=batch["query"],
            device=device,
        )

        y = torch.tensor(
            [
                answer_to_class[a]
                for a in batch["answer"]
            ],
            dtype=torch.long,
        )

        fact_representations.append(
            fact_vectors.cpu()
        )

        query_representations.append(
            query_vectors.cpu()
        )

        labels.append(
            y
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
            fact_representations,
            dim=0,
        ),

        torch.cat(
            query_representations,
            dim=0,
        ),

        torch.cat(
            labels,
            dim=0,
        ),
    )


# ============================================================
# TRAIN DATASET
# ============================================================

class RepresentationDataset(Dataset):

    def __init__(
        self,
        fact_x,
        query_x,
        labels,
    ):

        self.fact_x = fact_x
        self.query_x = query_x
        self.labels = labels

    def __len__(self):

        return self.labels.size(0)

    def __getitem__(
        self,
        index,
    ):

        return (
            self.fact_x[index],
            self.query_x[index],
            self.labels[index],
        )


# ============================================================
# VALUE GEOMETRY
# ============================================================

@torch.no_grad()
def value_geometry(values):

    values = values.float().cpu()

    n = values.size(0)

    if n > 500:

        indices = torch.linspace(
            0,
            n - 1,
            500,
        ).long()

        values = values[
            indices
        ]

    norm = (
        values.norm(
            dim=-1
        ).mean()
    )

    normalized = F.normalize(
        values,
        p=2,
        dim=-1,
    )

    cosine = (
        normalized
        @ normalized.T
    )

    distance = torch.cdist(
        values,
        values,
    )

    n = values.size(0)

    upper = torch.triu(
        torch.ones(
            n,
            n,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    pair_cos = cosine[
        upper
    ]

    pair_dist = distance[
        upper
    ]

    mean_distance = (
        pair_dist.mean()
    )

    return {
        "mean_norm": float(
            norm.item()
        ),

        "mean_pairwise_l2": float(
            mean_distance.item()
        ),

        "relative_l2": float(
            (
                mean_distance
                / (
                    norm
                    + 1e-8
                )
            ).item()
        ),

        "mean_pairwise_cosine": float(
            pair_cos.mean().item()
        ),
    }


# ============================================================
# EVALUATION
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

    # ========================================================
    # MATCHED
    # ========================================================

    matched_logits, values = model(
        fact_x,
        query_x,
    )

    matched_loss_per_example = (
        F.cross_entropy(
            matched_logits,
            labels,
            reduction="none",
        )
    )

    matched_loss = float(
        matched_loss_per_example
        .mean()
        .item()
    )

    matched_predictions = (
        matched_logits.argmax(
            dim=-1
        )
    )

    matched_accuracy = float(
        (
            matched_predictions
            == labels
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    # ========================================================
    # MISMATCHED
    #
    # Rotate VALUE by one example.
    #
    # Query A receives Value B.
    # ========================================================

    mismatched_values = torch.roll(
        values,
        shifts=1,
        dims=0,
    )

    mismatched_logits = (
        model.read(
            query_x,
            mismatched_values,
        )
    )

    mismatch_loss_per_example = (
        F.cross_entropy(
            mismatched_logits,
            labels,
            reduction="none",
        )
    )

    mismatch_loss = float(
        mismatch_loss_per_example
        .mean()
        .item()
    )

    mismatch_predictions = (
        mismatched_logits.argmax(
            dim=-1
        )
    )

    mismatch_accuracy = float(
        (
            mismatch_predictions
            == labels
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    # ========================================================
    # QUERY ONLY CONTROL
    #
    # Zero VALUE.
    # ========================================================

    zero_values = torch.zeros_like(
        values
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

    query_only_accuracy = float(
        (
            query_only_logits
            .argmax(dim=-1)
            == labels
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    # ========================================================
    # MEMORY-SPECIFIC GAP
    #
    # Positive is good:
    #
    # mismatched loss > matched loss
    # ========================================================

    gap = (
        mismatch_loss
        - matched_loss
    )

    per_example_gap = (
        mismatch_loss_per_example
        - matched_loss_per_example
    )

    positive_gap_fraction = float(
        (
            per_example_gap > 0
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    geometry = (
        value_geometry(
            values
        )
    )

    return {
        "matched_accuracy": (
            matched_accuracy
        ),

        "mismatched_accuracy": (
            mismatch_accuracy
        ),

        "query_only_accuracy": (
            query_only_accuracy
        ),

        "matched_loss": (
            matched_loss
        ),

        "mismatched_loss": (
            mismatch_loss
        ),

        "query_only_loss": (
            query_only_loss
        ),

        "nll_gap": (
            gap
        ),

        "positive_gap_fraction": (
            positive_gap_fraction
        ),

        "value_geometry": (
            geometry
        ),
    }


# ============================================================
# PRINT EVALUATION
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

    print("-" * 52)

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
        "Examples with positive NLL gap: "
        f"{result['positive_gap_fraction']:.2f}%"
    )

    print()

    print(
        "VALUE geometry:"
    )

    for key, value in (
        result[
            "value_geometry"
        ].items()
    ):

        print(
            f"  {key:<24}"
            f"{value:.6f}"
        )


# ============================================================
# TRAIN ONE EPOCH
#
# Main loss:
#
# CE(matched)
#
# Additional ranking loss:
#
# We explicitly require:
#
# correct-class loss(mismatched)
# >
# correct-class loss(matched)
# + margin
#
# This prevents learning only a global 13-word prior.
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

    total_loss = 0.0
    total_matched_ce = 0.0
    total_rank_loss = 0.0
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
        # MATCHED VALUE
        # ----------------------------------------------------

        matched_logits, values = model(
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
        # MISMATCHED VALUE
        #
        # Random permutation within batch.
        #
        # Try to avoid identity matches.
        # ----------------------------------------------------

        batch_size = (
            fact_x.size(0)
        )

        if batch_size > 1:

            permutation = torch.randperm(
                batch_size,
                device=device,
            )

            identity = torch.arange(
                batch_size,
                device=device,
            )

            if torch.any(
                permutation == identity
            ):

                permutation = torch.roll(
                    identity,
                    shifts=1,
                )

            mismatched_values = values[
                permutation
            ]

            mismatched_logits = (
                model.read(
                    query_x,
                    mismatched_values,
                )
            )

            mismatch_losses = (
                F.cross_entropy(
                    mismatched_logits,
                    labels,
                    reduction="none",
                )
            )

            # -----------------------------------------------
            # Want:
            #
            # mismatch_loss >= matched_loss + margin
            #
            # Therefore:
            #
            # margin + matched - mismatch <= 0
            # -----------------------------------------------

            rank_loss = F.relu(
                mismatch_margin
                + matched_losses
                - mismatch_losses
            ).mean()

        else:

            rank_loss = (
                matched_ce
                * 0.0
            )

        loss = (
            matched_ce
            + mismatch_weight
            * rank_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        n = batch_size

        total += n

        total_loss += (
            float(loss.item())
            * n
        )

        total_matched_ce += (
            float(
                matched_ce.item()
            )
            * n
        )

        total_rank_loss += (
            float(
                rank_loss.item()
            )
            * n
        )

    return {
        "loss": (
            total_loss
            / total
        ),

        "matched_ce": (
            total_matched_ce
            / total
        ),

        "rank_loss": (
            total_rank_loss
            / total
        ),
    }


# ============================================================
# SAVE
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    result,
    args,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "validation": result,
            "arguments": vars(args),
            "experiment": (
                "level0_latent_value"
            ),
        },
        path,
    )


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
        "--fact-rep",
        type=str,
        choices=[
            "answer_token",
            "mean",
        ],
        default="answer_token",
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
        "--seed",
        type=int,
        default=2090,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/level0_latent_value",
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
    print("LEVEL-0 LATENT VALUE EXPERIMENT")
    print("=" * 90)

    print(
        "Device:",
        device,
    )

    print(
        "Checkpoint:",
        args.checkpoint,
    )

    print(
        "Fact representation:",
        args.fact_rep,
    )

    print()
    print(
        "NO CandidateWriter"
    )
    print(
        "NO router"
    )
    print(
        "NO write gate"
    )
    print(
        "NO orthogonalizer"
    )
    print(
        "NO MemoryBank"
    )
    print(
        "NO original MemoryReader"
    )
    print(
        "NO E5"
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
        word: i
        for i, word in enumerate(
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
        "Number classes:",
        number_classes,
    )

    print(
        "Chance accuracy:",
        f"{100 / number_classes:.2f}%",
    )

    print(
        "Chance CE:",
        f"{math.log(number_classes):.4f}",
    )

    # ========================================================
    # LOAD FROZEN GPT-2
    # ========================================================

    backbone, hidden_size = (
        load_backbone(
            checkpoint_path=(
                args.checkpoint
            ),
            model_name=(
                args.model_name
            ),
            device=device,
        )
    )

    print(
        "GPT-2 hidden size:",
        hidden_size,
    )

    # ========================================================
    # BUILD DATA
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

    # ========================================================
    # PRECOMPUTE REPRESENTATIONS
    # ========================================================

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
            fact_rep=args.fact_rep,
            batch_size=(
                args.batch_size
            ),
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
            fact_rep=args.fact_rep,
            batch_size=(
                args.batch_size
            ),
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
            fact_rep=args.fact_rep,
            batch_size=(
                args.batch_size
            ),
        )
    )

    # Backbone no longer needed.
    del backbone

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # LEVEL 0 MODEL
    # ========================================================

    model = Level0Memory(
        hidden_size=hidden_size,
        number_classes=number_classes,
    ).to(device)

    trainable = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "Level-0 trainable parameters:",
        f"{trainable:,}",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_dataset = (
        RepresentationDataset(
            train_fact,
            train_query,
            train_y,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # ========================================================
    # PRE-TRAIN
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
    #
    # SELECT BEST CHECKPOINT BY NLL GAP.
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
            f"{train_metrics['loss']:.4f}",
        )

        print(
            "Train matched CE:",
            f"{train_metrics['matched_ce']:.4f}",
        )

        print(
            "Train rank loss:",
            f"{train_metrics['rank_loss']:.4f}",
        )

        result = evaluate(
            model=model,
            fact_x=valid_fact,
            query_x=valid_query,
            labels=valid_y,
            device=device,
        )

        print_result(
            f"VALIDATION EPOCH {epoch}",
            result,
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": result,
            }
        )

        gap = result[
            "nll_gap"
        ]

        if gap > best_gap:

            best_gap = gap
            best_epoch = epoch

            save_checkpoint(
                output_dir
                / "checkpoint_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                result=result,
                args=args,
            )

            print(
                "Saved new best checkpoint."
            )

    # ========================================================
    # LOAD BEST
    # ========================================================

    print()
    print("=" * 90)
    print(
        f"LOADING BEST EPOCH {best_epoch}"
    )
    print("=" * 90)

    best_checkpoint = torch.load(
        output_dir
        / "checkpoint_best.pt",
        map_location=device,
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # ========================================================
    # FINAL TEST
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
    # FINAL VERDICT
    # ========================================================

    matched = (
        test_result[
            "matched_accuracy"
        ]
    )

    mismatched = (
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
    print("LEVEL-0 VERDICT")
    print("=" * 90)

    print(
        f"MATCHED:       "
        f"{matched:.2f}%"
    )

    print(
        f"MISMATCHED:    "
        f"{mismatched:.2f}%"
    )

    print(
        f"QUERY ONLY:    "
        f"{query_only:.2f}%"
    )

    print(
        f"NLL GAP:       "
        f"{gap:+.6f}"
    )

    print(
        f"POSITIVE GAP:  "
        f"{test_result['positive_gap_fraction']:.2f}% "
        f"of examples"
    )

    print()

    accuracy_gap = (
        matched
        - mismatched
    )

    if (
        matched >= 80
        and accuracy_gap >= 40
        and gap > 0.5
    ):

        verdict = (
            "STRONG PASS: a simple latent VALUE "
            "clearly stores usable fact-specific "
            "information."
        )

    elif (
        matched >= 60
        and accuracy_gap >= 20
        and gap > 0.2
    ):

        verdict = (
            "PASS: the minimal latent VALUE system "
            "works. We can now add original memory "
            "components back one at a time."
        )

    elif (
        accuracy_gap >= 10
        and gap > 0
    ):

        verdict = (
            "PARTIAL PASS: VALUE information is "
            "being used, but the Level-0 reader/"
            "training setup needs improvement."
        )

    else:

        verdict = (
            "FAIL: even the minimal latent VALUE "
            "system cannot produce a reliable "
            "matched-vs-mismatched gap. Stop adding "
            "memory components and diagnose the "
            "reader/training formulation."
        )

    print(verdict)

    # ========================================================
    # SAVE JSON
    # ========================================================

    summary = {
        "fact_representation": (
            args.fact_rep
        ),

        "best_epoch": (
            best_epoch
        ),

        "best_validation_gap": (
            best_gap
        ),

        "test": (
            test_result
        ),

        "history": (
            history
        ),
    }

    json_path = (
        output_dir
        / "level0_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(
        "Saved results:",
        json_path,
    )

    print(
        "Saved best checkpoint:",
        output_dir
        / "checkpoint_best.pt",
    )


if __name__ == "__main__":
    main()