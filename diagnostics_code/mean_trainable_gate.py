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
# SYNTHETIC ANSWERS
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
# MEMORY CONFIG
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
        fact_templates = TRAIN_FACT_TEMPLATES
        query_templates = TRAIN_QUERY_TEMPLATES
    else:
        fact_templates = EVAL_FACT_TEMPLATES
        query_templates = EVAL_QUERY_TEMPLATES

    examples = []

    for i in range(n):

        entity = f"person_{start_id + i}"
        answer = rng.choice(answers)

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
# TOKENIZATION
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
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
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
# PRECOMPUTE MASKED-MEAN FACT SUMMARY + QUERY
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

        # ----------------------------------------------------
        # FACT
        # ----------------------------------------------------

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
            weights.sum(dim=1)
            .clamp_min(1.0)
        )

        # ----------------------------------------------------
        # QUERY
        # ----------------------------------------------------

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
            summary.cpu()
        )

        queries.append(
            query.cpu()
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
    }


# ============================================================
# DATASET
# ============================================================

class RepDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data[
            "labels"
        ].size(0)

    def __getitem__(self, i):

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
# TRAINABLE GATE MODEL
#
# mean summary
#     ↓
# trainable Linear VALUE
#     ↓
# TRAINABLE ORIGINAL VECTOR GATE
#     ↓
# frozen MemoryBank
#     ↓
# simple reader
# ============================================================

class MeanTrainableGate(nn.Module):

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

        # ----------------------------------------------------
        # ORIGINAL MEMORY BANK
        # ----------------------------------------------------

        self.memory_bank = (
            original.memory_bank
        )

        # ----------------------------------------------------
        # ORIGINAL VECTOR GATE
        #
        # THIS ONE WILL BE TRAINABLE
        # ----------------------------------------------------

        self.write_gate = (
            original.write_gate_module
        )

        # Freeze MemoryBank
        for p in (
            self.memory_bank
            .parameters()
        ):
            p.requires_grad = False

        # TRAIN VectorGate
        for p in (
            self.write_gate
            .parameters()
        ):
            p.requires_grad = True

        # ----------------------------------------------------
        # TRAINABLE MEAN -> VALUE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # TRAINABLE SIMPLE READER
        # ----------------------------------------------------

        self.reader = (
            SimpleReader(
                self.d_model,
                classes,
            )
        )

    # ========================================================
    # SLOT MASK
    # ========================================================

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

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        summary,
        query,
    ):

        batch = summary.size(0)

        # ----------------------------------------------------
        # LATENT VALUE
        # ----------------------------------------------------

        value = (
            self.value_projection(
                summary
            )
        )

        # ----------------------------------------------------
        # INITIAL MEMORY
        # ----------------------------------------------------

        state = (
            self.memory_bank
            .initialize(
                batch_size=batch,
                device=summary.device,
                dtype=summary.dtype,
            )
        )

        # ----------------------------------------------------
        # VALUE AS CANDIDATE
        # ----------------------------------------------------

        candidate = (
            state.slots.clone()
        )

        candidate[
            :,
            self.forced_slot,
            :,
        ] = value

        # ----------------------------------------------------
        # ONLY ONE SLOT
        # ----------------------------------------------------

        slot_mask = (
            self.slot_mask(
                batch,
                summary.device,
            )
        )

        # ----------------------------------------------------
        # TRAINABLE ORIGINAL VECTOR GATE
        # ----------------------------------------------------

        write_gate = (
            self.write_gate(
                summary,
                slot_mask=slot_mask,
            )
        )

        # ----------------------------------------------------
        # REAL MEMORY BANK
        # ----------------------------------------------------

        new_state = (
            self.memory_bank(
                state=state,
                candidate=candidate,
                write_gate=write_gate,
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
                :,
            ]
        )

        logits = (
            self.reader(
                query,
                stored,
            )
        )

        forced_gate = (
            write_gate[
                :,
                self.forced_slot,
                0,
            ]
        )

        return {
            "logits": logits,
            "raw_value": value,
            "stored": stored,
            "gate": forced_gate,
        }


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

            if cpu[j] != cpu[i]:

                chosen = j
                break

        if chosen is None:

            raise RuntimeError(
                "Could not make mismatch."
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

    distance = torch.cdist(
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
        distance[
            mask
        ].mean()
    )

    mean_cos = (
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
                / (
                    mean_norm
                    + 1e-8
                )
            ).item()
        ),

        "mean_cosine": float(
            mean_cos.item()
        ),
    }


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

    model.eval()

    loader = DataLoader(
        RepDataset(data),
        batch_size=batch_size,
        shuffle=False,
    )

    logits_all = []
    stored_all = []
    raw_all = []
    gate_all = []

    query_all = []
    label_all = []

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

        output = model(
            summary,
            query,
        )

        logits_all.append(
            output["logits"]
        )

        stored_all.append(
            output["stored"]
        )

        raw_all.append(
            output["raw_value"]
        )

        gate_all.append(
            output["gate"]
        )

        query_all.append(
            query
        )

        label_all.append(
            labels
        )

    logits = torch.cat(
        logits_all,
        dim=0,
    )

    stored = torch.cat(
        stored_all,
        dim=0,
    )

    raw = torch.cat(
        raw_all,
        dim=0,
    )

    gates = torch.cat(
        gate_all,
        dim=0,
    )

    query = torch.cat(
        query_all,
        dim=0,
    )

    labels = torch.cat(
        label_all,
        dim=0,
    )

    # --------------------------------------------------------
    # MATCHED
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MISMATCHED MEMORY
    # --------------------------------------------------------

    wrong_idx = (
        deterministic_mismatch(
            labels
        )
    )

    wrong_values = (
        stored[
            wrong_idx
        ]
    )

    wrong_logits = (
        model.reader(
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

    wrong_accuracy = (
        wrong_logits
        .argmax(dim=-1)
        .eq(labels)
        .float()
        .mean()
        .item()
        * 100
    )

    # --------------------------------------------------------
    # QUERY ONLY
    # --------------------------------------------------------

    zero_values = (
        torch.zeros_like(
            stored
        )
    )

    query_logits = (
        model.reader(
            query,
            zero_values,
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

    query_loss = float(
        F.cross_entropy(
            query_logits,
            labels,
        ).item()
    )

    gap = (
        wrong_losses
        - matched_losses
    )

    return {
        "matched_accuracy": float(
            matched_accuracy
        ),

        "mismatched_accuracy": float(
            wrong_accuracy
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
            wrong_losses
            .mean()
            .item()
        ),

        "query_only_loss": (
            query_loss
        ),

        "nll_gap": float(
            gap.mean()
            .item()
        ),

        "positive_gap_fraction": float(
            (
                gap > 0
            )
            .float()
            .mean()
            .item()
            * 100
        ),

        "gate_mean": float(
            gates.mean().item()
        ),

        "gate_std": float(
            gates.std(
                unbiased=False
            ).item()
        ),

        "gate_min": float(
            gates.min().item()
        ),

        "gate_max": float(
            gates.max().item()
        ),

        "raw_geometry": geometry(
            raw
        ),

        "stored_geometry": geometry(
            stored
        ),
    }


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

    # MemoryBank stays frozen/eval
    model.memory_bank.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_rank = 0.0
    count = 0

    all_gates = []

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

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            summary,
            query,
        )

        logits = (
            output["logits"]
        )

        stored = (
            output["stored"]
        )

        matched_losses = (
            F.cross_entropy(
                logits,
                labels,
                reduction="none",
            )
        )

        ce = (
            matched_losses.mean()
        )

        # ----------------------------------------------------
        # WRONG MEMORY
        # ----------------------------------------------------

        wrong_idx = (
            random_mismatch(
                labels
            )
        )

        wrong_values = (
            stored[
                wrong_idx
            ]
        )

        wrong_logits = (
            model.reader(
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

        rank_loss = F.relu(
            mismatch_margin
            + matched_losses
            - wrong_losses
        ).mean()

        loss = (
            ce
            + mismatch_weight
            * rank_loss
        )

        loss.backward()

        trainable = [
            p
            for p in model.parameters()
            if p.requires_grad
        ]

        torch.nn.utils.clip_grad_norm_(
            trainable,
            max_norm=5.0,
        )

        optimizer.step()

        n = labels.size(0)

        count += n

        total_loss += (
            float(loss.item())
            * n
        )

        total_ce += (
            float(ce.item())
            * n
        )

        total_rank += (
            float(rank_loss.item())
            * n
        )

        all_gates.append(
            output["gate"]
            .detach()
            .cpu()
        )

    gates = torch.cat(
        all_gates,
        dim=0,
    )

    return {
        "loss": (
            total_loss / count
        ),

        "ce": (
            total_ce / count
        ),

        "rank": (
            total_rank / count
        ),

        "gate_mean": float(
            gates.mean().item()
        ),

        "gate_std": float(
            gates.std(
                unbiased=False
            ).item()
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

    # Separate gate LR in case we want it smaller later.
    parser.add_argument(
        "--gate-learning-rate",
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
        default=(
            "outputs/"
            "mean_trainable_gate"
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
    print(
        "MEAN VALUE + TRAINABLE VECTOR GATE"
    )
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
        "Pipeline:"
    )

    print(
        "masked mean"
    )

    print(
        "  -> trainable Linear VALUE"
    )

    print(
        "  -> TRAINABLE original VectorGate"
    )

    print(
        "  -> frozen MemoryBank"
    )

    print(
        "  -> trainable simple reader"
    )

    print()
    print(
        "Disabled:"
    )

    print(
        "  CandidateWriter"
    )

    print(
        "  OrthogonalUpdate"
    )

    print(
        "  Router"
    )

    print(
        "  Original MemoryReader"
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
        "Number classes:",
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
    # MODEL
    # ========================================================

    original = load_original(
        args.checkpoint,
        args.model_name,
        device,
    )

    # ========================================================
    # DATA
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

    valid_data = precompute(
        original,
        tokenizer,
        valid_examples,
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
    # TRAINABLE MODEL
    # ========================================================

    model = MeanTrainableGate(
        original=original,
        classes=len(answers),
        forced_slot=(
            args.forced_slot
        ),
    ).to(device)

    value_and_reader_params = []

    value_and_reader_params += list(
        model.value_projection
        .parameters()
    )

    value_and_reader_params += list(
        model.reader
        .parameters()
    )

    gate_params = list(
        model.write_gate
        .parameters()
    )

    print()
    print(
        "Value + reader params:",
        f"{sum(p.numel() for p in value_and_reader_params):,}"
    )

    print(
        "Gate params:",
        f"{sum(p.numel() for p in gate_params):,}"
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

    print(
        "VectorGate trainable:",
        any(
            p.requires_grad
            for p in (
                model.write_gate
                .parameters()
            )
        ),
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    value_and_reader_params
                ),
                "lr": (
                    args.learning_rate
                ),
            },

            {
                "params": (
                    gate_params
                ),
                "lr": (
                    args.gate_learning_rate
                ),
            },
        ],
        weight_decay=1e-4,
    )

    train_loader = DataLoader(
        RepDataset(
            train_data
        ),
        batch_size=(
            args.batch_size
        ),
        shuffle=True,
    )

    # ========================================================
    # PRETRAIN
    # ========================================================

    pre = evaluate(
        model,
        valid_data,
        device,
        args.batch_size,
    )

    print()
    print("=" * 90)
    print(
        "PRETRAIN VALIDATION"
    )
    print("=" * 90)

    print(
        f"Matched:     "
        f"{pre['matched_accuracy']:.2f}%"
    )

    print(
        f"Mismatched:  "
        f"{pre['mismatched_accuracy']:.2f}%"
    )

    print(
        f"NLL gap:     "
        f"{pre['nll_gap']:+.6f}"
    )

    print(
        f"Gate mean:   "
        f"{pre['gate_mean']:.6f}"
    )

    print(
        f"Gate std:    "
        f"{pre['gate_std']:.6f}"
    )

    print(
        f"Stored relL2:"
        f" {pre['stored_geometry']['relative_l2']:.6f}"
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

        validation = evaluate(
            model,
            valid_data,
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
            f"gate={validation['gate_mean']:.4f} "
            f"±{validation['gate_std']:.4f} | "
            f"relL2={validation['stored_geometry']['relative_l2']:.4f}"
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

        if (
            validation["nll_gap"]
            > best_gap
        ):

            best_gap = (
                validation[
                    "nll_gap"
                ]
            )

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
                    "args": vars(args),
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
    # TEST
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
        f"FINAL TEST — BEST EPOCH {best_epoch}"
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

    print()
    print(
        "VECTOR GATE"
    )

    print(
        f"mean:          "
        f"{test['gate_mean']:.6f}"
    )

    print(
        f"std:           "
        f"{test['gate_std']:.6f}"
    )

    print(
        f"min:           "
        f"{test['gate_min']:.6f}"
    )

    print(
        f"max:           "
        f"{test['gate_max']:.6f}"
    )

    print()
    print(
        "RAW VALUE GEOMETRY"
    )

    for key, value in (
        test[
            "raw_geometry"
        ].items()
    ):

        print(
            f"{key:<24}"
            f"{value:.6f}"
        )

    print()
    print(
        "STORED VALUE GEOMETRY"
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

    # ========================================================
    # VERDICT
    # ========================================================

    print()
    print("=" * 90)
    print(
        "VERDICT"
    )
    print("=" * 90)

    if (
        test[
            "matched_accuracy"
        ] >= 70
        and
        test[
            "matched_accuracy"
        ]
        - test[
            "mismatched_accuracy"
        ] >= 50
    ):

        print(
            "PASS: training the VectorGate recovers "
            "the mean-pooled latent VALUE pathway."
        )

        print(
            "Conclusion: the VectorGate architecture "
            "can work; the frozen checkpoint gate was "
            "poorly calibrated."
        )

    elif (
        test[
            "matched_accuracy"
        ] >= 40
    ):

        print(
            "PARTIAL RECOVERY."
        )

        print(
            "Gate training helps, but the gating "
            "mechanism still attenuates the VALUE strongly."
        )

    else:

        print(
            "FAIL: trainable VectorGate did not recover "
            "the mean-pooled VALUE."
        )

        print(
            "Next step: test gate formulation itself "
            "(minimum write floor / residual update)."
        )

    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        output_dir
        / "trainable_gate_results.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "best_epoch": (
                    best_epoch
                ),
                "best_validation_gap": (
                    best_gap
                ),
                "pretrain": pre,
                "test": test,
                "history": history,
            },
            f,
            indent=2,
        )

    print()
    print(
        "Saved:",
        output_dir
        / "trainable_gate_results.json",
    )


if __name__ == "__main__":
    main()