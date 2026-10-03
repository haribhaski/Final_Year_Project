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
# EXAMPLES
# ============================================================

def get_answers(tokenizer):

    usable = {}

    for word in ANSWER_POOL:

        ids = tokenizer(
            " " + word,
            add_special_tokens=False,
        )["input_ids"]

        if len(ids) == 1:
            usable[word] = ids[0]

    return usable


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

    output = []

    for i in range(n):

        entity = f"person_{start_id + i}"
        answer = rng.choice(answers)

        output.append(
            {
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

                "answer": answer,
            }
        )

    return output


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
# LOAD ORIGINAL
# ============================================================

def load_original(
    checkpoint_path,
    model_name,
    device,
):

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
# PRECOMPUTE
#
# FACT SOURCE = MASKED MEAN ONLY
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
            start:
            start + batch_size
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
# MEAN LADDER
#
# M0 = mean -> Linear -> reader
#
# M1 = mean -> Linear
#             -> MemoryBank
#
# M2 = mean -> Linear
#             -> frozen original gate
#             -> MemoryBank
#
# M3 = mean -> Linear
#             -> OrthogonalUpdate
#             -> frozen gate
#             -> MemoryBank
# ============================================================

class MeanValueLevel(nn.Module):

    def __init__(
        self,
        original,
        classes,
        level,
        forced_slot,
    ):

        super().__init__()

        self.level = level

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

        self.orthogonalizer = copy.deepcopy(
            original.orthogonalizer
        )

        # Freeze original modules.
        for module in [
            self.memory_bank,
            self.write_gate,
            self.orthogonalizer,
        ]:

            for p in module.parameters():
                p.requires_grad = False

        # Trainable mean -> VALUE map.
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

        self.reader = SimpleReader(
            self.d_model,
            classes,
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

        value = self.value_projection(
            summary
        )

        # ----------------------------------------------------
        # M0
        # ----------------------------------------------------

        if self.level == 0:

            stored = value

            logits = self.reader(
                query,
                stored,
            )

            return {
                "logits": logits,
                "stored": stored,
                "raw": value,
                "gate": None,
            }

        batch = value.size(0)

        state = (
            self.memory_bank
            .initialize(
                batch_size=batch,
                device=value.device,
                dtype=value.dtype,
            )
        )

        mask = self.slot_mask(
            batch,
            value.device,
        )

        # ----------------------------------------------------
        # M1
        #
        # MemoryBank with gate exactly 1.
        # ----------------------------------------------------

        if self.level == 1:

            candidate = (
                state.slots.clone()
            )

            candidate[
                :,
                self.forced_slot,
                :,
            ] = value

            gate = torch.zeros(
                batch,
                self.num_slots,
                1,
                device=value.device,
                dtype=value.dtype,
            )

            gate[
                :,
                self.forced_slot,
                0,
            ] = 1.0

        # ----------------------------------------------------
        # M2
        #
        # Frozen real gate + MemoryBank
        # ----------------------------------------------------

        elif self.level == 2:

            candidate = (
                state.slots.clone()
            )

            candidate[
                :,
                self.forced_slot,
                :,
            ] = value

            gate = self.write_gate(
                summary,
                slot_mask=mask,
            )

        # ----------------------------------------------------
        # M3
        #
        # OrthogonalUpdate + real gate + MemoryBank
        # ----------------------------------------------------

        elif self.level == 3:

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

            ortho = (
                self.orthogonalizer(
                    updates=updates,
                    memory_slots=(
                        state.slots
                    ),
                )
            )

            candidate = (
                state.slots
                + ortho.updates
            )

            gate = self.write_gate(
                summary,
                slot_mask=mask,
            )

        else:

            raise ValueError(
                f"Invalid level: {self.level}"
            )

        new_state = (
            self.memory_bank(
                state=state,
                candidate=candidate,
                write_gate=gate,
                write_mask=(
                    mask.unsqueeze(-1)
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

        logits = self.reader(
            query,
            stored,
        )

        return {
            "logits": logits,
            "stored": stored,
            "raw": value,
            "gate": (
                gate[
                    :,
                    self.forced_slot,
                    0,
                ]
            ),
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

        for offset in range(
            1,
            labels.size(0),
        ):

            j = (
                i + offset
            ) % labels.size(0)

            if cpu[j] != cpu[i]:

                result.append(j)
                break

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

    x = x.detach().float().cpu()

    if x.size(0) > 500:

        ids = torch.linspace(
            0,
            x.size(0) - 1,
            500,
        ).long()

        x = x[ids]

    mean_norm = (
        x.norm(dim=-1)
        .mean()
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

    mean_l2 = distance[
        mask
    ].mean()

    mean_cos = cosine[
        mask
    ].mean()

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
    values_all = []
    raw_all = []
    gates_all = []

    labels_all = []
    query_all = []

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

        values_all.append(
            output["stored"]
        )

        raw_all.append(
            output["raw"]
        )

        if output["gate"] is not None:

            gates_all.append(
                output["gate"]
            )

        labels_all.append(
            labels
        )

        query_all.append(
            query
        )

    logits = torch.cat(
        logits_all,
        dim=0,
    )

    values = torch.cat(
        values_all,
        dim=0,
    )

    raw_values = torch.cat(
        raw_all,
        dim=0,
    )

    labels = torch.cat(
        labels_all,
        dim=0,
    )

    query = torch.cat(
        query_all,
        dim=0,
    )

    matched_losses = (
        F.cross_entropy(
            logits,
            labels,
            reduction="none",
        )
    )

    matched_acc = (
        logits.argmax(dim=-1)
        .eq(labels)
        .float()
        .mean()
        .item()
        * 100
    )

    # --------------------------------------------------------
    # MISMATCH
    # --------------------------------------------------------

    wrong_idx = (
        deterministic_mismatch(
            labels
        )
    )

    wrong_values = values[
        wrong_idx
    ]

    wrong_logits = model.reader(
        query,
        wrong_values,
    )

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

    # --------------------------------------------------------
    # QUERY ONLY
    # --------------------------------------------------------

    zero = torch.zeros_like(
        values
    )

    query_logits = model.reader(
        query,
        zero,
    )

    query_acc = (
        query_logits
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

    result = {
        "matched_accuracy": float(
            matched_acc
        ),

        "mismatched_accuracy": float(
            wrong_acc
        ),

        "query_only_accuracy": float(
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

        "raw_geometry": geometry(
            raw_values
        ),

        "stored_geometry": geometry(
            values
        ),
    }

    if len(gates_all) > 0:

        gates = torch.cat(
            gates_all,
            dim=0,
        )

        result[
            "gate_mean"
        ] = float(
            gates.mean().item()
        )

        result[
            "gate_std"
        ] = float(
            gates.std(
                unbiased=False
            ).item()
        )

    else:

        result[
            "gate_mean"
        ] = None

        result[
            "gate_std"
        ] = None

    return result


# ============================================================
# TRAIN
# ============================================================

def train_level(
    level,
    original,
    train_data,
    valid_data,
    test_data,
    classes,
    device,
    args,
):

    print()
    print("#" * 90)
    print(
        f"MEAN LEVEL M{level}"
    )
    print("#" * 90)

    set_seed(
        args.seed
    )

    model = MeanValueLevel(
        original=original,
        classes=classes,
        level=level,
        forced_slot=(
            args.forced_slot
        ),
    ).to(device)

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = (
        torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            weight_decay=1e-4,
        )
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

    best_gap = -float("inf")
    best_epoch = -1
    best_state = None

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        model.memory_bank.eval()
        model.write_gate.eval()
        model.orthogonalizer.eval()

        total_loss = 0.0
        total_ce = 0.0
        total_rank = 0.0
        count = 0

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

            values = (
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

            # -----------------------------------------------
            # DIFFERENT-ANSWER MISMATCH
            # -----------------------------------------------

            wrong_idx = (
                random_mismatch(
                    labels
                )
            )

            wrong_values = (
                values[
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
                args.mismatch_margin
                + matched_losses
                - wrong_losses
            ).mean()

            loss = (
                ce
                + args.mismatch_weight
                * rank_loss
            )

            loss.backward()

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

        validation = evaluate(
            model,
            valid_data,
            device,
            args.batch_size,
        )

        print(
            f"M{level} E{epoch:02d} | "
            f"loss={total_loss / count:.4f} | "
            f"match={validation['matched_accuracy']:.2f}% | "
            f"mismatch={validation['mismatched_accuracy']:.2f}% | "
            f"query={validation['query_only_accuracy']:.2f}% | "
            f"gap={validation['nll_gap']:+.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": (
                    total_loss / count
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

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model
                .state_dict()
                .items()
            }

    model.load_state_dict(
        best_state,
        strict=True,
    )

    model.to(device)

    test = evaluate(
        model,
        test_data,
        device,
        args.batch_size,
    )

    print()
    print(
        f"M{level} TEST"
    )

    print(
        f"Matched:       "
        f"{test['matched_accuracy']:.2f}%"
    )

    print(
        f"Mismatched:    "
        f"{test['mismatched_accuracy']:.2f}%"
    )

    print(
        f"Query only:    "
        f"{test['query_only_accuracy']:.2f}%"
    )

    print(
        f"NLL gap:       "
        f"{test['nll_gap']:+.6f}"
    )

    print(
        f"Raw rel L2:    "
        f"{test['raw_geometry']['relative_l2']:.6f}"
    )

    print(
        f"Stored rel L2: "
        f"{test['stored_geometry']['relative_l2']:.6f}"
    )

    print(
        f"Stored cosine: "
        f"{test['stored_geometry']['mean_cosine']:.6f}"
    )

    if (
        test["gate_mean"]
        is not None
    ):

        print(
            f"Gate mean:     "
            f"{test['gate_mean']:.6f}"
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
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
            "test": test,
        },
        output_dir
        / f"mean_level_{level}.pt",
    )

    return {
        "best_epoch": best_epoch,
        "test": test,
        "history": history,
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
            "mean_value_ladder"
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
        "MEAN-POOL VALUE LADDER"
    )
    print("=" * 90)

    print(
        "Device:",
        device,
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

    answers_map = (
        get_answers(
            tokenizer
        )
    )

    answers = list(
        answers_map.keys()
    )

    answer_to_class = {
        word: i
        for i, word
        in enumerate(answers)
    }

    print(
        "Chance accuracy:",
        f"{100 / len(answers):.2f}%"
    )

    print(
        "Chance CE:",
        f"{math.log(len(answers)):.4f}"
    )

    original = load_original(
        args.checkpoint,
        args.model_name,
        device,
    )

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

    results = {}

    # ========================================================
    # RUN M0-M3 INDEPENDENTLY
    # ========================================================

    for level in [
        0,
        1,
        2,
        3,
    ]:

        result = train_level(
            level=level,
            original=original,
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            classes=len(answers),
            device=device,
            args=args,
        )

        results[
            f"M{level}"
        ] = result

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print("=" * 100)
    print(
        "FINAL MEAN-POOL LADDER"
    )
    print("=" * 100)

    print(
        f"{'LEVEL':<12}"
        f"{'MATCH':>12}"
        f"{'MISMATCH':>14}"
        f"{'QUERY':>12}"
        f"{'NLL GAP':>14}"
        f"{'REL L2':>12}"
    )

    print("-" * 78)

    for level in [
        0,
        1,
        2,
        3,
    ]:

        r = results[
            f"M{level}"
        ]["test"]

        print(
            f"M{level:<11}"
            f"{r['matched_accuracy']:>11.2f}%"
            f"{r['mismatched_accuracy']:>13.2f}%"
            f"{r['query_only_accuracy']:>11.2f}%"
            f"{r['nll_gap']:>14.4f}"
            f"{r['stored_geometry']['relative_l2']:>12.4f}"
        )

    print()
    print(
        "Meaning:"
    )

    print(
        "M0 = Mean -> Linear"
    )

    print(
        "M1 = M0 + MemoryBank"
    )

    print(
        "M2 = M1 + frozen original VectorGate"
    )

    print(
        "M3 = M2 + original OrthogonalUpdate"
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir
        / "mean_ladder_results.json",
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
        "Saved:",
        output_dir
        / "mean_ladder_results.json",
    )


if __name__ == "__main__":
    main()