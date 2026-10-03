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
# ORIGINAL MEMORY CONFIG
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
# ANSWER TOKENS
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
                "entity": entity,
                "answer": answer,
                "fact": fact,
                "query": query,
            }
        )

    return examples


# ============================================================
# LOAD ORIGINAL MODEL
# ============================================================

def load_original_model(
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
        add_special_tokens=False,
        return_tensors="pt",
    )

    return (
        encoded[
            "input_ids"
        ].to(device),

        encoded[
            "attention_mask"
        ].to(device),
    )


# ============================================================
# FEATURE EXTRACTION
#
# For each example we need:
#
# summary:
#   masked-mean final GPT2 states
#
# query:
#   last query hidden state
#
# token_states:
#   all fact token states
#
# ============================================================

@torch.no_grad()
def precompute(
    original,
    tokenizer,
    examples,
    answer_to_class,
    device,
):

    summaries = []
    queries = []
    labels = []

    token_states = []
    token_masks = []

    for i, example in enumerate(
        examples
    ):

        # ----------------------------------------------------
        # FACT
        # ----------------------------------------------------

        fact_ids, fact_mask = (
            tokenize(
                tokenizer,
                [example["fact"]],
                device,
            )
        )

        fact_output = (
            original
            .backbone
            .transformer(
                input_ids=fact_ids,
                attention_mask=fact_mask,
                return_dict=True,
            )
        )

        hidden = (
            fact_output
            .last_hidden_state
        )

        weights = (
            fact_mask
            .unsqueeze(-1)
            .to(hidden.dtype)
        )

        summary = (
            (hidden * weights)
            .sum(dim=1)
            /
            weights.sum(dim=1)
            .clamp_min(1.0)
        )[0]

        # ----------------------------------------------------
        # QUERY
        # ----------------------------------------------------

        query_ids, query_mask = (
            tokenize(
                tokenizer,
                [example["query"]],
                device,
            )
        )

        query_output = (
            original
            .backbone
            .transformer(
                input_ids=query_ids,
                attention_mask=query_mask,
                return_dict=True,
            )
        )

        query_hidden = (
            query_output
            .last_hidden_state
        )

        query_last = (
            query_hidden[
                0,
                int(
                    query_mask
                    .sum()
                    .item()
                ) - 1,
                :
            ]
        )

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        summaries.append(
            summary.cpu()
        )

        queries.append(
            query_last.cpu()
        )

        labels.append(
            answer_to_class[
                example["answer"]
            ]
        )

        token_states.append(
            hidden[0].cpu()
        )

        token_masks.append(
            fact_mask[0].cpu()
        )

        if (
            (i + 1) % 500 == 0
            or i + 1
            == len(examples)
        ):

            print(
                f"  {i + 1}/"
                f"{len(examples)}"
            )

    return {
        "summary": torch.stack(
            summaries
        ),

        "query": torch.stack(
            queries
        ),

        "labels": torch.tensor(
            labels,
            dtype=torch.long,
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
# PAD FACT TOKEN STATES FOR CANDIDATE WRITER
# ============================================================

def build_token_batch(
    data,
    indices,
    device,
):

    tensors = [
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
        for x in tensors
    )

    d_model = (
        tensors[0]
        .size(-1)
    )

    states = torch.zeros(
        len(tensors),
        max_len,
        d_model,
        dtype=tensors[0].dtype,
        device=device,
    )

    attention_mask = torch.zeros(
        len(tensors),
        max_len,
        dtype=torch.long,
        device=device,
    )

    for j, x in enumerate(
        tensors
    ):

        length = x.size(0)

        states[
            j,
            :length,
            :,
        ] = x.to(device)

        attention_mask[
            j,
            :length,
        ] = masks[j].to(
            device
        )

    return (
        states,
        attention_mask,
    )


# ============================================================
# SIMPLE READER
#
# EXACT SAME READER FOR BOTH VARIANTS
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
# WRITER MICRO-ABLATION MODEL
# ============================================================

class WriterExperiment(nn.Module):

    def __init__(
        self,
        original,
        num_classes,
        variant,
        forced_slot,
    ):

        super().__init__()

        self.variant = (
            variant
        )

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
        # COPY ORIGINAL COMPONENTS
        # ----------------------------------------------------

        self.memory_bank = (
            copy.deepcopy(
                original.memory_bank
            )
        )

        self.write_gate = (
            copy.deepcopy(
                original
                .write_gate_module
            )
        )

        self.orthogonalizer = (
            copy.deepcopy(
                original
                .orthogonalizer
            )
        )

        self.writer = (
            copy.deepcopy(
                original.writer
            )
        )

        # ----------------------------------------------------
        # Freeze all original modules
        # ----------------------------------------------------

        for module in [
            self.memory_bank,
            self.write_gate,
            self.orthogonalizer,
            self.writer,
        ]:

            for p in module.parameters():
                p.requires_grad = False

        # ----------------------------------------------------
        # DIRECT SUMMARY writer
        #
        # Used ONLY by direct_summary variant.
        # ----------------------------------------------------

        self.summary_value = (
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
        # SAME READER FOR BOTH
        # ----------------------------------------------------

        self.simple_reader = (
            SimpleReader(
                self.d_model,
                num_classes,
            )
        )

    # ========================================================
    # INITIAL MEMORY
    # ========================================================

    def initialize(
        self,
        batch,
        device,
        dtype,
    ):

        return (
            self.memory_bank
            .initialize(
                batch_size=batch,
                device=device,
                dtype=dtype,
            )
        )

    # ========================================================
    # SLOT MASK
    # ========================================================

    def get_slot_mask(
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
    # DIRECT SUMMARY WRITER
    # ========================================================

    def direct_summary_write(
        self,
        summary,
    ):

        batch = (
            summary.size(0)
        )

        state = self.initialize(
            batch,
            summary.device,
            summary.dtype,
        )

        # -----------------------------------------------
        # Trainable summary → latent VALUE
        # -----------------------------------------------

        value = (
            self.summary_value(
                summary
            )
        )

        # -----------------------------------------------
        # Convert into slot update.
        #
        # This mirrors Level 3:
        #
        # update = desired_value - old_slot
        # -----------------------------------------------

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

        # -----------------------------------------------
        # ORIGINAL ORTHOGONAL UPDATE
        # -----------------------------------------------

        ortho = self.orthogonalizer(
            updates=updates,
            memory_slots=(
                state.slots
            ),
        )

        candidate = (
            state.slots
            + ortho.updates
        )

        # -----------------------------------------------
        # ORIGINAL WRITE GATE
        # -----------------------------------------------

        slot_mask = (
            self.get_slot_mask(
                batch,
                summary.device,
            )
        )

        gate = self.write_gate(
            summary,
            slot_mask=slot_mask,
        )

        # -----------------------------------------------
        # ORIGINAL MEMORY BANK
        # -----------------------------------------------

        new_state = (
            self.memory_bank(
                state=state,
                candidate=candidate,
                write_gate=gate,
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

        return (
            stored,
            value,
            gate[
                :,
                self.forced_slot,
                0,
            ],
        )

    # ========================================================
    # ORIGINAL CANDIDATE WRITER
    # ========================================================

    def original_writer_write(
        self,
        summary,
        token_states,
        attention_mask,
    ):

        batch = (
            summary.size(0)
        )

        state = self.initialize(
            batch,
            summary.device,
            summary.dtype,
        )

        slot_mask = (
            self.get_slot_mask(
                batch,
                summary.device,
            )
        )

        # -----------------------------------------------
        # Force writer to slot 0.
        #
        # We are NOT testing routing here.
        # -----------------------------------------------

        routing_weights = (
            torch.zeros(
                batch,
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
            self.forced_slot,
        ] = 1.0

        # -----------------------------------------------
        # ORIGINAL CANDIDATE WRITER
        # -----------------------------------------------

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

        # -----------------------------------------------
        # ORIGINAL ORTHOGONAL UPDATE
        # -----------------------------------------------

        ortho = (
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
            + ortho.updates
        )

        # -----------------------------------------------
        # ORIGINAL WRITE GATE
        # -----------------------------------------------

        gate = self.write_gate(
            summary,
            slot_mask=slot_mask,
        )

        # -----------------------------------------------
        # ORIGINAL MEMORY BANK
        # -----------------------------------------------

        new_state = (
            self.memory_bank(
                state=state,
                candidate=candidate,
                write_gate=gate,
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

        # CandidateWriter value before downstream modules
        writer_candidate = (
            writer_output
            .candidates[
                :,
                self.forced_slot,
                :,
            ]
        )

        return (
            stored,
            writer_candidate,
            gate[
                :,
                self.forced_slot,
                0,
            ],
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        summary,
        query,
        token_states=None,
        attention_mask=None,
    ):

        if (
            self.variant
            == "direct_summary"
        ):

            (
                stored,
                source_value,
                gate,
            ) = (
                self.direct_summary_write(
                    summary
                )
            )

        elif (
            self.variant
            == "original_writer"
        ):

            if token_states is None:
                raise RuntimeError(
                    "token_states required "
                    "for original writer."
                )

            (
                stored,
                source_value,
                gate,
            ) = (
                self
                .original_writer_write(
                    summary,
                    token_states,
                    attention_mask,
                )
            )

        else:

            raise ValueError(
                self.variant
            )

        logits = (
            self.simple_reader(
                query,
                stored,
            )
        )

        return {
            "logits": logits,
            "stored": stored,
            "source_value": (
                source_value
            ),
            "gate": gate,
        }


# ============================================================
# MISMATCH INDICES
#
# GUARANTEE DIFFERENT ANSWER
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

        options = torch.nonzero(
            labels != labels[i],
            as_tuple=False,
        ).flatten()

        if options.numel() == 0:
            raise RuntimeError(
                "Need multiple answer classes "
                "inside batch."
            )

        result[i] = options[
            torch.randint(
                0,
                options.numel(),
                (1,),
                device=labels.device,
            )
        ]

    return result


def deterministic_mismatch(
    labels,
):

    labels_cpu = (
        labels.cpu()
    )

    result = []

    n = labels.size(0)

    for i in range(n):

        chosen = None

        for offset in range(
            1,
            n,
        ):

            j = (
                i + offset
            ) % n

            if (
                labels_cpu[j]
                != labels_cpu[i]
            ):

                chosen = j
                break

        if chosen is None:
            raise RuntimeError(
                "Unable to build mismatch."
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

        idx = torch.linspace(
            0,
            x.size(0) - 1,
            500,
        ).long()

        x = x[idx]

    norm = (
        x.norm(
            dim=-1
        ).mean()
    )

    normalized = F.normalize(
        x,
        dim=-1,
    )

    cosine = (
        normalized
        @ normalized.T
    )

    distance = torch.cdist(
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

    mean_l2 = (
        distance[
            mask
        ].mean()
    )

    return {
        "mean_norm": float(
            norm.item()
        ),

        "mean_pairwise_l2": float(
            mean_l2.item()
        ),

        "relative_l2": float(
            (
                mean_l2
                / (
                    norm
                    + 1e-8
                )
            ).item()
        ),

        "mean_cosine": float(
            cosine[
                mask
            ].mean().item()
        ),
    }


# ============================================================
# RUN MODEL ON COMPLETE DATASET
# ============================================================

@torch.no_grad()
def collect_outputs(
    model,
    data,
    device,
    batch_size,
):

    loader = DataLoader(
        IndexDataset(data),
        batch_size=batch_size,
        shuffle=False,
    )

    logits_all = []
    stored_all = []
    source_all = []
    gate_all = []

    for (
        indices,
        summary,
        query,
        _,
    ) in loader:

        summary = summary.to(
            device
        )

        query = query.to(
            device
        )

        if (
            model.variant
            == "original_writer"
        ):

            token_states, mask = (
                build_token_batch(
                    data,
                    indices,
                    device,
                )
            )

        else:

            token_states = None
            mask = None

        output = model(
            summary=summary,
            query=query,
            token_states=(
                token_states
            ),
            attention_mask=mask,
        )

        logits_all.append(
            output[
                "logits"
            ]
        )

        stored_all.append(
            output[
                "stored"
            ]
        )

        source_all.append(
            output[
                "source_value"
            ]
        )

        gate_all.append(
            output[
                "gate"
            ]
        )

    return {
        "logits": torch.cat(
            logits_all,
            dim=0,
        ),

        "stored": torch.cat(
            stored_all,
            dim=0,
        ),

        "source": torch.cat(
            source_all,
            dim=0,
        ),

        "gate": torch.cat(
            gate_all,
            dim=0,
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    data,
    device,
    batch_size,
):

    model.eval()

    labels = (
        data[
            "labels"
        ].to(device)
    )

    query = (
        data[
            "query"
        ].to(device)
    )

    output = collect_outputs(
        model,
        data,
        device,
        batch_size,
    )

    matched_logits = (
        output[
            "logits"
        ]
    )

    stored = (
        output[
            "stored"
        ]
    )

    matched_losses = (
        F.cross_entropy(
            matched_logits,
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

    # --------------------------------------------------------
    # MISMATCHED VALUE
    # --------------------------------------------------------

    wrong_indices = (
        deterministic_mismatch(
            labels
        )
    )

    wrong_values = (
        stored[
            wrong_indices
        ]
    )

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

    query_only_logits = (
        model.simple_reader(
            query,
            zero_values,
        )
    )

    query_only_accuracy = (
        query_only_logits
        .argmax(dim=-1)
        .eq(labels)
        .float()
        .mean()
        .item()
        * 100
    )

    query_only_loss = (
        F.cross_entropy(
            query_only_logits,
            labels,
        ).item()
    )

    # --------------------------------------------------------
    # GAP
    # --------------------------------------------------------

    gap = (
        wrong_losses
        - matched_losses
    )

    gate = (
        output[
            "gate"
        ]
    )

    return {
        "matched_accuracy": float(
            matched_accuracy
        ),

        "mismatched_accuracy": float(
            wrong_accuracy
        ),

        "query_only_accuracy": float(
            query_only_accuracy
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

        "query_only_loss": float(
            query_only_loss
        ),

        "nll_gap": float(
            gap.mean().item()
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
            gate.mean().item()
        ),

        "gate_std": float(
            gate.std(
                unbiased=False
            ).item()
        ),

        "source_geometry": (
            geometry(
                output[
                    "source"
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
# TRAIN ONE VARIANT
# ============================================================

def train_variant(
    variant,
    original,
    train_data,
    valid_data,
    test_data,
    num_classes,
    device,
    args,
):

    print()
    print("#" * 90)
    print(
        f"VARIANT: {variant.upper()}"
    )
    print("#" * 90)

    set_seed(
        args.seed
    )

    model = WriterExperiment(
        original=original,
        num_classes=num_classes,
        variant=variant,
        forced_slot=(
            args.forced_slot
        ),
    ).to(device)

    # --------------------------------------------------------
    # Train only:
    #
    # DIRECT SUMMARY:
    #     summary_value + simple reader
    #
    # ORIGINAL WRITER:
    #     simple reader only
    #
    # Original CandidateWriter remains frozen.
    # --------------------------------------------------------

    if (
        variant
        == "original_writer"
    ):

        for p in (
            model.summary_value
            .parameters()
        ):
            p.requires_grad = False

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    print(
        "Trainable parameters:",
        f"{sum(p.numel() for p in trainable):,}"
    )

    print(
        "Original CandidateWriter trainable:",
        any(
            p.requires_grad
            for p in (
                model.writer
                .parameters()
            )
        ),
    )

    optimizer = (
        torch.optim.AdamW(
            trainable,
            lr=(
                args.learning_rate
            ),
            weight_decay=1e-4,
        )
    )

    loader = DataLoader(
        IndexDataset(
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

        # Frozen original modules remain eval.
        model.memory_bank.eval()
        model.write_gate.eval()
        model.orthogonalizer.eval()
        model.writer.eval()

        total_loss = 0
        total_ce = 0
        total_rank = 0
        total_count = 0

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

            if (
                variant
                == "original_writer"
            ):

                (
                    token_states,
                    token_mask,
                ) = build_token_batch(
                    train_data,
                    indices,
                    device,
                )

            else:

                token_states = None
                token_mask = None

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
                    token_mask
                ),
            )

            logits = (
                output[
                    "logits"
                ]
            )

            values = (
                output[
                    "stored"
                ]
            )

            matched_losses = (
                F.cross_entropy(
                    logits,
                    labels,
                    reduction="none",
                )
            )

            matched_ce = (
                matched_losses.mean()
            )

            # ------------------------------------------------
            # DIFFERENT-ANSWER MISMATCH
            # ------------------------------------------------

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

            rank_loss = F.relu(
                args.mismatch_margin
                + matched_losses
                - wrong_losses
            ).mean()

            loss = (
                matched_ce
                + args.mismatch_weight
                * rank_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable,
                5.0,
            )

            optimizer.step()

            n = labels.size(0)

            total_count += n

            total_loss += (
                float(
                    loss.item()
                )
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

        validation = evaluate(
            model=model,
            data=valid_data,
            device=device,
            batch_size=(
                args.batch_size
            ),
        )

        epoch_result = {
            "epoch": epoch,

            "train_loss": (
                total_loss
                / total_count
            ),

            "train_ce": (
                total_ce
                / total_count
            ),

            "train_rank": (
                total_rank
                / total_count
            ),

            "validation": (
                validation
            ),
        }

        history.append(
            epoch_result
        )

        print(
            f"E{epoch:02d} | "
            f"loss="
            f"{total_loss / total_count:.4f} | "
            f"match="
            f"{validation['matched_accuracy']:.2f}% | "
            f"mismatch="
            f"{validation['mismatched_accuracy']:.2f}% | "
            f"query="
            f"{validation['query_only_accuracy']:.2f}% | "
            f"gap="
            f"{validation['nll_gap']:+.4f}"
        )

        if (
            validation[
                "nll_gap"
            ]
            > best_gap
        ):

            best_gap = (
                validation[
                    "nll_gap"
                ]
            )

            best_epoch = epoch

            best_state = {
                key: value
                .detach()
                .cpu()
                .clone()

                for key, value
                in model
                .state_dict()
                .items()
            }

    # ========================================================
    # BEST CHECKPOINT
    # ========================================================

    model.load_state_dict(
        best_state,
        strict=True,
    )

    model.to(device)
    model.eval()

    test_result = evaluate(
        model=model,
        data=test_data,
        device=device,
        batch_size=(
            args.batch_size
        ),
    )

    print()
    print(
        f"{variant.upper()} TEST"
    )

    print(
        f"Matched:      "
        f"{test_result['matched_accuracy']:.2f}%"
    )

    print(
        f"Mismatched:   "
        f"{test_result['mismatched_accuracy']:.2f}%"
    )

    print(
        f"Query only:   "
        f"{test_result['query_only_accuracy']:.2f}%"
    )

    print(
        f"NLL gap:      "
        f"{test_result['nll_gap']:+.6f}"
    )

    print(
        f"Gate mean:    "
        f"{test_result['gate_mean']:.6f}"
    )

    print()

    print(
        "SOURCE VALUE GEOMETRY"
    )

    for key, value in (
        test_result[
            "source_geometry"
        ].items()
    ):

        print(
            f"  {key:<22}"
            f"{value:.6f}"
        )

    print()

    print(
        "FINAL STORED GEOMETRY"
    )

    for key, value in (
        test_result[
            "stored_geometry"
        ].items()
    ):

        print(
            f"  {key:<22}"
            f"{value:.6f}"
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

    torch.save(
        {
            "variant": variant,
            "best_epoch": (
                best_epoch
            ),
            "model_state_dict": (
                model.state_dict()
            ),
            "test": test_result,
        },
        output_dir
        / (
            f"{variant}_best.pt"
        ),
    )

    return {
        "best_epoch": (
            best_epoch
        ),

        "test": (
            test_result
        ),

        "history": (
            history
        ),
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
        "--epochs",
        type=int,
        default=7,
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
            "writer_micro_ablation"
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
        "WRITER MICRO-ABLATION"
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
        "A = Direct masked-mean summary -> Linear VALUE"
    )

    print(
        "B = Original CandidateWriter"
    )

    print()

    print(
        "Both use same:"
    )

    print(
        "  OrthogonalUpdate"
    )

    print(
        "  frozen VectorGate"
    )

    print(
        "  MemoryBank"
    )

    print(
        "  simple reader"
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

    token_map = (
        get_single_token_answers(
            tokenizer
        )
    )

    answers = list(
        token_map.keys()
    )

    answer_to_class = {
        answer: i

        for i, answer
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

    original = (
        load_original_model(
            checkpoint_path=(
                args.checkpoint
            ),
            model_name=(
                args.model_name
            ),
            device=device,
        )
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
    )

    # ========================================================
    # RUN BOTH VARIANTS INDEPENDENTLY
    # ========================================================

    direct_result = (
        train_variant(
            variant="direct_summary",
            original=original,
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            num_classes=(
                len(answers)
            ),
            device=device,
            args=args,
        )
    )

    original_result = (
        train_variant(
            variant="original_writer",
            original=original,
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            num_classes=(
                len(answers)
            ),
            device=device,
            args=args,
        )
    )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    results = {
        "direct_summary": (
            direct_result
        ),

        "original_writer": (
            original_result
        ),
    }

    print()
    print("=" * 100)
    print(
        "FINAL WRITER MICRO-ABLATION"
    )
    print("=" * 100)

    print(
        f"{'VARIANT':<24}"
        f"{'MATCH':>12}"
        f"{'MISMATCH':>14}"
        f"{'QUERY':>12}"
        f"{'NLL GAP':>14}"
    )

    print("-" * 76)

    for name in [
        "direct_summary",
        "original_writer",
    ]:

        r = (
            results[
                name
            ]["test"]
        )

        print(
            f"{name:<24}"
            f"{r['matched_accuracy']:>11.2f}%"
            f"{r['mismatched_accuracy']:>13.2f}%"
            f"{r['query_only_accuracy']:>11.2f}%"
            f"{r['nll_gap']:>14.4f}"
        )

    print()

    direct = (
        direct_result[
            "test"
        ]
    )

    original_r = (
        original_result[
            "test"
        ]
    )

    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    print("=" * 100)
    print(
        "DIAGNOSTIC INTERPRETATION"
    )
    print("=" * 100)

    if (
        direct[
            "matched_accuracy"
        ] >= 80
        and
        direct[
            "matched_accuracy"
        ]
        - direct[
            "mismatched_accuracy"
        ] >= 50
        and
        original_r[
            "matched_accuracy"
        ] < 30
    ):

        print(
            "STRONG WRITER BOTTLENECK:"
        )

        print(
            "The masked-mean summary contains enough "
            "information to construct a usable latent VALUE,"
        )

        print(
            "but the original CandidateWriter transforms "
            "that information into a representation that "
            "the downstream system cannot use."
        )

    elif (
        direct[
            "matched_accuracy"
        ] >= 60
        and
        original_r[
            "matched_accuracy"
        ] < direct[
            "matched_accuracy"
        ] - 20
    ):

        print(
            "WRITER BOTTLENECK SUPPORTED:"
        )

        print(
            "Direct summary substantially outperforms "
            "the original CandidateWriter."
        )

    elif (
        direct[
            "matched_accuracy"
        ] < 30
    ):

        print(
            "DIRECT SUMMARY ALSO FAILS."
        )

        print(
            "Do not redesign CandidateWriter yet. "
            "The earlier success may depend on the "
            "answer-token representation rather than "
            "the masked-mean summary."
        )

    else:

        print(
            "AMBIGUOUS RESULT."
        )

        print(
            "Inspect source/stored geometry and "
            "matched-vs-mismatched NLL before changing "
            "the architecture."
        )

    # ========================================================
    # SAVE JSON
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
        / "writer_ablation_results.json",
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
        / "writer_ablation_results.json",
    )


if __name__ == "__main__":
    main()