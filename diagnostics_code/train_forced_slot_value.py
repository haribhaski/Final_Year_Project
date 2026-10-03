from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from models.gpt2_memory import (
    MemoryAugmentedGPT2LMHeadModel,
    MemoryGPT2Config,
)


# ============================================================
# ANSWER CANDIDATES
#
# We later filter these and retain only words that are exactly
# ONE GPT-2 token when preceded by a space.
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
#
# Training and evaluation templates are deliberately different.
# We want to know whether the VALUE pathway stores the FACT,
# not merely memorizes one exact sentence pattern.
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
# MEMORY CONFIGURATION
#
# Architecture must match the checkpoint.
#
# Auxiliary weights are set to zero because this diagnostic
# trains directly on answer prediction.
# They are config values, not shape-changing parameters.
# ============================================================

def build_memory_config() -> MemoryGPT2Config:

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

        # We are testing direct VALUE learning.
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

def set_seed(seed: int) -> None:

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SINGLE TOKEN ANSWERS
# ============================================================

def get_single_token_answers(
    tokenizer: Any,
) -> dict[str, int]:

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
            "Need at least 4 single-token answer words."
        )

    return usable


# ============================================================
# DATASET CREATION
# ============================================================

def build_examples(
    number_examples: int,
    answer_words: list[str],
    seed: int,
    start_id: int,
    split: str,
) -> list[dict[str, Any]]:

    rng = random.Random(seed)

    if split == "train":
        fact_templates = TRAIN_FACT_TEMPLATES
        query_templates = TRAIN_QUERY_TEMPLATES
    else:
        fact_templates = EVAL_FACT_TEMPLATES
        query_templates = EVAL_QUERY_TEMPLATES

    examples = []

    for i in range(number_examples):

        entity = f"person_{start_id + i}"

        answer = rng.choice(
            answer_words
        )

        fact_template = rng.choice(
            fact_templates
        )

        query_template = rng.choice(
            query_templates
        )

        fact = fact_template.format(
            entity=entity,
            answer=answer,
        )

        query = query_template.format(
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


class FactDataset(Dataset):

    def __init__(
        self,
        examples: list[dict[str, Any]],
    ):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


# ============================================================
# COLLATE
# ============================================================

def collate_examples(batch):

    return {
        "id": [
            x["id"]
            for x in batch
        ],
        "entity": [
            x["entity"]
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
# LOAD MODEL
# ============================================================

def load_model(
    checkpoint_path: str,
    model_name: str,
    device: torch.device,
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

    return model


# ============================================================
# FREEZE EVERYTHING EXCEPT VALUE PATH
#
# Train:
#   CandidateWriter
#   OrthogonalUpdate
#   write gate
#   MemoryReader/fusion
#   confidence head
#   MemoryBank normalization parameters
#
# Do NOT train:
#   GPT-2
#   SlotRouter
#   initial slot vectors
#
# This asks:
#
# Can the writer + latent value + fusion pathway learn?
# ============================================================

def configure_value_training(
    model: MemoryAugmentedGPT2LMHeadModel,
):

    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_prefixes = (
        "writer.",
        "orthogonalizer.",
        "write_gate_module.",
        "reader.",
        "write_confidence_head.",
        "memory_bank.layer_norm.",
    )

    for name, parameter in (
        model.named_parameters()
    ):

        if name.startswith(
            trainable_prefixes
        ):
            parameter.requires_grad = True

    # Explicitly keep initial slot contents frozen.
    if hasattr(
        model.memory_bank,
        "initial_slots",
    ):
        model.memory_bank.initial_slots.requires_grad = False


def print_trainable_parameters(model):

    total = 0
    trainable = 0

    print()
    print("=" * 90)
    print("TRAINABLE PARAMETERS")
    print("=" * 90)

    for name, parameter in (
        model.named_parameters()
    ):

        total += parameter.numel()

        if parameter.requires_grad:

            trainable += parameter.numel()

            print(
                f"{name:<70}"
                f"{parameter.numel():>12,}"
            )

    print("-" * 90)

    print(
        f"Total parameters:     {total:,}"
    )

    print(
        f"Trainable parameters: {trainable:,}"
    )

    print(
        f"Trainable percentage: "
        f"{100.0 * trainable / total:.4f}%"
    )


# ============================================================
# TOKENIZATION
# ============================================================

def encode_batch(
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
# FORCE EXACTLY ONE SLOT
#
# slot 0 = available
# all others = unavailable
#
# This mask is used on BOTH write and read.
# ============================================================

def forced_slot_mask(
    batch_size: int,
    num_slots: int,
    slot_index: int,
    device: torch.device,
):

    mask = torch.zeros(
        batch_size,
        num_slots,
        dtype=torch.bool,
        device=device,
    )

    mask[:, slot_index] = True

    return mask


# ============================================================
# INITIAL MEMORY
# ============================================================

def initial_memory(
    model,
    batch_size,
    device,
):

    dtype = next(
        model.parameters()
    ).dtype

    return model.initialize_memory(
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )


# ============================================================
# LAST VALID TOKEN LOGITS
# ============================================================

def last_token_logits(
    logits,
    attention_mask,
):

    # [B]
    last_indices = (
        attention_mask.sum(dim=1)
        - 1
    )

    batch_indices = torch.arange(
        logits.size(0),
        device=logits.device,
    )

    return logits[
        batch_indices,
        last_indices,
        :
    ]


# ============================================================
# WRITE FACT INTO FORCED SLOT
#
# IMPORTANT:
# No detach here during training.
#
# Gradient from answer loss must travel:
#
# query loss
# -> reader
# -> memory value
# -> write gate/writer
# ============================================================

def write_batch(
    model,
    tokenizer,
    facts,
    device,
    forced_slot,
):

    batch_size = len(facts)

    input_ids, attention_mask = (
        encode_batch(
            tokenizer,
            facts,
            device,
        )
    )

    state = initial_memory(
        model,
        batch_size,
        device,
    )

    mask = forced_slot_mask(
        batch_size=batch_size,
        num_slots=model.num_slots,
        slot_index=forced_slot,
        device=device,
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        memory_state=state,
        memory_mask=mask,

        # WRITE
        update_memory=True,

        return_diagnostics=False,
    )

    return output.memory_state


# ============================================================
# READ QUESTION FROM SAME FORCED SLOT
# ============================================================

def read_query(
    model,
    tokenizer,
    queries,
    memory_state,
    device,
    forced_slot,
):

    input_ids, attention_mask = (
        encode_batch(
            tokenizer,
            queries,
            device,
        )
    )

    mask = forced_slot_mask(
        batch_size=len(queries),
        num_slots=model.num_slots,
        slot_index=forced_slot,
        device=device,
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        memory_state=memory_state,
        memory_mask=mask,

        # CRITICAL:
        # Never write the question into memory.
        update_memory=False,

        return_diagnostics=False,
    )

    logits = last_token_logits(
        output.logits,
        attention_mask,
    )

    return logits


# ============================================================
# RAW TEXT LOGITS
# ============================================================

@torch.no_grad()
def raw_text_logits(
    model,
    tokenizer,
    facts,
    queries,
    device,
):

    prompts = [
        fact + "\n" + query
        for fact, query
        in zip(facts, queries)
    ]

    input_ids, attention_mask = (
        encode_batch(
            tokenizer,
            prompts,
            device,
        )
    )

    output = model.backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )

    return last_token_logits(
        output.logits,
        attention_mask,
    )


# ============================================================
# TRAINING
# ============================================================

def train_epoch(
    model,
    tokenizer,
    loader,
    answer_token_map,
    optimizer,
    device,
    forced_slot,
    grad_clip,
):

    model.train()

    # Backbone remains frozen. eval() prevents accidental
    # dropout behavior inside frozen GPT-2.
    model.backbone.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_grad_norm = 0.0

    for batch in loader:

        answers = batch["answer"]

        target_ids = torch.tensor(
            [
                answer_token_map[a]
                for a in answers
            ],
            dtype=torch.long,
            device=device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # WRITE FACT TO KNOWN SLOT
        # ----------------------------------------------------

        memory_state = write_batch(
            model=model,
            tokenizer=tokenizer,
            facts=batch["fact"],
            device=device,
            forced_slot=forced_slot,
        )

        # ----------------------------------------------------
        # READ SAME KNOWN SLOT
        # ----------------------------------------------------

        logits = read_query(
            model=model,
            tokenizer=tokenizer,
            queries=batch["query"],
            memory_state=memory_state,
            device=device,
            forced_slot=forced_slot,
        )

        # ----------------------------------------------------
        # DIRECT ANSWER OBJECTIVE
        # ----------------------------------------------------

        loss = F.cross_entropy(
            logits,
            target_ids,
        )

        loss.backward()

        trainable_parameters = [
            p
            for p in model.parameters()
            if p.requires_grad
            and p.grad is not None
        ]

        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=grad_clip,
        )

        optimizer.step()

        predictions = logits.argmax(
            dim=-1
        )

        total_correct += int(
            (
                predictions
                == target_ids
            )
            .sum()
            .item()
        )

        batch_size = len(answers)

        total_examples += batch_size

        total_loss += (
            float(loss.item())
            * batch_size
        )

        total_grad_norm += float(
            grad_norm
        )

    return {
        "loss": (
            total_loss
            / max(total_examples, 1)
        ),
        "full_vocab_accuracy": (
            100.0
            * total_correct
            / max(total_examples, 1)
        ),
        "mean_grad_norm": (
            total_grad_norm
            / max(len(loader), 1)
        ),
    }


# ============================================================
# FOUR-WAY CANDIDATE GENERATION
# ============================================================

def candidate_lists(
    examples,
    answer_words,
    seed,
):

    result = []

    for example in examples:

        correct = example["answer"]

        negatives = [
            answer
            for answer in answer_words
            if answer != correct
        ]

        rng = random.Random(
            seed
            + example["id"]
            * 7919
        )

        choices = [
            correct,
            *rng.sample(
                negatives,
                3,
            ),
        ]

        rng.shuffle(choices)

        result.append(choices)

    return result


# ============================================================
# 4-WAY ACCURACY
# ============================================================

def four_way_predictions(
    logits,
    examples,
    candidates,
    answer_token_map,
):

    correct = 0

    correct_nll = []

    log_probs = F.log_softmax(
        logits.float(),
        dim=-1,
    )

    for i, example in enumerate(
        examples
    ):

        scores = {}

        for candidate in candidates[i]:

            token_id = (
                answer_token_map[candidate]
            )

            scores[candidate] = float(
                -log_probs[
                    i,
                    token_id,
                ].item()
            )

        prediction = min(
            scores,
            key=scores.get,
        )

        if (
            prediction
            == example["answer"]
        ):
            correct += 1

        correct_nll.append(
            scores[
                example["answer"]
            ]
        )

    return (
        correct,
        correct_nll,
    )


# ============================================================
# CONTENT DEPENDENCE
# ============================================================

def vector_statistics(
    vectors: torch.Tensor,
):

    vectors = vectors.float()

    if vectors.size(0) < 2:
        return {}

    norms = vectors.norm(
        dim=-1
    )

    normalized = F.normalize(
        vectors,
        p=2,
        dim=-1,
    )

    cosine = (
        normalized
        @ normalized.T
    )

    distance = torch.cdist(
        vectors,
        vectors,
    )

    n = vectors.size(0)

    mask = torch.triu(
        torch.ones(
            n,
            n,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    pairwise_cosine = (
        cosine.cpu()[mask]
    )

    pairwise_distance = (
        distance.cpu()[mask]
    )

    mean_norm = float(
        norms.mean().item()
    )

    mean_distance = float(
        pairwise_distance.mean().item()
    )

    return {
        "mean_norm": mean_norm,

        "mean_pairwise_l2": (
            mean_distance
        ),

        "relative_l2": (
            mean_distance
            / (mean_norm + 1e-8)
        ),

        "mean_pairwise_cosine": float(
            pairwise_cosine.mean().item()
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    examples,
    answer_token_map,
    device,
    forced_slot,
    seed,
    eval_batch_size=64,
):

    model.eval()

    answer_words = list(
        answer_token_map.keys()
    )

    candidates = candidate_lists(
        examples=examples,
        answer_words=answer_words,
        seed=seed,
    )

    condition_correct = {
        "EMPTY_MEMORY": 0,
        "MATCHED_MEMORY": 0,
        "MISMATCHED_MEMORY": 0,
        "RAW_TEXT": 0,
    }

    condition_nll = {
        name: []
        for name in condition_correct
    }

    # We first generate all written memories one example
    # at a time/batches so mismatched memories can be rotated.
    written_states = []

    slot_vectors = []

    slot_deltas = []

    # --------------------------------------------------------
    # Build memories
    # --------------------------------------------------------

    for start in range(
        0,
        len(examples),
        eval_batch_size,
    ):

        batch_examples = examples[
            start:
            start + eval_batch_size
        ]

        facts = [
            x["fact"]
            for x in batch_examples
        ]

        batch_size = len(
            batch_examples
        )

        initial_state = initial_memory(
            model,
            batch_size,
            device,
        )

        initial_slot = (
            initial_state
            .slots[
                :,
                forced_slot,
                :
            ]
            .detach()
            .float()
            .cpu()
        )

        memory_state = write_batch(
            model=model,
            tokenizer=tokenizer,
            facts=facts,
            device=device,
            forced_slot=forced_slot,
        )

        final_slot = (
            memory_state
            .slots[
                :,
                forced_slot,
                :
            ]
            .detach()
            .float()
            .cpu()
        )

        delta = (
            final_slot
            - initial_slot
        )

        slot_vectors.append(
            final_slot
        )

        slot_deltas.append(
            delta
        )

        # Store individual states by slicing.
        for j in range(batch_size):

            # Create independent single-example MemoryState
            # using the state's dataclass type.
            state_type = type(memory_state)

            written_states.append(
                state_type(
                    slots=memory_state.slots[
                        j:j + 1
                    ].clone(),

                    age=memory_state.age[
                        j:j + 1
                    ].clone(),

                    write_count=memory_state.write_count[
                        j:j + 1
                    ].clone(),

                    read_count=memory_state.read_count[
                        j:j + 1
                    ].clone(),

                    confidence=memory_state.confidence[
                        j:j + 1
                    ].clone(),
                )
            )

    # --------------------------------------------------------
    # Evaluate one example at a time.
    #
    # This is slower but removes any ambiguity from
    # mismatched-state indexing.
    # --------------------------------------------------------

    for i, example in enumerate(
        examples
    ):

        candidate_set = candidates[i]

        candidate_ids = torch.tensor(
            [
                answer_token_map[x]
                for x in candidate_set
            ],
            dtype=torch.long,
            device=device,
        )

        # ====================================================
        # 1. EMPTY MEMORY
        # ====================================================

        empty_state = initial_memory(
            model,
            1,
            device,
        )

        empty_logits = read_query(
            model=model,
            tokenizer=tokenizer,
            queries=[
                example["query"]
            ],
            memory_state=empty_state,
            device=device,
            forced_slot=forced_slot,
        )[0]

        empty_lp = F.log_softmax(
            empty_logits.float(),
            dim=-1,
        )

        empty_scores = (
            -empty_lp[candidate_ids]
        )

        empty_choice = int(
            empty_scores.argmin().item()
        )

        if (
            candidate_set[
                empty_choice
            ]
            == example["answer"]
        ):
            condition_correct[
                "EMPTY_MEMORY"
            ] += 1

        condition_nll[
            "EMPTY_MEMORY"
        ].append(
            float(
                -empty_lp[
                    answer_token_map[
                        example["answer"]
                    ]
                ]
                .item()
            )
        )

        # ====================================================
        # 2. MATCHED MEMORY
        # ====================================================

        matched_state = (
            written_states[i]
        )

        matched_logits = read_query(
            model=model,
            tokenizer=tokenizer,
            queries=[
                example["query"]
            ],
            memory_state=matched_state,
            device=device,
            forced_slot=forced_slot,
        )[0]

        matched_lp = F.log_softmax(
            matched_logits.float(),
            dim=-1,
        )

        matched_scores = (
            -matched_lp[
                candidate_ids
            ]
        )

        matched_choice = int(
            matched_scores
            .argmin()
            .item()
        )

        if (
            candidate_set[
                matched_choice
            ]
            == example["answer"]
        ):
            condition_correct[
                "MATCHED_MEMORY"
            ] += 1

        condition_nll[
            "MATCHED_MEMORY"
        ].append(
            float(
                -matched_lp[
                    answer_token_map[
                        example["answer"]
                    ]
                ]
                .item()
            )
        )

        # ====================================================
        # 3. MISMATCHED MEMORY
        #
        # Rotate by one example.
        # ====================================================

        mismatch_index = (
            i + 1
        ) % len(examples)

        mismatch_state = (
            written_states[
                mismatch_index
            ]
        )

        mismatch_logits = read_query(
            model=model,
            tokenizer=tokenizer,
            queries=[
                example["query"]
            ],
            memory_state=mismatch_state,
            device=device,
            forced_slot=forced_slot,
        )[0]

        mismatch_lp = F.log_softmax(
            mismatch_logits.float(),
            dim=-1,
        )

        mismatch_scores = (
            -mismatch_lp[
                candidate_ids
            ]
        )

        mismatch_choice = int(
            mismatch_scores
            .argmin()
            .item()
        )

        if (
            candidate_set[
                mismatch_choice
            ]
            == example["answer"]
        ):
            condition_correct[
                "MISMATCHED_MEMORY"
            ] += 1

        condition_nll[
            "MISMATCHED_MEMORY"
        ].append(
            float(
                -mismatch_lp[
                    answer_token_map[
                        example["answer"]
                    ]
                ]
                .item()
            )
        )

        # ====================================================
        # 4. RAW TEXT
        # ====================================================

        raw_logits = raw_text_logits(
            model=model,
            tokenizer=tokenizer,
            facts=[
                example["fact"]
            ],
            queries=[
                example["query"]
            ],
            device=device,
        )[0]

        raw_lp = F.log_softmax(
            raw_logits.float(),
            dim=-1,
        )

        raw_scores = (
            -raw_lp[
                candidate_ids
            ]
        )

        raw_choice = int(
            raw_scores
            .argmin()
            .item()
        )

        if (
            candidate_set[
                raw_choice
            ]
            == example["answer"]
        ):
            condition_correct[
                "RAW_TEXT"
            ] += 1

        condition_nll[
            "RAW_TEXT"
        ].append(
            float(
                -raw_lp[
                    answer_token_map[
                        example["answer"]
                    ]
                ]
                .item()
            )
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    result = {}

    total = len(examples)

    for condition in (
        condition_correct
    ):

        result[
            condition
        ] = {
            "accuracy": (
                100.0
                * condition_correct[
                    condition
                ]
                / total
            ),

            "correct": (
                condition_correct[
                    condition
                ]
            ),

            "mean_nll": (
                sum(
                    condition_nll[
                        condition
                    ]
                )
                / total
            ),
        }

    # --------------------------------------------------------
    # Matched-vs-mismatched fact specificity
    # --------------------------------------------------------

    nll_differences = [
        mismatch - match
        for mismatch, match
        in zip(
            condition_nll[
                "MISMATCHED_MEMORY"
            ],
            condition_nll[
                "MATCHED_MEMORY"
            ],
        )
    ]

    result[
        "memory_specific_nll_gain"
    ] = (
        sum(nll_differences)
        / len(nll_differences)
    )

    all_slot_vectors = torch.cat(
        slot_vectors,
        dim=0,
    )

    all_slot_deltas = torch.cat(
        slot_deltas,
        dim=0,
    )

    result["slot_vector_stats"] = (
        vector_statistics(
            all_slot_vectors
        )
    )

    result["slot_delta_stats"] = (
        vector_statistics(
            all_slot_deltas
        )
    )

    return result


# ============================================================
# PRINT RESULTS
# ============================================================

def print_evaluation(
    title,
    result,
):

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    print(
        f"{'Condition':<25}"
        f"{'Accuracy':>12}"
        f"{'Correct':>14}"
        f"{'Mean NLL':>14}"
    )

    print("-" * 67)

    for name in [
        "EMPTY_MEMORY",
        "MATCHED_MEMORY",
        "MISMATCHED_MEMORY",
        "RAW_TEXT",
    ]:

        row = result[name]

        print(
            f"{name:<25}"
            f"{row['accuracy']:>11.2f}%"
            f"{row['correct']:>9}"
            f"{row['mean_nll']:>14.4f}"
        )

    print()

    matched = result[
        "MATCHED_MEMORY"
    ]["accuracy"]

    mismatched = result[
        "MISMATCHED_MEMORY"
    ]["accuracy"]

    empty = result[
        "EMPTY_MEMORY"
    ]["accuracy"]

    print(
        "Matched - Empty:      "
        f"{matched - empty:+.2f} pp"
    )

    print(
        "Matched - Mismatched: "
        f"{matched - mismatched:+.2f} pp"
    )

    print(
        "Mean mismatched NLL - matched NLL: "
        f"{result['memory_specific_nll_gain']:+.6f}"
    )

    print()
    print("FORCED SLOT VALUE VECTORS")

    stats = result[
        "slot_vector_stats"
    ]

    for key, value in stats.items():
        print(
            f"  {key:<24} "
            f"{value:.6f}"
        )

    print()
    print("FORCED SLOT WRITE DELTAS")

    stats = result[
        "slot_delta_stats"
    ]

    for key, value in stats.items():
        print(
            f"  {key:<24} "
            f"{value:.6f}"
        )


# ============================================================
# CHECKPOINT SAVE
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    validation_result,
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

            "validation": (
                validation_result
            ),

            "arguments": vars(args),

            "experiment": (
                "forced_slot_value_training"
            ),
        },
        path,
    )

    print(
        "Saved:",
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
        "--train-examples",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--validation-examples",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--test-examples",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
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
            "forced_slot_value_training"
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
    print("FORCED-SLOT LATENT VALUE TRAINING")
    print("=" * 90)

    print(
        "Device:",
        device,
    )

    print(
        "Starting checkpoint:",
        args.checkpoint,
    )

    print(
        "Forced slot:",
        args.forced_slot,
    )

    print()
    print(
        "NO FILE UNDER models/ WILL BE MODIFIED."
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = (
        AutoTokenizer.from_pretrained(
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

    print()
    print(
        "Single-token answers:"
    )

    for answer, token_id in (
        answer_token_map.items()
    ):

        print(
            f"  {answer:<12} "
            f"token_id={token_id}"
        )

    print(
        f"\nTotal answer classes: "
        f"{len(answer_words)}"
    )

    # ========================================================
    # DATA
    # ========================================================

    train_examples = build_examples(
        number_examples=(
            args.train_examples
        ),
        answer_words=answer_words,
        seed=args.seed,
        start_id=0,
        split="train",
    )

    validation_examples = (
        build_examples(
            number_examples=(
                args.validation_examples
            ),
            answer_words=answer_words,
            seed=args.seed + 1000,
            start_id=100000,
            split="eval",
        )
    )

    test_examples = build_examples(
        number_examples=(
            args.test_examples
        ),
        answer_words=answer_words,
        seed=args.seed + 2000,
        start_id=200000,
        split="eval",
    )

    train_loader = DataLoader(
        FactDataset(
            train_examples
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_examples,
    )

    print()
    print(
        f"Train: {len(train_examples)}"
    )

    print(
        f"Validation: "
        f"{len(validation_examples)}"
    )

    print(
        f"Test: {len(test_examples)}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = load_model(
        checkpoint_path=(
            args.checkpoint
        ),
        model_name=args.model_name,
        device=device,
    )

    if not (
        0
        <= args.forced_slot
        < model.num_slots
    ):
        raise ValueError(
            "forced-slot outside valid range"
        )

    configure_value_training(
        model
    )

    print_trainable_parameters(
        model
    )

    trainable_parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ========================================================
    # PRE-TRAIN EVALUATION
    # ========================================================

    print()
    print("=" * 90)
    print("PRE-TRAIN VALIDATION")
    print("=" * 90)

    pre_result = evaluate(
        model=model,
        tokenizer=tokenizer,
        examples=validation_examples,
        answer_token_map=(
            answer_token_map
        ),
        device=device,
        forced_slot=(
            args.forced_slot
        ),
        seed=args.seed + 5000,
    )

    print_evaluation(
        "PRE-TRAIN VALIDATION RESULTS",
        pre_result,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    best_score = -math.inf
    best_epoch = -1

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            tokenizer=tokenizer,
            loader=train_loader,
            answer_token_map=(
                answer_token_map
            ),
            optimizer=optimizer,
            device=device,
            forced_slot=(
                args.forced_slot
            ),
            grad_clip=(
                args.grad_clip
            ),
        )

        print(
            f"Train loss:          "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            f"Train full-vocab acc:"
            f" "
            f"{train_metrics['full_vocab_accuracy']:.2f}%"
        )

        print(
            f"Mean grad norm:      "
            f"{train_metrics['mean_grad_norm']:.4f}"
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        validation_result = evaluate(
            model=model,
            tokenizer=tokenizer,
            examples=(
                validation_examples
            ),
            answer_token_map=(
                answer_token_map
            ),
            device=device,
            forced_slot=(
                args.forced_slot
            ),
            seed=args.seed + 5000,
        )

        print_evaluation(
            f"VALIDATION AFTER EPOCH {epoch}",
            validation_result,
        )

        matched = validation_result[
            "MATCHED_MEMORY"
        ]["accuracy"]

        mismatched = (
            validation_result[
                "MISMATCHED_MEMORY"
            ]["accuracy"]
        )

        # Primary target:
        # high matched accuracy.
        #
        # Secondary target:
        # matched should outperform mismatched.
        selection_score = (
            matched
            + 0.25
            * (
                matched
                - mismatched
            )
        )

        print()
        print(
            "Checkpoint selection score: "
            f"{selection_score:.4f}"
        )

        if selection_score > best_score:

            best_score = (
                selection_score
            )

            best_epoch = epoch

            save_checkpoint(
                path=(
                    output_dir
                    / "checkpoint_best.pt"
                ),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_result=(
                    validation_result
                ),
                args=args,
            )

        save_checkpoint(
            path=(
                output_dir
                / f"checkpoint_epoch_{epoch}.pt"
            ),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            validation_result=(
                validation_result
            ),
            args=args,
        )

    # ========================================================
    # LOAD BEST
    # ========================================================

    print()
    print("=" * 90)

    print(
        f"LOADING BEST EPOCH: "
        f"{best_epoch}"
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
        ],
        strict=True,
    )

    model.to(device)
    model.eval()

    # ========================================================
    # FINAL TEST
    # ========================================================

    final_result = evaluate(
        model=model,
        tokenizer=tokenizer,
        examples=test_examples,
        answer_token_map=(
            answer_token_map
        ),
        device=device,
        forced_slot=(
            args.forced_slot
        ),
        seed=args.seed + 9000,
    )

    print_evaluation(
        "FINAL HELD-OUT TEST",
        final_result,
    )

    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    matched = final_result[
        "MATCHED_MEMORY"
    ]["accuracy"]

    mismatch = final_result[
        "MISMATCHED_MEMORY"
    ]["accuracy"]

    empty = final_result[
        "EMPTY_MEMORY"
    ]["accuracy"]

    raw = final_result[
        "RAW_TEXT"
    ]["accuracy"]

    print()
    print("=" * 90)
    print("FINAL DIAGNOSTIC")
    print("=" * 90)

    print(
        f"EMPTY_MEMORY:      "
        f"{empty:.2f}%"
    )

    print(
        f"MATCHED_MEMORY:    "
        f"{matched:.2f}%"
    )

    print(
        f"MISMATCHED_MEMORY: "
        f"{mismatch:.2f}%"
    )

    print(
        f"RAW_TEXT:          "
        f"{raw:.2f}%"
    )

    print()

    print(
        "Matched - Empty:      "
        f"{matched - empty:+.2f} pp"
    )

    print(
        "Matched - Mismatched: "
        f"{matched - mismatch:+.2f} pp"
    )

    print()

    if (
        matched >= 70.0
        and matched - mismatch >= 20.0
    ):

        verdict = (
            "STRONG PASS: the forced-slot latent "
            "VALUE pathway learned fact-specific "
            "information."
        )

    elif (
        matched >= 50.0
        and matched - mismatch >= 10.0
    ):

        verdict = (
            "PASS: the latent VALUE pathway carries "
            "usable fact-specific information, but "
            "there is still substantial room for "
            "improvement."
        )

    elif (
        matched - mismatch >= 5.0
    ):

        verdict = (
            "PARTIAL: there is measurable "
            "fact-specific information in the latent "
            "VALUE, but the pathway is weak."
        )

    else:

        verdict = (
            "FAIL: even after direct forced-slot "
            "training, matched memory does not "
            "meaningfully outperform mismatched "
            "memory. The writer/value/fusion design "
            "needs architectural revision."
        )

    print(verdict)

    print()
    print(
        "This experiment intentionally bypasses "
        "slot selection. It does NOT evaluate E5, "
        "the router, or long-range retrieval."
    )

    # ========================================================
    # SAVE SUMMARY JSON
    # ========================================================

    summary = {
        "best_epoch": best_epoch,
        "pre_validation": (
            pre_result
        ),
        "final_test": (
            final_result
        ),
        "forced_slot": (
            args.forced_slot
        ),
        "starting_checkpoint": (
            args.checkpoint
        ),
    }

    json_path = (
        output_dir
        / "forced_slot_value_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print(
        "Saved results:",
        json_path,
    )

    print(
        "Best checkpoint:",
        output_dir
        / "checkpoint_best.pt",
    )


if __name__ == "__main__":
    main()