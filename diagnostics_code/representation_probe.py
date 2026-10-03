from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from models.gpt2_memory import (
    MemoryAugmentedGPT2LMHeadModel,
    MemoryGPT2Config,
)


# ============================================================
# ANSWER POOL
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

EVAL_FACT_TEMPLATES = [
    "The secret code associated with {entity} is {answer}.",
    "{entity} is linked to the secret code {answer}.",
]


# ============================================================
# MODEL CONFIG
# Must match checkpoint.
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
            "Need at least 4 single-token answers."
        )

    return result


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

    templates = (
        TRAIN_FACT_TEMPLATES
        if split == "train"
        else EVAL_FACT_TEMPLATES
    )

    examples = []

    for i in range(n):

        entity = f"person_{start_id + i}"

        answer = rng.choice(
            answer_words
        )

        template = rng.choice(
            templates
        )

        fact = template.format(
            entity=entity,
            answer=answer,
        )

        examples.append(
            {
                "id": start_id + i,
                "entity": entity,
                "answer": answer,
                "fact": fact,
            }
        )

    return examples


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    checkpoint_path,
    model_name,
    device,
):

    print("Loading model...")

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
# FIND ANSWER TOKEN POSITION
#
# We search for the one-token answer inside the tokenized fact.
# Since answer words are guaranteed one-token with leading space,
# this should normally find exactly one occurrence.
# ============================================================

def find_answer_position(
    input_ids,
    attention_mask,
    answer_token_id,
):

    valid_length = int(
        attention_mask.sum().item()
    )

    tokens = input_ids[
        :valid_length
    ]

    positions = (
        tokens == answer_token_id
    ).nonzero(
        as_tuple=False
    ).flatten()

    if positions.numel() == 0:

        raise RuntimeError(
            f"Could not locate answer token "
            f"{answer_token_id} in fact."
        )

    # Use last occurrence if repeated.
    return int(
        positions[-1].item()
    )


# ============================================================
# EXTRACT REPRESENTATIONS
# ============================================================

@torch.no_grad()
def extract_representation(
    model,
    tokenizer,
    example,
    answer_token_map,
    device,
    forced_slot=0,
):

    encoded = tokenizer(
        example["fact"],
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = encoded[
        "input_ids"
    ].to(device)

    attention_mask = encoded[
        "attention_mask"
    ].to(device)

    # --------------------------------------------------------
    # 1. RAW GPT-2 FINAL HIDDEN STATES
    # --------------------------------------------------------

    transformer_output = (
        model.backbone.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    )

    hidden = (
        transformer_output
        .last_hidden_state
    )

    # --------------------------------------------------------
    # MEAN-POOLED REPRESENTATION
    #
    # This matches the model's masked-mean summary behavior.
    # --------------------------------------------------------

    weights = (
        attention_mask
        .unsqueeze(-1)
        .to(hidden.dtype)
    )

    mean_rep = (
        hidden * weights
    ).sum(dim=1) / (
        weights.sum(dim=1)
        .clamp_min(1.0)
    )

    mean_rep = (
        mean_rep[0]
        .float()
        .cpu()
    )

    # --------------------------------------------------------
    # ANSWER TOKEN REPRESENTATION
    # --------------------------------------------------------

    answer_token_id = (
        answer_token_map[
            example["answer"]
        ]
    )

    answer_pos = (
        find_answer_position(
            input_ids=input_ids[0],
            attention_mask=(
                attention_mask[0]
            ),
            answer_token_id=(
                answer_token_id
            ),
        )
    )

    answer_rep = (
        hidden[
            0,
            answer_pos,
            :
        ]
        .float()
        .cpu()
    )

    # --------------------------------------------------------
    # 3. WRITTEN SLOT REPRESENTATION
    #
    # Force only slot 0 to be available.
    # This removes router allocation ambiguity.
    # --------------------------------------------------------

    dtype = next(
        model.parameters()
    ).dtype

    initial_state = (
        model.initialize_memory(
            batch_size=1,
            device=device,
            dtype=dtype,
        )
    )

    memory_mask = torch.zeros(
        1,
        model.num_slots,
        dtype=torch.bool,
        device=device,
    )

    memory_mask[
        0,
        forced_slot,
    ] = True

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        memory_state=initial_state,
        memory_mask=memory_mask,
        update_memory=True,
        return_diagnostics=False,
    )

    slot_rep = (
        output.memory_state.slots[
            0,
            forced_slot,
            :
        ]
        .detach()
        .float()
        .cpu()
    )

    # --------------------------------------------------------
    # WRITE DELTA ALSO USEFUL
    # --------------------------------------------------------

    initial_slot = (
        initial_state.slots[
            0,
            forced_slot,
            :
        ]
        .detach()
        .float()
        .cpu()
    )

    delta_rep = (
        slot_rep
        - initial_slot
    )

    return {
        "mean": mean_rep,
        "answer_token": answer_rep,
        "slot": slot_rep,
        "delta": delta_rep,
    }


# ============================================================
# EXTRACT WHOLE DATASET
# ============================================================

def extract_dataset(
    model,
    tokenizer,
    examples,
    answer_token_map,
    answer_to_class,
    device,
    forced_slot,
):

    representations = {
        "MEAN_POOL": [],
        "ANSWER_TOKEN": [],
        "WRITTEN_SLOT": [],
        "WRITE_DELTA": [],
    }

    labels = []

    print(
        f"Extracting {len(examples)} examples..."
    )

    for i, example in enumerate(
        examples
    ):

        reps = extract_representation(
            model=model,
            tokenizer=tokenizer,
            example=example,
            answer_token_map=(
                answer_token_map
            ),
            device=device,
            forced_slot=forced_slot,
        )

        representations[
            "MEAN_POOL"
        ].append(
            reps["mean"]
        )

        representations[
            "ANSWER_TOKEN"
        ].append(
            reps["answer_token"]
        )

        representations[
            "WRITTEN_SLOT"
        ].append(
            reps["slot"]
        )

        representations[
            "WRITE_DELTA"
        ].append(
            reps["delta"]
        )

        labels.append(
            answer_to_class[
                example["answer"]
            ]
        )

        if (
            (i + 1) % 500 == 0
            or i + 1 == len(examples)
        ):
            print(
                f"  {i + 1}/{len(examples)}"
            )

    output = {}

    for name, values in (
        representations.items()
    ):

        output[name] = (
            torch.stack(
                values,
                dim=0,
            )
        )

    labels = torch.tensor(
        labels,
        dtype=torch.long,
    )

    return output, labels


# ============================================================
# LINEAR PROBE
# ============================================================

class LinearProbe(nn.Module):

    def __init__(
        self,
        input_dim,
        number_classes,
    ):

        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            number_classes,
        )

    def forward(self, x):

        return self.linear(x)


# ============================================================
# STANDARDIZATION
#
# Fit ONLY on training data.
# ============================================================

def standardize(
    train_x,
    valid_x,
    test_x,
):

    mean = train_x.mean(
        dim=0,
        keepdim=True,
    )

    std = train_x.std(
        dim=0,
        keepdim=True,
        unbiased=False,
    )

    std = std.clamp_min(
        1e-5
    )

    return (
        (train_x - mean) / std,
        (valid_x - mean) / std,
        (test_x - mean) / std,
    )


# ============================================================
# EVALUATE PROBE
# ============================================================

@torch.no_grad()
def evaluate_probe(
    probe,
    x,
    y,
    device,
):

    probe.eval()

    logits = probe(
        x.to(device)
    )

    labels = y.to(device)

    loss = F.cross_entropy(
        logits,
        labels,
    )

    predictions = (
        logits.argmax(
            dim=-1
        )
    )

    accuracy = (
        predictions
        .eq(labels)
        .float()
        .mean()
        .item()
        * 100.0
    )

    return {
        "loss": float(
            loss.item()
        ),
        "accuracy": (
            accuracy
        ),
    }


# ============================================================
# TRAIN PROBE
# ============================================================

def train_probe(
    train_x,
    train_y,
    valid_x,
    valid_y,
    test_x,
    test_y,
    number_classes,
    device,
    epochs=30,
    batch_size=128,
    learning_rate=1e-3,
    weight_decay=1e-4,
):

    input_dim = (
        train_x.size(-1)
    )

    probe = LinearProbe(
        input_dim=input_dim,
        number_classes=number_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    dataset = TensorDataset(
        train_x,
        train_y,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    best_valid_acc = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(
        1,
        epochs + 1,
    ):

        probe.train()

        total_loss = 0.0
        total = 0

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(
                device
            )

            batch_y = batch_y.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = probe(
                batch_x
            )

            loss = F.cross_entropy(
                logits,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                float(loss.item())
                * batch_x.size(0)
            )

            total += (
                batch_x.size(0)
            )

        valid_metrics = (
            evaluate_probe(
                probe,
                valid_x,
                valid_y,
                device,
            )
        )

        if (
            valid_metrics[
                "accuracy"
            ]
            > best_valid_acc
        ):

            best_valid_acc = (
                valid_metrics[
                    "accuracy"
                ]
            )

            best_epoch = epoch

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v
                in probe.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == epochs
        ):

            print(
                f"    Epoch {epoch:02d} | "
                f"Train loss "
                f"{total_loss / total:.4f} | "
                f"Val acc "
                f"{valid_metrics['accuracy']:.2f}%"
            )

    probe.load_state_dict(
        best_state
    )

    probe.to(device)

    train_metrics = evaluate_probe(
        probe,
        train_x,
        train_y,
        device,
    )

    valid_metrics = evaluate_probe(
        probe,
        valid_x,
        valid_y,
        device,
    )

    test_metrics = evaluate_probe(
        probe,
        test_x,
        test_y,
        device,
    )

    return {
        "best_epoch": best_epoch,
        "train": train_metrics,
        "valid": valid_metrics,
        "test": test_metrics,
    }


# ============================================================
# REPRESENTATION GEOMETRY
# ============================================================

def representation_stats(x):

    x = x.float()

    n = x.size(0)

    norms = x.norm(
        dim=-1
    )

    # Sample at most 500 for pairwise statistics
    # so this stays cheap.
    if n > 500:

        indices = torch.linspace(
            0,
            n - 1,
            500,
        ).long()

        x = x[
            indices
        ]

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

    m = x.size(0)

    upper = torch.triu(
        torch.ones(
            m,
            m,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    pair_cos = cosine[
        upper
    ]

    pair_l2 = distances[
        upper
    ]

    mean_norm = float(
        norms.mean().item()
    )

    mean_l2 = float(
        pair_l2.mean().item()
    )

    return {
        "mean_norm": mean_norm,
        "mean_pairwise_l2": (
            mean_l2
        ),
        "relative_l2": (
            mean_l2
            / (mean_norm + 1e-8)
        ),
        "mean_cosine": float(
            pair_cos.mean().item()
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
        "--probe-epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
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
    print("REPRESENTATION PROBE")
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
        "Forced memory slot:",
        args.forced_slot,
    )

    print()
    print(
        "This script does NOT train or modify "
        "the memory architecture."
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

    answer_token_map = (
        get_single_token_answers(
            tokenizer
        )
    )

    answer_words = list(
        answer_token_map.keys()
    )

    answer_to_class = {
        answer: index
        for index, answer
        in enumerate(
            answer_words
        )
    }

    number_classes = len(
        answer_words
    )

    print()
    print(
        "Answer classes:"
    )

    for answer in answer_words:

        print(
            f"  {answer:<12} "
            f"class="
            f"{answer_to_class[answer]:2d} "
            f"token="
            f"{answer_token_map[answer]}"
        )

    print()

    print(
        "Number classes:",
        number_classes,
    )

    print(
        "Chance accuracy:",
        f"{100.0 / number_classes:.2f}%",
    )

    print(
        "Chance CE loss:",
        f"{torch.log(torch.tensor(float(number_classes))).item():.4f}",
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

    # ========================================================
    # DATA
    # ========================================================

    train_examples = (
        build_examples(
            n=args.train_examples,
            answer_words=(
                answer_words
            ),
            seed=args.seed,
            start_id=0,
            split="train",
        )
    )

    valid_examples = (
        build_examples(
            n=(
                args.validation_examples
            ),
            answer_words=(
                answer_words
            ),
            seed=args.seed + 1000,
            start_id=100000,
            split="eval",
        )
    )

    test_examples = (
        build_examples(
            n=args.test_examples,
            answer_words=(
                answer_words
            ),
            seed=args.seed + 2000,
            start_id=200000,
            split="eval",
        )
    )

    print()
    print(
        f"Train examples: "
        f"{len(train_examples)}"
    )

    print(
        f"Validation examples: "
        f"{len(valid_examples)}"
    )

    print(
        f"Test examples: "
        f"{len(test_examples)}"
    )

    # ========================================================
    # REPRESENTATION EXTRACTION
    # ========================================================

    print()
    print("=" * 90)
    print("TRAIN REPRESENTATIONS")
    print("=" * 90)

    train_reps, train_y = (
        extract_dataset(
            model=model,
            tokenizer=tokenizer,
            examples=train_examples,
            answer_token_map=(
                answer_token_map
            ),
            answer_to_class=(
                answer_to_class
            ),
            device=device,
            forced_slot=(
                args.forced_slot
            ),
        )
    )

    print()
    print("=" * 90)
    print("VALIDATION REPRESENTATIONS")
    print("=" * 90)

    valid_reps, valid_y = (
        extract_dataset(
            model=model,
            tokenizer=tokenizer,
            examples=valid_examples,
            answer_token_map=(
                answer_token_map
            ),
            answer_to_class=(
                answer_to_class
            ),
            device=device,
            forced_slot=(
                args.forced_slot
            ),
        )
    )

    print()
    print("=" * 90)
    print("TEST REPRESENTATIONS")
    print("=" * 90)

    test_reps, test_y = (
        extract_dataset(
            model=model,
            tokenizer=tokenizer,
            examples=test_examples,
            answer_token_map=(
                answer_token_map
            ),
            answer_to_class=(
                answer_to_class
            ),
            device=device,
            forced_slot=(
                args.forced_slot
            ),
        )
    )

    # ========================================================
    # RAW GEOMETRY FIRST
    # ========================================================

    print()
    print("=" * 90)
    print("RAW REPRESENTATION GEOMETRY")
    print("=" * 90)

    for name in train_reps:

        stats = representation_stats(
            test_reps[name]
        )

        print()
        print(name)

        print(
            f"  mean norm:          "
            f"{stats['mean_norm']:.6f}"
        )

        print(
            f"  mean pairwise L2:   "
            f"{stats['mean_pairwise_l2']:.6f}"
        )

        print(
            f"  L2 / norm:          "
            f"{stats['relative_l2']:.6f}"
        )

        print(
            f"  mean cosine:        "
            f"{stats['mean_cosine']:.6f}"
        )

    # ========================================================
    # PROBES
    # ========================================================

    probe_results = {}

    for name in [
        "MEAN_POOL",
        "ANSWER_TOKEN",
        "WRITTEN_SLOT",
        "WRITE_DELTA",
    ]:

        print()
        print("=" * 90)
        print(
            f"TRAINING LINEAR PROBE: {name}"
        )
        print("=" * 90)

        (
            train_x,
            valid_x,
            test_x,
        ) = standardize(
            train_reps[name],
            valid_reps[name],
            test_reps[name],
        )

        result = train_probe(
            train_x=train_x,
            train_y=train_y,
            valid_x=valid_x,
            valid_y=valid_y,
            test_x=test_x,
            test_y=test_y,
            number_classes=(
                number_classes
            ),
            device=device,
            epochs=(
                args.probe_epochs
            ),
            batch_size=(
                args.batch_size
            ),
            learning_rate=(
                args.learning_rate
            ),
        )

        probe_results[name] = (
            result
        )

        print()
        print(
            "Best epoch:",
            result["best_epoch"],
        )

        print(
            "Train:",
            f"acc={result['train']['accuracy']:.2f}% "
            f"loss={result['train']['loss']:.4f}"
        )

        print(
            "Valid:",
            f"acc={result['valid']['accuracy']:.2f}% "
            f"loss={result['valid']['loss']:.4f}"
        )

        print(
            "Test:",
            f"acc={result['test']['accuracy']:.2f}% "
            f"loss={result['test']['loss']:.4f}"
        )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print("=" * 90)
    print("FINAL REPRESENTATION PROBE RESULTS")
    print("=" * 90)

    print(
        f"{'Representation':<22}"
        f"{'Train':>12}"
        f"{'Valid':>12}"
        f"{'Test':>12}"
        f"{'Test Loss':>14}"
    )

    print("-" * 72)

    for name in [
        "MEAN_POOL",
        "ANSWER_TOKEN",
        "WRITTEN_SLOT",
        "WRITE_DELTA",
    ]:

        result = (
            probe_results[name]
        )

        print(
            f"{name:<22}"
            f"{result['train']['accuracy']:>11.2f}%"
            f"{result['valid']['accuracy']:>11.2f}%"
            f"{result['test']['accuracy']:>11.2f}%"
            f"{result['test']['loss']:>14.4f}"
        )

    # ========================================================
    # SIMPLE INTERPRETATION
    # ========================================================

    chance = (
        100.0
        / number_classes
    )

    mean_acc = (
        probe_results[
            "MEAN_POOL"
        ]["test"]["accuracy"]
    )

    token_acc = (
        probe_results[
            "ANSWER_TOKEN"
        ]["test"]["accuracy"]
    )

    slot_acc = (
        probe_results[
            "WRITTEN_SLOT"
        ]["test"]["accuracy"]
    )

    delta_acc = (
        probe_results[
            "WRITE_DELTA"
        ]["test"]["accuracy"]
    )

    print()
    print("=" * 90)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 90)

    print(
        f"Chance:       {chance:.2f}%"
    )

    print(
        f"Mean pool:    {mean_acc:.2f}%"
    )

    print(
        f"Answer token: {token_acc:.2f}%"
    )

    print(
        f"Written slot: {slot_acc:.2f}%"
    )

    print(
        f"Write delta:  {delta_acc:.2f}%"
    )

    print()

    if (
        token_acc > mean_acc + 15
    ):

        print(
            "Finding: answer-token representation "
            "contains substantially more answer "
            "information than mean pooling."
        )

    if (
        mean_acc >= chance + 20
        and slot_acc <= chance + 10
    ):

        print(
            "Finding: mean-pooled GPT-2 contains "
            "answer information, but much of it is "
            "lost before/inside the final memory slot."
        )

    if (
        token_acc >= chance + 20
        and slot_acc <= chance + 10
    ):

        print(
            "Finding: GPT-2 clearly contains the "
            "answer information, but the written "
            "slot does not preserve it well."
        )

    if (
        slot_acc >= chance + 20
    ):

        print(
            "Finding: written memory slots contain "
            "linearly recoverable answer information. "
            "The failure may therefore be downstream "
            "in reading/fusion rather than storage."
        )

    if (
        max(
            mean_acc,
            token_acc,
        )
        <= chance + 10
    ):

        print(
            "Finding: even the raw GPT-2 "
            "representations contain little linearly "
            "recoverable answer information under "
            "this setup. Do NOT redesign the writer "
            "until the representation/task is fixed."
        )


if __name__ == "__main__":
    main()