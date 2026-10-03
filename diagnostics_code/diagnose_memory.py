from __future__ import annotations

import random
from typing import Any
import copy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from synthetic_retrieval_train import (
    ANSWERS,
    build_dataset,
    build_memory_config,
)

from models.gpt2_memory import (
    MemoryAugmentedGPT2LMHeadModel,
)


# ================================================================
# Configuration
# ================================================================

CHECKPOINT = "outputs/retrieval_gradient_test/checkpoint_best.pt"
MODEL_NAME = "gpt2"

SEED = 42
DISTANCE = 0

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# Pretty printing
# ================================================================

def heading(text):
    print("\n")
    print("=" * 100)
    print(text)
    print("=" * 100)


def subheading(text):
    print("\n" + "-" * 100)
    print(text)
    print("-" * 100)


def tensor_summary(name, tensor):
    if tensor is None:
        print(f"{name}: None")
        return

    if not torch.is_tensor(tensor):
        print(f"{name}: {tensor}")
        return

    x = tensor.detach().float()

    print(
        f"{name}: "
        f"shape={tuple(x.shape)} | "
        f"mean={x.mean().item():.6f} | "
        f"std={x.std(unbiased=False).item():.6f} | "
        f"min={x.min().item():.6f} | "
        f"max={x.max().item():.6f} | "
        f"norm={x.norm().item():.6f}"
    )


def print_object_fields(name, obj):
    subheading(name)

    if obj is None:
        print("None")
        return

    if hasattr(obj, "__dict__"):
        fields = vars(obj)

        for key, value in fields.items():

            if torch.is_tensor(value):
                tensor_summary(key, value)

                # Print complete small tensors.
                if value.numel() <= 100:
                    print(
                        value.detach()
                        .float()
                        .cpu()
                    )

            else:
                print(
                    f"{key}: "
                    f"{type(value).__name__} "
                    f"{value}"
                )

    else:
        print(obj)


def print_diagnostics(diagnostics):
    subheading("BUILT-IN DIAGNOSTICS")

    if not diagnostics:
        print("No diagnostics returned.")
        return

    for key in sorted(diagnostics):

        value = diagnostics[key]

        if torch.is_tensor(value):

            value = value.detach().float().cpu()

            if value.numel() == 1:
                print(
                    f"{key:55s} "
                    f"{value.item():.8f}"
                )
            else:
                print(
                    f"{key:55s} "
                    f"shape={tuple(value.shape)} "
                    f"mean={value.mean().item():.6f} "
                    f"std={value.std(unbiased=False).item():.6f}"
                )

                if value.numel() <= 50:
                    print(value)

        else:
            print(f"{key:55s} {value}")


# ================================================================
# Memory-state inspection
# ================================================================

def print_memory_state(name, state):

    subheading(name)

    if state is None:
        print("Memory state: None")
        return

    slots = state.slots.detach().float()

    print("Slots shape:", tuple(slots.shape))

    norms = slots.norm(dim=-1)

    print("\nSLOT NORMS")
    print(norms.cpu())

    if hasattr(state, "confidence"):
        print("\nCONFIDENCE")
        print(state.confidence.detach().float().cpu())

    if hasattr(state, "age"):
        print("\nAGE")
        print(state.age.detach().cpu())

    if hasattr(state, "write_count"):
        print("\nWRITE COUNT")
        print(state.write_count.detach().cpu())

    if hasattr(state, "read_count"):
        print("\nREAD COUNT")
        print(state.read_count.detach().cpu())

    normalized = F.normalize(
        slots,
        dim=-1,
        eps=1e-8,
    )

    cosine = torch.matmul(
        normalized,
        normalized.transpose(-1, -2),
    )

    print("\nPAIRWISE SLOT COSINE MATRIX")
    print(cosine[0].cpu())


def compare_memory(before, after):

    subheading("MEMORY CHANGE CAUSED BY WRITE")

    if before is None or after is None:
        print(
            "Cannot directly compare because "
            "one memory state is None."
        )
        return

    delta = (
        after.slots.detach().float()
        - before.slots.detach().float()
    )

    slot_delta = delta.norm(dim=-1)[0]

    before_norm = (
        before.slots.detach()
        .float()
        .norm(dim=-1)[0]
    )

    after_norm = (
        after.slots.detach()
        .float()
        .norm(dim=-1)[0]
    )

    print(
        f"{'Slot':<8}"
        f"{'Before':>15}"
        f"{'After':>15}"
        f"{'Delta':>15}"
        f"{'Relative Δ':>15}"
    )

    for i in range(slot_delta.numel()):

        relative = (
            slot_delta[i]
            / (before_norm[i] + 1e-8)
        )

        print(
            f"{i:<8}"
            f"{before_norm[i].item():>15.6f}"
            f"{after_norm[i].item():>15.6f}"
            f"{slot_delta[i].item():>15.6f}"
            f"{relative.item():>15.6f}"
        )

    print("\nTOTAL MEMORY DELTA NORM:")
    print(delta.norm().item())

    print("\nMAX CHANGED SLOT:")
    print(slot_delta.argmax().item())


# ================================================================
# Model loading
# ================================================================

heading("LOADING MODEL")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = (
    MemoryAugmentedGPT2LMHeadModel
    .from_pretrained(
        MODEL_NAME,
        memory_config=build_memory_config(),
    )
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model.to(device)
model.eval()

print("Device:", device)
print("Checkpoint:", CHECKPOINT)


# ================================================================
# Build one controlled example
# ================================================================

examples = build_dataset(
    number_examples=10,
    distances=[DISTANCE],
    seed=SEED + 10000,
    start_id=100000,
)

example = examples[0]

heading("CONTROLLED EXAMPLE")

print("ID:       ", example["id"])
print("Entity:   ", example["entity"])
print("Answer:   ", example["answer"])
print("Distance: ", example["distance"])

print("\nFACT:")
print(example["fact"])

print("\nDISTRACTORS:")
print(example["distractors"])

print("\nQUERY:")
print(example["query"])


# ================================================================
# Token inspection
# ================================================================

heading("TOKENIZATION")

fact_encoding = tokenizer(
    example["fact"],
    return_tensors="pt",
    add_special_tokens=False,
)

fact_tokens = tokenizer.convert_ids_to_tokens(
    fact_encoding["input_ids"][0]
)

print("\nFACT TOKENS")

for i, token in enumerate(fact_tokens):
    print(
        f"{i:3d}: "
        f"{token!r} "
        f"id={fact_encoding['input_ids'][0, i].item()}"
    )


query_encoding = tokenizer(
    example["query"],
    return_tensors="pt",
    add_special_tokens=False,
)

query_tokens = tokenizer.convert_ids_to_tokens(
    query_encoding["input_ids"][0]
)

print("\nQUERY TOKENS")

for i, token in enumerate(query_tokens):
    print(
        f"{i:3d}: "
        f"{token!r} "
        f"id={query_encoding['input_ids'][0, i].item()}"
    )


# ================================================================
# STEP 1 — obtain initial/default memory
# ================================================================

heading("STEP 1 — INITIAL MEMORY")

dummy_ids = query_encoding["input_ids"].to(device)
dummy_mask = query_encoding["attention_mask"].to(device)

with torch.no_grad():

    initial_output = model(
        input_ids=dummy_ids,
        attention_mask=dummy_mask,
        memory_state=None,
        update_memory=False,
        return_diagnostics=True,
    )

initial_memory = initial_output.memory_state

print_memory_state(
    "INITIAL / UNWRITTEN MEMORY",
    initial_memory,
)


# ================================================================
# STEP 2 — write fact
# ================================================================

heading("STEP 2 — WRITE FACT INTO MEMORY")

fact_ids = fact_encoding["input_ids"].to(device)
fact_mask = fact_encoding["attention_mask"].to(device)

with torch.no_grad():

    write_output = model(
        input_ids=fact_ids,
        attention_mask=fact_mask,
        memory_state=initial_memory,
        update_memory=True,
        return_diagnostics=True,
    )

written_memory = write_output.memory_state


# ================================================================
# Inspect hidden representation
# ================================================================

subheading("FACT FORWARD REPRESENTATIONS")

tensor_summary(
    "fused hidden",
    write_output.hidden_states,
)

tensor_summary(
    "memory context",
    write_output.memory_context,
)

tensor_summary(
    "write gate",
    write_output.write_gate,
)


# ================================================================
# Writer internals
# ================================================================

print_object_fields(
    "ROUTER OUTPUT",
    write_output.routing_output,
)

print_object_fields(
    "WRITER OUTPUT",
    write_output.writer_output,
)

print_object_fields(
    "ORTHOGONAL OUTPUT",
    write_output.orthogonal_output,
)

print_object_fields(
    "READ OUTPUT DURING FACT",
    write_output.read_output,
)

print_diagnostics(
    write_output.diagnostics
)


# ================================================================
# Compare memory
# ================================================================

print_memory_state(
    "MEMORY AFTER FACT",
    written_memory,
)

compare_memory(
    initial_memory,
    written_memory,
)


# ================================================================
# STEP 3 — Query helper
# ================================================================

def run_query(memory_state, name):

    encoding = tokenizer(
        example["query"],
        return_tensors="pt",
        add_special_tokens=False,
    )

    ids = encoding["input_ids"].to(device)
    mask = encoding["attention_mask"].to(device)

    with torch.no_grad():

        output = model(
            input_ids=ids,
            attention_mask=mask,
            memory_state=memory_state,
            update_memory=False,
            return_diagnostics=True,
        )

    heading(f"QUERY CONDITION — {name}")

    tensor_summary(
        "fused hidden",
        output.hidden_states,
    )

    tensor_summary(
        "memory context",
        output.memory_context,
    )

    print_object_fields(
        "READER OUTPUT",
        output.read_output,
    )

    print_diagnostics(
        output.diagnostics
    )

    return output


# ================================================================
# STEP 4 — Correct memory
# ================================================================

correct_query = run_query(
    written_memory,
    "CORRECT MEMORY",
)


# ================================================================
# STEP 5 — No fact / default memory
# ================================================================

no_memory_query = run_query(
    initial_memory,
    "UNWRITTEN MEMORY",
)


# ================================================================
# STEP 6 — Build wrong memory
# ================================================================

wrong_example = None

for candidate in examples[1:]:

    if candidate["answer"] != example["answer"]:
        wrong_example = candidate
        break

assert wrong_example is not None

heading("WRONG MEMORY SOURCE")

print("Entity:", wrong_example["entity"])
print("Answer:", wrong_example["answer"])
print("Fact:")
print(wrong_example["fact"])

wrong_encoding = tokenizer(
    wrong_example["fact"],
    return_tensors="pt",
    add_special_tokens=False,
)

with torch.no_grad():

    wrong_write = model(
        input_ids=wrong_encoding[
            "input_ids"
        ].to(device),
        attention_mask=wrong_encoding[
            "attention_mask"
        ].to(device),
        memory_state=initial_memory,
        update_memory=True,
        return_diagnostics=True,
    )
    
wrong_memory = wrong_write.memory_state
wrong_query = run_query(
    wrong_memory,
    "WRONG / SHUFFLED MEMORY",
)

# ================================================================
# ROOT-CAUSE TEST:
# WHERE DO TWO DIFFERENT FACTS BECOME INDISTINGUISHABLE?
# ================================================================

heading("ROOT CAUSE — CORRECT FACT VS WRONG FACT")


def cosine_sim(a, b):
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)

    return F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0),
        dim=-1
    ).item()


def compare_tensor(name, a, b):

    print("\n" + "-" * 100)
    print(name)
    print("-" * 100)

    if a is None or b is None:
        print("Missing tensor.")
        return

    a = a.detach().float()
    b = b.detach().float()

    diff = a - b

    print("Shape:              ", tuple(a.shape))
    print("A norm:             ", a.norm().item())
    print("B norm:             ", b.norm().item())
    print("Difference L2:       ", diff.norm().item())

    denom = (
        a.norm().item()
        + b.norm().item()
        + 1e-8
    )

    print(
        "Relative difference:",
        2 * diff.norm().item() / denom
    )

    print(
        "Cosine similarity:  ",
        cosine_sim(a, b)
    )

    print(
        "Max absolute diff:   ",
        diff.abs().max().item()
    )


# ------------------------------------------------
# 1. ROUTER
# ------------------------------------------------

heading("A — ROUTER COMPARISON")

compare_tensor(
    "Router logits",
    write_output.routing_output.logits,
    wrong_write.routing_output.logits,
)

compare_tensor(
    "Router weights",
    write_output.routing_output.weights,
    wrong_write.routing_output.weights,
)

print("\nCORRECT selected slots:")
print(
    write_output.routing_output
    .selected_indices.detach().cpu()
)

print("\nWRONG selected slots:")
print(
    wrong_write.routing_output
    .selected_indices.detach().cpu()
)


# ------------------------------------------------
# 2. WRITER
# ------------------------------------------------

heading("B — WRITER COMPARISON")

compare_tensor(
    "Writer candidates",
    write_output.writer_output.candidates,
    wrong_write.writer_output.candidates,
)

compare_tensor(
    "Writer deltas",
    write_output.writer_output.deltas,
    wrong_write.writer_output.deltas,
)

compare_tensor(
    "Writer attended context",
    write_output.writer_output.attended_context,
    wrong_write.writer_output.attended_context,
)


# ------------------------------------------------
# 3. ORTHOGONAL UPDATE
# ------------------------------------------------

heading("C — ORTHOGONAL UPDATE COMPARISON")

compare_tensor(
    "Orthogonal updates",
    write_output.orthogonal_output.updates,
    wrong_write.orthogonal_output.updates,
)

compare_tensor(
    "Removed component",
    write_output.orthogonal_output.removed_component,
    wrong_write.orthogonal_output.removed_component,
)


# ------------------------------------------------
# 4. WRITE GATE
# ------------------------------------------------

heading("D — WRITE GATE COMPARISON")

compare_tensor(
    "Write gate",
    write_output.write_gate,
    wrong_write.write_gate,
)

print("\nCORRECT WRITE GATE")
print(
    write_output.write_gate
    .detach().float().cpu().squeeze()
)

print("\nWRONG WRITE GATE")
print(
    wrong_write.write_gate
    .detach().float().cpu().squeeze()
)


# ------------------------------------------------
# 5. FINAL MEMORY
# ------------------------------------------------

heading("E — FINAL MEMORY COMPARISON")

compare_tensor(
    "Entire memory",
    written_memory.slots,
    wrong_memory.slots,
)


correct_slots = (
    written_memory.slots
    .detach().float()[0]
)

wrong_slots = (
    wrong_memory.slots
    .detach().float()[0]
)

print(
    f"\n{'SLOT':<8}"
    f"{'COSINE':>15}"
    f"{'L2 DIFFERENCE':>20}"
)

print("-" * 50)

for i in range(correct_slots.shape[0]):

    cos = F.cosine_similarity(
        correct_slots[i].unsqueeze(0),
        wrong_slots[i].unsqueeze(0),
        dim=-1,
    ).item()

    delta = (
        correct_slots[i]
        - wrong_slots[i]
    ).norm().item()

    print(
        f"{i:<8}"
        f"{cos:>15.8f}"
        f"{delta:>20.8f}"
    )


# ------------------------------------------------
# 6. READER
# ------------------------------------------------

heading("F — READER COMPARISON")

compare_tensor(
    "Reader context",
    correct_query.read_output.context,
    wrong_query.read_output.context,
)

compare_tensor(
    "Reader attention weights",
    correct_query.read_output.attention_weights,
    wrong_query.read_output.attention_weights,
)

compare_tensor(
    "Reader slot usage",
    correct_query.read_output.slot_usage,
    wrong_query.read_output.slot_usage,
)

compare_tensor(
    "Reader confidence",
    correct_query.read_output.read_confidence,
    wrong_query.read_output.read_confidence,
)


# ------------------------------------------------
# 7. FUSED HIDDEN
# ------------------------------------------------

heading("G — FUSION COMPARISON")

compare_tensor(
    "Fused hidden",
    correct_query.hidden_states,
    wrong_query.hidden_states,
)

compare_tensor(
    "Memory context",
    correct_query.memory_context,
    wrong_query.memory_context,
)


# ------------------------------------------------
# 8. LOGITS
# ------------------------------------------------

heading("H — FINAL LOGIT COMPARISON")

compare_tensor(
    "All logits",
    correct_query.logits,
    wrong_query.logits,
)

compare_tensor(
    "Last-token logits",
    correct_query.logits[:, -1, :],
    wrong_query.logits[:, -1, :],
)


# ================================================================
# AUTOMATIC SUMMARY
# ================================================================

heading("ROOT CAUSE SUMMARY")


stages = [
    (
        "Router",
        write_output.routing_output.weights,
        wrong_write.routing_output.weights,
    ),
    (
        "Writer candidate",
        write_output.writer_output.candidates,
        wrong_write.writer_output.candidates,
    ),
    (
        "Writer delta",
        write_output.writer_output.deltas,
        wrong_write.writer_output.deltas,
    ),
    (
        "Orthogonal update",
        write_output.orthogonal_output.updates,
        wrong_write.orthogonal_output.updates,
    ),
    (
        "Final memory",
        written_memory.slots,
        wrong_memory.slots,
    ),
    (
        "Reader context",
        correct_query.read_output.context,
        wrong_query.read_output.context,
    ),
    (
        "Fused hidden",
        correct_query.hidden_states,
        wrong_query.hidden_states,
    ),
    (
        "Logits",
        correct_query.logits,
        wrong_query.logits,
    ),
]


print(
    f"{'STAGE':<25}"
    f"{'COSINE':>15}"
    f"{'RELATIVE Δ':>18}"
)

print("-" * 60)


for name, a, b in stages:

    a = a.detach().float()
    b = b.detach().float()

    cos = cosine_sim(a, b)

    delta = (
        a - b
    ).norm().item()

    relative = (
        2 * delta
        / (
            a.norm().item()
            + b.norm().item()
            + 1e-8
        )
    )

    print(
        f"{name:<25}"
        f"{cos:>15.8f}"
        f"{relative:>18.8f}"
    )


# ================================================================
# STEP 7 — Zero memory intervention
# ================================================================

zero_memory = copy.deepcopy(initial_memory)

zero_memory.slots.zero_()

if hasattr(zero_memory, "confidence"):
    zero_memory.confidence.zero_()

zero_query = run_query(
    zero_memory,
    "ZERO MEMORY",
)


# ================================================================
# STEP 8 — Random memory intervention
# ================================================================

random_memory = copy.deepcopy(initial_memory)

random_memory.slots.copy_(
    torch.randn_like(
        random_memory.slots
    )
)

if hasattr(random_memory, "confidence"):
    random_memory.confidence.fill_(1.0)

random_query = run_query(
    random_memory,
    "RANDOM MEMORY",
)


# ================================================================
# STEP 9 — Amplified correct memory
# ================================================================

amplified_memory = copy.deepcopy(written_memory)
amplified_memory.slots.mul_(10.0)

if hasattr(amplified_memory, "confidence"):
    amplified_memory.confidence.fill_(1.0)

amplified_query = run_query(
    amplified_memory,
    "10X CORRECT MEMORY",
)


# ================================================================
# Compare hidden states and logits
# ================================================================

heading("MEMORY INTERVENTION COMPARISON")


def compare_outputs(name, reference, other):

    hidden_delta = (
        reference.hidden_states.detach().float()
        - other.hidden_states.detach().float()
    )

    logits_delta = (
        reference.logits.detach().float()
        - other.logits.detach().float()
    )

    ref_logits = reference.logits[
        :, -1, :
    ].float()

    other_logits = other.logits[
        :, -1, :
    ].float()

    p = F.softmax(
        ref_logits,
        dim=-1,
    )

    q = F.softmax(
        other_logits,
        dim=-1,
    )

    kl = (
        p
        * (
            torch.log(p + 1e-12)
            - torch.log(q + 1e-12)
        )
    ).sum(dim=-1).mean()

    print(f"\n{name}")

    print(
        "Hidden Δ L2: ",
        hidden_delta.norm().item()
    )

    print(
        "Logits Δ L2: ",
        logits_delta.norm().item()
    )

    print(
        "Last-token logits Δ L2: ",
        (
            ref_logits - other_logits
        ).norm().item()
    )

    print(
        "KL divergence: ",
        kl.item()
    )


compare_outputs(
    "CORRECT vs UNWRITTEN",
    correct_query,
    no_memory_query,
)

compare_outputs(
    "CORRECT vs WRONG",
    correct_query,
    wrong_query,
)

compare_outputs(
    "CORRECT vs ZERO",
    correct_query,
    zero_query,
)

compare_outputs(
    "CORRECT vs RANDOM",
    correct_query,
    random_query,
)

compare_outputs(
    "CORRECT vs 10X CORRECT",
    correct_query,
    amplified_query,
)


# ================================================================
# STEP 10 — Score ALL 16 answers
# ================================================================

heading("ALL 16 ANSWER SCORES")


def answer_loss(
    memory_state,
    answer,
):

    query_ids = tokenizer(
        example["query"],
        add_special_tokens=False,
    )["input_ids"]

    answer_ids = tokenizer(
        " " + answer,
        add_special_tokens=False,
    )["input_ids"]

    full_ids = query_ids + answer_ids

    input_ids = torch.tensor(
        [full_ids],
        dtype=torch.long,
        device=device,
    )

    labels = input_ids.clone()

    labels[
        :, :len(query_ids)
    ] = -100

    attention_mask = torch.ones_like(
        input_ids
    )

    with torch.no_grad():

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            memory_state=memory_state,
            update_memory=False,
            return_diagnostics=False,
        )

    return float(
        output.lm_loss.detach().cpu()
    )


conditions = {
    "CORRECT": written_memory,
    "UNWRITTEN": initial_memory,
    "WRONG": wrong_memory,
    "ZERO": zero_memory,
    "RANDOM": random_memory,
    "10X": amplified_memory,
}


all_scores = {}

for condition, state in conditions.items():

    scores = {}

    for answer in ANSWERS:

        scores[answer] = answer_loss(
            state,
            answer,
        )

    all_scores[condition] = scores


print(
    f"\n{'ANSWER':<12}"
    + "".join(
        f"{condition:>14}"
        for condition in conditions
    )
)

print("-" * 100)

for answer in ANSWERS:

    marker = (
        "*"
        if answer == example["answer"]
        else " "
    )

    print(
        f"{marker}{answer:<11}"
        + "".join(
            f"{all_scores[c][answer]:>14.4f}"
            for c in conditions
        )
    )


print("\nPREDICTIONS")

for condition in conditions:

    prediction = min(
        all_scores[condition],
        key=all_scores[condition].get,
    )

    rank = sorted(
        all_scores[condition],
        key=all_scores[condition].get,
    ).index(
        example["answer"]
    ) + 1

    print(
        f"{condition:12s} "
        f"prediction={prediction:10s} "
        f"correct_rank={rank}/16 "
        f"correct_loss="
        f"{all_scores[condition][example['answer']]:.4f}"
    )


# ================================================================
# STEP 11 — Gradient tracing
# ================================================================

heading("GRADIENT FLOW TEST")

model.train()

for parameter in model.parameters():
    parameter.grad = None


# Re-run fact WITH graph
fact_out = model(
    input_ids=fact_ids,
    attention_mask=fact_mask,
    memory_state=None,
    update_memory=True,
    return_diagnostics=False,
)

gradient_memory = fact_out.memory_state


# Query + correct answer
query_ids = tokenizer(
    example["query"],
    add_special_tokens=False,
)["input_ids"]

answer_ids = tokenizer(
    " " + example["answer"],
    add_special_tokens=False,
)["input_ids"]

full_ids = query_ids + answer_ids

input_ids = torch.tensor(
    [full_ids],
    dtype=torch.long,
    device=device,
)

labels = input_ids.clone()

labels[
    :, :len(query_ids)
] = -100

attention_mask = torch.ones_like(
    input_ids
)


query_out = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    memory_state=gradient_memory,
    update_memory=False,
    return_diagnostics=False,
)

loss = query_out.lm_loss

print("Retrieval LM loss:", loss.item())

loss.backward()


# ================================================================
# Module gradient statistics
# ================================================================

groups = {
    "memory_bank": [],
    "write_gate": [],
    "router": [],
    "writer": [],
    "orthogonalizer": [],
    "reader": [],
    "write_confidence": [],
}


for name, parameter in model.named_parameters():

    if parameter.grad is None:
        continue

    grad_norm = (
        parameter.grad.detach()
        .float()
        .norm()
        .item()
    )

    if "memory_bank" in name:
        groups["memory_bank"].append(grad_norm)

    elif "write_gate" in name:
        groups["write_gate"].append(grad_norm)

    elif "router" in name:
        groups["router"].append(grad_norm)

    elif "writer" in name:
        groups["writer"].append(grad_norm)

    elif "orthogonal" in name:
        groups["orthogonalizer"].append(
            grad_norm
        )

    elif "reader" in name:
        groups["reader"].append(grad_norm)

    elif "write_confidence" in name:
        groups["write_confidence"].append(
            grad_norm
        )


print(
    f"\n{'MODULE':<25}"
    f"{'PARAMS W/ GRAD':>18}"
    f"{'SUM NORM':>18}"
    f"{'MAX NORM':>18}"
)

print("-" * 80)

for name, values in groups.items():

    if values:
        print(
            f"{name:<25}"
            f"{len(values):>18}"
            f"{sum(values):>18.8f}"
            f"{max(values):>18.8f}"
        )

    else:
        print(
            f"{name:<25}"
            f"{0:>18}"
            f"{0:>18.8f}"
            f"{0:>18.8f}"
        )


# ================================================================
# Final checklist
# ================================================================

heading("DIAGNOSTIC FINISHED")

print(
    """
Look especially at:

1. MEMORY CHANGE CAUSED BY WRITE
2. Router selected slots / weights
3. Write gate magnitude
4. Reader attention / slot usage
5. CORRECT vs WRONG hidden delta
6. CORRECT vs WRONG logits delta
7. All-16 answer ranking
8. Per-module gradient norms

Do NOT retrain yet.
"""
)