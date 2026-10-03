import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from synthetic_retrieval_train import (
    load_model,
    configure_memory_only_training,
)

# ============================================================
# CONFIG
# ============================================================

BASE_CHECKPOINT = "outputs/retrieval_gradient_test/checkpoint_best.pt"
CAPACITY_CHECKPOINT = "outputs/capacity_n64.pt"
MODEL_NAME = "gpt2"

ANSWERS = [
    "tiger", "apple", "blue", "horse",
    "green", "orange", "piano", "river",
    "chair", "lemon", "purple", "rabbit",
    "silver", "garden", "falcon", "banana",
]

A = {
    "entity": "Project-0000",
    "answer": "tiger",
}

B = {
    "entity": "Project-0001",
    "answer": "apple",
}


def complete_example(x):
    return {
        "entity": x["entity"],
        "answer": x["answer"],
        "fact": (
            f"The assigned keyword for {x['entity']} is {x['answer']}. "
            f"Remember that the keyword associated with "
            f"{x['entity']} is {x['answer']}."
        ),
        "query": (
            f"The assigned keyword for {x['entity']} is"
        ),
    }


A = complete_example(A)
B = complete_example(B)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# HELPERS
# ============================================================

def tokenize(text):
    x = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    return (
        x["input_ids"].to(device),
        x["attention_mask"].to(device),
    )


def tensor_summary(name, x):
    """
    Compact tensor printer.
    """
    if x is None:
        print(f"{name}: None")
        return

    if not torch.is_tensor(x):
        print(f"{name}: {x}")
        return

    x = x.detach().float().cpu()

    print(
        f"{name}: "
        f"shape={tuple(x.shape)} | "
        f"mean={x.mean().item():.6f} | "
        f"std={x.std().item():.6f} | "
        f"min={x.min().item():.6f} | "
        f"max={x.max().item():.6f}"
    )


def get_attr(obj, names):
    """
    Try several possible attribute names.
    Useful because diagnostic output class names may differ.
    """
    if obj is None:
        return None

    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value

    return None


# ============================================================
# LOAD TRAINED N=64 MODEL
# ============================================================

print("=" * 100)
print("LOADING MODEL")
print("=" * 100)

model = load_model(
    BASE_CHECKPOINT,
    MODEL_NAME,
    device,
)

checkpoint = torch.load(
    CAPACITY_CHECKPOINT,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

configure_memory_only_training(model)

model.eval()

print(f"Loaded: {CAPACITY_CHECKPOINT}")
print(f"Device: {device}")


# ============================================================
# WRITE ONE FACT
# ============================================================

@torch.no_grad()
def write_fact(model, example, memory_state):

    ids, mask = tokenize(example["fact"])

    output = model(
        input_ids=ids,
        attention_mask=mask,
        memory_state=memory_state,
        update_memory=True,
        return_diagnostics=True,
    )

    return output


# ============================================================
# SCORE ANSWER
# ============================================================

@torch.no_grad()
def score_answer(
    model,
    query,
    candidate,
    memory_state,
):

    q_ids = tokenizer(
        query,
        add_special_tokens=False,
    )["input_ids"]

    a_ids = tokenizer(
        " " + candidate,
        add_special_tokens=False,
    )["input_ids"]

    ids = torch.tensor(
        [q_ids + a_ids],
        dtype=torch.long,
        device=device,
    )

    mask = torch.ones_like(ids)

    labels = torch.full_like(ids, -100)

    labels[:, len(q_ids):] = ids[:, len(q_ids):]

    output = model(
        input_ids=ids,
        attention_mask=mask,
        labels=labels,
        memory_state=memory_state,
        update_memory=False,
        return_diagnostics=False,
    )

    return float(output.lm_loss)


# ============================================================
# QUERY + READER DIAGNOSTICS
# ============================================================

@torch.no_grad()
def query_fact(
    model,
    example,
    memory_state,
    title,
):

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    # --------------------------------------------------------
    # 16-way answer ranking
    # --------------------------------------------------------

    scores = {}

    for candidate in ANSWERS:
        scores[candidate] = score_answer(
            model,
            example["query"],
            candidate,
            memory_state,
        )

    ranking = sorted(
        scores.items(),
        key=lambda x: x[1],
    )

    prediction = ranking[0][0]

    correct_rank = next(
        i + 1
        for i, (answer, _) in enumerate(ranking)
        if answer == example["answer"]
    )

    print(f"Entity:       {example['entity']}")
    print(f"Correct:      {example['answer']}")
    print(f"Prediction:   {prediction}")
    print(f"Correct rank: {correct_rank}/16")

    print("\nTOP-5 ANSWERS")

    for i, (answer, loss) in enumerate(ranking[:5], 1):
        marker = "<-- CORRECT" if answer == example["answer"] else ""

        print(
            f"{i:2d}. "
            f"{answer:8s} "
            f"loss={loss:.6f} "
            f"{marker}"
        )

    print(
        f"\nCorrect-answer loss: "
        f"{scores[example['answer']]:.6f}"
    )

    # --------------------------------------------------------
    # Reader diagnostics using query alone
    # --------------------------------------------------------

    q_ids, q_mask = tokenize(example["query"])

    q_out = model(
        input_ids=q_ids,
        attention_mask=q_mask,
        memory_state=memory_state,
        update_memory=False,
        return_diagnostics=True,
    )

    read = q_out.read_output

    if read is None:
        print("\nReader diagnostics: None")

    else:
        print("\nREADER DIAGNOSTICS")

        attention = get_attr(
            read,
            [
                "attention_weights",
                "attention",
                "weights",
            ],
        )

        usage = get_attr(
            read,
            [
                "slot_usage",
                "usage",
            ],
        )

        confidence = get_attr(
            read,
            [
                "read_confidence",
                "confidence",
            ],
        )

        tensor_summary(
            "reader attention",
            attention,
        )

        if attention is not None:

            att = attention.detach().float()

            # Reduce every dimension except slot dimension.
            while att.ndim > 1:
                att = att.mean(dim=0)

            print(
                "mean reader attention per slot:",
                [
                    round(float(x), 6)
                    for x in att.cpu().flatten()
                ],
            )

        tensor_summary(
            "reader slot usage",
            usage,
        )

        if usage is not None:

            u = usage.detach().float()

            while u.ndim > 1:
                u = u.mean(dim=0)

            print(
                "mean slot usage:",
                [
                    round(float(x), 6)
                    for x in u.cpu().flatten()
                ],
            )

        tensor_summary(
            "reader confidence",
            confidence,
        )

    return {
        "prediction": prediction,
        "rank": correct_rank,
        "correct_loss": scores[example["answer"]],
        "scores": scores,
    }


# ============================================================
# WRITE DIAGNOSTICS
# ============================================================

def print_write_diagnostics(output, label):

    print("\n" + "-" * 100)
    print(f"WRITE DIAGNOSTICS: {label}")
    print("-" * 100)

    routing = output.routing_output
    writer = output.writer_output

    # --------------------------------------------------------
    # Router
    # --------------------------------------------------------

    if routing is None:

        print("Routing output: None")

    else:

        print("\nROUTER")

        weights = get_attr(
            routing,
            [
                "weights",
                "routing_weights",
                "soft_weights",
            ],
        )

        mask = get_attr(
            routing,
            [
                "mask",
                "routing_mask",
                "selected_mask",
            ],
        )

        logits = get_attr(
            routing,
            [
                "logits",
                "routing_logits",
            ],
        )

        tensor_summary(
            "router logits",
            logits,
        )

        tensor_summary(
            "router weights",
            weights,
        )

        if weights is not None:

            w = weights.detach().float()

            while w.ndim > 1:
                w = w.mean(dim=0)

            w = w.flatten()

            print(
                "router weights per slot:",
                [
                    round(float(x), 6)
                    for x in w.cpu()
                ],
            )

            k = min(2, w.numel())

            top = torch.topk(
                w,
                k=k,
            )

            print(
                "router top slots:",
                top.indices.cpu().tolist(),
            )

        tensor_summary(
            "router mask",
            mask,
        )

    # --------------------------------------------------------
    # Write gate
    # --------------------------------------------------------

    print("\nWRITE GATE")

    gate = output.write_gate

    tensor_summary(
        "write gate",
        gate,
    )

    if gate is not None:

        g = gate.detach().float()

        # Vector gate may have hidden dimension.
        # Reduce everything except slot dimension if possible.
        if g.ndim >= 3:
            g = g.mean(dim=-1)

        while g.ndim > 1:
            g = g.mean(dim=0)

        print(
            "mean gate per slot:",
            [
                round(float(x), 6)
                for x in g.cpu().flatten()
            ],
        )

    # --------------------------------------------------------
    # Writer
    # --------------------------------------------------------

    if writer is None:

        print("\nWriter output: None")

    else:

        print("\nWRITER")

        candidates = get_attr(
            writer,
            ["candidates"],
        )

        deltas = get_attr(
            writer,
            ["deltas"],
        )

        attended = get_attr(
            writer,
            ["attended_context"],
        )

        tensor_summary(
            "writer candidates",
            candidates,
        )

        tensor_summary(
            "writer deltas",
            deltas,
        )

        tensor_summary(
            "writer attended context",
            attended,
        )


# ============================================================
# MEMORY COMPARISON
# ============================================================

def compare_memories(
    before,
    after,
    title,
):

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    after_slots = (
        after.slots
        .detach()
        .float()
    )

    if before is None:

        # Compare against zero only for displaying write magnitude.
        before_slots = torch.zeros_like(
            after_slots
        )

        print(
            "NOTE: first write delta shown relative to zero "
            "(not necessarily the learned initial memory)."
        )

    else:

        before_slots = (
            before.slots
            .detach()
            .float()
        )

    diff = after_slots - before_slots

    # Assume [B, slots, hidden] or [slots, hidden]
    if diff.ndim == 3:
        diff_view = diff[0]
        before_view = before_slots[0]
        after_view = after_slots[0]
    else:
        diff_view = diff
        before_view = before_slots
        after_view = after_slots

    print("\nPER-SLOT CHANGE")

    print(
        f"{'SLOT':>6} "
        f"{'DELTA L2':>14} "
        f"{'BEFORE NORM':>14} "
        f"{'AFTER NORM':>14} "
        f"{'COSINE':>12}"
    )

    print("-" * 70)

    for i in range(diff_view.shape[0]):

        delta_l2 = torch.norm(
            diff_view[i]
        ).item()

        before_norm = torch.norm(
            before_view[i]
        ).item()

        after_norm = torch.norm(
            after_view[i]
        ).item()

        if (
            before_norm > 1e-12
            and after_norm > 1e-12
        ):

            cosine = F.cosine_similarity(
                before_view[i].unsqueeze(0),
                after_view[i].unsqueeze(0),
            ).item()

        else:
            cosine = float("nan")

        print(
            f"{i:>6d} "
            f"{delta_l2:>14.6f} "
            f"{before_norm:>14.6f} "
            f"{after_norm:>14.6f} "
            f"{cosine:>12.6f}"
        )

    total_delta = torch.norm(
        diff_view.flatten()
    ).item()

    after_norm = torch.norm(
        after_view.flatten()
    ).item()

    relative = (
        total_delta / after_norm
        if after_norm > 0
        else 0.0
    )

    print("\nTOTAL")

    print(
        f"L2 change:       {total_delta:.6f}"
    )

    print(
        f"Relative change: {relative:.6f}"
    )


# ============================================================
# ACTUAL TEST
# ============================================================

print("\n")
print("#" * 100)
print("TWO-FACT INTERFERENCE TEST")
print("#" * 100)

print(
    f"\nFACT A: {A['entity']} -> {A['answer']}"
)

print(
    f"FACT B: {B['entity']} -> {B['answer']}"
)


# ============================================================
# STEP 1: WRITE A
# ============================================================

print("\n")
print("#" * 100)
print("STEP 1 — WRITE FACT A")
print("#" * 100)

write_A = write_fact(
    model,
    A,
    None,
)

M1 = write_A.memory_state.detach()

print_write_diagnostics(
    write_A,
    "FACT A",
)

compare_memories(
    None,
    M1,
    "MEMORY AFTER WRITING A",
)


# ============================================================
# STEP 2: QUERY A FROM M1
# ============================================================

A_after_A = query_fact(
    model,
    A,
    M1,
    "QUERY A AFTER WRITING ONLY A",
)


# ============================================================
# STEP 3: WRITE B INTO SAME MEMORY
# ============================================================

print("\n")
print("#" * 100)
print("STEP 2 — WRITE FACT B INTO MEMORY CONTAINING A")
print("#" * 100)

write_B = write_fact(
    model,
    B,
    M1,
)

M2 = write_B.memory_state.detach()

print_write_diagnostics(
    write_B,
    "FACT B",
)

compare_memories(
    M1,
    M2,
    "MEMORY CHANGE CAUSED BY WRITING B",
)


# ============================================================
# STEP 4: QUERY A AFTER B
# ============================================================

A_after_AB = query_fact(
    model,
    A,
    M2,
    "QUERY A AFTER WRITING A + B",
)


# ============================================================
# STEP 5: QUERY B AFTER B
# ============================================================

B_after_AB = query_fact(
    model,
    B,
    M2,
    "QUERY B AFTER WRITING A + B",
)


# ============================================================
# CONTROL: B ALONE
# ============================================================

print("\n")
print("#" * 100)
print("CONTROL — WRITE B INTO FRESH MEMORY")
print("#" * 100)

write_B_fresh = write_fact(
    model,
    B,
    None,
)

MB = write_B_fresh.memory_state.detach()

B_alone = query_fact(
    model,
    B,
    MB,
    "QUERY B AFTER WRITING ONLY B",
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 100)
print("FINAL INTERFERENCE SUMMARY")
print("=" * 100)

print(
    f"{'TEST':<28}"
    f"{'PREDICTION':<14}"
    f"{'RANK':>8}"
    f"{'CORRECT LOSS':>18}"
)

print("-" * 100)

rows = [
    (
        "A after A",
        A_after_A,
        A["answer"],
    ),
    (
        "A after A+B",
        A_after_AB,
        A["answer"],
    ),
    (
        "B after A+B",
        B_after_AB,
        B["answer"],
    ),
    (
        "B alone",
        B_alone,
        B["answer"],
    ),
]

for name, result, correct_answer in rows:

    status = (
        "OK"
        if result["prediction"] == correct_answer
        else "FAIL"
    )

    print(
        f"{name:<28}"
        f"{result['prediction']:<14}"
        f"{result['rank']:>8d}"
        f"{result['correct_loss']:>18.6f} "
        f"{status}"
    )


# ============================================================
# INTERFERENCE MAGNITUDE
# ============================================================

print("\n")
print("=" * 100)
print("INTERFERENCE MAGNITUDE")
print("=" * 100)

print(
    f"A correct loss after A:   "
    f"{A_after_A['correct_loss']:.6f}"
)

print(
    f"A correct loss after A+B: "
    f"{A_after_AB['correct_loss']:.6f}"
)

loss_change = (
    A_after_AB["correct_loss"]
    - A_after_A["correct_loss"]
)

print(
    f"A loss increase caused by B: "
    f"{loss_change:+.6f}"
)

print("\nDone.")