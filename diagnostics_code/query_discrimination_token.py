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

# Architecture/base checkpoint used only to construct model.
BASE_CHECKPOINT = "outputs/retrieval_gradient_test/checkpoint_best.pt"

# THIS is the model from the latest experiment.
TRAINED_CHECKPOINT = "outputs/two_fact_token_collision.pt"

MODEL_NAME = "gpt2"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

A = {
    "entity": "Project-A",
    "answer": "rabbit",
}

B = {
    "entity": "Project-B",
    "answer": "river",
}


# ============================================================
# BUILD EXAMPLES
# ============================================================

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


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


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


# ============================================================
# GENERIC HELPERS
# ============================================================

def get_attr(obj, names):

    if obj is None:
        return None

    for name in names:

        if hasattr(obj, name):

            value = getattr(obj, name)

            if value is not None:
                return value

    return None


def flatten(x):

    if x is None:
        return None

    return x.detach().float().flatten()


def compare(name, a, b):

    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)

    if a is None or b is None:
        print("Unavailable")
        return

    a = flatten(a)
    b = flatten(b)

    if a.numel() != b.numel():

        print(
            f"Shape mismatch: "
            f"{a.numel()} vs {b.numel()}"
        )

        return

    cosine = F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0),
        dim=-1,
    ).item()

    delta = torch.norm(
        a - b
    ).item()

    norm_a = torch.norm(a).item()
    norm_b = torch.norm(b).item()

    relative = delta / (
        ((norm_a + norm_b) / 2.0)
        + 1e-12
    )

    print(f"A norm:        {norm_a:.8f}")
    print(f"B norm:        {norm_b:.8f}")
    print(f"L2 difference: {delta:.8f}")
    print(f"Relative diff: {relative:.8f}")
    print(f"Cosine:        {cosine:.8f}")


def metric(a, b):

    if a is None or b is None:
        return None

    a = flatten(a)
    b = flatten(b)

    if a.numel() != b.numel():
        return None

    cosine = F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0),
        dim=-1,
    ).item()

    delta = torch.norm(
        a - b
    ).item()

    denom = (
        torch.norm(a).item()
        + torch.norm(b).item()
    ) / 2.0

    relative = (
        delta / (denom + 1e-12)
    )

    return cosine, relative


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 100)
print("LOADING TOKEN + COLLISION TRAINED MODEL")
print("=" * 100)

model = load_model(
    BASE_CHECKPOINT,
    MODEL_NAME,
    device,
)

checkpoint = torch.load(
    TRAINED_CHECKPOINT,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

# IMPORTANT:
# The saved state_dict does not store the Python string
# model.reader.mode.
#
# The model was trained using token mode, so restore it.
print(
    "Reader mode after architecture construction:",
    model.reader.mode,
)

model.reader.mode = "token"

print(
    "Reader mode used for diagnostic:",
    model.reader.mode,
)

configure_memory_only_training(model)

model.eval()

print(
    f"Loaded trained checkpoint: "
    f"{TRAINED_CHECKPOINT}"
)

print(
    f"Device: {device}"
)


# ============================================================
# WRITE FACT
# ============================================================

@torch.no_grad()
def write_fact(example, memory_state):

    ids, mask = tokenize(
        example["fact"]
    )

    out = model(
        input_ids=ids,
        attention_mask=mask,
        memory_state=memory_state,
        update_memory=True,
        return_diagnostics=True,
    )

    return out


# ============================================================
# BUILD A -> B SEQUENTIAL MEMORY
# ============================================================

print("\n" + "=" * 100)
print("BUILDING FINAL A+B MEMORY")
print("=" * 100)

with torch.no_grad():

    write_A = write_fact(
        A,
        None,
    )

    M1 = write_A.memory_state.detach()

    write_B = write_fact(
        B,
        M1,
    )

    M2 = write_B.memory_state.detach()


print("\nWritten sequentially:")

print(
    f"  {A['entity']} -> {A['answer']}"
)

print(
    f"  {B['entity']} -> {B['answer']}"
)


# ============================================================
# WRITE ROUTING
# ============================================================

def print_write_router(
    name,
    output,
):

    routing = output.routing_output

    print("\n" + name)

    if routing is None:

        print("Router unavailable")
        return

    weights = (
        routing.weights[0]
        .detach()
        .float()
        .cpu()
    )

    print(
        "weights:",
        [
            round(float(x), 6)
            for x in weights
        ],
    )

    if routing.selected_indices is not None:

        selected = (
            routing.selected_indices[0]
            .detach()
            .cpu()
            .tolist()
        )

        print(
            "selected:",
            selected,
        )


print("\n" + "=" * 100)
print("WRITE ROUTING")
print("=" * 100)

print_write_router(
    "A WRITE",
    write_A,
)

print_write_router(
    "B WRITE",
    write_B,
)


# ============================================================
# QUERY
# ============================================================

@torch.no_grad()
def run_query(example):

    ids, mask = tokenize(
        example["query"]
    )

    out = model(
        input_ids=ids,
        attention_mask=mask,
        memory_state=M2,
        update_memory=False,
        return_diagnostics=True,
    )

    return out


out_A = run_query(A)
out_B = run_query(B)


# ============================================================
# READ OUTPUT
# ============================================================

read_A = out_A.read_output
read_B = out_B.read_output


att_A = get_attr(
    read_A,
    [
        "attention_weights",
        "attention",
        "weights",
    ],
)

att_B = get_attr(
    read_B,
    [
        "attention_weights",
        "attention",
        "weights",
    ],
)


usage_A = get_attr(
    read_A,
    [
        "slot_usage",
        "usage",
    ],
)

usage_B = get_attr(
    read_B,
    [
        "slot_usage",
        "usage",
    ],
)


context_A = get_attr(
    read_A,
    [
        "context",
        "memory_context",
        "retrieved_context",
        "read_context",
    ],
)

context_B = get_attr(
    read_B,
    [
        "context",
        "memory_context",
        "retrieved_context",
        "read_context",
    ],
)


if context_A is None:

    context_A = getattr(
        out_A,
        "memory_context",
        None,
    )


if context_B is None:

    context_B = getattr(
        out_B,
        "memory_context",
        None,
    )


hidden_A = getattr(
    out_A,
    "hidden_states",
    None,
)

hidden_B = getattr(
    out_B,
    "hidden_states",
    None,
)


logits_A = getattr(
    out_A,
    "logits",
    None,
)

logits_B = getattr(
    out_B,
    "logits",
    None,
)


# ============================================================
# ATTENTION SHAPE
# ============================================================

print("\n" + "=" * 100)
print("READER ATTENTION SHAPES")
print("=" * 100)

if att_A is not None:

    print(
        "A attention shape:",
        tuple(att_A.shape),
    )

else:

    print(
        "A attention unavailable"
    )


if att_B is not None:

    print(
        "B attention shape:",
        tuple(att_B.shape),
    )

else:

    print(
        "B attention unavailable"
    )


# ============================================================
# SLOT USAGE
# ============================================================

print("\n" + "=" * 100)
print("READER SLOT USAGE")
print("=" * 100)


def slot_vector(x):

    if x is None:
        return None

    x = (
        x.detach()
        .float()
    )

    # Keep reducing dimensions until only slots remain.
    while x.ndim > 1:
        x = x.mean(dim=0)

    return x


uA = slot_vector(
    usage_A
)

uB = slot_vector(
    usage_B
)


if uA is not None:

    print(
        "Query A:",
        [
            round(float(x), 6)
            for x in uA.cpu()
        ],
    )


if uB is not None:

    print(
        "Query B:",
        [
            round(float(x), 6)
            for x in uB.cpu()
        ],
    )


compare(
    "A vs B SLOT USAGE",
    usage_A,
    usage_B,
)


# ============================================================
# LAST-TOKEN READER DISTRIBUTION
#
# This is especially important because the last query token
# predicts the answer.
# ============================================================

print("\n" + "=" * 100)
print("LAST-TOKEN READER DISTRIBUTION")
print("=" * 100)


def last_token_slot_distribution(attention):

    if attention is None:
        return None

    x = (
        attention.detach()
        .float()
    )

    print(
        "Raw attention shape:",
        tuple(x.shape),
    )

    # Expected common shape:
    # [B, H, T, N]
    if x.ndim == 4:

        # Last query token
        x = x[:, :, -1, :]

        # Average heads
        x = x.mean(dim=1)

        return x


    # Possible:
    # [B, T, N]
    elif x.ndim == 3:

        return x[:, -1, :]


    # Possible:
    # [B, N]
    elif x.ndim == 2:

        return x


    print(
        "WARNING: Unexpected attention shape."
    )

    return None


last_att_A = (
    last_token_slot_distribution(
        att_A
    )
)

last_att_B = (
    last_token_slot_distribution(
        att_B
    )
)


if last_att_A is not None:

    print(
        "\nQuery A last-token slots:"
    )

    print(
        [
            round(float(x), 8)
            for x in last_att_A[0].cpu()
        ]
    )


if last_att_B is not None:

    print(
        "\nQuery B last-token slots:"
    )

    print(
        [
            round(float(x), 8)
            for x in last_att_B[0].cpu()
        ]
    )


compare(
    "LAST-TOKEN READER ATTENTION",
    last_att_A,
    last_att_B,
)


# ============================================================
# FULL READER ATTENTION
# ============================================================

compare(
    "FULL READER ATTENTION",
    att_A,
    att_B,
)


# ============================================================
# RETRIEVED CONTEXT
# ============================================================

compare(
    "RETRIEVED MEMORY CONTEXT",
    context_A,
    context_B,
)


# ============================================================
# BASE GPT-2 QUERY REPRESENTATION
# ============================================================

print("\n" + "=" * 100)
print("BASE GPT-2 QUERY REPRESENTATION")
print("=" * 100)


@torch.no_grad()
def base_hidden(example):

    ids, mask = tokenize(
        example["query"]
    )

    base = model.backbone.transformer(
        input_ids=ids,
        attention_mask=mask,
        return_dict=True,
    )

    return base.last_hidden_state


base_A = base_hidden(A)
base_B = base_hidden(B)


compare(
    "FULL BASE QUERY HIDDEN",
    base_A,
    base_B,
)

compare(
    "LAST-TOKEN BASE QUERY HIDDEN",
    base_A[:, -1, :],
    base_B[:, -1, :],
)


# ============================================================
# FUSED HIDDEN
# ============================================================

print("\n" + "=" * 100)
print("FUSED HIDDEN STATE")
print("=" * 100)

compare(
    "FULL FUSED HIDDEN",
    hidden_A,
    hidden_B,
)


last_hidden_A = None
last_hidden_B = None

if (
    hidden_A is not None
    and hidden_B is not None
    and hidden_A.ndim >= 3
):

    last_hidden_A = (
        hidden_A[:, -1, :]
    )

    last_hidden_B = (
        hidden_B[:, -1, :]
    )

    compare(
        "LAST-TOKEN FUSED HIDDEN",
        last_hidden_A,
        last_hidden_B,
    )


# ============================================================
# LOGITS
# ============================================================

print("\n" + "=" * 100)
print("FINAL LOGITS")
print("=" * 100)

compare(
    "FULL LOGITS",
    logits_A,
    logits_B,
)


last_logits_A = None
last_logits_B = None


if (
    logits_A is not None
    and logits_B is not None
):

    last_logits_A = (
        logits_A[:, -1, :]
    )

    last_logits_B = (
        logits_B[:, -1, :]
    )

    compare(
        "LAST-TOKEN LOGITS",
        last_logits_A,
        last_logits_B,
    )


# ============================================================
# KL DIVERGENCE
# ============================================================

if (
    last_logits_A is not None
    and last_logits_B is not None
):

    log_prob_A = F.log_softmax(
        last_logits_A.float(),
        dim=-1,
    )

    log_prob_B = F.log_softmax(
        last_logits_B.float(),
        dim=-1,
    )

    prob_A = log_prob_A.exp()
    prob_B = log_prob_B.exp()


    kl_A_B = F.kl_div(
        log_prob_B,
        prob_A,
        reduction="batchmean",
    ).item()


    kl_B_A = F.kl_div(
        log_prob_A,
        prob_B,
        reduction="batchmean",
    ).item()


    print("\n" + "=" * 100)
    print("OUTPUT DISTRIBUTION DIFFERENCE")
    print("=" * 100)

    print(
        f"KL(A || B): {kl_A_B:.10f}"
    )

    print(
        f"KL(B || A): {kl_B_A:.10f}"
    )


# ============================================================
# RABBIT / RIVER TOKEN LOGITS
# ============================================================

print("\n" + "=" * 100)
print("RABBIT vs RIVER LOGITS")
print("=" * 100)


def answer_token(answer):

    ids = tokenizer(
        " " + answer,
        add_special_tokens=False,
    )["input_ids"]

    if len(ids) != 1:

        print(
            f"WARNING: '{answer}' "
            f"uses {len(ids)} tokens: {ids}"
        )

    return ids[0]


rabbit_id = answer_token(
    "rabbit"
)

river_id = answer_token(
    "river"
)


print(
    "rabbit token id:",
    rabbit_id,
)

print(
    "river token id:",
    river_id,
)


if (
    logits_A is not None
    and logits_B is not None
):

    A_rabbit = float(
        logits_A[
            0,
            -1,
            rabbit_id,
        ]
    )

    A_river = float(
        logits_A[
            0,
            -1,
            river_id,
        ]
    )

    B_rabbit = float(
        logits_B[
            0,
            -1,
            rabbit_id,
        ]
    )

    B_river = float(
        logits_B[
            0,
            -1,
            river_id,
        ]
    )


    print(
        "\nQuery A = Project-A"
    )

    print(
        f"rabbit logit = {A_rabbit:.6f}"
    )

    print(
        f"river  logit = {A_river:.6f}"
    )

    print(
        f"rabbit - river = "
        f"{A_rabbit - A_river:+.6f}"
    )


    print(
        "\nQuery B = Project-B"
    )

    print(
        f"rabbit logit = {B_rabbit:.6f}"
    )

    print(
        f"river  logit = {B_river:.6f}"
    )

    print(
        f"river - rabbit = "
        f"{B_river - B_rabbit:+.6f}"
    )


# ============================================================
# WRITE vs READ ALIGNMENT
# ============================================================

print("\n" + "=" * 100)
print("WRITE vs READ ALIGNMENT")
print("=" * 100)


write_route_A = (
    write_A.routing_output.weights
    .detach()
    .float()
)

write_route_B = (
    write_B.routing_output.weights
    .detach()
    .float()
)


def alignment(name, write_route, read_route):

    print("\n" + name)

    if (
        write_route is None
        or read_route is None
    ):

        print(
            "Unavailable"
        )

        return

    w = write_route.flatten()
    r = read_route.flatten()

    if w.numel() != r.numel():

        print(
            "Shape mismatch:",
            w.shape,
            r.shape,
        )

        return

    cosine = F.cosine_similarity(
        w.unsqueeze(0),
        r.unsqueeze(0),
    ).item()

    dot = torch.sum(
        w * r
    ).item()

    print(
        f"cosine = {cosine:.8f}"
    )

    print(
        f"dot     = {dot:.8f}"
    )


alignment(
    "Query A read vs A write",
    write_route_A,
    last_att_A,
)

alignment(
    "Query A read vs B write",
    write_route_B,
    last_att_A,
)

alignment(
    "Query B read vs B write",
    write_route_B,
    last_att_B,
)

alignment(
    "Query B read vs A write",
    write_route_A,
    last_att_B,
)


# ============================================================
# COMPACT SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("COMPACT DISCRIMINATION SUMMARY")
print("=" * 100)


rows = [
    (
        "Last reader attention",
        last_att_A,
        last_att_B,
    ),
    (
        "Full reader attention",
        att_A,
        att_B,
    ),
    (
        "Reader context",
        context_A,
        context_B,
    ),
    (
        "Base query hidden",
        base_A,
        base_B,
    ),
    (
        "Fused hidden",
        hidden_A,
        hidden_B,
    ),
    (
        "Logits",
        logits_A,
        logits_B,
    ),
]


print(
    f"{'STAGE':<28}"
    f"{'COSINE':>15}"
    f"{'RELATIVE DIFF':>20}"
)

print("-" * 68)


for name, x, y in rows:

    result = metric(
        x,
        y,
    )

    if result is None:

        print(
            f"{name:<28}"
            f"{'N/A':>15}"
            f"{'N/A':>20}"
        )

    else:

        cosine, relative = result

        print(
            f"{name:<28}"
            f"{cosine:>15.8f}"
            f"{relative:>20.8f}"
        )


print("\n" + "=" * 100)
print("WHAT TO LOOK FOR")
print("=" * 100)

print(
    """
1. A and B WRITE ROUTING:
   Did A mainly write slot 2 while B mainly wrote slot 7?

2. LAST-TOKEN READER DISTRIBUTION:
   Does Query A read a different slot distribution from Query B?

3. WRITE vs READ ALIGNMENT:
   Ideally:
       Query A read ~ A write
       Query B read ~ B write

   and NOT:
       Query A read ~ B write
       Query B read ~ A write

4. If write routing differs but reader attention is still identical:
   that directly confirms a write-read binding/addressing failure.

5. If reader attention differs but logits remain almost identical:
   then the next bottleneck is fusion/output rather than addressing.
"""
)

print("Done.")