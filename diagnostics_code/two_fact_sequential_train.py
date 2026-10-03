import torch
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

STEPS = 300
LR = 5e-5

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
# TOKENIZATION
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


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 90)
print("LOADING MODEL")
print("=" * 90)

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
model.train()

trainable = [
    p for p in model.parameters()
    if p.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable,
    lr=LR,
)

print(f"Loaded: {CAPACITY_CHECKPOINT}")
print(f"Device: {device}")
print(
    "Trainable parameters:",
    sum(p.numel() for p in trainable),
)


# ============================================================
# WRITE
# ============================================================

def write_fact(example, memory_state):

    ids, mask = tokenize(example["fact"])

    output = model(
        input_ids=ids,
        attention_mask=mask,
        memory_state=memory_state,
        update_memory=True,
        return_diagnostics=False,
    )

    return output.memory_state


# ============================================================
# RETRIEVAL LOSS
# ============================================================

def retrieval_loss(example, memory_state):

    q_ids = tokenizer(
        example["query"],
        add_special_tokens=False,
    )["input_ids"]

    a_ids = tokenizer(
        " " + example["answer"],
        add_special_tokens=False,
    )["input_ids"]

    ids = torch.tensor(
        [q_ids + a_ids],
        dtype=torch.long,
        device=device,
    )

    mask = torch.ones_like(ids)

    labels = torch.full_like(
        ids,
        -100,
    )

    labels[:, len(q_ids):] = ids[:, len(q_ids):]

    output = model(
        input_ids=ids,
        attention_mask=mask,
        labels=labels,
        memory_state=memory_state,
        update_memory=False,
        return_diagnostics=False,
    )

    return output.lm_loss


# ============================================================
# SCORE CANDIDATE
# ============================================================

@torch.no_grad()
def score_candidate(
    example,
    candidate,
    memory_state,
):

    q_ids = tokenizer(
        example["query"],
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

    labels = torch.full_like(
        ids,
        -100,
    )

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
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate():

    model.eval()

    # IMPORTANT:
    # SAME sequential write procedure used during training.

    M1 = write_fact(A, None)
    M2 = write_fact(B, M1)

    M2 = M2.detach()

    results = {}

    for name, example in [
        ("A", A),
        ("B", B),
    ]:

        scores = {}

        for candidate in ANSWERS:

            scores[candidate] = score_candidate(
                example,
                candidate,
                M2,
            )

        ranking = sorted(
            scores.items(),
            key=lambda x: x[1],
        )

        prediction = ranking[0][0]

        rank = next(
            i + 1
            for i, (answer, _) in enumerate(ranking)
            if answer == example["answer"]
        )

        results[name] = {
            "prediction": prediction,
            "rank": rank,
            "correct_loss": scores[example["answer"]],
            "best_loss": ranking[0][1],
        }

    model.train()

    return results


# ============================================================
# INITIAL EVALUATION
# ============================================================

print("\n")
print("=" * 90)
print("BEFORE SEQUENTIAL TRAINING")
print("=" * 90)

before = evaluate()

print(
    f"A -> true=tiger | "
    f"pred={before['A']['prediction']} | "
    f"rank={before['A']['rank']} | "
    f"loss={before['A']['correct_loss']:.6f}"
)

print(
    f"B -> true=apple | "
    f"pred={before['B']['prediction']} | "
    f"rank={before['B']['rank']} | "
    f"loss={before['B']['correct_loss']:.6f}"
)


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 90)
print("SEQUENTIAL TWO-FACT TRAINING")
print("=" * 90)

for step in range(1, STEPS + 1):

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    # --------------------------------------------------------
    # SAME MEMORY:
    #
    # empty -> write A -> M1
    # M1    -> write B -> M2
    #
    # DO NOT DETACH M1 OR M2.
    #
    # We need gradients from BOTH retrieval losses to flow
    # through the sequential writes.
    # --------------------------------------------------------

    M1 = write_fact(
        A,
        None,
    )

    M2 = write_fact(
        B,
        M1,
    )

    # --------------------------------------------------------
    # BOTH A and B must be retrievable from FINAL MEMORY M2
    # --------------------------------------------------------

    loss_A = retrieval_loss(
        A,
        M2,
    )

    loss_B = retrieval_loss(
        B,
        M2,
    )

    loss = (
        loss_A +
        loss_B
    ) / 2.0

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable,
        1.0,
    )

    optimizer.step()

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    if (
        step == 1
        or step % 25 == 0
        or step == STEPS
    ):

        result = evaluate()

        print(
            f"\nStep {step:03d} | "
            f"loss={loss.item():.6f} | "
            f"loss_A={loss_A.item():.6f} | "
            f"loss_B={loss_B.item():.6f} | "
            f"grad={float(grad_norm):.4f}"
        )

        print(
            f"  A: "
            f"pred={result['A']['prediction']:8s} | "
            f"rank={result['A']['rank']:2d} | "
            f"loss={result['A']['correct_loss']:.6f}"
        )

        print(
            f"  B: "
            f"pred={result['B']['prediction']:8s} | "
            f"rank={result['B']['rank']:2d} | "
            f"loss={result['B']['correct_loss']:.6f}"
        )


# ============================================================
# FINAL EVALUATION
# ============================================================

final = evaluate()

print("\n")
print("=" * 90)
print("FINAL RESULT")
print("=" * 90)

print(
    f"A after A+B | "
    f"true=tiger | "
    f"pred={final['A']['prediction']} | "
    f"rank={final['A']['rank']} | "
    f"loss={final['A']['correct_loss']:.6f}"
)

print(
    f"B after A+B | "
    f"true=apple | "
    f"pred={final['B']['prediction']} | "
    f"rank={final['B']['rank']} | "
    f"loss={final['B']['correct_loss']:.6f}"
)


# ============================================================
# SAVE
# ============================================================

save_path = "outputs/two_fact_sequential_trained.pt"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "steps": STEPS,
        "learning_rate": LR,
        "facts": [
            ("Project-0000", "tiger"),
            ("Project-0001", "apple"),
        ],
    },
    save_path,
)

print(
    f"\nSaved model to: {save_path}"
)