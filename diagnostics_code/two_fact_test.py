import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from synthetic_retrieval_train import (
    load_model,
    configure_memory_only_training,
)


CHECKPOINT = "outputs/retrieval_gradient_test/checkpoint_best.pt"
MODEL_NAME = "gpt2"

STEPS = 300
LR = 5e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATA
# ============================================================

examples = [
    {
        "name": "A",
        "fact": (
            "The assigned keyword for Project-A is rabbit. "
            "Remember that the keyword associated with Project-A is rabbit."
        ),
        "query": "The assigned keyword for Project-A is",
        "answer": "rabbit",
    },
    {
        "name": "B",
        "fact": (
            "The assigned keyword for Project-B is river. "
            "Remember that the keyword associated with Project-B is river."
        ),
        "query": "The assigned keyword for Project-B is",
        "answer": "river",
    },
]


# ============================================================
# LOAD
# ============================================================

print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = load_model(
    CHECKPOINT,
    MODEL_NAME,
    device,
)

configure_memory_only_training(model)

model.train()

trainable = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(
    trainable,
    lr=LR,
)


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


def build_memory(example):
    """
    Write exactly ONE fact.

    IMPORTANT:
    We return the graph-connected memory during training.
    """

    ids, mask = tokenize(example["fact"])

    output = model(
        input_ids=ids,
        attention_mask=mask,
        memory_state=None,
        update_memory=True,
        return_diagnostics=True,
    )

    return output


def query_loss(example, memory_state):
    """
    Query the written memory.

    Loss is calculated ONLY on the answer token(s).
    """

    query = example["query"]
    answer = " " + example["answer"]

    q_ids = tokenizer(
        query,
        add_special_tokens=False,
    )["input_ids"]

    a_ids = tokenizer(
        answer,
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
        return_diagnostics=True,
    )

    return output.lm_loss, output


# ============================================================
# CANDIDATE SCORING
# ============================================================

@torch.no_grad()
def score_answer(example, memory_state, answer):

    query = example["query"]
    answer_text = " " + answer

    q_ids = tokenizer(
        query,
        add_special_tokens=False,
    )["input_ids"]

    a_ids = tokenizer(
        answer_text,
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


@torch.no_grad()
def evaluate():

    model.eval()

    memories = {}

    print("\n" + "=" * 75)
    print("EVALUATION")
    print("=" * 75)

    for example in examples:

        write = build_memory(example)

        memory = write.memory_state

        # detach because evaluation doesn't need graph
        memory = memory.detach()

        memories[example["name"]] = memory

        rabbit_loss = score_answer(
            example,
            memory,
            "rabbit",
        )

        river_loss = score_answer(
            example,
            memory,
            "river",
        )

        prediction = (
            "rabbit"
            if rabbit_loss < river_loss
            else "river"
        )

        print(
            f"{example['name']} | "
            f"correct={example['answer']:7s} | "
            f"rabbit={rabbit_loss:.4f} | "
            f"river={river_loss:.4f} | "
            f"prediction={prediction}"
        )

    # --------------------------------------------------------
    # Compare A memory vs B memory
    # --------------------------------------------------------

    A = memories["A"].slots.flatten()
    B = memories["B"].slots.flatten()

    cosine = F.cosine_similarity(
        A.unsqueeze(0),
        B.unsqueeze(0),
    ).item()

    delta = torch.norm(A - B).item()

    relative_delta = (
        delta /
        (torch.norm(A).item() + 1e-12)
    )

    print("\nMEMORY DIFFERENCE")

    print(
        f"A/B cosine:        {cosine:.8f}"
    )

    print(
        f"A/B L2 difference: {delta:.8f}"
    )

    print(
        f"A/B relative diff: {relative_delta:.8f}"
    )

    model.train()


# ============================================================
# BEFORE TRAINING
# ============================================================

print("\nBEFORE TRAINING")

evaluate()


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 75)
print("TRAINING TWO FACTS")
print("=" * 75)

for step in range(1, STEPS + 1):

    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0

    # Both associations contribute to every update.
    for example in examples:

        write_output = build_memory(example)

        memory_state = write_output.memory_state

        loss, query_output = query_loss(
            example,
            memory_state,
        )

        total_loss = total_loss + loss

    total_loss = total_loss / len(examples)

    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        trainable,
        1.0,
    )

    optimizer.step()

    if (
        step == 1
        or step % 25 == 0
        or step == STEPS
    ):

        print(
            f"\nSTEP {step:03d} | "
            f"loss={total_loss.item():.6f}"
        )

        evaluate()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("FINAL RESULT")
print("=" * 75)

evaluate()