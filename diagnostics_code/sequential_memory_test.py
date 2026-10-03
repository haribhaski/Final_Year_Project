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

TEST_SIZES = [1, 2, 4, 8, 12, 16]

ANSWERS = [
    "tiger", "apple", "blue", "horse",
    "green", "orange", "piano", "river",
    "chair", "lemon", "purple", "rabbit",
    "silver", "garden", "falcon", "banana",
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# DATA
# ============================================================

def make_examples(n):

    examples = []

    for i in range(n):

        entity = f"Project-{i:04d}"
        answer = ANSWERS[i % len(ANSWERS)]

        examples.append({
            "entity": entity,
            "answer": answer,

            "fact": (
                f"The assigned keyword for {entity} is {answer}. "
                f"Remember that the keyword associated with "
                f"{entity} is {answer}."
            ),

            "query": (
                f"The assigned keyword for {entity} is"
            ),
        })

    return examples


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
# LOAD N=64 TRAINED MODEL
# ============================================================

print("Loading model...")

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

print("Loaded:", CAPACITY_CHECKPOINT)


# ============================================================
# WRITE MULTIPLE FACTS INTO SAME MEMORY
# ============================================================

@torch.no_grad()
def write_all_facts(model, examples):

    memory_state = None

    print("\nWRITE SEQUENCE")
    print("-" * 70)

    for i, example in enumerate(examples):

        ids, mask = tokenize(example["fact"])

        output = model(
            input_ids=ids,
            attention_mask=mask,
            memory_state=memory_state,
            update_memory=True,
            return_diagnostics=True,
        )

        memory_state = output.memory_state.detach()

        print(
            f"Write {i+1:02d}: "
            f"{example['entity']} -> {example['answer']}"
        )

    return memory_state


# ============================================================
# SCORE CANDIDATE
# ============================================================

@torch.no_grad()
def score_answer(
    model,
    query,
    answer,
    memory_state,
):

    q_ids = tokenizer(
        query,
        add_special_tokens=False,
    )["input_ids"]

    a_ids = tokenizer(
        " " + answer,
        add_special_tokens=False,
    )["input_ids"]

    ids = torch.tensor(
        [q_ids + a_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(ids)

    labels = torch.full_like(ids, -100)

    labels[:, len(q_ids):] = ids[:, len(q_ids):]

    output = model(
        input_ids=ids,
        attention_mask=attention_mask,
        labels=labels,
        memory_state=memory_state,
        update_memory=False,
        return_diagnostics=False,
    )

    return float(output.lm_loss)


# ============================================================
# EVALUATE ALL FACTS FROM SAME MEMORY
# ============================================================

@torch.no_grad()
def evaluate(model, examples, memory_state):

    correct = 0
    reciprocal_rank = 0.0
    total_rank = 0.0

    results = []

    for example in examples:

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

        rank = next(
            i + 1
            for i, (answer, _) in enumerate(ranking)
            if answer == example["answer"]
        )

        is_correct = (
            prediction == example["answer"]
        )

        correct += int(is_correct)

        reciprocal_rank += 1.0 / rank
        total_rank += rank

        results.append({
            "entity": example["entity"],
            "answer": example["answer"],
            "prediction": prediction,
            "rank": rank,
            "correct_loss": scores[example["answer"]],
            "best_loss": ranking[0][1],
            "correct": is_correct,
        })

    n = len(examples)

    return {
        "accuracy": correct / n,
        "mrr": reciprocal_rank / n,
        "mean_rank": total_rank / n,
        "results": results,
    }


# ============================================================
# RUN TEST
# ============================================================

all_results = []

print("\n")
print("=" * 90)
print("SEQUENTIAL MULTI-FACT MEMORY TEST")
print("=" * 90)

for n in TEST_SIZES:

    print("\n")
    print("=" * 90)
    print(f"TESTING {n} FACT(S) IN SAME MEMORY")
    print("=" * 90)

    examples = make_examples(n)

    memory = write_all_facts(
        model,
        examples,
    )

    result = evaluate(
        model,
        examples,
        memory,
    )

    print("\nRETRIEVAL RESULTS")
    print("-" * 90)

    for r in result["results"]:

        status = "OK" if r["correct"] else "FAIL"

        print(
            f"{r['entity']:14s} "
            f"true={r['answer']:8s} "
            f"pred={r['prediction']:8s} "
            f"rank={r['rank']:2d} "
            f"loss={r['correct_loss']:.4f} "
            f"{status}"
        )

    print("\nSUMMARY")

    print(
        f"N={n} | "
        f"accuracy={100*result['accuracy']:.2f}% | "
        f"MRR={result['mrr']:.4f} | "
        f"mean_rank={result['mean_rank']:.2f}"
    )

    all_results.append({
        "n": n,
        "accuracy": result["accuracy"],
        "mrr": result["mrr"],
        "mean_rank": result["mean_rank"],
    })


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("FINAL SEQUENTIAL MEMORY SUMMARY")
print("=" * 90)

print(
    f"{'FACTS':>8} "
    f"{'ACCURACY':>12} "
    f"{'MRR':>10} "
    f"{'MEAN RANK':>12}"
)

print("-" * 90)

for r in all_results:

    print(
        f"{r['n']:>8} "
        f"{100*r['accuracy']:>11.2f}% "
        f"{r['mrr']:>10.4f} "
        f"{r['mean_rank']:>12.2f}"
    )

print("\n16-way chance accuracy = 6.25%")