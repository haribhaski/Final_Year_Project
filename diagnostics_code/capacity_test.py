import random
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

CHECKPOINT = "outputs/retrieval_gradient_test/checkpoint_best.pt"
MODEL_NAME = "gpt2"

CAPACITIES = [64]
STEPS = 300
LR = 5e-5
SEED = 42

ANSWERS = [
    "tiger", "apple", "blue", "horse",
    "green", "orange", "piano", "river",
    "chair", "lemon", "purple", "rabbit",
    "silver", "garden", "falcon", "banana",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATA
# ============================================================

def make_examples(n):

    examples = []

    for i in range(n):

        # Balanced cyclic assignment.
        answer = ANSWERS[i % len(ANSWERS)]

        entity = f"Project-{i:04d}"

        examples.append(
            {
                "id": i,
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
            }
        )

    return examples


# ============================================================
# TOKENIZATION
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
# MEMORY WRITE
# ============================================================

def build_memory(model, example, diagnostics=False):

    ids, mask = tokenize(example["fact"])

    output = model(
        input_ids=ids,
        attention_mask=mask,
        memory_state=None,
        update_memory=True,
        return_diagnostics=diagnostics,
    )

    return output


# ============================================================
# QUERY LOSS
# ============================================================

def query_loss(model, example, memory_state):

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

    return output.lm_loss


# ============================================================
# SCORE ONE CANDIDATE ANSWER
# ============================================================

@torch.no_grad()
def score_answer(model, example, memory_state, answer):

    q_ids = tokenizer(
        example["query"],
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
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, examples):

    model.eval()

    correct = 0
    reciprocal_rank_sum = 0.0
    rank_sum = 0.0

    memories = []

    slot_usage_sum = None
    slot_usage_count = 0

    for example in examples:

        write = build_memory(
            model,
            example,
            diagnostics=True,
        )

        memory = write.memory_state.detach()

        memories.append(
            memory.slots.detach().flatten()
        )

        # ----------------------------------------------------
        # Score ALL 16 answers
        # ----------------------------------------------------

        scores = {}

        for answer in ANSWERS:

            scores[answer] = score_answer(
                model,
                example,
                memory,
                answer,
            )

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
        )

        predicted = ranked[0][0]

        if predicted == example["answer"]:
            correct += 1

        correct_rank = next(
            i + 1
            for i, (answer, _) in enumerate(ranked)
            if answer == example["answer"]
        )

        rank_sum += correct_rank

        reciprocal_rank_sum += 1.0 / correct_rank

        # ----------------------------------------------------
        # Reader slot usage
        # Query again only for diagnostics.
        # ----------------------------------------------------

        q_ids, q_mask = tokenize(example["query"])

        q_out = model(
            input_ids=q_ids,
            attention_mask=q_mask,
            memory_state=memory,
            update_memory=False,
            return_diagnostics=True,
        )

        if (
            q_out.read_output is not None
            and q_out.read_output.slot_usage is not None
        ):

            usage = (
                q_out.read_output
                .slot_usage
                .detach()
                .float()
                .cpu()
            )

            if usage.ndim > 1:
                usage = usage.mean(dim=0)

            if slot_usage_sum is None:
                slot_usage_sum = torch.zeros_like(usage)

            slot_usage_sum += usage
            slot_usage_count += 1

    n = len(examples)

    accuracy = correct / n
    mrr = reciprocal_rank_sum / n
    mean_rank = rank_sum / n

    # --------------------------------------------------------
    # Mean pairwise cosine between memories
    # --------------------------------------------------------

    memory_tensor = torch.stack(memories)

    memory_tensor = F.normalize(
        memory_tensor,
        dim=-1,
    )

    cosine_matrix = (
        memory_tensor @ memory_tensor.T
    )

    if n > 1:

        mask = ~torch.eye(
            n,
            dtype=torch.bool,
            device=cosine_matrix.device,
        )

        mean_memory_cosine = (
            cosine_matrix[mask]
            .mean()
            .item()
        )

    else:
        mean_memory_cosine = 1.0

    # --------------------------------------------------------
    # Aggregate slot usage
    # --------------------------------------------------------

    if (
        slot_usage_sum is not None
        and slot_usage_count > 0
    ):

        mean_slot_usage = (
            slot_usage_sum /
            slot_usage_count
        )

        active_slots = int(
            (mean_slot_usage > 0.01)
            .sum()
            .item()
        )

        slot_usage_text = (
            "[" +
            ", ".join(
                f"{x:.3f}"
                for x in mean_slot_usage.tolist()
            ) +
            "]"
        )

    else:

        active_slots = -1
        slot_usage_text = "N/A"

    model.train()

    return {
        "accuracy": accuracy,
        "mrr": mrr,
        "mean_rank": mean_rank,
        "memory_cosine": mean_memory_cosine,
        "active_slots": active_slots,
        "slot_usage": slot_usage_text,
    }


# ============================================================
# ONE CAPACITY EXPERIMENT
# ============================================================

def run_capacity(n):

    print("\n")
    print("=" * 80)
    print(f"CAPACITY TEST: N={n}")
    print("=" * 80)

    # IMPORTANT:
    # Every N starts from SAME checkpoint.
    # N=64 must NOT inherit training from N=32.

    model = load_model(
        CHECKPOINT,
        MODEL_NAME,
        device,
    )

    configure_memory_only_training(model)

    model.train()

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LR,
    )

    examples = make_examples(n)

    print("\nBefore training:")

    before = evaluate(
        model,
        examples,
    )

    print(
        f"accuracy={100*before['accuracy']:.2f}% | "
        f"MRR={before['mrr']:.4f} | "
        f"mean_rank={before['mean_rank']:.2f} | "
        f"memory_cos={before['memory_cosine']:.4f}"
    )

    # ========================================================
    # TRAIN
    # ========================================================

    for step in range(1, STEPS + 1):

        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0

        # Every association participates in every optimizer step.
        for example in examples:

            write = build_memory(
                model,
                example,
                diagnostics=False,
            )

            loss = query_loss(
                model,
                example,
                write.memory_state,
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
            or step % 50 == 0
            or step == STEPS
        ):

            print(
                f"Step {step:03d} | "
                f"loss={total_loss.item():.6f}"
            )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    result = evaluate(
        model,
        examples,
    )

    result["n"] = n
    result["final_loss"] = float(
        total_loss.item()
    )

    print("\nFINAL")

    print(
        f"N={n} | "
        f"loss={result['final_loss']:.4f} | "
        f"accuracy={100*result['accuracy']:.2f}% | "
        f"MRR={result['mrr']:.4f} | "
        f"mean_rank={result['mean_rank']:.2f} | "
        f"memory_cos={result['memory_cosine']:.4f} | "
        f"active_slots={result['active_slots']}"
    )

    print(
        "slot_usage="
        + result["slot_usage"]
    )

    if n == 64:
        save_path = "outputs/capacity_n64.pt"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "capacity": n,
                "steps": STEPS,
                "learning_rate": LR,
            },
            save_path,
        )

        print(f"\nSaved N=64 model to: {save_path}")

    del optimizer
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result



# ============================================================
# RUN ALL CAPACITIES
# ============================================================

results = []

for n in CAPACITIES:

    result = run_capacity(n)

    results.append(result)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 100)
print("CAPACITY SUMMARY")
print("=" * 100)

print(
    f"{'N':>5} "
    f"{'LOSS':>10} "
    f"{'ACC':>10} "
    f"{'MRR':>10} "
    f"{'MEAN RANK':>12} "
    f"{'MEM COS':>10} "
    f"{'ACTIVE':>8}"
)

print("-" * 100)

for r in results:

    print(
        f"{r['n']:>5} "
        f"{r['final_loss']:>10.4f} "
        f"{100*r['accuracy']:>9.2f}% "
        f"{r['mrr']:>10.4f} "
        f"{r['mean_rank']:>12.2f} "
        f"{r['memory_cosine']:>10.4f} "
        f"{r['active_slots']:>8}"
    )

print("\nChance accuracy with 16 candidates = 6.25%")
print("Ideal MRR = 1.0")
print("Ideal mean rank = 1.0")