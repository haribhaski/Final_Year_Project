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
MODEL_NAME = "gpt2"

STEPS = 300
LR = 5e-5
LOG_EVERY = 25

SAVE_PATH = "outputs/two_fact_forced_slots.pt"

# Force different addresses
SLOT_A = 2
SLOT_B = 7

BINDING_WEIGHT = 1.0

ANSWERS = [
    "tiger", "apple", "blue", "horse",
    "green", "orange", "piano", "river",
    "chair", "lemon", "purple", "rabbit",
    "silver", "garden", "falcon", "banana",
]

A = {
    "entity": "Project-A",
    "answer": "rabbit",
}

B = {
    "entity": "Project-B",
    "answer": "river",
}


# ============================================================
# TEXT
# ============================================================

def fact_text(example):
    return (
        f"The assigned keyword for {example['entity']} is "
        f"{example['answer']}. "
        f"Remember that the keyword associated with "
        f"{example['entity']} is {example['answer']}."
    )


def query_text(example):
    return (
        f"The assigned keyword for {example['entity']} is"
    )


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_context(tokenizer, text, device):
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    return (
        enc["input_ids"].to(device),
        enc["attention_mask"].to(device),
    )


def prepare_query_answer(
    tokenizer,
    example,
    candidate_answer,
    device,
):
    query_ids = tokenizer(
        query_text(example),
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    answer_ids = tokenizer(
        " " + candidate_answer,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    input_ids = torch.cat(
        [query_ids, answer_ids],
        dim=1,
    ).to(device)

    attention_mask = torch.ones_like(input_ids)

    labels = input_ids.clone()

    # Only answer tokens contribute to LM loss
    labels[:, :query_ids.size(1)] = -100

    return input_ids, attention_mask, labels


# ============================================================
# FORCED ROUTING
# ============================================================

class ForcedRouter:
    """
    Temporarily intercepts the router output so a fact is
    written to one explicitly chosen slot.

    We patch router.forward for the duration of the write.
    """

    def __init__(self, model, slot):
        self.model = model
        self.slot = slot
        self.original_forward = None

    def __enter__(self):

        router = self.model.router

        self.original_forward = router.forward

        slot = self.slot
        original_forward = self.original_forward

        def forced_forward(*args, **kwargs):

            # First get a normal output so we preserve the
            # expected RoutingOutput object/fields.
            output = original_forward(
                *args,
                **kwargs,
            )

            weights = torch.zeros_like(
                output.weights
            )

            weights[..., slot] = 1.0

            output.weights = weights

            # Update mask if it exists.
            if hasattr(output, "mask"):
                if output.mask is not None:
                    mask = torch.zeros_like(
                        output.mask
                    )

                    mask[..., slot] = 1

                    output.mask = mask

            # Update selected indices if available.
            if hasattr(
                output,
                "selected_indices",
            ):
                if output.selected_indices is not None:
                    batch_size = weights.size(0)

                    output.selected_indices = (
                        torch.full(
                            (batch_size, 1),
                            slot,
                            dtype=torch.long,
                            device=weights.device,
                        )
                    )

            return output

        router.forward = forced_forward

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.model.router.forward = (
            self.original_forward
        )


# ============================================================
# WRITE
# ============================================================

def write_fact_forced(
    model,
    tokenizer,
    example,
    memory_state,
    slot,
    device,
):

    ids, mask = tokenize_context(
        tokenizer,
        fact_text(example),
        device,
    )

    with ForcedRouter(model, slot):

        output = model(
            input_ids=ids,
            attention_mask=mask,
            memory_state=memory_state,
            update_memory=True,
            return_diagnostics=True,
        )

    return output


# ============================================================
# QUERY
# ============================================================

def answer_loss(
    model,
    tokenizer,
    example,
    candidate_answer,
    memory_state,
    device,
):

    ids, mask, labels = prepare_query_answer(
        tokenizer,
        example,
        candidate_answer,
        device,
    )

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
# READER
# ============================================================

def get_last_reader_distribution(output):

    attention = (
        output.read_output.attention_weights
    )

    if attention is None:
        raise RuntimeError(
            "Reader attention unavailable."
        )

    if attention.ndim != 4:
        raise RuntimeError(
            f"Expected [B,H,T,N], "
            f"got {tuple(attention.shape)}"
        )

    return (
        attention[:, :, -1, :]
        .mean(dim=1)
    )


def forced_target(
    batch_size,
    num_slots,
    slot,
    device,
):

    target = torch.zeros(
        batch_size,
        num_slots,
        device=device,
    )

    target[:, slot] = 1.0

    return target


# ============================================================
# RANKING
# ============================================================

@torch.no_grad()
def rank_example(
    model,
    tokenizer,
    example,
    memory_state,
    device,
):

    scores = []

    for candidate in ANSWERS:

        loss, _ = answer_loss(
            model,
            tokenizer,
            example,
            candidate,
            memory_state,
            device,
        )

        scores.append(
            (
                candidate,
                float(loss.item()),
            )
        )

    scores.sort(
        key=lambda x: x[1]
    )

    correct = example["answer"]

    rank = next(
        i + 1
        for i, (answer, _) in enumerate(scores)
        if answer == correct
    )

    correct_loss = next(
        loss
        for answer, loss in scores
        if answer == correct
    )

    return {
        "prediction": scores[0][0],
        "rank": rank,
        "correct_loss": correct_loss,
        "scores": scores,
    }


# ============================================================
# MEMORY CHANGE
# ============================================================

@torch.no_grad()
def slot_changes(before, after):

    diff = (
        after.slots.detach()
        - before.slots.detach()
    )

    return (
        torch.norm(
            diff,
            dim=-1,
        )[0]
        .cpu()
        .tolist()
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    device,
):

    model.eval()

    # --------------------------------------------------------
    # WRITE A -> SLOT 2
    # --------------------------------------------------------

    out_a = write_fact_forced(
        model,
        tokenizer,
        A,
        None,
        SLOT_A,
        device,
    )

    memory_a = out_a.memory_state

    # Save memory before B
    memory_before_b = memory_a

    # --------------------------------------------------------
    # WRITE B -> SLOT 7
    # --------------------------------------------------------

    out_b = write_fact_forced(
        model,
        tokenizer,
        B,
        memory_a,
        SLOT_B,
        device,
    )

    memory_ab = out_b.memory_state

    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    result_a = rank_example(
        model,
        tokenizer,
        A,
        memory_ab,
        device,
    )

    result_b = rank_example(
        model,
        tokenizer,
        B,
        memory_ab,
        device,
    )

    # --------------------------------------------------------
    # QUERY FOR ATTENTION
    # --------------------------------------------------------

    _, query_a = answer_loss(
        model,
        tokenizer,
        A,
        A["answer"],
        memory_ab,
        device,
    )

    _, query_b = answer_loss(
        model,
        tokenizer,
        B,
        B["answer"],
        memory_ab,
        device,
    )

    read_a = (
        get_last_reader_distribution(
            query_a
        )
    )

    read_b = (
        get_last_reader_distribution(
            query_b
        )
    )

    changes = slot_changes(
        memory_before_b,
        memory_ab,
    )

    model.train()

    return {
        "A": result_a,
        "B": result_b,

        "read_A":
            read_a[0].cpu().tolist(),

        "read_B":
            read_b[0].cpu().tolist(),

        "B_slot_changes":
            changes,
    }


# ============================================================
# PRINT HELPERS
# ============================================================

def pretty(x):

    return [
        round(float(v), 4)
        for v in x
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 90)
    print(
        "TWO-FACT FORCED SLOT EXPERIMENT"
    )
    print("=" * 90)

    print("Device:", device)

    print(
        "Base checkpoint:",
        BASE_CHECKPOINT,
    )

    print(
        f"\nFORCED ADDRESS:"
        f"\n  A -> slot {SLOT_A}"
        f"\n  B -> slot {SLOT_B}"
    )

    # ========================================================
    # LOAD
    # ========================================================

    model = load_model(
        BASE_CHECKPOINT,
        MODEL_NAME,
        device,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print(
        "\nOriginal reader mode:",
        model.reader.mode,
    )

    model.reader.mode = "token"

    print(
        "Reader mode:",
        model.reader.mode,
    )

    print(
        "Reader top-k:",
        model.reader.top_k,
    )

    configure_memory_only_training(
        model
    )

    # --------------------------------------------------------
    # Freeze router.
    #
    # There is nothing for the router to learn in this test.
    # --------------------------------------------------------

    for param in model.router.parameters():
        param.requires_grad = False

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LR,
    )

    print(
        "\nA:",
        A,
    )

    print(
        "B:",
        B,
    )

    # ========================================================
    # BEFORE
    # ========================================================

    print("\n" + "=" * 90)
    print("BEFORE TRAINING")
    print("=" * 90)

    before = evaluate(
        model,
        tokenizer,
        device,
    )

    print(
        f"A expected={A['answer']} "
        f"| pred={before['A']['prediction']} "
        f"| rank={before['A']['rank']}/16 "
        f"| loss={before['A']['correct_loss']:.4f}"
    )

    print(
        f"B expected={B['answer']} "
        f"| pred={before['B']['prediction']} "
        f"| rank={before['B']['rank']}/16 "
        f"| loss={before['B']['correct_loss']:.4f}"
    )

    print(
        "\nA reader:",
        pretty(
            before["read_A"]
        ),
    )

    print(
        "B reader:",
        pretty(
            before["read_B"]
        ),
    )

    print(
        "\nSlot changes caused by B:",
        pretty(
            before["B_slot_changes"]
        ),
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print("\n" + "=" * 90)
    print("TRAINING")
    print("=" * 90)

    for step in range(
        1,
        STEPS + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        # ====================================================
        # WRITE A -> SLOT 2
        # ====================================================

        out_a = write_fact_forced(
            model,
            tokenizer,
            A,
            None,
            SLOT_A,
            device,
        )

        memory_a = (
            out_a.memory_state
        )

        # ====================================================
        # WRITE B -> SLOT 7
        #
        # IMPORTANT:
        # memory_a is NOT detached.
        # ====================================================

        out_b = write_fact_forced(
            model,
            tokenizer,
            B,
            memory_a,
            SLOT_B,
            device,
        )

        memory_ab = (
            out_b.memory_state
        )

        # ====================================================
        # RETRIEVE A
        # ====================================================

        loss_a, query_a = answer_loss(
            model,
            tokenizer,
            A,
            A["answer"],
            memory_ab,
            device,
        )

        # ====================================================
        # RETRIEVE B
        # ====================================================

        loss_b, query_b = answer_loss(
            model,
            tokenizer,
            B,
            B["answer"],
            memory_ab,
            device,
        )

        retrieval_loss = (
            loss_a + loss_b
        ) / 2.0

        # ====================================================
        # READER DISTRIBUTIONS
        # ====================================================

        read_a = (
            get_last_reader_distribution(
                query_a
            )
        )

        read_b = (
            get_last_reader_distribution(
                query_b
            )
        )

        # ====================================================
        # EXPLICIT TARGETS
        #
        # Query A MUST read slot 2.
        # Query B MUST read slot 7.
        # ====================================================

        target_a = forced_target(
            read_a.size(0),
            read_a.size(1),
            SLOT_A,
            device,
        )

        target_b = forced_target(
            read_b.size(0),
            read_b.size(1),
            SLOT_B,
            device,
        )

        eps = 1e-8

        bind_a = -(
            target_a
            * torch.log(
                read_a.clamp_min(eps)
            )
        ).sum(
            dim=-1
        ).mean()

        bind_b = -(
            target_b
            * torch.log(
                read_b.clamp_min(eps)
            )
        ).sum(
            dim=-1
        ).mean()

        binding_loss = (
            bind_a + bind_b
        ) / 2.0

        # ====================================================
        # TOTAL
        # ====================================================

        loss = (
            retrieval_loss
            + BINDING_WEIGHT
            * binding_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable,
            1.0,
        )

        optimizer.step()

        # ====================================================
        # LOG
        # ====================================================

        if (
            step == 1
            or step % LOG_EVERY == 0
            or step == STEPS
        ):

            result = evaluate(
                model,
                tokenizer,
                device,
            )

            print(
                "\n" + "-" * 90
            )

            print(
                f"STEP {step:03d}"
                f" | total={loss.item():.6f}"
                f" | retrieval={retrieval_loss.item():.6f}"
                f" | binding={binding_loss.item():.6f}"
                f" | bindA={bind_a.item():.6f}"
                f" | bindB={bind_b.item():.6f}"
            )

            print(
                f"A: pred={result['A']['prediction']:>7}"
                f" | rank={result['A']['rank']:2d}/16"
                f" | loss={result['A']['correct_loss']:.4f}"
            )

            print(
                f"B: pred={result['B']['prediction']:>7}"
                f" | rank={result['B']['rank']:2d}/16"
                f" | loss={result['B']['correct_loss']:.4f}"
            )

            print(
                "A read:",
                pretty(
                    result["read_A"]
                ),
            )

            print(
                "B read:",
                pretty(
                    result["read_B"]
                ),
            )

            print(
                "B write slot changes:",
                pretty(
                    result[
                        "B_slot_changes"
                    ]
                ),
            )

    # ========================================================
    # FINAL
    # ========================================================

    final = evaluate(
        model,
        tokenizer,
        device,
    )

    print("\n" + "=" * 90)
    print("FINAL RESULT")
    print("=" * 90)

    print(
        f"A ({A['answer']}): "
        f"prediction={final['A']['prediction']} "
        f"| rank={final['A']['rank']}/16 "
        f"| loss={final['A']['correct_loss']:.4f}"
    )

    print(
        f"B ({B['answer']}): "
        f"prediction={final['B']['prediction']} "
        f"| rank={final['B']['rank']}/16 "
        f"| loss={final['B']['correct_loss']:.4f}"
    )

    print("\nEXPECTED ADDRESSING")

    print(
        f"A write -> slot {SLOT_A}"
    )

    print(
        f"B write -> slot {SLOT_B}"
    )

    print("\nACTUAL READER DISTRIBUTIONS")

    print(
        "A:",
        pretty(
            final["read_A"]
        ),
    )

    print(
        "B:",
        pretty(
            final["read_B"]
        ),
    )

    print(
        "\nB-induced per-slot memory changes:",
        pretty(
            final["B_slot_changes"]
        ),
    )

    # ========================================================
    # READER METRICS
    # ========================================================

    ra = torch.tensor(
        final["read_A"]
    )

    rb = torch.tensor(
        final["read_B"]
    )

    reader_cosine = (
        F.cosine_similarity(
            ra.unsqueeze(0),
            rb.unsqueeze(0),
        ).item()
    )

    a_target_mass = float(
        ra[SLOT_A]
    )

    b_target_mass = float(
        rb[SLOT_B]
    )

    print("\n" + "=" * 90)
    print("ADDRESSING DIAGNOSTICS")
    print("=" * 90)

    print(
        f"A reader mass on slot {SLOT_A}: "
        f"{a_target_mass:.6f}"
    )

    print(
        f"B reader mass on slot {SLOT_B}: "
        f"{b_target_mass:.6f}"
    )

    print(
        f"A/B reader cosine: "
        f"{reader_cosine:.6f}"
    )

    # ========================================================
    # TOP ANSWERS
    # ========================================================

    print("\nTop-5 A:")

    for answer, score in (
        final["A"]["scores"][:5]
    ):
        print(
            f"  {answer:>8}: {score:.4f}"
        )

    print("\nTop-5 B:")

    for answer, score in (
        final["B"]["scores"][:5]
    ):
        print(
            f"  {answer:>8}: {score:.4f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "forced_slot_A":
                SLOT_A,

            "forced_slot_B":
                SLOT_B,

            "reader_mode":
                model.reader.mode,
        },
        SAVE_PATH,
    )

    print(
        "\nSaved:",
        SAVE_PATH,
    )

    # ========================================================
    # AUTOMATIC INTERPRETATION
    # ========================================================

    a_ok = (
        final["A"]["rank"] == 1
    )

    b_ok = (
        final["B"]["rank"] == 1
    )

    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)

    if a_ok and b_ok:

        print(
            "SUCCESS: Both associations are rank-1 "
            "when given separate memory addresses."
        )

        print(
            "This strongly isolates the main failure "
            "to learned address allocation/routing."
        )

    elif (
        a_target_mass > 0.7
        and b_target_mass > 0.7
    ):

        print(
            "Reader learned the correct separate addresses, "
            "but retrieval still failed."
        )

        print(
            "The next bottleneck is memory content encoding "
            "or memory-to-GPT fusion, not addressing."
        )

    else:

        print(
            "Even with forced separate writes, the reader "
            "did not reliably learn query-specific retrieval."
        )

        print(
            "The read-side addressing mechanism also needs "
            "investigation."
        )


if __name__ == "__main__":
    main()