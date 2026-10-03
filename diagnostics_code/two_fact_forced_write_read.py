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

SAVE_PATH = "outputs/two_fact_forced_write_read.pt"

SLOT_A = 2
SLOT_B = 7

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

    attention_mask = torch.ones_like(
        input_ids
    )

    labels = input_ids.clone()

    # Only answer tokens contribute to LM loss
    labels[:, :query_ids.size(1)] = -100

    return (
        input_ids,
        attention_mask,
        labels,
    )


# ============================================================
# FORCED WRITE ROUTER
# ============================================================

class ForcedRouter:

    def __init__(self, model, slot):
        self.model = model
        self.slot = slot
        self.original_forward = None

    def __enter__(self):

        router = self.model.router

        self.original_forward = (
            router.forward
        )

        original_forward = (
            self.original_forward
        )

        slot = self.slot

        def forced_forward(*args, **kwargs):

            output = original_forward(
                *args,
                **kwargs,
            )

            # ----------------------------------------------
            # Force router weight
            # ----------------------------------------------

            weights = torch.zeros_like(
                output.weights
            )

            weights[..., slot] = 1.0

            output.weights = weights

            # ----------------------------------------------
            # Force mask
            # ----------------------------------------------

            if hasattr(output, "mask"):

                if output.mask is not None:

                    mask = torch.zeros_like(
                        output.mask
                    )

                    mask[..., slot] = 1

                    output.mask = mask

            # ----------------------------------------------
            # Force selected index
            # ----------------------------------------------

            if hasattr(
                output,
                "selected_indices",
            ):

                if (
                    output.selected_indices
                    is not None
                ):

                    batch_size = (
                        weights.size(0)
                    )

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
# WRITE FACT
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

    with ForcedRouter(
        model,
        slot,
    ):

        output = model(
            input_ids=ids,
            attention_mask=mask,
            memory_state=memory_state,
            update_memory=True,
            return_diagnostics=True,
        )

    return output


# ============================================================
# FORCED READ
# ============================================================

class ForcedReader:
    """
    Force the reader to use exactly one memory slot.

    We accomplish this using the memory mask passed to the
    reader.

    Only the requested slot is visible.
    """

    def __init__(self, model, slot):
        self.model = model
        self.slot = slot

        self.original_forward = None

    def __enter__(self):

        reader = self.model.reader

        self.original_forward = (
            reader.forward
        )

        original_forward = (
            self.original_forward
        )

        slot = self.slot

        def forced_forward(
            hidden_states,
            memory_slots,
            *args,
            **kwargs,
        ):

            batch_size = (
                memory_slots.size(0)
            )

            num_slots = (
                memory_slots.size(1)
            )

            # ----------------------------------------------
            # Build mask:
            #
            # 0 = hidden
            # 1 = available
            # ----------------------------------------------

            forced_mask = torch.zeros(
                batch_size,
                num_slots,
                dtype=torch.bool,
                device=memory_slots.device,
            )

            forced_mask[:, slot] = True

            # Override any existing memory mask.
            kwargs["memory_mask"] = (
                forced_mask
            )

            return original_forward(
                hidden_states,
                memory_slots,
                *args,
                **kwargs,
            )

        reader.forward = forced_forward

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.model.reader.forward = (
            self.original_forward
        )


# ============================================================
# QUERY WITH FORCED READ
# ============================================================

def answer_loss_forced_read(
    model,
    tokenizer,
    example,
    candidate_answer,
    memory_state,
    slot,
    device,
):

    ids, mask, labels = (
        prepare_query_answer(
            tokenizer,
            example,
            candidate_answer,
            device,
        )
    )

    with ForcedReader(
        model,
        slot,
    ):

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
# READER DISTRIBUTION
# ============================================================

def get_last_reader_distribution(
    output,
):

    attention = (
        output
        .read_output
        .attention_weights
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


# ============================================================
# RANKING
# ============================================================

@torch.no_grad()
def rank_example(
    model,
    tokenizer,
    example,
    memory_state,
    forced_slot,
    device,
):

    scores = []

    for candidate in ANSWERS:

        loss, _ = (
            answer_loss_forced_read(
                model,
                tokenizer,
                example,
                candidate,
                memory_state,
                forced_slot,
                device,
            )
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
        "prediction":
            scores[0][0],

        "rank":
            rank,

        "correct_loss":
            correct_loss,

        "scores":
            scores,
    }


# ============================================================
# MEMORY DIFFERENCE
# ============================================================

@torch.no_grad()
def slot_changes(
    before,
    after,
):

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
# EVALUATE
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    device,
):

    model.eval()

    # ========================================================
    # WRITE A -> SLOT 2
    # ========================================================

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

    # ========================================================
    # WRITE B -> SLOT 7
    # ========================================================

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

    # ========================================================
    # FORCE A READ -> SLOT 2
    # ========================================================

    result_a = rank_example(
        model,
        tokenizer,
        A,
        memory_ab,
        SLOT_A,
        device,
    )

    # ========================================================
    # FORCE B READ -> SLOT 7
    # ========================================================

    result_b = rank_example(
        model,
        tokenizer,
        B,
        memory_ab,
        SLOT_B,
        device,
    )

    # ========================================================
    # GET QUERY OUTPUTS
    # ========================================================

    _, query_a = (
        answer_loss_forced_read(
            model,
            tokenizer,
            A,
            A["answer"],
            memory_ab,
            SLOT_A,
            device,
        )
    )

    _, query_b = (
        answer_loss_forced_read(
            model,
            tokenizer,
            B,
            B["answer"],
            memory_ab,
            SLOT_B,
            device,
        )
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
        memory_a,
        memory_ab,
    )

    model.train()

    return {
        "A":
            result_a,

        "B":
            result_b,

        "read_A":
            read_a[0]
            .detach()
            .cpu()
            .tolist(),

        "read_B":
            read_b[0]
            .detach()
            .cpu()
            .tolist(),

        "B_slot_changes":
            changes,
    }


# ============================================================
# PRINT HELPER
# ============================================================

def pretty(x):

    return [
        round(
            float(v),
            4,
        )
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
        "TWO-FACT FORCED WRITE + "
        "FORCED READ EXPERIMENT"
    )

    print("=" * 90)

    print(
        "Device:",
        device,
    )

    print(
        "Base checkpoint:",
        BASE_CHECKPOINT,
    )

    print()

    print(
        f"A: {A['entity']} -> "
        f"{A['answer']} "
        f"| WRITE slot {SLOT_A} "
        f"| READ slot {SLOT_A}"
    )

    print(
        f"B: {B['entity']} -> "
        f"{B['answer']} "
        f"| WRITE slot {SLOT_B} "
        f"| READ slot {SLOT_B}"
    )

    # ========================================================
    # LOAD MODEL
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

    # ========================================================
    # MEMORY-ONLY TRAINING
    # ========================================================

    configure_memory_only_training(
        model
    )

    # Router is irrelevant because write address is forced.
    for param in model.router.parameters():
        param.requires_grad = False

    # ========================================================
    # IMPORTANT
    #
    # Reader addressing itself is also not what we are
    # testing here.
    #
    # We still allow reader value/output/fusion parameters
    # to learn, but its attention address is externally
    # constrained by the mask.
    # ========================================================

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in trainable
        ),
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LR,
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
        "\nA forced reader:",
        pretty(
            before["read_A"]
        ),
    )

    print(
        "B forced reader:",
        pretty(
            before["read_B"]
        ),
    )

    print(
        "\nB-induced slot changes:",
        pretty(
            before[
                "B_slot_changes"
            ]
        ),
    )

    # ========================================================
    # TRAINING
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
        # QUERY A -> FORCE READ SLOT 2
        # ====================================================

        loss_a, query_a = (
            answer_loss_forced_read(
                model,
                tokenizer,
                A,
                A["answer"],
                memory_ab,
                SLOT_A,
                device,
            )
        )

        # ====================================================
        # QUERY B -> FORCE READ SLOT 7
        # ====================================================

        loss_b, query_b = (
            answer_loss_forced_read(
                model,
                tokenizer,
                B,
                B["answer"],
                memory_ab,
                SLOT_B,
                device,
            )
        )

        retrieval_loss = (
            loss_a + loss_b
        ) / 2.0

        # No collision loss.
        # No binding loss.
        #
        # Addressing is already perfect by construction.

        loss = retrieval_loss

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
                f" | retrieval="
                f"{retrieval_loss.item():.6f}"
                f" | lossA={loss_a.item():.6f}"
                f" | lossB={loss_b.item():.6f}"
            )

            print(
                f"A: pred="
                f"{result['A']['prediction']:>7}"
                f" | rank="
                f"{result['A']['rank']:2d}/16"
                f" | loss="
                f"{result['A']['correct_loss']:.4f}"
            )

            print(
                f"B: pred="
                f"{result['B']['prediction']:>7}"
                f" | rank="
                f"{result['B']['rank']:2d}/16"
                f" | loss="
                f"{result['B']['correct_loss']:.4f}"
            )

            print(
                "A forced read:",
                pretty(
                    result["read_A"]
                ),
            )

            print(
                "B forced read:",
                pretty(
                    result["read_B"]
                ),
            )

            print(
                "B-induced slot changes:",
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
        f"prediction="
        f"{final['A']['prediction']} "
        f"| rank="
        f"{final['A']['rank']}/16 "
        f"| loss="
        f"{final['A']['correct_loss']:.4f}"
    )

    print(
        f"B ({B['answer']}): "
        f"prediction="
        f"{final['B']['prediction']} "
        f"| rank="
        f"{final['B']['rank']}/16 "
        f"| loss="
        f"{final['B']['correct_loss']:.4f}"
    )

    print("\nFORCED ADDRESSING")

    print(
        f"A: write slot {SLOT_A} "
        f"-> read slot {SLOT_A}"
    )

    print(
        f"B: write slot {SLOT_B} "
        f"-> read slot {SLOT_B}"
    )

    print("\nACTUAL READER ATTENTION")

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
        "\nB-induced slot changes:",
        pretty(
            final[
                "B_slot_changes"
            ]
        ),
    )

    # ========================================================
    # TOP-5
    # ========================================================

    print("\nTop-5 A:")

    for answer, score in (
        final["A"]["scores"][:5]
    ):

        print(
            f"  {answer:>8}: "
            f"{score:.4f}"
        )

    print("\nTop-5 B:")

    for answer, score in (
        final["B"]["scores"][:5]
    ):

        print(
            f"  {answer:>8}: "
            f"{score:.4f}"
        )

    # ========================================================
    # VERIFY MASK REALLY WORKED
    # ========================================================

    ra = torch.tensor(
        final["read_A"]
    )

    rb = torch.tensor(
        final["read_B"]
    )

    a_mass = float(
        ra[SLOT_A]
    )

    b_mass = float(
        rb[SLOT_B]
    )

    print("\n" + "=" * 90)
    print("FORCED READ VERIFICATION")
    print("=" * 90)

    print(
        f"A attention on slot {SLOT_A}: "
        f"{a_mass:.6f}"
    )

    print(
        f"B attention on slot {SLOT_B}: "
        f"{b_mass:.6f}"
    )

    if (
        a_mass < 0.99
        or b_mass < 0.99
    ):

        print(
            "\nWARNING:"
        )

        print(
            "The memory mask did NOT produce "
            "near-one-hot attention."
        )

        print(
            "Do NOT interpret the retrieval "
            "result as a valid forced-read test."
        )

    # ========================================================
    # SAVE
    # ========================================================

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "slot_A":
                SLOT_A,

            "slot_B":
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
    # INTERPRET
    # ========================================================

    a_ok = (
        final["A"]["rank"] == 1
    )

    b_ok = (
        final["B"]["rank"] == 1
    )

    forced_ok = (
        a_mass > 0.99
        and b_mass > 0.99
    )

    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)

    if not forced_ok:

        print(
            "INVALID FORCED-READ TEST:"
        )

        print(
            "Reader attention was not actually "
            "restricted to the intended slots."
        )

    elif a_ok and b_ok:

        print(
            "SUCCESS."
        )

        print(
            "With perfect write and read addressing, "
            "the memory system can simultaneously "
            "store and retrieve both associations."
        )

        print()

        print(
            "This isolates the major failure to "
            "ADDRESSING:"
        )

        print(
            "1. learned write allocation/router"
        )

        print(
            "2. learned query-specific read selection"
        )

    else:

        print(
            "DEEPER FAILURE."
        )

        print(
            "Even with perfect write and read "
            "addresses, both associations were not "
            "retrieved correctly."
        )

        print()

        print(
            "Next investigate:"
        )

        print(
            "1. writer candidate representation"
        )

        print(
            "2. whether slot 2 actually encodes rabbit"
        )

        print(
            "3. whether slot 7 actually encodes river"
        )

        print(
            "4. reader value projection"
        )

        print(
            "5. memory-to-GPT fusion"
        )


if __name__ == "__main__":
    main()