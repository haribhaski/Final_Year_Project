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

COLLISION_WEIGHT = 1.0
ENTROPY_WEIGHT = 0.5
BINDING_WEIGHT = 1.0

SAVE_PATH = "outputs/two_fact_soft_routing_trained.pt"

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
        f"The assigned keyword for "
        f"{example['entity']} is"
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

    # GPT-2 needs leading space for answer token.
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

    # Only answer tokens contribute to loss.
    labels[:, :query_ids.size(1)] = -100

    return (
        input_ids,
        attention_mask,
        labels,
    )


# ============================================================
# WRITE
# ============================================================

def write_fact(
    model,
    tokenizer,
    example,
    memory_state,
    device,
):
    ids, mask = tokenize_context(
        tokenizer,
        fact_text(example),
        device,
    )

    return model(
        input_ids=ids,
        attention_mask=mask,
        memory_state=memory_state,
        update_memory=True,
        return_diagnostics=True,
    )


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
# ROUTING
# ============================================================

def dense_route(model, output):
    """
    Dense differentiable routing distribution BEFORE
    top-k sparsification.
    """

    logits = output.routing_output.logits

    return torch.softmax(
        logits / model.router.temperature,
        dim=-1,
    )


def route_entropy(p):
    """
    Entropy of routing distribution.

    MINIMIZING this encourages a sharp/specialized route.
    """

    eps = 1e-8

    return -(
        p * torch.log(
            p.clamp_min(eps)
        )
    ).sum(dim=-1).mean()


def pretty_tensor(x):
    return [
        round(float(v), 4)
        for v in x.detach()[0].cpu()
    ]


def router_info(output):
    routing = output.routing_output

    selected = None

    if routing.selected_indices is not None:
        selected = (
            routing.selected_indices[0]
            .detach()
            .cpu()
            .tolist()
        )

    weights = (
        routing.weights[0]
        .detach()
        .cpu()
        .tolist()
    )

    return {
        "selected": selected,
        "weights": [
            round(float(v), 4)
            for v in weights
        ],
    }


# ============================================================
# READER
# ============================================================

def get_last_reader_distribution(output):
    """
    attention_weights:
        [B, H, T, N]

    Return:
        [B, N]
    """

    attention = (
        output.read_output.attention_weights
    )

    if attention is None:
        raise RuntimeError(
            "Reader attention unavailable."
        )

    if attention.ndim != 4:
        raise RuntimeError(
            f"Expected [B,H,T,N], got "
            f"{tuple(attention.shape)}"
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

    prediction = scores[0][0]

    correct_loss = next(
        loss
        for answer, loss in scores
        if answer == correct
    )

    return {
        "rank": rank,
        "prediction": prediction,
        "correct_loss": correct_loss,
        "scores": scores,
    }


# ============================================================
# MEMORY DIFFERENCE
# ============================================================

@torch.no_grad()
def memory_difference(before, after):

    a = before.slots.detach()
    b = after.slots.detach()

    diff = b - a

    total_l2 = torch.norm(
        diff
    ).item()

    relative = (
        torch.norm(diff)
        / torch.norm(a).clamp_min(1e-8)
    ).item()

    slot_l2 = torch.norm(
        diff,
        dim=-1,
    )[0]

    return {
        "total_l2": total_l2,
        "relative": relative,
        "slot_l2":
            slot_l2.cpu().tolist(),
    }


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

    # Write A
    out_a = write_fact(
        model,
        tokenizer,
        A,
        None,
        device,
    )

    memory_a = out_a.memory_state

    # Write B into same memory
    out_b = write_fact(
        model,
        tokenizer,
        B,
        memory_a,
        device,
    )

    memory_ab = out_b.memory_state

    # Rank both
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

    # Query outputs for reader diagnostics
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

    dense_a = dense_route(
        model,
        out_a,
    )

    dense_b = dense_route(
        model,
        out_b,
    )

    change = memory_difference(
        memory_a,
        memory_ab,
    )

    model.train()

    return {
        "A": result_a,
        "B": result_b,

        "dense_A":
            pretty_tensor(dense_a),

        "dense_B":
            pretty_tensor(dense_b),

        "sparse_A":
            router_info(out_a),

        "sparse_B":
            router_info(out_b),

        "read_A":
            pretty_tensor(read_a),

        "read_B":
            pretty_tensor(read_b),

        "entropy_A":
            float(
                route_entropy(
                    dense_a
                ).item()
            ),

        "entropy_B":
            float(
                route_entropy(
                    dense_b
                ).item()
            ),

        "memory_change":
            change,
    }


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
        "TWO-FACT SOFT ROUTING + "
        "COLLISION + ENTROPY + BINDING"
    )
    print("=" * 90)

    print("Device:", device)

    print(
        "Base checkpoint:",
        BASE_CHECKPOINT,
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
        "Original reader mode:",
        model.reader.mode,
    )

    # Keep token reader.
    model.reader.mode = "token"

    print(
        "Reader mode:",
        model.reader.mode,
    )

    print(
        "Original router top-k:",
        model.router.top_k,
    )

    # ========================================================
    # IMPORTANT:
    # DISABLE TOP-K
    #
    # This means actual writes now use all slots according
    # to the dense routing distribution.
    #
    # No discrete top-k bottleneck during this experiment.
    # ========================================================

    original_top_k = model.router.top_k

    model.router.top_k = None

    print(
        "Router top-k during experiment:",
        model.router.top_k,
    )

    print(
        "Router temperature:",
        model.router.temperature,
    )

    configure_memory_only_training(
        model
    )

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LR,
    )

    print()
    print(
        f"A: {A['entity']} -> {A['answer']}"
    )

    print(
        f"B: {B['entity']} -> {B['answer']}"
    )

    print()
    print(
        "Collision weight:",
        COLLISION_WEIGHT,
    )

    print(
        "Entropy weight:",
        ENTROPY_WEIGHT,
    )

    print(
        "Binding weight:",
        BINDING_WEIGHT,
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
        f"A: pred={before['A']['prediction']} "
        f"| rank={before['A']['rank']}/16 "
        f"| loss={before['A']['correct_loss']:.4f}"
    )

    print(
        f"B: pred={before['B']['prediction']} "
        f"| rank={before['B']['rank']}/16 "
        f"| loss={before['B']['correct_loss']:.4f}"
    )

    print(
        "\nA dense/write:",
        before["dense_A"],
    )

    print(
        "B dense/write:",
        before["dense_B"],
    )

    print(
        "\nA read:",
        before["read_A"],
    )

    print(
        "B read:",
        before["read_B"],
    )

    print(
        "\nA entropy:",
        round(
            before["entropy_A"],
            6,
        ),
    )

    print(
        "B entropy:",
        round(
            before["entropy_B"],
            6,
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
        # 1. WRITE A
        # ====================================================

        out_a = write_fact(
            model,
            tokenizer,
            A,
            None,
            device,
        )

        memory_a = (
            out_a.memory_state
        )

        # ====================================================
        # 2. WRITE B INTO A MEMORY
        #
        # DO NOT DETACH.
        # ====================================================

        out_b = write_fact(
            model,
            tokenizer,
            B,
            memory_a,
            device,
        )

        memory_ab = (
            out_b.memory_state
        )

        # ====================================================
        # 3. QUERY BOTH
        # ====================================================

        loss_a, query_a = answer_loss(
            model,
            tokenizer,
            A,
            A["answer"],
            memory_ab,
            device,
        )

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
        # 4. DENSE WRITE ROUTES
        #
        # Because top-k=None, these are ALSO the routing
        # distributions used for actual writing.
        # ====================================================

        p_a = dense_route(
            model,
            out_a,
        )

        p_b = dense_route(
            model,
            out_b,
        )

        # ====================================================
        # 5. COLLISION
        #
        # Different facts should use different addresses.
        # ====================================================

        collision_loss = (
            p_a * p_b
        ).sum(
            dim=-1
        ).mean()

        # ====================================================
        # 6. ENTROPY / SHARPNESS
        #
        # Prevent:
        #
        # A = uniform
        # B = uniform
        #
        # which previously gave collision ~= 1/8.
        #
        # We MINIMIZE entropy.
        # ====================================================

        entropy_a = route_entropy(
            p_a
        )

        entropy_b = route_entropy(
            p_b
        )

        entropy_loss = (
            entropy_a + entropy_b
        ) / 2.0

        # ====================================================
        # 7. READER
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
        # 8. WRITE -> READ BINDING
        #
        # Reader should retrieve from the address used
        # during that fact's write.
        # ====================================================

        target_a = p_a.detach()
        target_b = p_b.detach()

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
        # 9. TOTAL
        # ====================================================

        loss = (
            retrieval_loss

            + COLLISION_WEIGHT
            * collision_loss

            + ENTROPY_WEIGHT
            * entropy_loss

            + BINDING_WEIGHT
            * binding_loss
        )

        # ====================================================
        # BACKPROP
        # ====================================================

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable,
            max_norm=1.0,
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
                f"STEP {step:03d} "
                f"| total={loss.item():.6f} "
                f"| retrieval={retrieval_loss.item():.6f} "
                f"| collision={collision_loss.item():.6f} "
                f"| entropy={entropy_loss.item():.6f} "
                f"| binding={binding_loss.item():.6f}"
            )

            print(
                f"A: pred={result['A']['prediction']:>7} "
                f"| rank={result['A']['rank']:2d}/16 "
                f"| loss={result['A']['correct_loss']:.4f}"
            )

            print(
                f"B: pred={result['B']['prediction']:>7} "
                f"| rank={result['B']['rank']:2d}/16 "
                f"| loss={result['B']['correct_loss']:.4f}"
            )

            print(
                "\nA write:",
                result["dense_A"],
            )

            print(
                "B write:",
                result["dense_B"],
            )

            print(
                "A read :",
                result["read_A"],
            )

            print(
                "B read :",
                result["read_B"],
            )

            print(
                "A entropy:",
                round(
                    result["entropy_A"],
                    6,
                ),
            )

            print(
                "B entropy:",
                round(
                    result["entropy_B"],
                    6,
                ),
            )

            change = (
                result["memory_change"]
            )

            print(
                "B-write memory change:"
                f" L2={change['total_l2']:.4f}"
                f" | relative={change['relative']:.4f}"
            )

            print(
                "Per-slot B-write L2:",
                [
                    round(x, 4)
                    for x in change[
                        "slot_l2"
                    ]
                ],
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
        f"| rank={final['A']['rank']}/16"
    )

    print(
        f"B ({B['answer']}): "
        f"prediction={final['B']['prediction']} "
        f"| rank={final['B']['rank']}/16"
    )

    print("\nWRITE ROUTES")

    print(
        "A:",
        final["dense_A"],
    )

    print(
        "B:",
        final["dense_B"],
    )

    print("\nREADER ROUTES")

    print(
        "A:",
        final["read_A"],
    )

    print(
        "B:",
        final["read_B"],
    )

    print("\nROUTING ENTROPY")

    print(
        "A:",
        round(
            final["entropy_A"],
            6,
        ),
    )

    print(
        "B:",
        round(
            final["entropy_B"],
            6,
        ),
    )

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
    # FINAL ROUTE METRICS
    # ========================================================

    # Rebuild final distributions as tensors
    # for simple diagnostic metrics.

    pa = torch.tensor(
        final["dense_A"]
    )

    pb = torch.tensor(
        final["dense_B"]
    )

    route_cosine = (
        F.cosine_similarity(
            pa.unsqueeze(0),
            pb.unsqueeze(0),
        ).item()
    )

    route_overlap = (
        pa * pb
    ).sum().item()

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

    print("\n" + "=" * 90)
    print("FINAL ADDRESSING DIAGNOSTICS")
    print("=" * 90)

    print(
        f"Write route cosine: "
        f"{route_cosine:.6f}"
    )

    print(
        f"Write route overlap: "
        f"{route_overlap:.6f}"
    )

    print(
        f"Reader route cosine: "
        f"{reader_cosine:.6f}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "router_top_k":
                model.router.top_k,

            "reader_mode":
                model.reader.mode,
        },
        SAVE_PATH,
    )

    print(
        f"\nSaved model to: "
        f"{SAVE_PATH}"
    )

    # ========================================================
    # INTERPRETATION
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
            "SUCCESS: Both associations are rank-1."
        )

        print(
            "Soft differentiable routing + sharpness + "
            "write-read binding can learn the two-fact "
            "sequential task."
        )

    elif route_cosine < 0.5:

        print(
            "WRITE ROUTES SEPARATED, but retrieval "
            "is not fully solved."
        )

        print(
            "The next bottleneck is reader addressing, "
            "memory content, or fusion."
        )

    elif final["entropy_A"] < 1.0 and final["entropy_B"] < 1.0:

        print(
            "Routes became sharp but still collapsed "
            "onto similar addresses."
        )

        print(
            "Next use an explicit fact-specific routing "
            "assignment / sequential occupancy mechanism."
        )

    else:

        print(
            "Routes did not become sufficiently sharp "
            "and distinct."
        )

        print(
            "The router itself is still failing to form "
            "fact-specific addresses."
        )


if __name__ == "__main__":
    main()