import argparse
import random

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.gpt2_memory import (
    MemoryAugmentedGPT2LMHeadModel,
    MemoryGPT2Config,
)


# ============================================================
# ANSWER VOCABULARY
# ============================================================

ANSWERS = [
    "amber",
    "birch",
    "cedar",
    "dune",
    "ember",
    "frost",
    "glade",
    "harbor",
    "iris",
    "jade",
    "kelp",
    "lilac",
    "maple",
    "nova",
    "olive",
    "pearl",
]


# ============================================================
# MEMORY CONFIG
# Must match the checkpoint architecture.
# ============================================================

def build_memory_config():

    return MemoryGPT2Config(
        num_slots=8,

        gate_type="vector",
        gate_mode="sigmoid",
        gate_init_bias=-2.0,

        router_enabled=True,
        router_mode="softmax",
        router_top_k=2,
        router_temperature=0.7,

        writer_mode="attention",
        writer_attention_heads=8,

        orthogonal_mode="other_slots",
        orthogonal_strength=0.5,

        reader_mode="hybrid",
        reader_fusion="gated",
        reader_heads=8,
        reader_top_k=3,
        reader_temperature=0.8,

        candidate_diversity_weight=0.01,
        update_orthogonality_weight=0.01,
        router_balance_weight=0.01,
        reader_balance_weight=0.01,
        memory_collapse_weight=0.01,
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    checkpoint_path,
    model_name,
    device,
):

    print("Loading model...")

    model = (
        MemoryAugmentedGPT2LMHeadModel
        .from_pretrained(
            model_name,
            memory_config=build_memory_config(),
        )
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_text(
    tokenizer,
    text,
    device,
):

    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    return (
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
    )


def prepare_query_answer(
    tokenizer,
    query,
    answer,
    device,
):

    query_ids = tokenizer(
        query,
        add_special_tokens=False,
    )["input_ids"]

    # GPT-2 usually expects the leading space.
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

    attention_mask = torch.ones_like(
        input_ids
    )

    labels = input_ids.clone()

    # Only evaluate answer tokens.
    labels[:, :len(query_ids)] = -100

    return (
        input_ids,
        attention_mask,
        labels,
    )


# ============================================================
# CONTROLLED TEST DATA
# ============================================================

def build_examples(
    number_examples=100,
    seed=2090,
):

    rng = random.Random(seed)

    examples = []

    for i in range(number_examples):

        entity = f"person_{10000 + i}"

        answer = rng.choice(ANSWERS)

        fact = (
            f"The secret code associated with "
            f"{entity} is {answer}."
        )

        query = (
            f"What is the secret code associated "
            f"with {entity}?"
        )

        examples.append(
            {
                "id": i,
                "entity": entity,
                "fact": fact,
                "query": query,
                "answer": answer,
            }
        )

    return examples


# ============================================================
# WRITE FACT
# ============================================================

@torch.no_grad()
def write_fact(
    model,
    tokenizer,
    fact,
    device,
):

    input_ids, attention_mask = tokenize_text(
        tokenizer,
        fact,
        device,
    )

    dtype = next(model.parameters()).dtype

    initial_state = model.initialize_memory(
        batch_size=1,
        device=device,
        dtype=dtype,
    )

    # Clone initial slots for comparison.
    before_slots = (
        initial_state.slots
        .detach()
        .float()
        .clone()
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        memory_state=initial_state,
        update_memory=True,
        return_diagnostics=False,
    )

    written_state = output.memory_state.detach()

    after_slots = (
        written_state.slots
        .detach()
        .float()
    )

    # How much did each slot change?
    delta = after_slots - before_slots

    slot_change = delta.norm(
        dim=-1
    )[0]

    # For this one-fact diagnostic, define the oracle
    # slot as the slot that changed the most.
    oracle_slot = int(
        slot_change.argmax().item()
    )

    return (
        written_state,
        oracle_slot,
        slot_change,
    )


# ============================================================
# CUSTOM MASKED LM LOSS
# ============================================================

def masked_causal_loss(
    logits,
    labels,
):

    shift_logits = (
        logits[..., :-1, :]
        .contiguous()
    )

    shift_labels = (
        labels[..., 1:]
        .contiguous()
    )

    return F.cross_entropy(
        shift_logits.view(
            -1,
            shift_logits.size(-1),
        ),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ============================================================
# CONDITION 1:
# NO MEMORY
#
# Important:
# We directly call the GPT-2 backbone.
# No MemoryReader is executed.
# ============================================================

@torch.no_grad()
def score_no_memory(
    model,
    tokenizer,
    query,
    answer,
    device,
):

    (
        input_ids,
        attention_mask,
        labels,
    ) = prepare_query_answer(
        tokenizer,
        query,
        answer,
        device,
    )

    output = model.backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )

    logits = output.logits

    loss = masked_causal_loss(
        logits,
        labels,
    )

    return float(loss.item())


# ============================================================
# CONDITIONS 2/3:
# MEMORY MODEL
# ============================================================

@torch.no_grad()
def score_with_memory(
    model,
    tokenizer,
    query,
    answer,
    memory_state,
    memory_mask,
    device,
):

    (
        input_ids,
        attention_mask,
        labels,
    ) = prepare_query_answer(
        tokenizer,
        query,
        answer,
        device,
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        memory_state=memory_state,
        memory_mask=memory_mask,
        update_memory=False,
        return_diagnostics=False,
    )

    if output.lm_loss is None:
        raise RuntimeError(
            "Memory model returned lm_loss=None."
        )

    return float(
        output.lm_loss.item()
    )


# ============================================================
# BUILD FIXED CANDIDATES
#
# SAME candidates must be used for:
#   no memory
#   normal reader
#   oracle slot
# ============================================================

def build_candidates(
    correct_answer,
    example_id,
    seed,
):

    negatives = [
        answer
        for answer in ANSWERS
        if answer != correct_answer
    ]

    rng = random.Random(
        seed
        + 100000
        + example_id
    )

    sampled = rng.sample(
        negatives,
        3,
    )

    candidates = [
        correct_answer,
        *sampled,
    ]

    # Shuffle position so correct answer isn't always first.
    rng.shuffle(candidates)

    return candidates


# ============================================================
# SCORE ONE CONDITION
# ============================================================

def evaluate_candidates(
    score_function,
    candidates,
):

    scores = {}

    for candidate in candidates:

        scores[candidate] = (
            score_function(candidate)
        )

    prediction = min(
        scores,
        key=scores.get,
    )

    return prediction, scores


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "outputs/"
            "retrieval_gradient_test/"
            "checkpoint_best.pt"
        ),
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2090,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print("ORACLE LATENT VALUE TEST")
    print("=" * 78)

    print("Device:", device)
    print("Checkpoint:", args.checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    model = load_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        device=device,
    )

    examples = build_examples(
        number_examples=args.examples,
        seed=args.seed,
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    correct_counts = {
        "NO_MEMORY": 0,
        "NORMAL_READER": 0,
        "ORACLE_SLOT": 0,
    }

    answer_losses = {
        "NO_MEMORY": [],
        "NORMAL_READER": [],
        "ORACLE_SLOT": [],
    }

    oracle_slot_histogram = [
        0 for _ in range(model.num_slots)
    ]

    print()
    print(
        f"Running {len(examples)} "
        f"controlled examples..."
    )

    # ========================================================
    # EXAMPLES
    # ========================================================

    for example_index, example in enumerate(
        examples
    ):

        # ----------------------------------------------------
        # Write the fact once.
        # ----------------------------------------------------

        (
            written_state,
            oracle_slot,
            slot_change,
        ) = write_fact(
            model=model,
            tokenizer=tokenizer,
            fact=example["fact"],
            device=device,
        )

        oracle_slot_histogram[
            oracle_slot
        ] += 1

        # ----------------------------------------------------
        # Same answer candidates for all conditions.
        # ----------------------------------------------------

        candidates = build_candidates(
            correct_answer=example["answer"],
            example_id=example["id"],
            seed=args.seed,
        )

        # ====================================================
        # CONDITION 1: NO MEMORY
        # ====================================================

        prediction, scores = (
            evaluate_candidates(
                score_function=lambda candidate: (
                    score_no_memory(
                        model=model,
                        tokenizer=tokenizer,
                        query=example["query"],
                        answer=candidate,
                        device=device,
                    )
                ),
                candidates=candidates,
            )
        )

        if prediction == example["answer"]:
            correct_counts[
                "NO_MEMORY"
            ] += 1

        answer_losses[
            "NO_MEMORY"
        ].append(
            scores[example["answer"]]
        )

        # ====================================================
        # CONDITION 2: NORMAL MEMORY READER
        #
        # All memory slots available.
        # ====================================================

        normal_mask = torch.ones(
            1,
            model.num_slots,
            dtype=torch.bool,
            device=device,
        )

        prediction, scores = (
            evaluate_candidates(
                score_function=lambda candidate: (
                    score_with_memory(
                        model=model,
                        tokenizer=tokenizer,
                        query=example["query"],
                        answer=candidate,
                        memory_state=written_state,
                        memory_mask=normal_mask,
                        device=device,
                    )
                ),
                candidates=candidates,
            )
        )

        if prediction == example["answer"]:
            correct_counts[
                "NORMAL_READER"
            ] += 1

        answer_losses[
            "NORMAL_READER"
        ].append(
            scores[example["answer"]]
        )

        # ====================================================
        # CONDITION 3: ORACLE SLOT
        #
        # Only the slot that received the strongest write
        # is exposed to the reader.
        #
        # This removes the slot-selection problem.
        # ====================================================

        oracle_mask = torch.zeros(
            1,
            model.num_slots,
            dtype=torch.bool,
            device=device,
        )

        oracle_mask[
            0,
            oracle_slot,
        ] = True

        prediction, scores = (
            evaluate_candidates(
                score_function=lambda candidate: (
                    score_with_memory(
                        model=model,
                        tokenizer=tokenizer,
                        query=example["query"],
                        answer=candidate,
                        memory_state=written_state,
                        memory_mask=oracle_mask,
                        device=device,
                    )
                ),
                candidates=candidates,
            )
        )

        if prediction == example["answer"]:
            correct_counts[
                "ORACLE_SLOT"
            ] += 1

        answer_losses[
            "ORACLE_SLOT"
        ].append(
            scores[example["answer"]]
        )

        # ----------------------------------------------------
        # Print first 5 examples for inspection.
        # ----------------------------------------------------

        if example_index < 5:

            print()
            print("-" * 78)

            print(
                f"EXAMPLE {example_index + 1}"
            )

            print(
                "Fact:      ",
                example["fact"],
            )

            print(
                "Question:  ",
                example["query"],
            )

            print(
                "Answer:    ",
                example["answer"],
            )

            print(
                "Candidates:",
                candidates,
            )

            print(
                "Oracle slot:",
                oracle_slot,
            )

            print(
                "Slot Δ:",
                [
                    round(
                        float(value),
                        4,
                    )
                    for value in (
                        slot_change
                        .detach()
                        .cpu()
                        .tolist()
                    )
                ],
            )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    total = len(examples)

    print()
    print("=" * 78)
    print("FINAL RESULTS")
    print("=" * 78)

    print()
    print(
        f"{'Condition':<20}"
        f"{'Accuracy':>12}"
        f"{'Correct':>12}"
        f"{'Answer Loss':>16}"
    )

    print("-" * 60)

    result_percentages = {}

    for condition in [
        "NO_MEMORY",
        "NORMAL_READER",
        "ORACLE_SLOT",
    ]:

        accuracy = (
            100.0
            * correct_counts[condition]
            / total
        )

        result_percentages[
            condition
        ] = accuracy

        mean_loss = (
            sum(
                answer_losses[condition]
            )
            /
            len(
                answer_losses[condition]
            )
        )

        print(
            f"{condition:<20}"
            f"{accuracy:>11.2f}%"
            f"{correct_counts[condition]:>8}"
            f"/{total:<3}"
            f"{mean_loss:>16.4f}"
        )

    # ========================================================
    # SLOT DISTRIBUTION
    # ========================================================

    print()
    print("=" * 78)
    print("MOST-WRITTEN SLOT DISTRIBUTION")
    print("=" * 78)

    for slot_index, count in enumerate(
        oracle_slot_histogram
    ):

        print(
            f"Slot {slot_index}: "
            f"{count:4d} / {total}"
        )

    # ========================================================
    # DIFFERENCES
    # ========================================================

    no_memory = result_percentages[
        "NO_MEMORY"
    ]

    normal_reader = result_percentages[
        "NORMAL_READER"
    ]

    oracle_slot = result_percentages[
        "ORACLE_SLOT"
    ]

    print()
    print("=" * 78)
    print("DIFFERENCES")
    print("=" * 78)

    print(
        "Normal reader - No memory:"
        f" {normal_reader - no_memory:+.2f} pp"
    )

    print(
        "Oracle slot   - No memory:"
        f" {oracle_slot - no_memory:+.2f} pp"
    )

    print(
        "Oracle slot   - Normal reader:"
        f" {oracle_slot - normal_reader:+.2f} pp"
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print()
    print("=" * 78)
    print("AUTOMATIC INTERPRETATION")
    print("=" * 78)

    oracle_gain = (
        oracle_slot
        - no_memory
    )

    addressing_gain = (
        oracle_slot
        - normal_reader
    )

    if oracle_gain >= 10.0:

        print(
            "VALUE TEST: PASS"
        )

        print(
            "The latent memory contains useful "
            "information when the likely written "
            "slot is forced."
        )

    elif oracle_gain >= 3.0:

        print(
            "VALUE TEST: PARTIAL"
        )

        print(
            "The latent memory provides some benefit, "
            "but the effect is modest."
        )

    else:

        print(
            "VALUE TEST: FAIL / INCONCLUSIVE"
        )

        print(
            "Forcing the likely written slot does not "
            "substantially outperform GPT-2 without "
            "memory."
        )

    print()

    if addressing_gain >= 5.0:

        print(
            "ADDRESSING RESULT:"
        )

        print(
            "The oracle slot substantially beats the "
            "normal reader, suggesting slot selection "
            "is an important bottleneck."
        )

    elif addressing_gain > 0:

        print(
            "ADDRESSING RESULT:"
        )

        print(
            "The oracle slot helps somewhat over the "
            "normal reader."
        )

    else:

        print(
            "ADDRESSING RESULT:"
        )

        print(
            "Forcing one slot does not improve over "
            "the normal reader."
        )

    print()
    print(
        "IMPORTANT: this is a controlled one-fact "
        "VALUE sanity test, not the final LoCoMo "
        "experiment."
    )


if __name__ == "__main__":
    main()