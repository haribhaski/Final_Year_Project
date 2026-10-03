import argparse
import random
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.gpt2_memory import (
    MemoryAugmentedGPT2LMHeadModel,
    MemoryGPT2Config,
)


# ============================================================
# ANSWER WORDS
# ============================================================

ANSWER_POOL = [
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
    "rose",
    "silver",
    "stone",
    "violet",
    "willow",
    "cloud",
    "river",
    "ocean",
]


# ============================================================
# MODEL CONFIG
#
# Must match retrieval_gradient_test checkpoint.
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
# LOAD
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
# ONLY USE SINGLE GPT-2 TOKEN ANSWERS
#
# This avoids the mean-token-loss confound.
# ============================================================

def get_single_token_answers(
    tokenizer,
):

    usable = {}

    rejected = {}

    for word in ANSWER_POOL:

        ids = tokenizer(
            " " + word,
            add_special_tokens=False,
        )["input_ids"]

        if len(ids) == 1:

            usable[word] = ids[0]

        else:

            rejected[word] = ids

    return usable, rejected


# ============================================================
# SYNTHETIC EXAMPLES
# ============================================================

def build_examples(
    answers,
    number_examples,
    seed,
):

    rng = random.Random(seed)

    answer_words = list(answers.keys())

    if len(answer_words) < 4:

        raise RuntimeError(
            "Need at least four single-token answers."
        )

    examples = []

    for i in range(number_examples):

        entity = f"person_{10000 + i}"

        answer = rng.choice(
            answer_words
        )

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
                "answer": answer,
                "fact": fact,
                "query": query,
            }
        )

    return examples


# ============================================================
# TOKENIZE PROMPT
# ============================================================

def tokenize_prompt(
    tokenizer,
    prompt,
    device,
):

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    return (
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
    )


# ============================================================
# INITIAL / EMPTY MEMORY STATE
# ============================================================

@torch.no_grad()
def create_initial_state(
    model,
    device,
):

    dtype = next(
        model.parameters()
    ).dtype

    state = model.initialize_memory(
        batch_size=1,
        device=device,
        dtype=dtype,
    )

    return state.detach()


# ============================================================
# WRITE A FACT INTO MEMORY
# ============================================================

@torch.no_grad()
def write_fact(
    model,
    tokenizer,
    fact,
    device,
):

    initial_state = create_initial_state(
        model,
        device,
    )

    input_ids, attention_mask = tokenize_prompt(
        tokenizer,
        fact,
        device,
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        memory_state=initial_state,
        update_memory=True,
        return_diagnostics=False,
    )

    written_state = (
        output.memory_state.detach()
    )

    before = (
        initial_state.slots
        .detach()
        .float()
    )

    after = (
        written_state.slots
        .detach()
        .float()
    )

    delta = after - before

    slot_change = (
        delta.norm(dim=-1)[0]
    )

    max_changed_slot = int(
        slot_change.argmax().item()
    )

    return {
        "initial_state": initial_state,
        "written_state": written_state,

        "before_slots": before,
        "after_slots": after,

        "delta": delta,

        "slot_change": slot_change,

        "max_changed_slot": (
            max_changed_slot
        ),

        "write_gate": (
            output.write_gate.detach().float()
            if output.write_gate is not None
            else None
        ),

        "routing_output": (
            output.routing_output
        ),
    }


# ============================================================
# SCORE ONE-TOKEN ANSWERS
#
# Lower NLL = model prefers candidate more.
#
# Because every candidate is exactly ONE GPT-2 token,
# this is directly comparable.
# ============================================================

@torch.no_grad()
def score_memory_prompt(
    model,
    tokenizer,
    query,
    memory_state,
    answer_token_ids,
    device,
):

    input_ids, attention_mask = tokenize_prompt(
        tokenizer,
        query,
        device,
    )

    memory_mask = torch.ones(
        1,
        model.num_slots,
        dtype=torch.bool,
        device=device,
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        memory_state=memory_state,
        memory_mask=memory_mask,
        update_memory=False,
        return_diagnostics=False,
    )

    # Prediction immediately after final query token.
    next_logits = (
        output.logits[0, -1]
        .float()
    )

    log_probs = F.log_softmax(
        next_logits,
        dim=-1,
    )

    scores = {}

    for answer, token_id in (
        answer_token_ids.items()
    ):

        scores[answer] = float(
            -log_probs[token_id].item()
        )

    return scores


# ============================================================
# RAW TEXT CONTROL
#
# Fact is explicitly inside GPT-2 context.
# No latent memory needed.
# ============================================================

@torch.no_grad()
def score_raw_text_prompt(
    model,
    tokenizer,
    fact,
    query,
    answer_token_ids,
    device,
):

    prompt = (
        fact
        + "\n"
        + query
    )

    input_ids, attention_mask = tokenize_prompt(
        tokenizer,
        prompt,
        device,
    )

    output = model.backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )

    next_logits = (
        output.logits[0, -1]
        .float()
    )

    log_probs = F.log_softmax(
        next_logits,
        dim=-1,
    )

    scores = {}

    for answer, token_id in (
        answer_token_ids.items()
    ):

        scores[answer] = float(
            -log_probs[token_id].item()
        )

    return scores


# ============================================================
# CHOOSE SAME FOUR CANDIDATES FOR ALL CONDITIONS
# ============================================================

def choose_candidates(
    correct_answer,
    answer_words,
    example_id,
    seed,
):

    negatives = [
        answer
        for answer in answer_words
        if answer != correct_answer
    ]

    rng = random.Random(
        seed
        + 100003
        + example_id
    )

    candidates = [
        correct_answer,
        *rng.sample(
            negatives,
            3,
        ),
    ]

    rng.shuffle(candidates)

    return candidates


# ============================================================
# RESTRICT SCORES TO FOUR CANDIDATES
# ============================================================

def candidate_prediction(
    all_scores,
    candidates,
):

    scores = {
        candidate: all_scores[candidate]
        for candidate in candidates
    }

    prediction = min(
        scores,
        key=scores.get,
    )

    return prediction, scores


# ============================================================
# PAIRWISE CONTENT DEPENDENCE
# ============================================================

def pairwise_statistics(
    vectors,
):

    if len(vectors) < 2:

        return {
            "mean_norm": float("nan"),
            "mean_pairwise_l2": float("nan"),
            "relative_l2": float("nan"),
            "mean_cosine": float("nan"),
        }

    x = torch.stack(
        vectors,
        dim=0,
    ).float()

    norms = x.norm(
        dim=-1
    )

    mean_norm = float(
        norms.mean().item()
    )

    # Full pairwise Euclidean distance.
    dist = torch.cdist(
        x,
        x,
        p=2,
    )

    n = x.size(0)

    upper = torch.triu(
        torch.ones(
            n,
            n,
            dtype=torch.bool,
            device=x.device,
        ),
        diagonal=1,
    )

    pairwise_l2 = dist[upper]

    mean_pairwise_l2 = float(
        pairwise_l2.mean().item()
    )

    relative_l2 = (
        mean_pairwise_l2
        / (mean_norm + 1e-8)
    )

    x_norm = F.normalize(
        x,
        p=2,
        dim=-1,
    )

    cosine = (
        x_norm
        @ x_norm.T
    )

    pairwise_cosine = cosine[upper]

    mean_cosine = float(
        pairwise_cosine.mean().item()
    )

    return {
        "mean_norm": mean_norm,
        "mean_pairwise_l2": (
            mean_pairwise_l2
        ),
        "relative_l2": relative_l2,
        "mean_cosine": mean_cosine,
    }


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
        default=200,
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

    print("=" * 90)
    print("LATENT VALUE CONTROL EXPERIMENT")
    print("=" * 90)

    print("Device:", device)
    print(
        "Checkpoint:",
        args.checkpoint,
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.model_name
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    usable_answers, rejected_answers = (
        get_single_token_answers(
            tokenizer
        )
    )

    print()
    print("=" * 90)
    print("ANSWER TOKEN CHECK")
    print("=" * 90)

    print(
        "Single-token answers:",
        list(usable_answers.keys()),
    )

    print(
        "Rejected multi-token answers:",
        list(rejected_answers.keys()),
    )

    print(
        "Number usable:",
        len(usable_answers),
    )

    if len(usable_answers) < 4:

        raise RuntimeError(
            "Not enough single-token answers "
            "for four-way evaluation."
        )

    # ========================================================
    # MODEL
    # ========================================================

    model = load_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        device=device,
    )

    # ========================================================
    # DATASET
    # ========================================================

    examples = build_examples(
        answers=usable_answers,
        number_examples=args.examples,
        seed=args.seed,
    )

    answer_words = list(
        usable_answers.keys()
    )

    # ========================================================
    # FIRST WRITE ALL FACTS
    #
    # We do this first so mismatched-memory states are available.
    # ========================================================

    print()
    print("=" * 90)
    print("WRITING FACTS TO MEMORY")
    print("=" * 90)

    writes = []

    slot_histogram = Counter()

    for i, example in enumerate(
        examples
    ):

        write_info = write_fact(
            model=model,
            tokenizer=tokenizer,
            fact=example["fact"],
            device=device,
        )

        writes.append(
            write_info
        )

        slot_histogram[
            write_info[
                "max_changed_slot"
            ]
        ] += 1

        if i < 5:

            print()
            print(
                f"Example {i + 1}"
            )

            print(
                "Fact:",
                example["fact"],
            )

            print(
                "Max changed slot:",
                write_info[
                    "max_changed_slot"
                ],
            )

            print(
                "Slot delta norms:",
                [
                    round(
                        float(x),
                        4,
                    )
                    for x in (
                        write_info[
                            "slot_change"
                        ]
                        .cpu()
                        .tolist()
                    )
                ],
            )

    # ========================================================
    # DOMINANT SLOT
    # ========================================================

    dominant_slot = (
        slot_histogram
        .most_common(1)[0][0]
    )

    print()
    print("=" * 90)
    print("WRITE DISTRIBUTION")
    print("=" * 90)

    for slot in range(
        model.num_slots
    ):

        print(
            f"Slot {slot}: "
            f"{slot_histogram[slot]}"
            f"/{len(examples)}"
        )

    print()
    print(
        "Dominant slot:",
        dominant_slot,
    )

    # ========================================================
    # EVALUATION CONDITIONS
    # ========================================================

    condition_names = [
        "EMPTY_MEMORY",
        "MATCHED_MEMORY",
        "MISMATCHED_MEMORY",
        "RAW_TEXT",
    ]

    correct = {
        name: 0
        for name in condition_names
    }

    correct_nll = {
        name: []
        for name in condition_names
    }

    # Difference between matched and mismatched correct-answer NLL.
    matched_minus_mismatch = []

    print()
    print("=" * 90)
    print("EVALUATING")
    print("=" * 90)

    for i, example in enumerate(
        examples
    ):

        # ----------------------------------------------------
        # Same candidates across all conditions.
        # ----------------------------------------------------

        candidates = choose_candidates(
            correct_answer=example["answer"],
            answer_words=answer_words,
            example_id=example["id"],
            seed=args.seed,
        )

        candidate_token_ids = {
            answer: (
                usable_answers[answer]
            )
            for answer in candidates
        }

        # ----------------------------------------------------
        # EMPTY INITIAL MEMORY
        #
        # Same memory model + reader.
        # Nothing from this fact has been written.
        # ----------------------------------------------------

        empty_state = create_initial_state(
            model,
            device,
        )

        scores = score_memory_prompt(
            model=model,
            tokenizer=tokenizer,
            query=example["query"],
            memory_state=empty_state,
            answer_token_ids=(
                candidate_token_ids
            ),
            device=device,
        )

        prediction, scores = (
            candidate_prediction(
                scores,
                candidates,
            )
        )

        if prediction == example["answer"]:
            correct["EMPTY_MEMORY"] += 1

        correct_nll[
            "EMPTY_MEMORY"
        ].append(
            scores[example["answer"]]
        )

        # ----------------------------------------------------
        # MATCHED MEMORY
        #
        # Fact A written.
        # Ask question about fact A.
        # ----------------------------------------------------

        matched_state = (
            writes[i][
                "written_state"
            ]
        )

        scores = score_memory_prompt(
            model=model,
            tokenizer=tokenizer,
            query=example["query"],
            memory_state=matched_state,
            answer_token_ids=(
                candidate_token_ids
            ),
            device=device,
        )

        prediction, matched_scores = (
            candidate_prediction(
                scores,
                candidates,
            )
        )

        if prediction == example["answer"]:
            correct["MATCHED_MEMORY"] += 1

        correct_nll[
            "MATCHED_MEMORY"
        ].append(
            matched_scores[
                example["answer"]
            ]
        )

        # ----------------------------------------------------
        # MISMATCHED MEMORY
        #
        # Use another example's written memory.
        # Ask current example's question.
        #
        # Memory content is intentionally wrong.
        # ----------------------------------------------------

        mismatch_index = (
            i + 1
        ) % len(examples)

        mismatched_state = (
            writes[
                mismatch_index
            ][
                "written_state"
            ]
        )

        scores = score_memory_prompt(
            model=model,
            tokenizer=tokenizer,
            query=example["query"],
            memory_state=mismatched_state,
            answer_token_ids=(
                candidate_token_ids
            ),
            device=device,
        )

        (
            prediction,
            mismatched_scores,
        ) = candidate_prediction(
            scores,
            candidates,
        )

        if prediction == example["answer"]:
            correct[
                "MISMATCHED_MEMORY"
            ] += 1

        correct_nll[
            "MISMATCHED_MEMORY"
        ].append(
            mismatched_scores[
                example["answer"]
            ]
        )

        matched_minus_mismatch.append(
            mismatched_scores[
                example["answer"]
            ]
            -
            matched_scores[
                example["answer"]
            ]
        )

        # ----------------------------------------------------
        # RAW-TEXT CEILING
        #
        # Directly place fact before question.
        # ----------------------------------------------------

        scores = score_raw_text_prompt(
            model=model,
            tokenizer=tokenizer,
            fact=example["fact"],
            query=example["query"],
            answer_token_ids=(
                candidate_token_ids
            ),
            device=device,
        )

        prediction, raw_scores = (
            candidate_prediction(
                scores,
                candidates,
            )
        )

        if prediction == example["answer"]:
            correct["RAW_TEXT"] += 1

        correct_nll[
            "RAW_TEXT"
        ].append(
            raw_scores[
                example["answer"]
            ]
        )

        # ----------------------------------------------------
        # FIRST FIVE EXAMPLES
        # ----------------------------------------------------

        if i < 5:

            print()
            print("-" * 90)

            print(
                f"Example {i + 1}"
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
                "Matched correct-answer NLL:",
                round(
                    matched_scores[
                        example["answer"]
                    ],
                    4,
                ),
            )

            print(
                "Mismatch correct-answer NLL:",
                round(
                    mismatched_scores[
                        example["answer"]
                    ],
                    4,
                ),
            )

            print(
                "Raw-text correct-answer NLL:",
                round(
                    raw_scores[
                        example["answer"]
                    ],
                    4,
                ),
            )

    # ========================================================
    # ACCURACY TABLE
    # ========================================================

    print()
    print("=" * 90)
    print("CONTROL RESULTS")
    print("=" * 90)

    total = len(examples)

    accuracies = {}

    print(
        f"{'Condition':<25}"
        f"{'Accuracy':>12}"
        f"{'Correct':>14}"
        f"{'Mean NLL':>14}"
    )

    print("-" * 65)

    for name in condition_names:

        accuracy = (
            100.0
            * correct[name]
            / total
        )

        accuracies[name] = (
            accuracy
        )

        mean_nll = (
            sum(
                correct_nll[name]
            )
            /
            len(
                correct_nll[name]
            )
        )

        print(
            f"{name:<25}"
            f"{accuracy:>11.2f}%"
            f"{correct[name]:>9}"
            f"/{total:<4}"
            f"{mean_nll:>14.4f}"
        )

    # ========================================================
    # MATCHED VS MISMATCHED
    # ========================================================

    mean_memory_specific_gain = (
        sum(
            matched_minus_mismatch
        )
        /
        len(
            matched_minus_mismatch
        )
    )

    print()
    print("=" * 90)
    print("MATCHED VS MISMATCHED MEMORY")
    print("=" * 90)

    print(
        "Matched accuracy:    "
        f"{accuracies['MATCHED_MEMORY']:.2f}%"
    )

    print(
        "Mismatched accuracy: "
        f"{accuracies['MISMATCHED_MEMORY']:.2f}%"
    )

    print(
        "Accuracy difference: "
        f"{accuracies['MATCHED_MEMORY'] - accuracies['MISMATCHED_MEMORY']:+.2f} pp"
    )

    print()

    print(
        "Mean "
        "(mismatched correct-answer NLL "
        "- matched correct-answer NLL): "
        f"{mean_memory_specific_gain:+.4f}"
    )

    print(
        "Positive means the matching memory "
        "assigns higher probability to the "
        "correct answer."
    )

    # ========================================================
    # CONTENT DEPENDENCE
    #
    # Compare dominant slot vectors across facts.
    # ========================================================

    dominant_slot_vectors = []

    dominant_slot_deltas = []

    for write_info in writes:

        dominant_slot_vectors.append(
            write_info[
                "after_slots"
            ][
                0,
                dominant_slot,
            ]
            .detach()
            .cpu()
        )

        dominant_slot_deltas.append(
            write_info[
                "delta"
            ][
                0,
                dominant_slot,
            ]
            .detach()
            .cpu()
        )

    slot_stats = pairwise_statistics(
        dominant_slot_vectors
    )

    delta_stats = pairwise_statistics(
        dominant_slot_deltas
    )

    print()
    print("=" * 90)
    print("CONTENT-DEPENDENCE CHECK")
    print("=" * 90)

    print(
        "Dominant slot:",
        dominant_slot,
    )

    print()

    print(
        "FINAL SLOT VECTORS"
    )

    print(
        "Mean vector norm:       "
        f"{slot_stats['mean_norm']:.6f}"
    )

    print(
        "Mean pairwise L2:       "
        f"{slot_stats['mean_pairwise_l2']:.6f}"
    )

    print(
        "Pairwise L2 / norm:     "
        f"{slot_stats['relative_l2']:.6f}"
    )

    print(
        "Mean pairwise cosine:   "
        f"{slot_stats['mean_cosine']:.6f}"
    )

    print()

    print(
        "WRITE DELTA VECTORS"
    )

    print(
        "Mean delta norm:        "
        f"{delta_stats['mean_norm']:.6f}"
    )

    print(
        "Mean pairwise delta L2: "
        f"{delta_stats['mean_pairwise_l2']:.6f}"
    )

    print(
        "Delta L2 / delta norm:  "
        f"{delta_stats['relative_l2']:.6f}"
    )

    print(
        "Mean delta cosine:      "
        f"{delta_stats['mean_cosine']:.6f}"
    )

    # ========================================================
    # QUICK INTERPRETATION
    # ========================================================

    print()
    print("=" * 90)
    print("INTERPRETATION GUIDE")
    print("=" * 90)

    print(
        """
1. RAW_TEXT should be clearly above chance (25%).
   If not, stop: the evaluation prompt/candidate setup is unreliable.

2. MATCHED_MEMORY should beat MISMATCHED_MEMORY if the latent
   memory carries fact-specific information.

3. If MATCHED ≈ MISMATCHED, but both beat EMPTY_MEMORY,
   the memory is likely changing the model's prior rather than
   retrieving the correct fact.

4. For content dependence:
      relative L2 very small
      + cosine near 1.0
   means different facts produce nearly identical slot values.

5. Slot collapse:
   if the same slot receives the maximum write almost every time,
   routing/write allocation is collapsed.

This still tests the CURRENT CHECKPOINT only.
It does not prove what the architecture could learn after
direct value-path training.
"""
    )


if __name__ == "__main__":
    main()