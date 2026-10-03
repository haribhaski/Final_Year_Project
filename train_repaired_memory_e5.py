from __future__ import annotations

"""
E5-addressed version of train_repaired_memory.py

WHAT CHANGES:
    OLD:
        early GPT-2 hidden -> address key

    NEW:
        E5("passage: fact") -> address key
        E5("query: question") -> read query

WHAT STAYS THE SAME:
    CandidateWriter
    VectorGate
    OrthogonalUpdate
    MemoryBank
    occupancy-aware SlotRouter
    MemoryReader
    fusion
    frozen GPT-2 LM head
    losses / evaluation

REQUIREMENT:
    The current train_repaired_memory.py must already be the version
    containing AddressEncoder/address_slots support.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

# Import your CURRENT repaired training script.
import train_repaired_memory as base


# ============================================================
# CONFIG
# ============================================================

E5_MODEL_NAME = "intfloat/e5-base-v2"

# E5 recommends query:/passage: prefixes.
E5_MAX_LENGTH = 128

# Smaller => sharper E5 addressing prior.
E5_ADDRESS_TEMPERATURE = 0.05


# ============================================================
# GLOBAL E5 CACHE
# ============================================================

_E5_TOKENIZER = None
_E5_MODEL = None
_E5_DEVICE = None


def get_e5(device):
    global _E5_TOKENIZER
    global _E5_MODEL
    global _E5_DEVICE

    if _E5_MODEL is None:

        print()
        print("=" * 100)
        print("LOADING E5 ADDRESS ENCODER")
        print("=" * 100)
        print("Model:", E5_MODEL_NAME)

        _E5_TOKENIZER = AutoTokenizer.from_pretrained(
            E5_MODEL_NAME
        )

        _E5_MODEL = AutoModel.from_pretrained(
            E5_MODEL_NAME
        )

        _E5_MODEL.to(device)
        _E5_MODEL.eval()

        for p in _E5_MODEL.parameters():
            p.requires_grad = False

        _E5_DEVICE = device

        print("E5 loaded.")
        print("E5 parameters frozen.")
        print()

    elif str(_E5_DEVICE) != str(device):

        _E5_MODEL.to(device)
        _E5_DEVICE = device

    return (
        _E5_TOKENIZER,
        _E5_MODEL,
    )


# ============================================================
# E5 MEAN POOL
# ============================================================

def average_pool(
    last_hidden_states,
    attention_mask,
):

    mask = (
        attention_mask
        .unsqueeze(-1)
        .to(last_hidden_states.dtype)
    )

    summed = (
        last_hidden_states
        * mask
    ).sum(dim=1)

    denominator = (
        mask
        .sum(dim=1)
        .clamp_min(1e-8)
    )

    return summed / denominator


# ============================================================
# E5 ENCODING
# ============================================================

@torch.no_grad()
def encode_e5_texts(
    texts,
    prefix,
    device,
    batch_size,
):

    tokenizer, model = get_e5(
        device
    )

    embeddings = []

    prefixed = [
        prefix + text
        for text in texts
    ]

    for start in range(
        0,
        len(prefixed),
        batch_size,
    ):

        batch = prefixed[
            start:
            start + batch_size
        ]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=E5_MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value
            in encoded.items()
        }

        output = model(
            **encoded
        )

        pooled = average_pool(
            output.last_hidden_state,
            encoded[
                "attention_mask"
            ],
        )

        # Important for E5 cosine retrieval.
        pooled = F.normalize(
            pooled,
            p=2,
            dim=-1,
        )

        embeddings.extend(
            pooled
            .detach()
            .cpu()
            .float()
        )

    return embeddings


# ============================================================
# GET FINAL GPT-2 HIDDEN STATES
#
# We only use GPT-2 here for the VALUE pathway.
# E5 handles ADDRESSING.
# ============================================================

@torch.no_grad()
def encode_gpt2_final(
    model,
    tokenizer,
    texts,
    device,
    batch_size,
):

    result = []

    for start in range(
        0,
        len(texts),
        batch_size,
    ):

        batch = texts[
            start:
            start + batch_size
        ]

        ids, mask = base.tokenize(
            tokenizer,
            batch,
            device,
        )

        output = (
            model
            .backbone
            .transformer(
                input_ids=ids,
                attention_mask=mask,
                return_dict=True,
            )
        )

        hidden = (
            output.last_hidden_state
        )

        for i in range(
            hidden.size(0)
        ):

            length = int(
                mask[i]
                .sum()
                .item()
            )

            seq = (
                hidden[
                    i,
                    :length,
                    :
                ]
                .detach()
                .cpu()
                .half()
            )

            result.append(
                seq
            )

    return result


# ============================================================
# REPLACEMENT PRECOMPUTE
#
# VALUE:
#     GPT-2 final hidden
#
# ADDRESS:
#     E5 passage/query embeddings
# ============================================================

@torch.no_grad()
def precompute_episodes_e5(
    model,
    tokenizer,
    episodes,
    answer_to_token,
    answer_to_class,
    device,
    batch_size,
):

    flat_facts = []
    flat_queries = []
    metadata = []

    for episode_id, episode in enumerate(
        episodes
    ):

        for position, item in enumerate(
            episode
        ):

            flat_facts.append(
                item["fact"]
            )

            flat_queries.append(
                item["query"]
            )

            metadata.append(
                {
                    "episode": (
                        episode_id
                    ),

                    "position": (
                        position
                    ),

                    "target_token": (
                        answer_to_token[
                            item["answer"]
                        ]
                    ),

                    "target_class": (
                        answer_to_class[
                            item["answer"]
                        ]
                    ),
                }
            )

    # ========================================================
    # VALUE representations
    # ========================================================

    fact_hidden = encode_gpt2_final(
        model=model,
        tokenizer=tokenizer,
        texts=flat_facts,
        device=device,
        batch_size=batch_size,
    )

    query_hidden = encode_gpt2_final(
        model=model,
        tokenizer=tokenizer,
        texts=flat_queries,
        device=device,
        batch_size=batch_size,
    )

    # ========================================================
    # ADDRESS representations
    # ========================================================

    print(
        "Encoding E5 passage addresses..."
    )

    fact_addresses = encode_e5_texts(
        texts=flat_facts,
        prefix="passage: ",
        device=device,
        batch_size=batch_size,
    )

    print(
        "Encoding E5 query addresses..."
    )

    query_addresses = encode_e5_texts(
        texts=flat_queries,
        prefix="query: ",
        device=device,
        batch_size=batch_size,
    )

    # E5-base-v2 should be 768 dimensions,
    # same as GPT-2 small.
    e5_dim = (
        fact_addresses[0]
        .numel()
    )

    if e5_dim != model.d_model:

        raise RuntimeError(
            f"E5 dimension={e5_dim}, "
            f"GPT-2 d_model={model.d_model}. "
            "Use intfloat/e5-base-v2 with GPT-2 small "
            "or add an address projection."
        )

    cached = [
        []
        for _ in episodes
    ]

    for i, meta in enumerate(
        metadata
    ):

        fact_final = (
            fact_hidden[i]
            .float()
        )

        # Existing writer/gate summary.
        summary = (
            fact_final
            .mean(dim=0)
        )

        # ====================================================
        # E5 FACT ADDRESS
        # ====================================================

        fact_address = (
            fact_addresses[i]
            .float()
        )

        # ====================================================
        # E5 QUERY ADDRESS
        #
        # MemoryReader expects [T,D].
        # E5 gives one semantic query vector [D].
        #
        # Broadcast the SAME semantic query vector over the
        # GPT-2 query token length.
        # ====================================================

        q_len = (
            query_hidden[i]
            .size(0)
        )

        query_address_hidden = (
            query_addresses[i]
            .unsqueeze(0)
            .expand(
                q_len,
                -1,
            )
            .contiguous()
        )

        f_len = (
            fact_hidden[i]
            .size(0)
        )

        fact_address_hidden = (
            fact_address
            .unsqueeze(0)
            .expand(
                f_len,
                -1,
            )
            .contiguous()
        )

        cached[
            meta["episode"]
        ].append(
            {
                # ----------------------------------------
                # VALUE pathway
                # ----------------------------------------

                "fact_hidden": (
                    fact_hidden[i]
                ),

                "summary": (
                    summary.half()
                ),

                "query_hidden": (
                    query_hidden[i]
                ),

                # ----------------------------------------
                # ADDRESS pathway
                # ----------------------------------------

                "fact_address_hidden": (
                    fact_address_hidden
                    .half()
                ),

                "address_summary": (
                    fact_address
                    .half()
                ),

                "query_address_hidden": (
                    query_address_hidden
                    .half()
                ),

                # ----------------------------------------
                # TARGETS
                # ----------------------------------------

                "target_token": (
                    meta[
                        "target_token"
                    ]
                ),

                "target_class": (
                    meta[
                        "target_class"
                    ]
                ),
            }
        )

    return cached


# ============================================================
# E5-AWARE REPAIRED SYSTEM
# ============================================================

class E5RepairedMemorySystem(
    base.RepairedMemorySystem
):

    def __init__(
        self,
        original,
        router_temperature=0.7,
    ):

        super().__init__(
            original=original,
            router_temperature=(
                router_temperature
            ),
        )

        print()
        print(
            "ADDRESS MODE: FROZEN E5"
        )

        print(
            "Address similarity temperature:",
            E5_ADDRESS_TEMPERATURE,
        )

    # ========================================================
    # READ
    #
    # E5 provides a semantic routing prior.
    #
    # Reader still:
    #   - performs multi-head attention
    #   - retrieves VALUE vectors
    #   - projects context
    #   - performs fusion
    #
    # E5 DOES NOT REPLACE THE MEMORY READER.
    # ========================================================

    def read_query(
        self,
        state,
        query_hidden,
        query_address_hidden,
        query_mask,
    ):

        memory_mask = (
            state.write_count
            > 0
        )

        if state.address_slots is None:

            raise RuntimeError(
                "Address slots are missing."
            )

        # ====================================================
        # Same address transform on QUERY and stored KEY.
        # ====================================================

        transformed_query = (
            self.address_encoder(
                query_address_hidden
            )
        )

        # Since every query token currently contains the same
        # E5 semantic vector, token 0 is sufficient for the
        # explicit E5 similarity prior.
        query_vector = (
            transformed_query[
                :,
                0,
                :
            ]
        )

        query_vector = F.normalize(
            query_vector,
            p=2,
            dim=-1,
        )

        key_vectors = F.normalize(
            state.address_slots,
            p=2,
            dim=-1,
        )

        # [B,N]
        cosine_scores = torch.einsum(
            "bd,bnd->bn",
            query_vector,
            key_vectors,
        )

        # Never route to unused memory.
        minimum = torch.finfo(
            cosine_scores.dtype
        ).min

        masked_scores = (
            cosine_scores
            .masked_fill(
                ~memory_mask,
                minimum,
            )
        )

        # ====================================================
        # Semantic E5 retrieval distribution
        # ====================================================

        routing_prior = torch.softmax(
            masked_scores
            /
            E5_ADDRESS_TEMPERATURE,
            dim=-1,
        )

        # ====================================================
        # EXISTING MEMORY READER
        #
        # Q/K addressing receives:
        #   address_hidden_states
        #   address_slots
        #
        # routing_prior strongly encourages the semantically
        # matching E5 slot.
        #
        # VALUES still come from state.slots.
        # ====================================================

        output = self.reader(
            hidden_states=(
                query_hidden
            ),

            # VALUE
            memory_slots=(
                state.slots
            ),

            # KEY
            address_slots=(
                state.address_slots
            ),

            address_hidden_states=(
                transformed_query
            ),

            attention_mask=(
                query_mask
            ),

            memory_mask=(
                memory_mask
            ),

            routing_prior=(
                routing_prior
            ),

            memory_confidence=None,

            return_attention=True,
        )

        lengths = (
            query_mask
            .sum(dim=-1)
            .long()
            .sub(1)
            .clamp_min(0)
        )

        rows = torch.arange(
            query_hidden.size(0),
            device=(
                query_hidden.device
            ),
        )

        fused_last = (
            output
            .fused_hidden[
                rows,
                lengths,
                :
            ]
        )

        logits = (
            self.lm_head(
                fused_last
            )
        )

        return (
            logits,
            output,
            lengths,
        )


# ============================================================
# PATCH THE EXISTING TRAINING SCRIPT
# ============================================================

base.precompute_episodes = (
    precompute_episodes_e5
)

base.RepairedMemorySystem = (
    E5RepairedMemorySystem
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 100)
    print("E5-ADDRESSED MEMORY INTEGRATION")
    print("=" * 100)

    print(
        "E5 model:",
        E5_MODEL_NAME,
    )

    print(
        "WRITE ADDRESS:",
        "passage: <fact>",
    )

    print(
        "READ ADDRESS:",
        "query: <question>",
    )

    print(
        "VALUE pathway:",
        "existing GPT-2 + CandidateWriter",
    )

    print(
        "E5 frozen:",
        True,
    )

    print("=" * 100)
    print()

    base.main()