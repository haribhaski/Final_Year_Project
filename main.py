''' run the file: 
/home/zeus/miniconda3/envs/cloudspace/bin/python main.py

'''
from __future__ import annotations

import torch
from transformers import AutoTokenizer

from data.dataset import WikiText103DocumentDataset
from data.preprocessing import prepare_chunk
from models.gpt2_memory import (
    MemoryAugmentedGPT2LMHeadModel,
    MemoryGPT2Config,
)


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = WikiText103DocumentDataset(
        data_dir="data/wikitext-103",
        tokenizer=tokenizer,
        split="train",
        chunk_size=256,
        min_document_tokens=256,

        # Use only two documents for the initial check.
        max_documents=2,
    )

    memory_config = MemoryGPT2Config(
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

    print("Loading real pretrained GPT-2...")

    model = MemoryAugmentedGPT2LMHeadModel.from_pretrained(
        "gpt2",
        memory_config=memory_config,
    )

    model.to(device)
    model.eval()

    document = dataset[0]

    print("\nDocument:", document["title"])
    print("Number of tokens:", document["num_tokens"])
    print("Number of chunks:", len(document["chunks"]))

    memory_state = None

    # Test only the first three chunks.
    chunks = document["chunks"][:3]

    with torch.no_grad():
        for chunk_index, chunk in enumerate(chunks):
            batch = prepare_chunk(
                chunk=chunk,
                tokenizer=tokenizer,
                device=device,
            )

            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                memory_state=memory_state,
                update_memory=True,
                return_diagnostics=True,
            )

            memory_state = output.memory_state.detach()

            print(f"\nChunk {chunk_index + 1}")
            print("Input shape:", batch["input_ids"].shape)
            print("Logits shape:", output.logits.shape)
            print("LM loss:", float(output.lm_loss))
            print(
                "Auxiliary loss:",
                float(output.auxiliary_loss),
            )
            print(
                "Memory shape:",
                output.memory_state.slots.shape,
            )
            print(
                "Memory effective rank:",
                float(
                    output.diagnostics[
                        "memory/effective_rank"
                    ]
                ),
            )
            print(
                "Mean write count:",
                float(
                    output.memory_state.write_count
                    .float()
                    .mean()
                ),
            )


if __name__ == "__main__":
    main()