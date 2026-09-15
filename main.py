from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2LMHeadModel

# Auto-resolve path to project root containing 'models'
current_dir = Path(__file__).resolve().parent
candidates = [current_dir] + list(current_dir.glob("**/models")) + list(current_dir.parent.glob("**/models"))
for c in candidates:
    target = c.parent if c.name == "models" else c
    if (target / "models").is_dir() and str(target) not in sys.path:
        sys.path.insert(0, str(target))
        break

from models.gpt2_memory import MemoryAugmentedGPT2LMHeadModel, MemoryGPT2Config

try:
    from data.dataset import WikiText103DocumentDataset
    from data.preprocessing import prepare_chunk
    HAS_WIKITEXT_PIPELINE = True
except ImportError:
    HAS_WIKITEXT_PIPELINE = False


# ---------------------------------------------------------------------------
# SYNTHETIC DATA GENERATOR (TARGET-ONLY MASKING)
# ---------------------------------------------------------------------------
NAMES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel"]
CODES = ["1029", "3847", "5612", "7394", "9281", "2468", "4135", "6570"]

def generate_synthetic_batch(
    tokenizer: AutoTokenizer,
    distractor_len: int = 128,
    num_entities: int = 4,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Constructs a multi-entity memory stream with TARGET-ONLY loss masking.
    All prompt and distractor tokens are masked to -100; only the answer tokens generate loss.
    """
    items = list(zip(NAMES, CODES))
    random.shuffle(items)
    active_items = items[:num_entities]
    target_name, target_code = random.choice(active_items)

    # 1. Fact token sequences to write sequentially into memory
    fact_token_list = [
        tokenizer(f"Agent {name}'s secret passcode is {code}.", return_tensors="pt").input_ids.to(device)
        for name, code in active_items
    ]

    # 2. Distractor tokens to stream through memory
    filler = " The regional weather in the valley remains unpredictable with light rain. " * (distractor_len // 10 + 2)
    tok_distractor = tokenizer(filler, return_tensors="pt").input_ids[:, :distractor_len].to(device)

    # 3. Query and target answer
    query_prompt = f" What is the secret passcode for Agent {target_name}? The passcode is"
    target_str = f" {target_code}"

    tok_query = tokenizer(query_prompt, return_tensors="pt").input_ids.to(device)
    tok_ans = tokenizer(target_str, return_tensors="pt").input_ids.to(device)
    ans_len = tok_ans.size(1)

    full_eval_seq = torch.cat([tok_query, tok_ans], dim=1)

    # STRICT TARGET-ONLY LOSS MASKING:
    # Everything is masked to -100 except the answer positions
    labels = torch.full_like(full_eval_seq, fill_value=-100)
    labels[:, -ans_len:] = full_eval_seq[:, -ans_len:]

    target_first_token_id = tok_ans[0, 0].item()

    return fact_token_list, tok_distractor, full_eval_seq, labels, target_first_token_id


# ---------------------------------------------------------------------------
# MODEL LOADER & DIAGNOSTIC UTILITIES
# ---------------------------------------------------------------------------
def load_memory_model(config: MemoryGPT2Config, device: torch.device) -> MemoryAugmentedGPT2LMHeadModel:
    print("Loading pretrained GPT-2 backbone...")
    backbone = GPT2LMHeadModel.from_pretrained("gpt2")
    model = MemoryAugmentedGPT2LMHeadModel(backbone=backbone, memory_config=config)
    return model.to(device)


def compute_slot_cosine_similarity(memory_state) -> float:
    """Computes mean off-diagonal pairwise cosine similarity across memory slots."""
    slots = getattr(memory_state, "slots", None)
    if slots is None and isinstance(memory_state, tuple):
        slots = memory_state[0]
    if slots is None or slots.ndim < 2:
        return 0.0

    s = slots[0] if slots.ndim == 3 else slots
    norm_s = F.normalize(s, p=2, dim=-1)
    cos_matrix = torch.mm(norm_s, norm_s.t())
    mask = ~torch.eye(cos_matrix.size(0), dtype=torch.bool, device=cos_matrix.device)
    return float(cos_matrix[mask].mean().item())


# ---------------------------------------------------------------------------
# TRAINING & EVALUATION ROUTINES
# ---------------------------------------------------------------------------
def train_synthetic_task(model: nn.Module, tokenizer: AutoTokenizer, args: argparse.Namespace, device: torch.device) -> None:
    print(f"\n>>> Starting Synthetic Retrieval Training ({args.max_steps} steps, Target-Only Masking Enabled)")
    
    # Freeze backbone; train only memory augmentations
    for name, param in model.named_parameters():
        param.requires_grad = "backbone" not in name

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()

    running_lm_loss = 0.0
    running_aux_loss = 0.0
    correct_retrievals = 0

    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad()

        facts, dist_toks, eval_seq, labels, target_token_id = generate_synthetic_batch(
            tokenizer=tokenizer,
            distractor_len=args.distractor_len,
            num_entities=args.num_entities,
            device=device,
        )

        # Pass 1: Write entities into memory sequentially
        mem_state = None
        for f_tokens in facts:
            out_fact = model(input_ids=f_tokens, memory_state=mem_state)
            mem_state = out_fact.memory_state

        # Pass 2: Stream distractors
        if dist_toks is not None and dist_toks.size(1) > 0:
            out_dist = model(input_ids=dist_toks, memory_state=mem_state)
            mem_state = out_dist.memory_state

        # Pass 3: Query target entity with target-only masked labels
        out_query = model(
            input_ids=eval_seq,
            labels=labels,
            memory_state=mem_state,
        )

        # ENSURE AUXILIARY LOSS IS BACKPROPAGATED:
        # out_query.loss automatically combines: total_loss = lm_loss + auxiliary_loss
        total_loss = out_query.loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        running_lm_loss += float(out_query.lm_loss) if out_query.lm_loss is not None else 0.0
        running_aux_loss += float(out_query.auxiliary_loss)

        # Check prediction at the first answer token position
        ans_start_idx = (labels[0] != -100).nonzero(as_tuple=True)[0][0].item()
        pred_token_id = out_query.logits[0, ans_start_idx - 1].argmax().item()
        if pred_token_id == target_token_id:
            correct_retrievals += 1

        if step % args.log_interval == 0 or step == 1:
            step_acc = (correct_retrievals / step) * 100.0
            cos_sim = compute_slot_cosine_similarity(out_query.memory_state)
            gate_mean = out_query.write_gate.mean().item() if out_query.write_gate is not None else 0.0

            print(
                f"Step {step:04d}/{args.max_steps:04d} | "
                f"Target Loss: {float(out_query.lm_loss):.4f} | "
                f"Aux Loss: {float(out_query.auxiliary_loss):.4f} | "
                f"Gate Mean: {gate_mean:.3f} | "
                f"Slot CosSim: {cos_sim:.4f} | "
                f"Cumulative Acc: {step_acc:.1f}%"
            )

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "memory_checkpoint.pt")
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}, save_path)
    print(f"\n[Saved] Trained memory weights saved to: {save_path}")


def run_wikitext_task(model: nn.Module, tokenizer: AutoTokenizer, args: argparse.Namespace, device: torch.device) -> None:
    if not HAS_WIKITEXT_PIPELINE:
        print("[Error] WikiText modules not found in data/. Run with --task synthetic instead.")
        return

    print(f"\n>>> Running WikiText-103 Pipeline (Train={args.train}, Max Docs={args.max_docs})")
    dataset = WikiText103DocumentDataset(
        data_dir="data/wikitext-103",
        tokenizer=tokenizer,
        split="train" if args.train else "valid",
        chunk_size=256,
        min_document_tokens=256,
        max_documents=args.max_docs,
    )

    if args.train:
        for name, param in model.named_parameters():
            param.requires_grad = "backbone" not in name
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        model.train()
    else:
        model.eval()

    for doc_idx in range(min(args.max_docs, len(dataset))):
        doc = dataset[doc_idx]
        print(f"\n--- Document [{doc_idx + 1}/{args.max_docs}]: {doc['title']} ---")
        mem_state = None
        chunks = doc["chunks"][:args.max_chunks_per_doc]

        for c_idx, chunk in enumerate(chunks):
            batch = prepare_chunk(chunk=chunk, tokenizer=tokenizer, device=device)

            if args.train:
                optimizer.zero_grad()
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    memory_state=mem_state,
                )
                total_loss = output.loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                mem_state = output.memory_state.detach() if hasattr(output.memory_state, "detach") else output.memory_state
            else:
                with torch.no_grad():
                    output = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        memory_state=mem_state,
                    )
                    mem_state = output.memory_state.detach() if hasattr(output.memory_state, "detach") else output.memory_state

            cos_sim = compute_slot_cosine_similarity(output.memory_state)
            gate_val = f"{output.write_gate.mean().item():.3f}" if output.write_gate is not None else "N/A"

            print(
                f"Chunk {c_idx + 1:02d}/{len(chunks):02d} | "
                f"LM Loss: {float(output.lm_loss):.4f} | "
                f"Aux Loss: {float(output.auxiliary_loss):.4f} | "
                f"Gate: {gate_val} | "
                f"Slot CosSim: {cos_sim:.4f}"
            )


# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="MemAttn Execution Harness")
    parser.add_argument("--task", type=str, default="synthetic", choices=["synthetic", "wikitext"], help="Task to execute")
    parser.add_argument("--train", action="store_true", help="Enable gradient backprop")
    parser.add_argument("--num_slots", type=int, default=16, help="Latent memory slots")
    parser.add_argument("--gate_type", type=str, default="vector", choices=["scalar", "vector"], help="Gate type")
    parser.add_argument("--gate_init_bias", type=float, default=0.0, help="Initial bias for gate (0.0 = un-choked)")
    parser.add_argument("--ortho_weight", type=float, default=0.05, help="Orthogonality regularization loss weight")
    parser.add_argument("--distractor_len", type=int, default=128, help="Distractor tokens between fact and query")
    parser.add_argument("--num_entities", type=int, default=4, help="Number of competing entities in synthetic stream")
    parser.add_argument("--max_steps", type=int, default=500, help="Total training steps for synthetic task")
    parser.add_argument("--max_docs", type=int, default=2, help="Number of documents for wikitext")
    parser.add_argument("--max_chunks_per_doc", type=int, default=4, help="Max chunks per document for wikitext")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=50, help="Logging step frequency")
    parser.add_argument("--output_dir", type=str, default="outputs/run_benchmark", help="Directory to save checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== MemAttn Harness Initialized on {device.type.upper()} ===")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mem_config = MemoryGPT2Config(
        num_slots=args.num_slots,
        gate_type=args.gate_type,
        gate_mode="sigmoid",
        gate_init_bias=args.gate_init_bias,
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
        update_orthogonality_weight=args.ortho_weight,
        router_balance_weight=0.01,
        reader_balance_weight=0.01,
        memory_collapse_weight=0.01,
        detach_memory_between_steps=False,
    )

    model = load_memory_model(mem_config, device)

    if args.task == "synthetic":
        train_synthetic_task(model, tokenizer, args, device)
    elif args.task == "wikitext":
        run_wikitext_task(model, tokenizer, args, device)


if __name__ == "__main__":
    main()