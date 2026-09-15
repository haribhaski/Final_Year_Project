import os
import sys
import math
import time
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2LMHeadModel

# Ensure repo root is on python path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from models.gpt2_memory import MemoryAugmentedGPT2LMHeadModel, MemoryGPT2Config
from data.dataset import WikiText103DocumentDataset
from data.preprocessing import prepare_chunk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "outputs/memattn_wikitext_15k"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"=== Initializing WikiText-103 Benchmark on {device.type.upper()} ===", flush=True)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Full MemAttn Architecture (Vector Gating + Ortho Regularization)
mem_config = MemoryGPT2Config(
    num_slots=16,
    gate_type="vector",
    gate_mode="sigmoid",
    gate_init_bias=0.0,
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
    update_orthogonality_weight=0.05,
    router_balance_weight=0.01,
    reader_balance_weight=0.01,
    memory_collapse_weight=0.01,
    detach_memory_between_steps=False,
)

backbone = GPT2LMHeadModel.from_pretrained("gpt2")
model = MemoryAugmentedGPT2LMHeadModel(backbone, mem_config).to(device)

# Freeze GPT-2 backbone
for name, param in model.named_parameters():
    param.requires_grad = "backbone" not in name

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01)

train_dataset = WikiText103DocumentDataset(
    data_dir="data/wikitext-103",
    tokenizer=tokenizer,
    split="train",
    chunk_size=256,
    min_document_tokens=256,
    max_documents=5000,
)

valid_dataset = WikiText103DocumentDataset(
    data_dir="data/wikitext-103",
    tokenizer=tokenizer,
    split="validation",
    chunk_size=256,
    min_document_tokens=256,
    max_documents=10,
)

def evaluate_valid(eval_steps=20):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        num_val = len(valid_dataset)
        for d_i in range(num_val):
            doc = valid_dataset[d_i]
            mem = None
            for chunk in doc["chunks"][:4]:
                batch = prepare_chunk(chunk, tokenizer, device)
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    memory_state=mem
                )
                total_loss += float(out.lm_loss)
                count += 1
                mem = out.memory_state.detach() if hasattr(out.memory_state, "detach") else out.memory_state
                if count >= eval_steps:
                    break
            if count >= eval_steps:
                break
    model.train()
    avg_loss = total_loss / max(1, count)
    return avg_loss, math.exp(min(avg_loss, 20.0))

TOTAL_STEPS = 15000
LOG_INTERVAL = 100
SAVE_INTERVAL = 1000

step = 0
running_loss = 0.0
start_time = time.time()
model.train()

print(f">>> Commencing 15,000-Step Training Run. Checkpoints: {OUTPUT_DIR}/ ...\n", flush=True)

num_docs = len(train_dataset)
while step < TOTAL_STEPS:
    for doc_idx in range(num_docs):
        doc = train_dataset[doc_idx]
        mem_state = None
        for chunk in doc["chunks"]:
            step += 1
            batch = prepare_chunk(chunk, tokenizer, device)

            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                memory_state=mem_state,
            )

            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            mem_state = out.memory_state.detach() if hasattr(out.memory_state, "detach") else out.memory_state
            running_loss += float(out.lm_loss)

            if step % LOG_INTERVAL == 0:
                avg_lm_loss = running_loss / LOG_INTERVAL
                ppl = math.exp(min(avg_lm_loss, 20.0))
                gate_val = out.write_gate.mean().item() if out.write_gate is not None else 0.0
                elapsed = time.time() - start_time
                hrs = elapsed / 3600.0

                print(
                    f"Step {step:05d}/{TOTAL_STEPS} | "
                    f"LM Loss: {avg_lm_loss:.4f} | "
                    f"Train PPL: {ppl:6.2f} | "
                    f"Gate: {gate_val:.3f} | "
                    f"Elapsed: {hrs:.2f} hrs",
                    flush=True
                )
                running_loss = 0.0

            if step % SAVE_INTERVAL == 0:
                val_loss, val_ppl = evaluate_valid()
                ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_step_{step}.pt")
                torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}, ckpt_path)
                print(f"--- [EVAL & SAVE] Step {step} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:6.2f} | Saved: {ckpt_path} ---", flush=True)

            if step >= TOTAL_STEPS:
                break
        if step >= TOTAL_STEPS:
            break

final_path = os.path.join(OUTPUT_DIR, "memory_checkpoint_wikitext_15k.pt")
torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}, final_path)
print(f"\n[Finished] Full 15,000 steps complete! Final weights: {final_path}", flush=True)
