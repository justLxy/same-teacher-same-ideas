"""Fine-tune one base model on one teacher's data with a LoRA adapter.

Our own training loop rather than a Trainer. Steps are fixed, so every cell of the
experiment gets the same training budget. Runs on CUDA, MPS, or CPU. With --init-adapter the
run continues training an existing adapter (a second SFT stage) with a new optimizer and schedule.
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from sameteacher.core.config import read_jsonl, write_json

from sameteacher.train.prompt_format import format_prompt

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def pick_dtype(device: str) -> torch.dtype:
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode(row: dict, tokenizer, max_length: int) -> dict:
    prompt = tokenizer(format_prompt(row["prompt"]), add_special_tokens=True)["input_ids"]
    answer = tokenizer(row["response"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    input_ids = (prompt + answer)[:max_length]
    labels = ([-100] * len(prompt) + answer)[:max_length]
    return {"input_ids": input_ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    width = max(len(row["input_ids"]) for row in batch)
    input_ids, labels, mask = [], [], []
    for row in batch:
        gap = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * gap)
        labels.append(row["labels"] + [-100] * gap)
        mask.append([1] * len(row["input_ids"]) + [0] * gap)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}


def train(args) -> dict:
    seed_everything(args.seed)
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=pick_dtype(device))
    model.config.use_cache = False
    if device == "cuda":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if args.init_adapter:
        # The adapter keeps its own rank, alpha and target modules; only training continues.
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True).to(device)
    else:
        model = get_peft_model(
            model,
            LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM"),
        ).to(device)

    rows = read_jsonl(args.train_file)
    if not rows:
        raise SystemExit(f"No training rows in {args.train_file}")
    examples = [encode(r, tokenizer, args.max_length) for r in rows]
    loader = DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, args.max_steps // 20), args.max_steps)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train_log.jsonl"

    model.train()
    step, micro, running = 0, 0, 0.0
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        while step < args.max_steps:
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss / args.grad_accum
                if not math.isfinite(loss.item()):
                    raise SystemExit(f"Loss became {loss.item()} at step {step}; lower the learning rate.")
                loss.backward()
                running += loss.item()
                micro += 1
                if micro % args.grad_accum:
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % 20 == 0 or step == args.max_steps:
                    log.write(json.dumps({"step": step, "loss": running, "seconds": round(time.time() - started, 1)}) + "\n")
                    log.flush()
                last_loss = running
                running = 0.0
                if step >= args.max_steps:
                    break

    model.save_pretrained(output / "adapter")
    meta = {
        "base_model": args.base_model,
        "train_file": str(args.train_file),
        "train_sha256": hashlib.sha256(Path(args.train_file).read_bytes()).hexdigest(),
        "examples": len(rows),
        "seed": args.seed,
        "init_adapter": str(args.init_adapter) if args.init_adapter else None,
        "max_steps": args.max_steps,
        "final_loss": last_loss,
        "seconds": round(time.time() - started, 1),
        "device": device,
    }
    write_json(output / "train_meta.json", meta)
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--init-adapter", help="Continue training this LoRA adapter instead of starting a new one")
    return parser


def main() -> None:
    print(train(build_parser().parse_args()))


if __name__ == "__main__":
    main()
