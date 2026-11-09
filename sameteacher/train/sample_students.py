"""Sample a trained student on the evaluation prompts, resuming after an interruption.

Without --adapter the base model is sampled as it is. --prefix-file puts few-shot examples in
the training format before each prompt; generation then also stops at the next "###" header,
since an untuned base does not end its answer with the end-of-sequence token. --chat samples a
released chat model with its own chat template instead of the training format; with --thinking
the answer is the text after the model's last </think>, and --answer-words cuts each answer.
"""
import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from sameteacher.core.config import read_jsonl, stable_seed, write_jsonl

from sameteacher.train.prompt_format import format_prompt, trim_response
from sameteacher.train.lora import pick_device, pick_dtype


class StopAtHeader(StoppingCriteria):
    """Stop each sequence once its new text contains the next "###" header."""

    def __init__(self, tokenizer, start: int):
        self.tokenizer, self.start = tokenizer, start

    def __call__(self, input_ids, scores, **kwargs):
        texts = self.tokenizer.batch_decode(input_ids[:, self.start:], skip_special_tokens=True)
        return torch.tensor(["###" in t for t in texts], dtype=torch.bool, device=input_ids.device)


THINK_END = "</think>"


def cut_words(text: str, n: int) -> str:
    """The first n words, keeping the original spacing and line breaks."""
    match = re.match(r"\s*(?:\S+\s+){0,%d}\S+" % max(n - 1, 0), text)
    return text[: match.end()].strip() if match else text.strip()


def chat_answer(raw: str, thinking: bool, answer_words: int | None) -> dict:
    """The answer part of a chat model's output; a reasoning model's answer follows its last </think>."""
    reasoning_words = None
    if thinking:
        reasoning, end, answer = raw.rpartition(THINK_END)
        if not end:
            # The reasoning never closed, so there is no answer.
            return {"text": "", "answer_closed": False, "reasoning_words": len(raw.split()), "answer_words": 0}
        reasoning_words, raw = len(reasoning.split()), answer
    raw = raw.strip()
    text = cut_words(raw, answer_words) if answer_words else raw
    return {"text": text, "answer_closed": True, "reasoning_words": reasoning_words, "answer_words": len(raw.split())}


def completed_rows(out_path: Path, samples: int) -> list[dict]:
    """Keep prompts with every sample; a partial prompt is redone."""
    rows = read_jsonl(out_path)
    counts: dict = {}
    for row in rows:
        counts[row["prompt_id"]] = counts.get(row["prompt_id"], 0) + 1
    return [r for r in rows if counts[r["prompt_id"]] >= samples]


def sample_prompt(model, tokenizer, encoded, start, stop, args, prompt, batch, prefix) -> list:
    """All samples of one prompt, drawn in batches of at most `batch` from the prompt's seed."""
    torch.manual_seed(stable_seed(args.student, prompt["prompt_id"]))
    texts = []
    eos = (model.generation_config.eos_token_id or tokenizer.eos_token_id) if args.chat else tokenizer.eos_token_id
    while len(texts) < args.samples:
        n = min(batch, args.samples - len(texts))
        with torch.no_grad():
            tokens = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id,
                # Training ends every answer with this token. Olmo 3 sets no eos
                # in its generation config, so it would otherwise never stop. Chat models
                # stop at the end-of-turn tokens of their own generation config.
                eos_token_id=eos,
                **stop,
            )
        for text in tokenizer.batch_decode(tokens[:, start:], skip_special_tokens=True):
            if args.chat:
                texts.append(chat_answer(text, args.thinking, args.answer_words))
                continue
            text = trim_response(text)
            if prefix:
                # Generation stops as soon as "###" appears, before the header name.
                text = text.split("###")[0].strip()
            texts.append(text)
    return texts


def generate(args) -> int:
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=pick_dtype(device))
    if args.adapter:
        # Merging the adapter gives the same outputs and generates faster.
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model = model.to(device).eval()
    prefix = Path(args.prefix_file).read_text(encoding="utf-8") if args.prefix_file else ""

    out_path = Path(args.output)
    kept = completed_rows(out_path, args.samples)
    write_jsonl(out_path, kept)
    done = {r["prompt_id"] for r in kept}
    prompts = [p for p in read_jsonl(args.prompts) if p["prompt_id"] not in done]

    batch = args.gen_batch
    with open(out_path, "a", encoding="utf-8") as out:
        for prompt in prompts:
            if args.chat:
                # The template adds its own begin-of-text token and, for the R1 distills, "<think>\n".
                encoded = tokenizer.apply_chat_template([{"role": "user", "content": prompt["prompt"]}], add_generation_prompt=True,
                                                        return_tensors="pt", return_dict=True).to(device)
            else:
                encoded = tokenizer(prefix + format_prompt(prompt["prompt"]), return_tensors="pt").to(device)
            start = encoded["input_ids"].shape[1]
            stop = {"stopping_criteria": StoppingCriteriaList([StopAtHeader(tokenizer, start)])} if prefix else {}
            while True:
                try:
                    texts = sample_prompt(model, tokenizer, encoded, start, stop, args, prompt, batch, prefix)
                    break
                except torch.OutOfMemoryError:
                    if batch == 1:
                        raise
                    # A smaller batch changes speed, not the distribution; the prompt restarts from its seed.
                    batch = max(1, batch // 2)
                    torch.cuda.empty_cache()
                    print(f"out of memory on {prompt['prompt_id']}; generation batch is now {batch}", flush=True)
            for index, text in enumerate(texts):
                fields = text if isinstance(text, dict) else {"text": text}
                row = {"student": args.student, "prompt_id": prompt["prompt_id"], "sample_idx": index, **fields}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
    return len(prompts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", help="LoRA adapter; leave out to sample the base model as it is")
    parser.add_argument("--prefix-file", help="Few-shot text placed before each formatted prompt")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--gen-batch", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--chat", action="store_true", help="Use the model's own chat template, with the prompt as the user turn")
    parser.add_argument("--thinking", action="store_true", help="The answer is the text after the last </think> (chat models that reason)")
    parser.add_argument("--answer-words", type=int, help="Cut each chat answer to its first n words")
    return parser


def main() -> None:
    print("prompts generated:", generate(build_parser().parse_args()))


if __name__ == "__main__":
    main()
