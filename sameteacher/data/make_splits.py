"""Draw prompts at random from the full Infinity-Chat pool and fix the splits.

The 100 prompts in infinite-chats-eval are excluded: they form Infinity-Chat100, the subset on
which Jiang et al. (2025) measure homogeneity, so our test prompts stay disjoint from it.
Everything not drawn is reserved as an untouched test pool.
"""
import argparse
import json
import random
import re

from sameteacher.core.config import add_common_args, load_yaml, write_json, write_jsonl

from sameteacher.core.paths import prompts_dir, prompts_path

UNSAFE = re.compile(
    r"\b(sex\w*|nude|naked|porn\w*|erotic\w*|nsfw|fetish\w*|kink\w*|lewd|seduc\w*|breasts?|thighs?|headscissors?|chokehold\w*|"
    r"rape|incest|loli|underage|minors?|suicid\w*|self-harm|bomb|genocide)\b",
    re.I,
)
NEEDS_CONTEXT = re.compile(
    r"\b(following|above|below|attached|given (text|passage|article)|the text|this (article|text|passage|paragraph|code|email|document|essay|image))\b",
    re.I,
)


PREFIX_CHARS = 50


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def flagged(annotation: dict, field: str):
    """The classifier's verdict: each field holds {"classification": bool, "reasoning": str}."""
    return (annotation.get(field) or {}).get("classification")


def keep_prompt(text: str, min_chars: int, max_chars: int) -> bool:
    if not min_chars <= len(text) <= max_chars:
        return False
    if sum(ch.isascii() for ch in text) / len(text) < 0.95:
        return False
    return not (UNSAFE.search(text) or NEEDS_CONTEXT.search(text))


def first_user_message(row: dict) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            return (message.get("content") or "").strip()
    return ""


def load_pool(args) -> tuple[list[dict], set[str]]:
    if args.input_jsonl:
        with open(args.input_jsonl, encoding="utf-8") as f:
            pool = [json.loads(line) for line in f]
        with open(args.exclude_jsonl, encoding="utf-8") as f:
            excluded = {normalize(json.loads(line)["query"]) for line in f}
        return pool, excluded
    from datasets import load_dataset

    pool = list(load_dataset("liweijiang/infinite-chats-taxonomy", split="train"))
    excluded = {normalize(r["query"]) for r in load_dataset("liweijiang/infinite-chats-eval", split="train")}
    return pool, excluded


def shuffled_pool(pool, excluded, min_chars: int, max_chars: int, seed: int) -> list[dict]:
    """Every usable prompt in the fixed random order that all splits are cut from."""
    rows, seen = [], set()
    for row in pool:
        annotation = row.get("lm_judge_annotation") or {}
        if flagged(annotation, "meaningful_information") is False or flagged(annotation, "greeting_model_inquiry") is True:
            continue
        text = first_user_message(row)
        key = normalize(text)
        # One user often sends near-identical prompts, which would leak across splits.
        prefix = key[:PREFIX_CHARS]
        if prefix in seen or key in excluded or not keep_prompt(text, min_chars, max_chars):
            continue
        seen.add(prefix)
        categories = row.get("categories") or [{}]
        rows.append({"prompt_id": row["conversation_id"], "prompt": text, "category": categories[0].get("category", "unknown")})
    rows.sort(key=lambda r: r["prompt_id"])
    random.Random(seed).shuffle(rows)
    return rows


def add_pool_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-jsonl", help="Local copy of the taxonomy data, for tests")
    parser.add_argument("--exclude-jsonl", help="Local copy of the eval queries, for tests")
    parser.add_argument("--min-chars", type=int, default=25)
    parser.add_argument("--max-chars", type=int, default=400)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_pool_args(parser)
    args = parser.parse_args()

    if prompts_path(args.run_dir, "screen").exists():
        raise SystemExit("Splits already exist. They are fixed once drawn; delete them only to restart from nothing.")

    cfg = load_yaml(args.config)["splits"]
    pool, excluded = load_pool(args)
    rows = shuffled_pool(pool, excluded, args.min_chars, args.max_chars, cfg["seed"])
    needed = cfg["screen"] + cfg["confirm"] + cfg["pilot_eval"]
    if len(rows) < needed:
        raise SystemExit(f"Only {len(rows)} prompts survive filtering; {needed} are needed.")

    start = 0
    for split in ("screen", "confirm", "pilot_eval"):
        write_jsonl(prompts_path(args.run_dir, split), rows[start : start + cfg[split]])
        start += cfg[split]
    write_json(prompts_dir(args.run_dir) / "reserved_ids.json", [r["prompt_id"] for r in rows[start:]])
    print(f"kept {len(rows)} of {len(pool)} prompts; {len(rows) - needed} reserved as the untouched test pool")


if __name__ == "__main__":
    main()
