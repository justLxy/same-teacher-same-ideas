"""How close the test prompts are to the training prompts, in the locked `main_prompt_overlap` section.

For each H1 test prompt: its nearest training prompt by embedding cosine and by word-set Jaccard, the same
two quantities to its nearest other test prompt as a reference, and whether the shared-teacher effect S
changes when the test prompts most similar to training are left out.

  python3 -m sameteacher.analysis.prompt_overlap  ->  runs/results/main/prompt_overlap/overlap.json
"""
import argparse
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from sameteacher.core.config import add_common_args, load_yaml, read_json, read_jsonl, write_json
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, prompts_dir, results_dir
from sameteacher.eval.embed import embed_texts, rows_dense
from sameteacher.eval.stats import percentile_interval

SECTION = "main_prompt_overlap"
TRAIN_FILE = "deepseek_rep0.jsonl"     # the complete draw: every one of the 4,938 training prompts


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def max_jaccard(query: set[str], pool: list[set[str]], skip: int | None = None) -> float:
    best = 0.0
    for k, other in enumerate(pool):
        if k == skip or not query or not other:
            continue
        best = max(best, len(query & other) / len(query | other))
    return best


def nearest(test_vecs: np.ndarray, pool_vecs: np.ndarray, exclude_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    sims = test_vecs @ pool_vecs.T
    if exclude_self:
        np.fill_diagonal(sims, -np.inf)
    idx = sims.argmax(axis=1)
    return sims[np.arange(len(idx)), idx], idx


def summary(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


def mean_ci(values, n_boot: int, seed: int, level: float) -> dict:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)]
    low, high = percentile_interval(draws, level)
    return {"estimate": float(v.mean()), "ci_low": low, "ci_high": high, "prompts": int(v.size)}


def analyze(test: list[dict], train: list[str], per_prompt_s: dict, section: dict) -> dict:
    texts = [t["prompt"] for t in test] + train
    vecs = embed_texts(texts, section["embedder"])
    vecs = rows_dense(vecs, list(range(len(texts))))
    test_vecs, train_vecs = vecs[: len(test)], vecs[len(test):]
    cos_train, idx_train = nearest(test_vecs, train_vecs)
    cos_test, _ = nearest(test_vecs, test_vecs, exclude_self=True)
    train_words = [words(t) for t in train]
    test_words = [words(t["prompt"]) for t in test]
    jac_train = [max_jaccard(w, train_words) for w in test_words]
    jac_test = [max_jaccard(w, test_words, skip=k) for k, w in enumerate(test_words)]

    s = np.array([per_prompt_s[t["prompt_id"]] for t in test])
    keep = cos_train < np.quantile(cos_train, 1 - section["drop_top_share"])
    n, seed, level = section["bootstrap"], section["bootstrap_seed"], section["ci_level"]
    rng = np.random.default_rng(seed)
    rho = spearmanr(s, cos_train).statistic
    rho_draws = []
    for _ in range(n):
        pick = rng.integers(0, s.size, s.size)
        rho_draws.append(spearmanr(s[pick], cos_train[pick]).statistic)
    order = np.argsort(-cos_train)[:5]
    return {
        "prompts": len(test), "training_prompts": len(train), "embedder": section["embedder"],
        "nearest training cosine": summary(cos_train),
        "nearest test cosine (reference)": summary(cos_test),
        "nearest training jaccard": summary(jac_train),
        "nearest test jaccard (reference)": summary(jac_test),
        "share with nearest training cosine above threshold": {
            "threshold": section["near_duplicate_cosine"],
            "share": float((cos_train > section["near_duplicate_cosine"]).mean())},
        "share closer to a training prompt than to any other test prompt": float((cos_train > cos_test).mean()),
        "most similar": [{"test": test[k]["prompt"], "train": train[idx_train[k]], "cosine": float(cos_train[k]),
                          "jaccard": float(jac_train[k])} for k in order],
        "S all prompts": mean_ci(s, n, seed, level),
        "S without the most similar": {**mean_ci(s[keep], n, seed, level), "dropped": int((~keep).sum())},
        "spearman S vs nearest training cosine": {"estimate": float(rho),
                                                  **dict(zip(("ci_low", "ci_high"), percentile_interval(rho_draws, level)))},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for name in ("main_grid", "main_shared_teacher", SECTION):
        verify_lock(cfg, args.run_dir, name)
    h1 = read_json(results_dir(args.run_dir, MAIN, "shared_teacher") / "h1.json")
    per_prompt_s = {pid: v["S"] for pid, v in h1["per_prompt"].items()}
    test = [t for t in read_jsonl(prompts_dir(args.run_dir) / f"{cfg['main_grid']['test_split']}.jsonl")
            if t["prompt_id"] in per_prompt_s]
    train = [r["prompt"] for r in read_jsonl(Path(args.run_dir) / "train_data" / TRAIN_FILE)]
    result = analyze(test, train, per_prompt_s, cfg[SECTION])
    path = results_dir(args.run_dir, MAIN, "prompt_overlap") / "overlap.json"
    write_json(path, result)
    print(path)


if __name__ == "__main__":
    main()
