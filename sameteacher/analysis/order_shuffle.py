"""Does the H2 retention depend on the order of the rewritten sentences? (locked `main_order_shuffle`)

The main_style_norm rewriter keeps every idea in its original order, so part of what survives the
rewrite could be the order in which a student presents its points. This repeats the locked H2
analysis on the same answers, prompts, word budgets, embedder and bootstrap, and adds a third stage:
each rewritten answer, after the cut to its prompt's word budget, with its sentences in a random
order other than the original one. Only the order changes, not the words.

Usage: python3 -m sameteacher.analysis.order_shuffle
Result: runs/results/main/style_norm/order_shuffle.json
"""
import argparse
import random
import re
from collections import defaultdict

import numpy as np

from sameteacher.analysis.main_shared_teacher import prompt_table, quantities, student_columns
from sameteacher.core.config import add_common_args, load_yaml, stable_seed, write_json
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, results_dir
from sameteacher.core.text import clean, common_word_budget, first_words
from sameteacher.eval.embed import embed_texts
from sameteacher.eval.stats import bootstrap_means, percentile_interval
from sameteacher.style_norm import main_retention as h2
from sameteacher.style_norm.pilot_retention import STAGES

SECTION = "main_order_shuffle"
SHUFFLED = "shuffled"
ALL_STAGES = (*STAGES, SHUFFLED)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_END.split(text.strip()) if s]


def shuffle_sentences(text: str, seed: int) -> str:
    """The sentences of `text` in a random order that differs from the original when there are two or more."""
    parts = sentences(text)
    if len(parts) < 2:
        return text.strip()
    rng, order = random.Random(seed), list(range(len(parts)))
    while order == sorted(order):
        rng.shuffle(order)
    return " ".join(parts[i] for i in order)


def build_texts(pairs, prompts, cells, style, section) -> tuple[list[str], list[str], dict, list[int]]:
    """H2's texts in H2's order with its per-prompt budget, the shuffled rewrites after them, the index of
    both, and the sentence count of each cut rewrite."""
    texts, shuffled, places, counts = [], [], {}, []
    for pid in prompts:
        groups = {(s, cell): [clean(pair[s]) for pair in pairs[cell][pid]] for s in range(len(STAGES)) for cell in cells}
        budget = common_word_budget(list(groups.values()), style["min_words"])
        for (s, cell), values in groups.items():
            cut = [first_words(t, budget) for t in values]
            places[(pid, (STAGES[s], cell))] = ("h2", len(texts), len(cut))
            texts.extend(cut)
            if STAGES[s] == "rewritten":
                counts += [len(sentences(t)) for t in cut]
                places[(pid, (SHUFFLED, cell))] = ("shuffled", len(shuffled), len(cut))
                shuffled.extend(shuffle_sentences(t, stable_seed(section["shuffle_seed"], pid, *cell, j)) for j, t in enumerate(cut))
    offset = {"h2": 0, "shuffled": len(texts)}
    index = {key: list(range(offset[part] + start, offset[part] + start + n)) for key, (part, start, n) in places.items()}
    return texts, shuffled, index, counts


def embed(texts: list[str], shuffled: list[str], name: str):
    """Rows for H2's texts and then the shuffled ones, all in H2's space: TF-IDF is fit on H2's texts only."""
    if name != "tfidf":
        return embed_texts(texts + shuffled, name)
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(texts)
    return sp.vstack([normalize(vectorizer.transform(texts)), normalize(vectorizer.transform(shuffled))]).tocsr()


def derive(m: dict, bases, teachers) -> dict:
    q = {s: quantities(h2.stage_view(m, s), bases, teachers) for s in ALL_STAGES}
    out = {}
    for group in q["written"]:
        w, r, x = q["written"][group], q["rewritten"][group], q[SHUFFLED][group]
        out[group] = {
            "S written": w["S"], "S rewritten": r["S"], "S shuffled": x["S"],
            "R": r["S"] / w["S"], "R shuffled": x["S"] / w["S"], "R shuffled - R": (x["S"] - r["S"]) / w["S"],
            "carryover rewritten": r["carryover"], "carryover shuffled": x["carryover"],
        }
    return out


def analyze(run_dir, cfg) -> dict:
    main_cfg, style, section = cfg["main_grid"], cfg[h2.SECTION], cfg[SECTION]
    bases, teachers, seeds, cells = h2.grid(main_cfg)
    sources = h2.select_sources(run_dir, main_cfg, style["samples_per_cell"])
    pairs, prompts = h2.matched_pairs(run_dir, cells, sources, style["min_pairs"])
    if not prompts:
        raise SystemExit("No prompt has enough usable rewrites for every student. Check the rewrites.")
    texts, shuffled, index, counts = build_texts(pairs, prompts, cells, style, section)
    matrix = embed(texts, shuffled, style["embedder"])

    columns = defaultdict(list)
    for pid in prompts:
        for stage in ALL_STAGES:
            table = prompt_table(matrix, index, pid, [(stage, cell) for cell in cells])

            def dist(first, second, table=table, stage=stage):
                return table[((stage, first), (stage, second))]

            for k, v in student_columns(dist, bases, teachers, seeds).items():
                columns[f"{stage} {k}"].append(v)

    point, draws = bootstrap_means({k: np.array(v) for k, v in columns.items()}, style["bootstrap"], style["bootstrap_seed"])
    estimate, sampled = derive(point, bases, teachers), [derive(d, bases, teachers) for d in draws]
    summary = {}
    for group, values in estimate.items():
        summary[group] = {}
        for name, value in values.items():
            low, high = percentile_interval([s[group][name] for s in sampled], style["ci_level"])
            summary[group][name] = {"estimate": float(value), "ci_low": low, "ci_high": high}
    return {
        "note": "main_order_shuffle: H2 repeated with a third stage whose rewritten sentences are shuffled; no state",
        "embedder": style["embedder"],
        "prompts_used": len(prompts),
        "answers_used": sum(len(pairs[c][p]) for c in cells for p in prompts),
        "rewrites_with_two_or_more_sentences": float(np.mean([c >= 2 for c in counts])),
        "mean_sentences_per_rewrite": float(np.mean(counts)),
        **summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for name in ("main_grid", h2.SECTION, SECTION):
        verify_lock(cfg, args.run_dir, name)
    out = analyze(args.run_dir, cfg)
    path = results_dir(args.run_dir, MAIN, "style_norm") / "order_shuffle.json"
    write_json(path, out)
    pooled = out["pooled"]
    for name in ("S written", "S rewritten", "S shuffled", "R", "R shuffled", "R shuffled - R", "carryover rewritten", "carryover shuffled"):
        v = pooled[name]
        print(f"  {name:22s} {v['estimate']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]")
    print(f"  rewrites with two or more sentences {out['rewrites_with_two_or_more_sentences']:.3f}; "
          f"mean sentences {out['mean_sentences_per_rewrite']:.2f}; {out['prompts_used']} prompts -> {path}")


if __name__ == "__main__":
    main()
