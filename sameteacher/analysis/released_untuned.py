"""The released pairs next to the untuned bases they start from (locked `released_untuned` section).

The four released models of stretch_students.released and their three untuned bases enter one
analysis: the prompts where every group has enough usable source and rewrite pairs, one word
budget per prompt across all groups and both stages, energy distances on embeddings averaged over
prompts, and one prompt bootstrap for every quantity. If too few prompts pass with all seven
groups, the R1 pair is left out and the other pair is analyzed with its own four groups.

Usage: python3 -m sameteacher.analysis.released_untuned
Result: runs/results/main/stretch/released_untuned.json
"""
import argparse
import statistics
from collections import defaultdict

import numpy as np

from sameteacher.analysis.extra_baselines import family_texts, mean
from sameteacher.analysis.main_shared_teacher import prompt_table
from sameteacher.analysis.stretch import style
from sameteacher.analysis.teacher_match import embed_for
from sameteacher.core.config import add_common_args, load_yaml, read_json, read_jsonl, write_json
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, results_dir, students_path
from sameteacher.core.text import clean, common_word_budget, is_answer
from sameteacher.data import extra_students, released_untuned, stretch_students
from sameteacher.eval.embed import rows_dense
from sameteacher.eval.stats import bootstrap_means, percentile_interval
from sameteacher.style_norm import main_retention as h2
from sameteacher.style_norm.pilot_retention import STAGES, source_hits
from sameteacher.style_norm.rewrite_students import rewrites_path

SECTION = released_untuned.SECTION
UNTUNED = "untuned"
# Each pair: the two released models and the two untuned bases they start from, as group keys.
PAIRS = {
    "r1": ((("qwen", "r1", 0), ("llama", "r1", 0)), (("qwen_math", UNTUNED, 0), ("llama", UNTUNED, 0))),
    "other": ((("qwen", "other", 0), ("llama", "other", 0)), (("qwen", UNTUNED, 0), ("llama", UNTUNED, 0))),
}


def groups_of(run_dir) -> dict:
    """{(family or base, source, 0): (stage, student)} for the released models and their bases."""
    released = read_json(stretch_students.manifest_path(run_dir))["runs"]
    out = {(r["base"], r["source"], 0): (r["stage"], r["student"]) for r in released if r["stage"] == "released"}
    untuned = read_json(released_untuned.manifest_path(run_dir))["runs"]
    return out | {(r["base"], UNTUNED, 0): (r["stage"], r["student"]) for r in untuned}


def usable_pairs(run_dir, cfg, groups) -> dict:
    style_cfg = cfg[h2.SECTION]
    students = [(stage, name, key) for key, (stage, name) in groups.items()]
    sources = h2.select_student_sources(run_dir, students, style_cfg["samples_per_cell"])
    files = [rewrites_path(run_dir, "stretch"), rewrites_path(run_dir, SECTION)]
    pairs, _ = h2.matched_pairs(run_dir, list(groups), sources, style_cfg["min_pairs"], [f for f in files if f.exists()])
    return pairs


def passing(pairs, keys, min_pairs) -> list[str]:
    sets = [{p for p, v in pairs[k].items() if len(v) >= min_pairs} for k in keys]
    return sorted(set.intersection(*sets)) if sets else []


def choose(pairs, cfg) -> tuple[list[str], list, list[str]]:
    """The pairs analyzed, their groups and the prompts every one of those groups passes."""
    min_pairs, section = cfg[h2.SECTION]["min_pairs"], cfg[SECTION]
    every = sorted({k for name in PAIRS for side in PAIRS[name] for k in side})
    prompts = passing(pairs, every, min_pairs)
    if len(prompts) >= section["min_prompts"]:
        return list(PAIRS), every, prompts
    keys = [k for side in PAIRS["other"] for k in side]
    return ["other"], keys, passing(pairs, keys, min_pairs)


def derive(m, names) -> dict:
    out = {}
    for stage in STAGES:
        for name in names:
            d, u = m[f"{stage} d {name}"], m[f"{stage} U {name}"]
            out |= {f"{stage} d {name}": d, f"{stage} U {name}": u, f"{stage} change {name}": d - u}
        if len(names) == 2:
            out[f"{stage} change r1 - other"] = out[f"{stage} change r1"] - out[f"{stage} change other"]
            out[f"{stage} U r1 - other"] = out[f"{stage} U r1"] - out[f"{stage} U other"]
            out[f"{stage} d r1 - other"] = out[f"{stage} d r1"] - out[f"{stage} d other"]
    return out


def summarize(columns, names, section) -> dict:
    point, draws = bootstrap_means({k: np.array(v) for k, v in columns.items()}, section["bootstrap"], section["bootstrap_seed"])
    estimate, sampled = derive(point, names), [derive(d, names) for d in draws]
    out = {}
    for key, value in estimate.items():
        low, high = percentile_interval([s[key] for s in sampled], section["ci_level"])
        out[key] = {"estimate": float(value), "ci_low": low, "ci_high": high}
    return out


def budgets(pairs, prompts, keys, min_words) -> list[int]:
    """The per-prompt word budgets that family_texts applies."""
    return [common_word_budget([[clean(pair[s]) for pair in pairs[k][pid]] for s in range(len(STAGES)) for k in keys], min_words)
            for pid in prompts]


def usable(run_dir, cfg, groups, pairs) -> dict:
    """For every group: samples, share that are answers, share reaching the cut, prompts with enough pairs."""
    cut, min_pairs = cfg[stretch_students.SECTION]["released"]["answer_words"], cfg[h2.SECTION]["min_pairs"]
    out = {}
    for key, (stage, name) in groups.items():
        rows = read_jsonl(students_path(run_dir, stage, name))
        out[f"{key[0]} {key[1]}"] = {
            "samples": len(rows),
            "answer_share": mean(is_answer(r["text"]) for r in rows),
            "reaching_cut_share": mean(r.get("answer_words", len(r["text"].split())) >= cut for r in rows),
            "prompts_with_min_pairs": sum(len(v) >= min_pairs for v in pairs[key].values()),
        }
    return out


def analyze(run_dir, cfg) -> dict:
    section, min_words = cfg[SECTION], cfg[h2.SECTION]["min_words"]
    groups = groups_of(run_dir)
    pairs = usable_pairs(run_dir, cfg, groups)
    names, keys, prompts = choose(pairs, cfg)
    if not prompts:
        raise SystemExit("No prompt has enough usable rewrites in every group. Check the samples and rewrites.")
    texts, index = family_texts(pairs, prompts, keys, min_words)
    matrix = embed_for(run_dir, texts, index, section["embedder"], SECTION,
                       note="Rows are the released models and their untuned bases, both stages, one word budget per prompt. Keys: prompt|stage|base|source|tag.")

    columns, hits = defaultdict(list), []
    for pid in prompts:
        for stage in STAGES:
            table = prompt_table(matrix, index, pid, [(stage, *k) for k in keys])
            for name in names:
                (a, b), (c, e) = PAIRS[name]
                columns[f"{stage} d {name}"].append(table[((stage, *a), (stage, *b))])
                columns[f"{stage} U {name}"].append(table[((stage, *c), (stage, *e))])
        for key in keys:
            written, rewritten = (rows_dense(matrix, index[(pid, (stage, *key))]) for stage in STAGES)
            hits += source_hits(written, rewritten, [clean(pair[0]) for pair in pairs[key][pid]])

    words = budgets(pairs, prompts, keys, min_words)
    return {
        "embedder": section["embedder"],
        "rewriter": cfg[h2.SECTION]["rewriter"],
        "students": {"|".join(map(str, k)): name for k, (_, name) in groups.items()},
        "pairs_analyzed": names,
        "r1_left_out": "r1" not in names,
        "prompts_used": len(prompts),
        "answers_used": sum(len(pairs[k][p]) for k in keys for p in prompts),
        "word_budget": {"median": statistics.median(words), "min": min(words), "max": max(words)},
        "source_match": mean(hits),
        **summarize(columns, names, section),
        "style": style(pairs, prompts, keys, "released"),
        "usable": usable(run_dir, cfg, groups, pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for name in ("main_grid", h2.SECTION, extra_students.SECTION, stretch_students.SECTION, "stretch_analysis", SECTION):
        verify_lock(cfg, args.run_dir, name)
    out = analyze(args.run_dir, cfg)
    path = results_dir(args.run_dir, MAIN, "stretch") / f"{SECTION}.json"
    write_json(path, out)
    print(f"pairs {out['pairs_analyzed']}; {out['prompts_used']} prompts; word budget median {out['word_budget']['median']}")
    for key in [k for k in out if k.split(" ", 1)[0] in STAGES]:
        v = out[key]
        print(f"  {key:32s} {v['estimate']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]")
    print(path)


if __name__ == "__main__":
    main()
