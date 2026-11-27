"""Do H1 and H2 depend on cutting every answer to its prompt's word budget? (locked `main_cut_robustness`)

The locked H1 and H2 analyses cut each cleaned answer to the first `budget` words of its prompt,
which drops more of the longer answers and ends most cut answers inside a sentence. This repeats
both analyses on their own prompts, answers, budgets, embedder and bootstrap under each cut in
the section: `words` (the locked cut, which reproduces h1.json and h2.json), `sentences` (whole
sentences from the start up to the budget) and `full` (no cut). One prompt bootstrap per analysis
covers every cut, so the differences from `words` have their own intervals.

Usage: python3 -m sameteacher.analysis.cut_robustness [--parts h1 h2]
Result: runs/results/main/robustness/cut.json
"""
import argparse
import hashlib
import json
from collections import defaultdict

import numpy as np

from sameteacher.analysis import main_shared_teacher as h1
from sameteacher.analysis.order_shuffle import sentences
from sameteacher.analysis.teacher_match import cache_paths, embed_cached
from sameteacher.core.config import add_common_args, load_yaml, write_json
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, results_dir
from sameteacher.core.text import clean, common_word_budget, first_words
from sameteacher.data.teacher_answers import load_groups
from sameteacher.eval.embed import embed_texts
from sameteacher.eval.stats import bootstrap_means, percentile_interval
from sameteacher.style_norm import main_retention as h2
from sameteacher.style_norm.pilot_retention import STAGES

SECTION = "main_cut_robustness"
FIRST_CUT = "words"


def sentence_cut(text: str, budget: int) -> tuple[str, bool]:
    """Whole sentences from the start while their words fit the budget, and whether the first
    sentence alone was longer than the budget (it is then kept whole)."""
    kept, used = [], 0
    for part in sentences(text):
        words = len(part.split())
        if not kept and words > budget:
            return part, True
        if used + words > budget:
            break
        kept.append(part)
        used += words
    return " ".join(kept), False


def cut_text(text: str, budget: int, cut: str) -> tuple[str, bool]:
    if cut == "words":
        return first_words(text, budget), False
    if cut == "sentences":
        return sentence_cut(text, budget)
    if cut == "full":
        return text, False
    raise ValueError(f"unknown cut {cut}")


def h1_groups(run_dir, cfg) -> dict:
    """The H1 answer sets per prompt, cleaned and uncut, chosen exactly as main_shared_teacher does."""
    main_cfg, section = cfg["main_grid"], cfg[h1.SECTION]
    minimum = section["min_usable_per_group"]
    students = h1.load_students(run_dir, main_cfg)
    teacher_groups, _, _ = load_groups(run_dir, main_cfg["test_split"], 2)
    groups_by_prompt = {}
    candidates = set.intersection(*[{p for p, v in s.items() if len(v) >= minimum} for s in students.values()])
    for pid in sorted(candidates):
        if all(len(teacher_groups[t].get(pid, [])) >= minimum for t in main_cfg["teachers"]):
            groups_by_prompt[pid] = {key: s[pid] for key, s in students.items()} | {h1.teacher_key(t): teacher_groups[t][pid] for t in main_cfg["teachers"]}
    return groups_by_prompt


def h2_groups(run_dir, cfg) -> tuple[dict, list]:
    """The H2 answer sets per prompt, (stage, cell) -> cleaned and uncut texts, as main_style_norm chose them."""
    style = cfg[h2.SECTION]
    _, _, _, cells = h2.grid(cfg["main_grid"])
    sources = h2.select_sources(run_dir, cfg["main_grid"], style["samples_per_cell"])
    pairs, prompts = h2.matched_pairs(run_dir, cells, sources, style["min_pairs"])
    if not prompts:
        raise SystemExit("No prompt has enough usable rewrites for every student. Check the rewrites.")
    groups = {pid: {(STAGES[s], cell): [clean(pair[s]) for pair in pairs[cell][pid]] for s in range(len(STAGES)) for cell in cells} for pid in prompts}
    return groups, prompts


def cut_groups(groups_by_prompt: dict, min_words: int, cut: str) -> tuple[list[str], dict, dict]:
    """Texts in group order with one budget per prompt (from the uncut texts), their index, and per
    group the cut texts' word counts and how many answers were kept whole past the budget."""
    texts, index, stats = [], {}, {}
    for pid, groups in groups_by_prompt.items():
        budget = common_word_budget(list(groups.values()), min_words)
        for key, values in groups.items():
            out = [cut_text(t, budget, cut) for t in values]
            index[(pid, key)] = list(range(len(texts), len(texts) + len(out)))
            texts.extend(t for t, _ in out)
            stats[(pid, key)] = ([len(t.split()) for t, _ in out], sum(whole for _, whole in out))
    return texts, index, stats


def embed(run_dir, texts, index, embedder: str, name: str, reuse: str | None = None):
    """Embeddings of the texts. A dense embedder reuses the cache `reuse` when it holds exactly these
    texts, and otherwise caches them under `name`."""
    if embedder == "tfidf":
        return embed_texts(texts, embedder)
    if reuse:
        matrix_path, index_path = cache_paths(run_dir, embedder, reuse)
        digest = hashlib.sha256("\n".join(texts).encode()).hexdigest()
        if matrix_path.exists() and index_path.exists() and json.loads(index_path.read_text())["texts_sha256"] == digest:
            print(f"  reusing the {reuse} embeddings")
            return np.load(matrix_path)
    return embed_cached(run_dir, texts, index, embedder, name=name,
                        note=f"Rows are the {name} texts of main_cut_robustness. Keys: prompt|key.")


def mean_words(stats: dict, keep) -> float:
    counts = [n for key, (words, _) in stats.items() if keep(key[1]) for n in words]
    return float(np.mean(counts))


def whole_share(stats: dict, keep) -> float:
    whole = sum(w for key, (_, w) in stats.items() if keep(key[1]))
    total = sum(len(words) for key, (words, _) in stats.items() if keep(key[1]))
    return whole / total


def interval_summary(estimate: dict, sampled: list[dict], ci_level: float) -> dict:
    out = {}
    for group, values in estimate.items():
        out[group] = {}
        for name, value in values.items():
            low, high = percentile_interval([s[group][name] for s in sampled], ci_level)
            out[group][name] = {"estimate": float(value), "ci_low": low, "ci_high": high}
    return out


def derive_h1(m: dict, bases, teachers, cuts) -> dict:
    q = {c: h1.quantities(h2.stage_view(m, c), bases, teachers) for c in cuts}
    out = {}
    for group in q[FIRST_CUT]:
        row = {}
        for c in cuts:
            for name in ("S", "T", "N", "carryover"):
                row[f"{name} {c}"] = q[c][group][name]
            if c != FIRST_CUT:
                row[f"S {c} - {FIRST_CUT}"] = q[c][group]["S"] - q[FIRST_CUT][group]["S"]
                row[f"carryover {c} - {FIRST_CUT}"] = q[c][group]["carryover"] - q[FIRST_CUT][group]["carryover"]
        out[group] = row
    return out


def derive_h2(m: dict, bases, teachers, cuts) -> dict:
    q = {(c, s): h1.quantities(h2.stage_view(m, f"{c} {s}"), bases, teachers) for c in cuts for s in STAGES}
    out = {}
    for group in q[(FIRST_CUT, STAGES[0])]:
        row = {}
        for c in cuts:
            w, r = q[(c, STAGES[0])][group], q[(c, STAGES[1])][group]
            row |= {f"S written {c}": w["S"], f"S rewritten {c}": r["S"], f"R {c}": r["S"] / w["S"],
                    f"carryover written {c}": w["carryover"], f"carryover rewritten {c}": r["carryover"]}
        for c in cuts:
            if c != FIRST_CUT:
                row[f"R {c} - {FIRST_CUT}"] = row[f"R {c}"] - row[f"R {FIRST_CUT}"]
                row[f"S written {c} - {FIRST_CUT}"] = row[f"S written {c}"] - row[f"S written {FIRST_CUT}"]
        out[group] = row
    return out


def h1_part(run_dir, cfg, cuts) -> dict:
    main_cfg, section = cfg["main_grid"], cfg[h1.SECTION]
    bases, teachers, seeds = list(main_cfg["bases"]), list(main_cfg["teachers"]), list(main_cfg["seeds"])
    students = [(b, t, s) for b in bases for t in teachers for s in seeds]
    groups = h1_groups(run_dir, cfg)
    prompts = list(groups)
    columns, words, whole = {}, {}, {}
    for cut in cuts:
        texts, index, stats = cut_groups(groups, section["min_words"], cut)
        print(f"H1 {cut}: {len(texts)} texts on {len(prompts)} prompts")
        matrix = embed(run_dir, texts, index, section["embedder"], f"cut_h1_{cut}", reuse="main_h1" if cut == FIRST_CUT else None)
        tables = h1.distance_tables(matrix, index, prompts, students)
        for k, v in h1.columns_for(tables, prompts, bases, teachers, seeds, with_teachers=False).items():
            columns[f"{cut} {k}"] = v
        words[cut] = {t: mean_words(stats, lambda key, t=t: key[0] != "teacher" and key[1] == t) for t in teachers}
        if cut == "sentences":
            whole[cut] = whole_share(stats, lambda key: key[0] != "teacher")
        del matrix
    point, draws = bootstrap_means(columns, section["bootstrap"], section["bootstrap_seed"])
    summary = interval_summary(derive_h1(point, bases, teachers, cuts), [derive_h1(d, bases, teachers, cuts) for d in draws], section["ci_level"])
    readings = {c: {"S lower bound above 0": summary["pooled"][f"S {c}"]["ci_low"] > 0,
                    "carryover lower bound at least 0.10": summary["pooled"][f"carryover {c}"]["ci_low"] >= 0.10} for c in cuts}
    return {"prompts_used": len(prompts), "mean_words_of_compared_student_texts": words,
            "share_kept_whole_past_the_budget": whole, "readings": readings, **summary}


def h2_part(run_dir, cfg, cuts) -> dict:
    main_cfg, style = cfg["main_grid"], cfg[h2.SECTION]
    bases, teachers, seeds, cells = h2.grid(main_cfg)
    groups, prompts = h2_groups(run_dir, cfg)
    columns, words, whole = {}, {}, {}
    for cut in cuts:
        texts, index, stats = cut_groups(groups, style["min_words"], cut)
        print(f"H2 {cut}: {len(texts)} texts on {len(prompts)} prompts")
        matrix = embed(run_dir, texts, index, style["embedder"], f"cut_h2_{cut}", reuse="main_h2" if cut == FIRST_CUT else None)
        for pid in prompts:
            for stage in STAGES:
                table = h1.prompt_table(matrix, index, pid, [(stage, cell) for cell in cells])

                def dist(first, second, table=table, stage=stage):
                    return table[((stage, first), (stage, second))]

                for k, v in h1.student_columns(dist, bases, teachers, seeds).items():
                    columns.setdefault(f"{cut} {stage} {k}", []).append(v)
        words[cut] = {f"{t} {s}": mean_words(stats, lambda key, t=t, s=s: key[0] == s and key[1][1] == t) for t in teachers for s in STAGES}
        if cut == "sentences":
            whole[cut] = {s: whole_share(stats, lambda key, s=s: key[0] == s) for s in STAGES}
        del matrix
    point, draws = bootstrap_means({k: np.array(v) for k, v in columns.items()}, style["bootstrap"], style["bootstrap_seed"])
    summary = interval_summary(derive_h2(point, bases, teachers, cuts), [derive_h2(d, bases, teachers, cuts) for d in draws], style["ci_level"])
    pairs = [g for g in summary if g != "pooled"]
    readings = {c: {"R lower bound at least 0.5": summary["pooled"][f"R {c}"]["ci_low"] >= style["most_low"],
                    "every pair's R lower bound at least 0.5": all(summary[g][f"R {c}"]["ci_low"] >= style["most_low"] for g in pairs)} for c in cuts}
    return {"prompts_used": len(prompts), "answers_used": sum(len(v) for g in groups.values() for k, v in g.items() if k[0] == STAGES[0]),
            "mean_words_of_compared_student_texts": words, "share_kept_whole_past_the_budget": whole, "readings": readings, **summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--parts", nargs="+", choices=["h1", "h2"], default=["h1", "h2"])
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    needed = ["main_grid", SECTION] + (["main_shared_teacher"] if "h1" in args.parts else []) + (["main_style_norm"] if "h2" in args.parts else [])
    for name in needed:
        verify_lock(cfg, args.run_dir, name)
    cuts = list(cfg[SECTION]["cuts"])
    out = {"note": "main_cut_robustness: H1 and H2 under each cut of the cleaned answers; no state", "cuts": cuts}
    if "h1" in args.parts:
        out["embedder"] = cfg[h1.SECTION]["embedder"]
        out["h1"] = h1_part(args.run_dir, cfg, cuts)
    if "h2" in args.parts:
        out["embedder"] = cfg[h2.SECTION]["embedder"]
        out["h2"] = h2_part(args.run_dir, cfg, cuts)
    path = results_dir(args.run_dir, MAIN, "robustness") / "cut.json"
    write_json(path, out)
    for part, names in (("h1", ("S", "carryover")), ("h2", ("S written", "R", "carryover rewritten"))):
        if part not in out:
            continue
        for cut in cuts:
            row = "  ".join(f"{n} {out[part]['pooled'][f'{n} {cut}']['estimate']:.4f} [{out[part]['pooled'][f'{n} {cut}']['ci_low']:.4f}, {out[part]['pooled'][f'{n} {cut}']['ci_high']:.4f}]" for n in names)
            print(f"  {part} {cut:9s} {row}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
