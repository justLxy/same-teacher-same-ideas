"""H3: a student trained on one teacher's answers rewritten in another teacher's style.

For a student c with content teacher a and style teacher s on base b and seed k, the reference
groups are the main-grid students on base b of teacher a (C) and of teacher s (S), from the
main-grid seeds other than k. On one measure d,
    index = [d(c, S) - d(c, C)] / d(S, C)
is +1 when c sits with its content teacher's students and -1 when it sits with its style
teacher's students. The measures are energy distance on the answers rewritten into plain prose
(normalized) and as written, and absolute differences in mean words (length) and in the share of
formatted answers (formatting) on the full written answers. Self-rewrite students (s = a) get the
same index with S = the other teacher's students, as the control for rewriting alone.

Every number is a function of per-prompt means, so one prompt bootstrap gives joint intervals.
The rule is in the locked `h3_analysis` config section.
"""
import argparse
import itertools
from collections import defaultdict

import numpy as np

from sameteacher.analysis.extra_baselines import family_texts
from sameteacher.analysis.main_shared_teacher import prompt_table
from sameteacher.analysis.teacher_match import embed_for
from sameteacher.core.config import add_common_args, load_yaml, read_json, write_json
from sameteacher.core.grid import main_cells
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, results_dir
from sameteacher.core.text import LIST_MARKER, MARKDOWN, clean
from sameteacher.data import h3_data
from sameteacher.eval.embed import rows_dense
from sameteacher.eval.stats import bootstrap_means, percentile_interval
from sameteacher.style_norm import main_retention as h2
from sameteacher.style_norm.pilot_retention import STAGES, source_hits
from sameteacher.style_norm.rewrite_students import rewrites_path

SECTION = "h3_analysis"
# Measure name -> per-prompt column kind; energy measures use the stage's distance table.
MEASURES = {"normalized": STAGES[1], "written": STAGES[0], "length": "words", "formatting": "formatted"}


def mean(values) -> float:
    return float(np.mean(list(values)))


def formatted(text: str) -> bool:
    return bool(MARKDOWN.search(text) or LIST_MARKER.search(text) or "\n" in text.strip())


def name_of(key) -> str:
    return "|".join(map(str, key))


def design(run_dir, cfg) -> tuple[dict, dict]:
    """Groups {(base, source, seed): (stage, student)} and, per H3 student, its run and (C, S) references."""
    section = cfg[h3_data.SECTION]
    seeds = cfg["main_grid"]["seeds"]
    groups = {(c["base"], c["teacher"], c["seed"]): (MAIN, c["student"]) for c in main_cells(cfg["main_grid"])
              if c["base"] in section["bases"] and c["teacher"] in section["teachers"]}
    refs = {}
    for run in read_json(h3_data.manifest_path(run_dir))["runs"]:
        key = (run["base"], f"{run['content']}_in_{run['style']}", run["seed"])
        groups[key] = (run["stage"], run["student"])
        other = run["style"] if run["condition"] == "cross" else next(t for t in section["teachers"] if t != run["content"])
        kept = [s for s in seeds if s != run["seed"]]
        refs[key] = (run, [(run["base"], run["content"], s) for s in kept], [(run["base"], other, s) for s in kept])
    return groups, refs


def pair(kind, first, second) -> str:
    return f"{kind} " + "~".join(sorted((name_of(first), name_of(second))))


def prompt_columns(matrix, index, pid, pairs, keys, refs) -> dict:
    """Energy distances for every pair an index needs, and mean words and formatted share per group."""
    out = {}
    for stage in STAGES:
        table = prompt_table(matrix, index, pid, [(stage, *k) for k in keys])
        for key, (_, content, style) in refs.items():
            for g in content + style:
                out[pair(stage, key, g)] = table[((stage, *key), (stage, *g))]
            for g, h in itertools.product(style, content):
                out[pair(stage, g, h)] = table[((stage, *g), (stage, *h))]
    for key in keys:
        written = [p[0] for p in pairs[key][pid]]
        out[f"words {name_of(key)}"] = mean(len(t.split()) for t in written)
        out[f"formatted {name_of(key)}"] = mean(formatted(t) for t in written)
    return out


def student_index(m, key, content, style, kind) -> float:
    if kind in STAGES:
        def d(g, h):
            return m[pair(kind, g, h)]
    else:
        def d(g, h):
            return abs(m[f"{kind} {name_of(g)}"] - m[f"{kind} {name_of(h)}"])
    to_style = mean(d(key, g) for g in style)
    to_content = mean(d(key, g) for g in content)
    between = mean(d(g, h) for g, h in itertools.product(style, content))
    return (to_style - to_content) / between


def pools(refs) -> dict:
    """Pooled groups of H3 students: cross (all, per direction, per base) and self (all, per teacher)."""
    out = defaultdict(list)
    for key, (run, _, _) in refs.items():
        if run["condition"] == "cross":
            out["cross pooled"].append(key)
            out[f"cross {run['content']} in {run['style']}"].append(key)
            out[f"cross base {run['base']}"].append(key)
        else:
            out["self pooled"].append(key)
            out[f"self {run['content']}"].append(key)
    return dict(out)


def derive(m, refs) -> dict:
    out = {}
    for key, (run, content, style) in refs.items():
        for measure, kind in MEASURES.items():
            out[f"{measure} {run['student']}"] = student_index(m, key, content, style, kind)
    for group, members in pools(refs).items():
        for measure in MEASURES:
            out[f"{measure} {group}"] = mean(out[f"{measure} {refs[k][0]['student']}"] for k in members)
    return out


def state(summary) -> str:
    content = summary["normalized cross pooled"]["ci_low"] > 0
    style = summary["length cross pooled"]["ci_high"] < 0 and summary["formatting cross pooled"]["ci_high"] < 0
    return "SUPPORTED" if content and style else "PARTIAL" if content or style else "NOT_SUPPORTED"


def analyze(run_dir, cfg, embedder=None, source_match: bool = True) -> dict:
    """The locked analysis; `embedder` replaces the section's embedder (main_robustness), and
    `source_match` False skips that mpnet check of the rewriter."""
    section, norm = cfg[SECTION], cfg[h2.SECTION]
    embedder = embedder or section["embedder"]
    groups, refs = design(run_dir, cfg)
    keys = list(groups)
    students = [(stage, name, key) for key, (stage, name) in groups.items()]
    sources = h2.select_student_sources(run_dir, students, norm["samples_per_cell"])
    pairs, prompts = h2.matched_pairs(run_dir, keys, sources, norm["min_pairs"], [h2.rewrites_path(run_dir), rewrites_path(run_dir, "h3")])
    if not prompts:
        raise SystemExit("No prompt has enough usable rewrites for every H3 and reference student. Check the samples and rewrites.")
    texts, index = family_texts(pairs, prompts, keys, norm["min_words"])
    matrix = embed_for(run_dir, texts, index, embedder, "main_h3",
                       note="Rows are the h3_analysis texts, both stages, one word budget per prompt. Keys: prompt|stage|base|source|seed.")

    columns, hits = defaultdict(list), defaultdict(list)
    for pid in prompts:
        for name, value in prompt_columns(matrix, index, pid, pairs, keys, refs).items():
            columns[name].append(value)
        for key in refs if source_match else ():
            written, rewritten = (rows_dense(matrix, index[(pid, (stage, *key))]) for stage in STAGES)
            hits[key] += source_hits(written, rewritten, [clean(p[0]) for p in pairs[key][pid]])

    point, draws = bootstrap_means({k: np.array(v) for k, v in columns.items()}, section["bootstrap"], section["bootstrap_seed"])
    estimate, sampled = derive(point, refs), [derive(d, refs) for d in draws]
    summary = {}
    for name, value in estimate.items():
        low, high = percentile_interval([s[name] for s in sampled], section["ci_level"])
        summary[name] = {"estimate": value, "ci_low": low, "ci_high": high}

    style = {}
    for key, (stage, student) in groups.items():
        written = [p[0] for pid in prompts for p in pairs[key][pid]]
        style[student] = {"mean_words": mean(len(t.split()) for t in written), "formatted_share": mean(formatted(t) for t in written)}
    return {
        "state": state(summary),
        "embedder": embedder,
        "rewriter": norm["rewriter"],
        "prompts_used": len(prompts),
        "answers_used": sum(len(pairs[k][p]) for k in keys for p in prompts),
        "source_match": mean(h for v in hits.values() for h in v) if source_match else None,
        "source_match_per_student": {refs[k][0]["student"]: mean(v) for k, v in hits.items()} if source_match else None,
        "references": {refs[k][0]["student"]: {"content": [groups[g][1] for g in c], "style": [groups[g][1] for g in s]} for k, (_, c, s) in refs.items()},
        **summary,
        "style": style,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for name in ("main_grid", h2.SECTION, h3_data.SECTION, SECTION):
        verify_lock(cfg, args.run_dir, name)
    result = analyze(args.run_dir, cfg)
    path = results_dir(args.run_dir, MAIN, "h3") / "h3.json"
    write_json(path, result)
    print(f"H3: {result['state']}")
    for key, value in result.items():
        if isinstance(value, dict) and "ci_low" in value and (" cross " in key or " self " in key):
            print(f"{key:40s} {value['estimate']:+.3f} [{value['ci_low']:+.3f}, {value['ci_high']:+.3f}]")
    print(f"source match {result['source_match']:.3f}, {result['answers_used']} answers on {result['prompts_used']} prompts")
    print(path)


if __name__ == "__main__":
    main()
