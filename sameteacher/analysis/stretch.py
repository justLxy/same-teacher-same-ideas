"""Stretch goals, as written and in plain prose.

  persistence  whether the shared-teacher effect survives a second SFT stage on human answers
  released     whether released models distilled from DeepSeek-R1 onto two base families answer
               more alike than released models of the same families tuned by other developers

Answers, rewrites, prompts and word budgets are handled as in `extra_baselines`; every number is
a function of per-prompt means, so one prompt bootstrap gives joint intervals. The quantities and
readings are defined in the locked `stretch_analysis` config section.

Groups are keyed (base, source, tag): main-grid students (base, teacher, seed), second-stage
students (base, "teacher>human", seed), human-data students (base, "human", seed), released
models (family, source, 0).

Usage: python3 -m sameteacher.analysis.stretch {persistence | released}
Results: runs/results/main/stretch/{family}.json
"""
import argparse
import itertools
from collections import defaultdict

import numpy as np

from sameteacher.analysis.extra_baselines import family_texts, mean
from sameteacher.analysis.main_shared_teacher import prompt_table
from sameteacher.analysis.teacher_match import embed_for
from sameteacher.core.config import add_common_args, load_yaml, read_json, read_jsonl, write_json
from sameteacher.core.grid import main_cells
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, results_dir, students_path
from sameteacher.core.text import LIST_MARKER, MARKDOWN, clean
from sameteacher.data import extra_students, stretch_students
from sameteacher.eval.embed import rows_dense
from sameteacher.eval.stats import bootstrap_means, percentile_interval
from sameteacher.style_norm import main_retention as h2
from sameteacher.style_norm.pilot_retention import STAGES, source_hits
from sameteacher.style_norm.rewrite_students import rewrites_path

SECTION = "stretch_analysis"
FAMILIES = ("persistence", "released")
HUMAN, AFTER = "human", ">human"
WRITTEN, REWRITTEN = STAGES


# ---------------------------------------------------------------- groups and answers

def family_groups(run_dir, cfg, family) -> dict:
    """{(base, source, tag): (stage, student name)} for the students one family compares."""
    runs = read_json(stretch_students.manifest_path(run_dir))["runs"]
    if family == "released":
        return {(r["base"], r["source"], 0): (r["stage"], r["student"]) for r in runs if r["stage"] == "released"}
    spec = cfg[stretch_students.SECTION]["persistence"]
    wanted = lambda c: c["base"] in spec["bases"] and c["teacher"] in spec["teachers"] and c["seed"] in spec["seeds"]
    groups = {(c["base"], c["teacher"], c["seed"]): (MAIN, c["student"]) for c in main_cells(cfg["main_grid"]) if wanted(c)}
    groups |= {(r["base"], r["teacher"] + AFTER, r["seed"]): (r["stage"], r["student"]) for r in runs if r["stage"] == "persist"}
    groups |= {(b, HUMAN, s): ("human", extra_students.human_name(b, s)) for b in spec["bases"] for s in spec["seeds"]}
    return groups


def family_pairs(run_dir, cfg, groups) -> tuple[dict, list[str]]:
    section = cfg[h2.SECTION]
    students = [(stage, name, key) for key, (stage, name) in groups.items()]
    sources = h2.select_student_sources(run_dir, students, section["samples_per_cell"])
    files = [h2.rewrites_path(run_dir), rewrites_path(run_dir, "extra"), rewrites_path(run_dir, "stretch")]
    return h2.matched_pairs(run_dir, list(groups), sources, section["min_pairs"], [f for f in files if f.exists()])


# ---------------------------------------------------------------- per-prompt columns

def persistence_columns(d, cfg) -> dict:
    spec = cfg[stretch_students.SECTION]["persistence"]
    (x, y), (a, b), seeds = spec["bases"], spec["teachers"], spec["seeds"]
    out = {}
    for label, tag in (("before", ""), ("after", AFTER)):
        def cross(p, q, tag=tag):
            return mean((d((x, p + tag, i), (y, q + tag, j)) + d((y, p + tag, i), (x, q + tag, j))) / 2 for i in seeds for j in seeds)

        out[f"cross ab {label}"] = (cross(a, b) + cross(b, a)) / 2
        out[f"cross same {label}"] = (cross(a, a) + cross(b, b)) / 2
        out[f"T {label}"] = mean(d((z, a + tag, i), (z, b + tag, j)) for z in (x, y) for i in seeds for j in seeds)
        out[f"N {label}"] = mean(d((z, t + tag, i), (z, t + tag, j)) for z in (x, y) for t in (a, b) for i, j in itertools.combinations(seeds, 2))
        out[f"H {label}"] = mean(d((z, t + tag, i), (z, HUMAN, j)) for z in (x, y) for t in (a, b) for i in seeds for j in seeds)
    out["cross human"] = mean(d((x, HUMAN, i), (y, HUMAN, j)) for i in seeds for j in seeds)
    out["N human"] = mean(d((z, HUMAN, i), (z, HUMAN, j)) for z in (x, y) for i, j in itertools.combinations(seeds, 2))
    return out


def released_columns(d, cfg) -> dict:
    (q, l) = cfg[stretch_students.SECTION]["released"]["models"]
    r1, other = "r1", "other"
    return {
        "shared": d((q, r1, 0), (l, r1, 0)),
        "other pair": d((q, other, 0), (l, other, 0)),
        "mixed": (d((q, r1, 0), (l, other, 0)) + d((q, other, 0), (l, r1, 0))) / 2,
        f"T {q}": d((q, r1, 0), (q, other, 0)),
        f"T {l}": d((l, r1, 0), (l, other, 0)),
    }


COLUMNS = {"persistence": persistence_columns, "released": released_columns}


# ---------------------------------------------------------------- derived quantities

def persistence_derived(m, cfg) -> dict:
    out = {}
    for label in ("before", "after"):
        s, t, n = m[f"cross ab {label}"] - m[f"cross same {label}"], m[f"T {label}"], m[f"N {label}"]
        out |= {f"S {label}": s, f"T {label}": t, f"N {label}": n, f"carryover {label}": s / (t - n),
                f"G {label}": m[f"cross same {label}"] - n, f"H {label}": m[f"H {label}"]}
    out["G human"] = m["cross human"] - m["N human"]
    out |= {
        "P": out["S after"] / out["S before"],
        "S after - before": out["S after"] - out["S before"],
        "carryover after - before": out["carryover after"] - out["carryover before"],
        "T after / before": out["T after"] / out["T before"],
        "G after - before": out["G after"] - out["G before"],
        "G after - human": out["G after"] - out["G human"],
        "H after / before": out["H after"] / out["H before"],
    }
    return out


def released_derived(m, cfg) -> dict:
    families = list(cfg[stretch_students.SECTION]["released"]["models"])
    t = mean(m[f"T {f}"] for f in families)
    s = m["mixed"] - (m["shared"] + m["other pair"]) / 2
    out = {"S": s, "shared minus other": m["other pair"] - m["shared"], "T": t, "S/T": s / t,
           "d shared": m["shared"], "d other pair": m["other pair"], "d mixed": m["mixed"]}
    return out | {f"T {f}": m[f"T {f}"] for f in families}


DERIVED = {"persistence": persistence_derived, "released": released_derived}


def derive(m, cfg, family) -> dict:
    """The family's quantities in both stages, and R = rewritten / written for each S."""
    out = {}
    for stage in STAGES:
        view = {k[len(stage) + 1:]: v for k, v in m.items() if k.startswith(stage + " ")}
        out |= {f"{stage} {k}": v for k, v in DERIVED[family](view, cfg).items()}
    for key in [k for k in out if k.startswith(f"{WRITTEN} S") and " - " not in k and "/" not in k and "minus" not in k]:
        name = key[len(WRITTEN) + 1:]
        out[f"R{name[1:]}"] = out[f"{REWRITTEN} {name}"] / out[key]
    return out


def summarize(columns, cfg, family) -> dict:
    section = cfg[SECTION]
    point, draws = bootstrap_means({k: np.array(v) for k, v in columns.items()}, section["bootstrap"], section["bootstrap_seed"])
    estimate = derive(point, cfg, family)
    sampled = [derive(d, cfg, family) for d in draws]
    out = {}
    for key, value in estimate.items():
        low, high = percentile_interval([s[key] for s in sampled], section["ci_level"])
        out[key] = {"estimate": float(value), "ci_low": low, "ci_high": high}
    return out


# ---------------------------------------------------------------- readings

def persistence_reading(out, stage, section) -> str:
    before, after, p = out[f"{stage} S before"], out[f"{stage} S after"], out[f"{stage} P"]
    if before["ci_low"] <= 0:
        return "NO_EFFECT_BEFORE"
    if after["ci_low"] <= 0:
        return "ERASED"
    if p["ci_low"] >= section["survive_at"]:
        return "SURVIVES"
    if p["ci_high"] <= section["erased_at"]:
        return "MOSTLY_ERASED"
    return "PARTLY_SURVIVES"


def released_reading(out, cfg) -> dict:
    if out[f"{WRITTEN} S"]["ci_low"] <= 0:
        return {"state": "NOT_SUPPORTED", "R reading": None}
    style = cfg[h2.SECTION]
    r = out["R"]
    if r["ci_low"] >= style["most_low"]:
        reading = "CONTENT"
    elif r["ci_high"] <= style["voice_high"]:
        reading = "VOICE"
    else:
        reading = "MIXED"
    return {"state": "SUPPORTED", "R reading": reading}


# ---------------------------------------------------------------- style, answer rate, source match

def label_of(key, family) -> str:
    base, source, tag = key
    return f"{base} {source}" if family == "released" else (source if source == HUMAN else f"{source.replace(AFTER, ' then human')}")


def style(pairs, prompts, keys, family) -> dict:
    texts = defaultdict(lambda: ([], []))
    for key in keys:
        for pid in prompts:
            for pair in pairs[key][pid]:
                for s in range(len(STAGES)):
                    texts[label_of(key, family)][s].append(pair[s])
    out = {}
    for label, stages in sorted(texts.items()):
        for s, stage in enumerate(STAGES):
            out[f"{label} {stage}"] = {
                "mean_words": mean(len(t.split()) for t in stages[s]),
                "formatted_share": mean(bool(MARKDOWN.search(t) or LIST_MARKER.search(t) or "\n" in t.strip()) for t in stages[s]),
            }
    return out


def answer_rates(run_dir, groups) -> dict:
    """Share of samples with a closed answer, and mean reasoning and answer words, per released model."""
    out = {}
    for key, (stage, name) in groups.items():
        rows = read_jsonl(students_path(run_dir, stage, name))
        reasoning = [r["reasoning_words"] for r in rows if r.get("reasoning_words") is not None]
        out[f"{key[0]} {key[1]}"] = {
            "samples": len(rows),
            "answer_share": mean(bool(r.get("answer_closed", True) and r["text"].strip()) for r in rows),
            "mean_reasoning_words": mean(reasoning) if reasoning else None,
            "mean_full_answer_words": mean(r.get("answer_words", len(r["text"].split())) for r in rows),
        }
    return out


def analyze(run_dir, cfg, family) -> dict:
    groups = family_groups(run_dir, cfg, family)
    pairs, prompts = family_pairs(run_dir, cfg, groups)
    if not prompts:
        raise SystemExit(f"No prompt has enough usable rewrites for every {family} student. Check the samples and rewrites.")
    keys = list(groups)
    texts, index = family_texts(pairs, prompts, keys, cfg[h2.SECTION]["min_words"])
    matrix = embed_for(run_dir, texts, index, cfg[SECTION]["embedder"], f"stretch_{family}",
                       note=f"Rows are the {family} family texts of stretch_analysis, both stages, one word budget per prompt. Keys: prompt|stage|base|source|tag.")

    columns, hits = defaultdict(list), []
    for pid in prompts:
        for stage in STAGES:
            table = prompt_table(matrix, index, pid, [(stage, *k) for k in keys])

            def d(first, second, table=table, stage=stage):
                return table[((stage, *first), (stage, *second))]

            for name, value in COLUMNS[family](d, cfg).items():
                columns[f"{stage} {name}"].append(value)
        for key in keys:
            written, rewritten = (rows_dense(matrix, index[(pid, (stage, *key))]) for stage in STAGES)
            hits += source_hits(written, rewritten, [clean(pair[0]) for pair in pairs[key][pid]])

    out = {
        "family": family,
        "embedder": cfg[SECTION]["embedder"],
        "rewriter": cfg[h2.SECTION]["rewriter"],
        "students": {"|".join(map(str, k)): name for k, (_, name) in groups.items()},
        "prompts_used": len(prompts),
        "answers_used": sum(len(pairs[k][p]) for k in keys for p in prompts),
        "source_match": mean(hits),
        **summarize(columns, cfg, family),
        "style": style(pairs, prompts, keys, family),
    }
    if family == "persistence":
        out = {"state": persistence_reading(out, WRITTEN, cfg[SECTION]), "reading rewritten": persistence_reading(out, REWRITTEN, cfg[SECTION]), **out}
    else:
        out = {**released_reading(out, cfg), **out, "answers": answer_rates(run_dir, groups)}
    return out


HEADLINE = {
    "persistence": ("S before", "S after", "P", "carryover before", "carryover after", "G after - human", "H after / before", "R before", "R after"),
    "released": ("S", "shared minus other", "S/T", "R"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("family", choices=FAMILIES)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for name in ("main_grid", h2.SECTION, extra_students.SECTION, stretch_students.SECTION, SECTION):
        verify_lock(cfg, args.run_dir, name)
    result = analyze(args.run_dir, cfg, args.family)
    path = results_dir(args.run_dir, MAIN, "stretch") / f"{args.family}.json"
    write_json(path, result)
    print(f"{args.family}: {result['state']}" + (f", rewritten {result['reading rewritten']}" if "reading rewritten" in result else f", R {result.get('R reading')}"))
    for key, value in result.items():
        name = key.split(" ", 1)[-1] if key.startswith(STAGES) else key
        if isinstance(value, dict) and "ci_low" in value and name in HEADLINE[args.family]:
            print(f"  {key:38s} {value['estimate']:+.4f} [{value['ci_low']:+.4f}, {value['ci_high']:+.4f}]")
    print(f"source match {result['source_match']:.3f}, {result['answers_used']} answers on {result['prompts_used']} prompts")
    print(path)


if __name__ == "__main__":
    main()
