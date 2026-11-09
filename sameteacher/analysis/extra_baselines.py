"""Data scaling, human-data students and untuned bases, as written and in plain prose.

  scaling  S, T and carryover at each number of distinct training examples (two bases, two teachers)
  human    human-written training data as a fourth source: the base gap it leaves and its carryover
  untuned  how far apart the bases are before tuning, against the gap left after tuning

Each family takes the main_style_norm answers and rewrites (main-grid students reuse their H2
rewrites), keeps the prompts where every compared student has enough usable rewrites, and cuts
both stages to one word budget per prompt. Every number is a function of per-prompt means, so one
prompt bootstrap gives joint intervals. The quantities are defined in the locked `extra_analysis`
config section; none has a decision rule.

Groups are keyed (base, source, tag): main-grid students (base, teacher, seed), scaling students
(base, teacher, "n500"), human-data students (base, "human", seed), untuned bases (base, "untuned", 0).
"""
import argparse
import itertools
from collections import defaultdict

import numpy as np

from sameteacher.analysis.main_shared_teacher import prompt_table
from sameteacher.analysis.teacher_match import embed_for
from sameteacher.core.config import add_common_args, load_yaml, read_json, write_json
from sameteacher.core.grid import main_cells
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, results_dir
from sameteacher.core.text import LIST_MARKER, MARKDOWN, clean, common_word_budget, first_words
from sameteacher.data import extra_students
from sameteacher.eval.embed import rows_dense
from sameteacher.eval.stats import bootstrap_means, percentile_interval
from sameteacher.style_norm import main_retention as h2
from sameteacher.style_norm.pilot_retention import STAGES, source_hits
from sameteacher.style_norm.rewrite_students import rewrites_path

SECTION = "extra_analysis"
FAMILIES = ("scaling", "human", "untuned")
HUMAN, UNTUNED, MAIN_SIZE = "human", "untuned", "main"


def mean(values) -> float:
    return float(np.mean(list(values)))


# ---------------------------------------------------------------- groups and texts

def family_groups(run_dir, cfg, family) -> dict:
    """{(base, source, tag): (stage, student name)} for the students one family compares."""
    runs = read_json(extra_students.manifest_path(run_dir))["runs"]
    main = {(c["base"], c["teacher"], c["seed"]): (MAIN, c["student"]) for c in main_cells(cfg["main_grid"])}
    if family == "scaling":
        spec = cfg[extra_students.SECTION]["data_scaling"]
        groups = {k: v for k, v in main.items() if k[0] in spec["bases"] and k[1] in spec["teachers"]}
        return groups | {(r["base"], r["teacher"], f"n{r['examples']}"): (r["stage"], r["student"]) for r in runs if r["stage"] == "scaling"}
    stage = {"human": "human", "untuned": "untuned"}[family]
    source = HUMAN if family == "human" else UNTUNED
    return main | {(r["base"], source, r["seed"] if family == "human" else 0): (r["stage"], r["student"]) for r in runs if r["stage"] == stage}


def family_pairs(run_dir, cfg, groups) -> tuple[dict, list[str]]:
    section = cfg[h2.SECTION]
    students = [(stage, name, key) for key, (stage, name) in groups.items()]
    sources = h2.select_student_sources(run_dir, students, section["samples_per_cell"])
    files = [h2.rewrites_path(run_dir), rewrites_path(run_dir, "extra")]
    return h2.matched_pairs(run_dir, list(groups), sources, section["min_pairs"], files)


def family_texts(pairs, prompts, keys, min_words) -> tuple[list[str], dict]:
    """Texts of both stages cut to one word budget per prompt; index keys are (prompt, (stage, *group))."""
    texts, index = [], {}
    for pid in prompts:
        groups = {(stage, *key): [clean(pair[s]) for pair in pairs[key][pid]] for s, stage in enumerate(STAGES) for key in keys}
        budget = common_word_budget(list(groups.values()), min_words)
        for key, values in groups.items():
            start = len(texts)
            texts.extend(first_words(t, budget) for t in values)
            index[(pid, key)] = list(range(start, len(texts)))
    return texts, index


# ---------------------------------------------------------------- per-prompt columns

def scaling_columns(d, cfg) -> dict:
    spec = cfg[extra_students.SECTION]["data_scaling"]
    bases, teachers, seeds = spec["bases"], spec["teachers"], cfg["main_grid"]["seeds"]
    out = {}
    for label, tag in [(f"n{n}", f"n{n}") for n in spec["sizes"]] + [(MAIN_SIZE, spec["seed"])]:
        blocks = [(d((x, a, tag), (y, b, tag)) + d((x, b, tag), (y, a, tag))) / 2 - (d((x, a, tag), (y, a, tag)) + d((x, b, tag), (y, b, tag))) / 2
                  for x, y in itertools.combinations(bases, 2) for a, b in itertools.combinations(teachers, 2)]
        out[f"S {label}"] = mean(blocks)
        out[f"T {label}"] = mean(d((x, a, tag), (x, b, tag)) for x in bases for a, b in itertools.combinations(teachers, 2))
    out["N"] = mean(d((x, t, i), (x, t, j)) for x in bases for t in teachers for i, j in itertools.combinations(seeds, 2))
    return out


def seeds_of(cfg) -> dict:
    seeds = {t: cfg["main_grid"]["seeds"] for t in cfg["main_grid"]["teachers"]}
    return seeds | {HUMAN: cfg[extra_students.SECTION]["human_data"]["seeds"]}


def cross(d, p, q, bases, seeds) -> float:
    return mean(d((x, p, i), (y, q, j)) for x in bases for y in bases if x != y for i in seeds[p] for j in seeds[q])


def within(d, p, q, bases, seeds) -> float:
    return mean(d((x, p, i), (x, q, j)) for x in bases for i in seeds[p] for j in seeds[q])


def noise(d, p, bases, seeds) -> float:
    return mean(d((x, p, i), (x, p, j)) for x in bases for i, j in itertools.combinations(seeds[p], 2))


def human_columns(d, cfg) -> dict:
    bases, seeds = list(cfg["main_grid"]["bases"]), seeds_of(cfg)
    sources = list(cfg["main_grid"]["teachers"]) + [HUMAN]
    out = {}
    for p, q in itertools.combinations_with_replacement(sources, 2):
        out[f"cross {p}-{q}"] = cross(d, p, q, bases, seeds)
        if p != q:
            out[f"within {p}-{q}"] = within(d, p, q, bases, seeds)
    for p in sources:
        out[f"N {p}"] = noise(d, p, bases, seeds)
    return out


def untuned_columns(d, cfg) -> dict:
    bases, teachers, seeds = list(cfg["main_grid"]["bases"]), list(cfg["main_grid"]["teachers"]), cfg["main_grid"]["seeds"]
    out = {}
    for x, y in itertools.combinations(bases, 2):
        out[f"U {x}-{y}"] = d((x, UNTUNED, 0), (y, UNTUNED, 0))
        out[f"same {x}-{y}"] = mean(d((x, t, i), (y, t, j)) for t in teachers for i in seeds for j in seeds)
    for x in bases:
        out[f"N {x}"] = mean(d((x, t, i), (x, t, j)) for t in teachers for i, j in itertools.combinations(seeds, 2))
        out[f"shift {x}"] = mean(d((x, UNTUNED, 0), (x, t, s)) for t in teachers for s in seeds)
    by_teacher = {t: seeds for t in teachers}
    out["different"] = mean(cross(d, a, b, bases, by_teacher) for a, b in itertools.combinations(teachers, 2))
    return out


COLUMNS = {"scaling": scaling_columns, "human": human_columns, "untuned": untuned_columns}


# ---------------------------------------------------------------- derived quantities

def scaling_derived(m, cfg) -> dict:
    labels = [f"n{n}" for n in cfg[extra_students.SECTION]["data_scaling"]["sizes"]] + [MAIN_SIZE]
    out = {"N": m["N"]}
    for label in labels:
        s, t = m[f"S {label}"], m[f"T {label}"]
        out |= {f"S {label}": s, f"T {label}": t, f"carryover {label}": s / (t - m["N"]), f"S/T {label}": s / t}
    for label in labels[:-1]:
        out[f"S {label} - {MAIN_SIZE}"] = out[f"S {label}"] - out[f"S {MAIN_SIZE}"]
        out[f"carryover {label} - {MAIN_SIZE}"] = out[f"carryover {label}"] - out[f"carryover {MAIN_SIZE}"]
    out["S increasing"] = float(all(out[f"S {a}"] < out[f"S {b}"] for a, b in zip(labels, labels[1:])))
    return out


def human_derived(m, cfg) -> dict:
    teachers = list(cfg["main_grid"]["teachers"])
    sources = teachers + [HUMAN]

    def name(kind, p, q):
        p, q = sorted((p, q), key=sources.index)
        return f"{kind} {p}-{q}"

    def shared(p, q):
        return m[name("cross", p, q)] - (m[name("cross", p, p)] + m[name("cross", q, q)]) / 2

    def transfer(p, q):
        return m[name("within", p, q)] - (m[f"N {p}"] + m[f"N {q}"]) / 2

    out = {f"G {p}": m[name("cross", p, p)] - m[f"N {p}"] for p in sources}
    out["G teachers"] = mean(out[f"G {t}"] for t in teachers)
    out["G human - teachers"] = out[f"G {HUMAN}"] - out["G teachers"]
    for t in teachers:
        out[f"S human-{t}"] = shared(HUMAN, t)
        out[f"carryover human-{t}"] = shared(HUMAN, t) / transfer(HUMAN, t)
    out["S human pooled"] = mean(shared(HUMAN, t) for t in teachers)
    out["carryover human pooled"] = sum(shared(HUMAN, t) for t in teachers) / sum(transfer(HUMAN, t) for t in teachers)
    pairs = list(itertools.combinations(teachers, 2))
    out["S teacher pairs"] = mean(shared(a, b) for a, b in pairs)
    out["carryover teacher pairs"] = sum(shared(a, b) for a, b in pairs) / sum(transfer(a, b) for a, b in pairs)
    out["carryover human - teacher pairs"] = out["carryover human pooled"] - out["carryover teacher pairs"]
    return out


def untuned_derived(m, cfg) -> dict:
    bases = list(cfg["main_grid"]["bases"])
    base_pairs = list(itertools.combinations(bases, 2))
    out = {"U": mean(m[f"U {x}-{y}"] for x, y in base_pairs), "same": mean(m[f"same {x}-{y}"] for x, y in base_pairs),
           "different": m["different"], "N": mean(m[f"N {x}"] for x in bases)}
    out |= {"G same": out["same"] - out["N"], "G different": out["different"] - out["N"]}
    out |= {"U - G same": out["U"] - out["G same"], "U - G different": out["U"] - out["G different"]}
    for x, y in base_pairs:
        gap = m[f"same {x}-{y}"] - (m[f"N {x}"] + m[f"N {y}"]) / 2
        out |= {f"U {x}-{y}": m[f"U {x}-{y}"], f"G same {x}-{y}": gap, f"U - G same {x}-{y}": m[f"U {x}-{y}"] - gap}
    out |= {f"shift {x}": m[f"shift {x}"] for x in bases}
    return out


DERIVED = {"scaling": scaling_derived, "human": human_derived, "untuned": untuned_derived}


def derive(m, cfg, family) -> dict:
    """The family's quantities in both stages, and R = rewritten / written for each S."""
    out = {}
    for stage in STAGES:
        view = {k[len(stage) + 1:]: v for k, v in m.items() if k.startswith(stage + " ")}
        out |= {f"{stage} {k}": v for k, v in DERIVED[family](view, cfg).items()}
    for key in [k for k in out if k.startswith(f"{STAGES[0]} S ") and " - " not in k and not k.endswith("increasing")]:
        name = key[len(STAGES[0]) + 1:]
        out[f"R {name[2:]}"] = out[f"{STAGES[1]} {name}"] / out[key]
    return out


def summarize(columns, cfg, family) -> dict:
    section = cfg[SECTION]
    point, draws = bootstrap_means({k: np.array(v) for k, v in columns.items()}, section["bootstrap"], section["bootstrap_seed"])
    estimate = derive(point, cfg, family)
    sampled = [derive(d, cfg, family) for d in draws]
    out = {}
    for key, value in estimate.items():
        values = [s[key] for s in sampled]
        if key.endswith("increasing"):
            out[key] = {"estimate": value, "share_of_draws": mean(values)}
        else:
            low, high = percentile_interval(values, section["ci_level"])
            out[key] = {"estimate": float(value), "ci_low": low, "ci_high": high}
    return out


# ---------------------------------------------------------------- style and source match

def label_of(key, family) -> str:
    base, source, tag = key
    if family == "scaling":
        return f"{source} {tag}" if isinstance(tag, str) else f"{source} main seed {tag}"
    if source == UNTUNED:
        return f"untuned {base}"
    return source


def style(pairs, prompts, groups, family) -> dict:
    texts = defaultdict(lambda: ([], []))
    for key in groups:
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


def analyze(run_dir, cfg, family, embedder=None, source_match: bool = True) -> dict:
    """The locked analysis of one family; `embedder` replaces the section's embedder
    (main_robustness), and `source_match` False skips that mpnet check of the rewriter."""
    embedder = embedder or cfg[SECTION]["embedder"]
    groups = family_groups(run_dir, cfg, family)
    pairs, prompts = family_pairs(run_dir, cfg, groups)
    if not prompts:
        raise SystemExit(f"No prompt has enough usable rewrites for every {family} student. Check the samples and rewrites.")
    keys = list(groups)
    texts, index = family_texts(pairs, prompts, keys, cfg[h2.SECTION]["min_words"])
    matrix = embed_for(run_dir, texts, index, embedder, f"extra_{family}",
                       note=f"Rows are the {family} family texts of extra_analysis, both stages, one word budget per prompt. Keys: prompt|stage|base|source|tag.")

    columns, hits = defaultdict(list), []
    for pid in prompts:
        for stage in STAGES:
            table = prompt_table(matrix, index, pid, [(stage, *k) for k in keys])

            def d(first, second, table=table, stage=stage):
                return table[((stage, *first), (stage, *second))]

            for name, value in COLUMNS[family](d, cfg).items():
                columns[f"{stage} {name}"].append(value)
        for key in keys if source_match else ():
            written, rewritten = (rows_dense(matrix, index[(pid, (stage, *key))]) for stage in STAGES)
            hits += source_hits(written, rewritten, [clean(pair[0]) for pair in pairs[key][pid]])

    return {
        "family": family,
        "embedder": embedder,
        "rewriter": cfg[h2.SECTION]["rewriter"],
        "students": {"|".join(map(str, k)): name for k, (_, name) in groups.items()},
        "prompts_used": len(prompts),
        "answers_used": sum(len(pairs[k][p]) for k in keys for p in prompts),
        "source_match": mean(hits) if source_match else None,
        **summarize(columns, cfg, family),
        "style": style(pairs, prompts, keys, family),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("family", choices=FAMILIES)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for name in ("main_grid", h2.SECTION, extra_students.SECTION, SECTION):
        verify_lock(cfg, args.run_dir, name)
    result = analyze(args.run_dir, cfg, args.family)
    path = results_dir(args.run_dir, MAIN, "extra") / f"{args.family}.json"
    write_json(path, result)
    shown = {"scaling": ("S ", "carryover ", "R "), "human": ("G ", "carryover ", "R "), "untuned": ("U", "G ", "shift ")}[args.family]
    for key, value in result.items():
        name = key.split(" ", 1)[-1] if key.startswith(STAGES) else key
        if isinstance(value, dict) and "ci_low" in value and name.startswith(shown):
            print(f"{key:42s} {value['estimate']:+.4f} [{value['ci_low']:+.4f}, {value['ci_high']:+.4f}]")
        elif isinstance(value, dict) and "share_of_draws" in value:
            print(f"{key:42s} share of draws {value['share_of_draws']:.3f}")
    print(f"source match {result['source_match']:.3f}, {result['answers_used']} answers on {result['prompts_used']} prompts")
    print(path)


if __name__ == "__main__":
    main()
