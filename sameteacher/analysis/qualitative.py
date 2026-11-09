"""Per-category breakdown of H1 and H2, and examples from the prompts with the largest shared effect.

Per Infinity-Chat category: the mean per-prompt H1 S, and S written, S rewritten and R on the H2
answers, with intervals that bootstrap prompts within the category. Examples: for the top H1
prompts by S, the most central full answers of each teacher and of each base's students of one
teacher, and the share of each group's answers nearest to each teacher's centroid. Settings are
in the locked `main_qualitative` config section.

Usage: python3 -m sameteacher.analysis.qualitative [categories | examples]
Results: runs/results/main/qualitative/{categories,examples}.json
"""
import argparse
from collections import defaultdict

import numpy as np

from sameteacher.analysis import attribution_probe
from sameteacher.analysis.main_shared_teacher import prompt_table, quantities, student_columns
from sameteacher.analysis.teacher_match import embed_for
from sameteacher.core.config import add_common_args, load_yaml, read_json, read_jsonl, write_json
from sameteacher.core.grid import main_cells
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, prompts_path, results_dir, students_path, teacher_samples_path
from sameteacher.core.text import is_answer
from sameteacher.eval.embed import embed_texts
from sameteacher.eval.stats import percentile_interval
from sameteacher.style_norm import main_retention as h2
from sameteacher.style_norm.pilot_retention import STAGES

SECTION = "main_qualitative"


def categories_of(run_dir, cfg) -> dict:
    return {r["prompt_id"]: r["category"] for r in read_jsonl(prompts_path(run_dir, cfg["main_grid"]["test_split"]))}


def h2_per_prompt(run_dir, cfg) -> dict:
    """{prompt: {stage: S}} on the H2 answers, from the cached H2 embeddings."""
    bases, teachers, seeds, cells = h2.grid(cfg["main_grid"])
    texts, index, prompts, _ = attribution_probe.build_texts(run_dir, cfg)
    matrix = embed_for(run_dir, texts, index, cfg[h2.SECTION]["embedder"], "main_h2")
    out = {}
    for pid in prompts:
        out[pid] = {}
        for stage in STAGES:
            table = prompt_table(matrix, index, pid, [(stage, *c) for c in cells])

            def dist(first, second, table=table, stage=stage):
                return table[((stage, *first), (stage, *second))]

            out[pid][stage] = quantities(student_columns(dist, bases, teachers, seeds), bases, teachers)["pooled"]["S"]
    return out


def grouping(cats: dict, prompts, min_prompts: int) -> dict:
    counts = defaultdict(int)
    for pid in prompts:
        counts[cats[pid]] += 1
    return {pid: cats[pid] if counts[cats[pid]] >= min_prompts else "other" for pid in prompts}


def interval(values, level) -> dict:
    low, high = percentile_interval(values, level)
    return {"ci_low": low, "ci_high": high}


def categories(run_dir, cfg) -> dict:
    section = cfg[SECTION]
    cats = categories_of(run_dir, cfg)
    h1 = read_json(results_dir(run_dir, MAIN, "shared_teacher") / "h1.json")["per_prompt"]
    s2 = h2_per_prompt(run_dir, cfg)
    group = grouping(cats, list(h1), section["min_prompts"])
    rng = np.random.default_rng(section["bootstrap_seed"])
    out = {"h1_prompts": len(h1), "h2_prompts": len(s2), "categories": {}}
    for name in sorted(set(group.values()), key=lambda g: (g == "other", -sum(v == g for v in group.values()), g)):
        p1 = [p for p in h1 if group[p] == name]
        p2 = [p for p in p1 if p in s2]
        s = np.array([h1[p]["S"] for p in p1])
        w = np.array([s2[p][STAGES[0]] for p in p2])
        r = np.array([s2[p][STAGES[1]] for p in p2])
        draws1 = [s[rng.integers(0, len(s), len(s))].mean() for _ in range(section["bootstrap"])]
        draws2 = [rng.integers(0, len(w), len(w)) for _ in range(section["bootstrap"])]
        row = {
            "h1_prompts": len(p1),
            "h2_prompts": len(p2),
            "members": sorted({cats[p] for p in p1}) if name == "other" else [name],
            "S": {"estimate": float(s.mean()), **interval(draws1, section["ci_level"])},
            "S written": {"estimate": float(w.mean()), **interval([w[i].mean() for i in draws2], section["ci_level"])},
            "S rewritten": {"estimate": float(r.mean()), **interval([r[i].mean() for i in draws2], section["ci_level"])},
            "R": {"estimate": float(r.mean() / w.mean()), **interval([r[i].mean() / w[i].mean() for i in draws2], section["ci_level"])},
            "share of prompts with S > 0": float(np.mean(s > 0)),
        }
        out["categories"][name] = row
    return out


def central(vectors: np.ndarray, k: int) -> list[int]:
    gram = vectors @ vectors.T
    n = len(vectors)
    mean_sim = (gram.sum(axis=1) - np.diag(gram)) / max(n - 1, 1)
    return [int(i) for i in np.argsort(-mean_sim)[:k]]


def examples(run_dir, cfg) -> dict:
    section, main_cfg = cfg[SECTION], cfg["main_grid"]
    cats = categories_of(run_dir, cfg)
    prompt_text = {r["prompt_id"]: r["prompt"] for r in read_jsonl(prompts_path(run_dir, main_cfg["test_split"]))}
    h1 = read_json(results_dir(run_dir, MAIN, "shared_teacher") / "h1.json")["per_prompt"]
    top = sorted(h1, key=lambda p: -h1[p]["S"])[: section["top_k"]]
    wanted = set(top)

    answers = defaultdict(list)  # (pid, group) -> texts
    for row in read_jsonl(teacher_samples_path(run_dir, main_cfg["test_split"])):
        if row["prompt_id"] in wanted and row["teacher"] in main_cfg["teachers"] and is_answer(row["text"]):
            answers[(row["prompt_id"], f"teacher {row['teacher']}")].append(row["text"].strip())
    for cell in main_cells(main_cfg):
        for row in read_jsonl(students_path(run_dir, MAIN, cell["student"])):
            if row["prompt_id"] in wanted and is_answer(row["text"]):
                answers[(row["prompt_id"], f"{cell['base']} students of {cell['teacher']}")].append(row["text"].strip())

    keys = sorted(answers)
    texts = [t for k in keys for t in answers[k]]
    matrix = embed_texts(texts, section["embedder"])
    matrix = np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix, dtype=np.float64)
    rows, start = {}, 0
    for k in keys:
        rows[k] = slice(start, start + len(answers[k]))
        start += len(answers[k])

    out = {"prompts": []}
    for rank, pid in enumerate(top, 1):
        groups = [k for k in keys if k[0] == pid]
        centroids = {}
        for t in main_cfg["teachers"]:
            c = matrix[rows[(pid, f"teacher {t}")]].mean(axis=0)
            centroids[t] = c / np.linalg.norm(c)
        names = list(centroids)
        cmat = np.stack([centroids[t] for t in names])
        entry = {"rank": rank, "prompt_id": pid, "prompt": prompt_text[pid], "category": cats[pid], **{k: v for k, v in h1[pid].items()}, "groups": {}}
        for k in groups:
            vecs = matrix[rows[k]]
            nearest = np.argmax(vecs @ cmat.T, axis=1)
            entry["groups"][k[1]] = {
                "answers": len(answers[k]),
                "nearest teacher share": {t: float(np.mean(nearest == i)) for i, t in enumerate(names)},
                "central answers": [answers[k][i] for i in central(vecs, section["examples_per_group"])],
            }
        out["prompts"].append(entry)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("parts", nargs="*", choices=["categories", "examples"], default=["categories", "examples"])
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for name in ("main_grid", "main_shared_teacher", h2.SECTION, SECTION):
        verify_lock(cfg, args.run_dir, name)
    out_dir = results_dir(args.run_dir, MAIN, "qualitative")
    if "categories" in args.parts:
        result = categories(args.run_dir, cfg)
        write_json(out_dir / "categories.json", result)
        print(f"{result['h1_prompts']} H1 prompts, {result['h2_prompts']} H2 prompts")
        for name, row in result["categories"].items():
            s, rr = row["S"], row["R"]
            print(f"{name[:40]:40s} n {row['h1_prompts']:3d}  S {s['estimate']:.4f} [{s['ci_low']:.4f}, {s['ci_high']:.4f}]  R {rr['estimate']:.3f} [{rr['ci_low']:.3f}, {rr['ci_high']:.3f}]")
    if "examples" in args.parts:
        result = examples(args.run_dir, cfg)
        write_json(out_dir / "examples.json", result)
        for e in result["prompts"]:
            print(f"{e['rank']:2d}  S {e['S']:.4f}  {e['category'][:28]:28s}  {e['prompt'][:80]}")
    print(out_dir)


if __name__ == "__main__":
    main()
