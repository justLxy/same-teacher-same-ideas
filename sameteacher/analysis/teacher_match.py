"""Diagnostic: which teacher's test answers are the students of each teacher closest to?

H1 found that gemma's students are, on average, farther from gemma's own test answers than
from the other teachers'. This module separates two readings of that result:
  location  the students' answers center on another teacher's ideas
  spread    the students center on gemma, but their answers are more varied than gemma's,
            and the energy distance counts that difference too
Energy distance depends on both. On unit-length embeddings the squared distance between mean
embeddings (unbiased linear-kernel MMD) also depends on spread, because a more varied set has a
shorter mean vector. The direction distance, 1 - mu_s.mu_t / (|mu_s| |mu_t|) with every inner
product estimated without bias (cross terms, and within-set terms without the diagonal),
depends only on where the mean points. The mean pairwise distance within a set measures spread. Training answers (Dolly) are compared with test answers for style and for
spread across draws. No state: the settings are in the locked `main_teacher_match` section.
"""
import argparse
import hashlib
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from sameteacher.analysis import main_shared_teacher as h1
from sameteacher.core.config import add_common_args, load_yaml, read_jsonl, write_json
from sameteacher.core.grid import main_cells
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, results_dir, students_path, teacher_samples_path, train_file
from sameteacher.core.text import LIST_MARKER, MARKDOWN, clean, common_word_budget, first_words, is_answer
from sameteacher.data.teacher_answers import group_name
from sameteacher.eval.embed import embed_texts
from sameteacher.eval.stats import bootstrap_means, percentile_interval

SECTION = "main_teacher_match"


def cache_paths(run_dir, embedder: str, name: str = "main_h1") -> tuple[Path, Path]:
    stem = Path(run_dir) / "cache" / f"{name}__{embedder.replace('/', '-')}"
    return stem.with_suffix(".npy"), stem.with_suffix(".json")


def key_name(pid: str, key: tuple) -> str:
    return "|".join([pid, *map(str, key)])


def embed_cached(run_dir, texts, index, embedder: str, name: str = "main_h1", note: str = "") -> np.ndarray:
    """An embedding matrix saved with its text index for later analyses (the H1 texts by default)."""
    matrix_path, index_path = cache_paths(run_dir, embedder, name)
    digest = hashlib.sha256("\n".join(texts).encode()).hexdigest()
    if matrix_path.exists() and index_path.exists() and json.loads(index_path.read_text())["texts_sha256"] == digest:
        return np.load(matrix_path)
    matrix = embed_texts(texts, embedder)
    matrix = np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix, dtype=np.float32)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, matrix)
    index_path.write_text(json.dumps({
        "embedder": embedder,
        "texts_sha256": digest,
        "note": note or "Rows are the main_shared_teacher texts, cut to one word budget per prompt. Keys: prompt|base|teacher|seed for students, prompt|teacher|name for teachers.",
        "index": {key_name(pid, key): rows for (pid, key), rows in index.items()},
    }))
    return matrix


def embed_for(run_dir, texts, index, embedder: str, name: str, note: str = ""):
    """Sparse tf-idf rows, which are never densified (the bigram vocabulary is large), or cached dense embeddings."""
    if embedder == "tfidf":
        return embed_texts(texts, embedder)
    return embed_cached(run_dir, texts, index, embedder, name=name, note=note)


def prompt_stats(matrix, index, pid, students, teachers) -> dict:
    """Per-prompt energy and centroid distances from every student to every teacher, and spreads."""
    keys = students + teachers
    rows = [i for k in keys for i in index[(pid, k)]]
    block = np.asarray(matrix[rows], dtype=np.float64)
    gram = block @ block.T
    pairwise = np.sqrt(np.clip(2.0 - 2.0 * gram, 0.0, None))
    spans, start = {}, 0
    for k in keys:
        spans[k] = slice(start, start + len(index[(pid, k)]))
        start += len(index[(pid, k)])
    spread, dot = {}, {}
    for k, s in spans.items():
        n = s.stop - s.start
        spread[k] = float(pairwise[s, s].sum()) / (n * (n - 1))
        dot[k] = float(gram[s, s].sum() - np.trace(gram[s, s])) / (n * (n - 1))
    out = {}
    for student, teacher in itertools.product(students, teachers):
        a, b = spans[student], spans[teacher]
        cross = float(gram[a, b].mean())
        out[("E", student, teacher)] = 2 * float(pairwise[a, b].mean()) - spread[student] - spread[teacher]
        out[("M", student, teacher)] = dot[student] + dot[teacher] - 2 * cross
        out[("D", student, teacher)] = 1.0 - cross / np.sqrt(max(dot[student], 1e-12) * max(dot[teacher], 1e-12))
    for k in keys:
        out[("W", k)] = spread[k]
    return out


def column_name(key: tuple) -> str:
    return " ".join(str(x) if not isinstance(x, tuple) else ":".join(map(str, x)) for x in key)


def derive(m: dict, bases, teachers, seeds) -> dict:
    """Matrices, spreads and own-teacher contrasts from column means."""
    def student(base, teacher, seed):
        return ":".join(map(str, (base, teacher, seed)))

    def tk(t):
        return f"teacher:{t}"

    out = {}
    for kind, name in (("E", "energy"), ("M", "centroid"), ("D", "direction")):
        cell = {(a, b): float(np.mean([m[f"{kind} {student(x, a, s)} {tk(b)}"] for x in bases for s in seeds])) for a in teachers for b in teachers}
        out[name] = {a: {b: cell[(a, b)] for b in teachers} for a in teachers}
        out[f"{name} own"] = {a: float(np.mean([cell[(a, o)] for o in teachers if o != a])) - cell[(a, a)] for a in teachers}
        for x in bases:
            by = {(a, b): float(np.mean([m[f"{kind} {student(x, a, s)} {tk(b)}"] for s in seeds])) for a in teachers for b in teachers}
            out[f"{name} {x}"] = {a: {b: by[(a, b)] for b in teachers} for a in teachers}
            out[f"{name} own {x}"] = {a: float(np.mean([by[(a, o)] for o in teachers if o != a])) - by[(a, a)] for a in teachers}
    out["spread teacher"] = {t: m[f"W {tk(t)}"] for t in teachers}
    out["spread students"] = {t: float(np.mean([m[f"W {student(x, t, s)}"] for x in bases for s in seeds])) for t in teachers}
    out["spread students minus teacher"] = {t: out["spread students"][t] - out["spread teacher"][t] for t in teachers}
    return out


def with_intervals(point, draws, level) -> dict:
    """The same nested structure with each leaf as estimate and interval."""
    if isinstance(point, dict):
        return {k: with_intervals(v, [d[k] for d in draws], level) for k, v in point.items()}
    low, high = percentile_interval(draws, level)
    return {"estimate": point, "ci_low": low, "ci_high": high}


def text_style(texts: list[str]) -> dict:
    usable = [t for t in texts if is_answer(t)]
    return {
        "n": len(texts),
        "usable_share": len(usable) / len(texts) if texts else float("nan"),
        "mean_words": float(np.mean([len(t.split()) for t in usable])),
        "formatted_share": float(np.mean([bool(MARKDOWN.search(t) or LIST_MARKER.search(t) or "\n" in t.strip()) for t in usable])),
    }


def train_reps(main_cfg) -> list[int]:
    return sorted({rep for reps in main_cfg["draws"].values() for rep in reps})


def style_table(run_dir, main_cfg) -> dict:
    teachers = list(main_cfg["teachers"])
    out = {}
    for t in teachers:
        train = [row["response"] for rep in train_reps(main_cfg) for row in read_jsonl(train_file(run_dir, t, rep))]
        test = [row["text"] for row in read_jsonl(teacher_samples_path(run_dir, main_cfg["test_split"])) if group_name(row) == t]
        students = [row["text"] for cell in main_cells(main_cfg) if cell["teacher"] == t for row in read_jsonl(students_path(run_dir, MAIN, cell["student"]))]
        out[t] = {"training answers": text_style(train), "test answers": text_style(test), "students": text_style(students)}
    return out


def train_spread(run_dir, cfg) -> dict:
    """Spread of each teacher's training answers to one Dolly prompt across its draws."""
    main_cfg, section = cfg["main_grid"], cfg[SECTION]
    teachers, reps = list(main_cfg["teachers"]), train_reps(main_cfg)
    answers = defaultdict(lambda: defaultdict(list))
    for t, rep in itertools.product(teachers, reps):
        seen = set()
        for row in read_jsonl(train_file(run_dir, t, rep)):
            if row["prompt"] not in seen and is_answer(row["response"]):
                seen.add(row["prompt"])
                answers[row["prompt"]][t].append(clean(row["response"]))
    complete = sorted(p for p, by in answers.items() if all(len(by[t]) == len(reps) for t in teachers))
    chosen = random.Random(section["train_seed"]).sample(complete, min(section["train_prompts"], len(complete)))
    texts, index = [], {}
    for p in chosen:
        budget = common_word_budget([answers[p][t] for t in teachers], section["min_words"])
        for t in teachers:
            start = len(texts)
            texts.extend(first_words(x, budget) for x in answers[p][t])
            index[(p, t)] = list(range(start, len(texts)))
    matrix = embed_texts(texts, section["embedder"])
    matrix = np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix, dtype=np.float64)
    columns = defaultdict(list)
    for p in chosen:
        for t in teachers:
            block = matrix[index[(p, t)]]
            pairwise = np.sqrt(np.clip(2.0 - 2.0 * block @ block.T, 0.0, None))
            n = len(block)
            columns[t].append(float(pairwise.sum()) / (n * (n - 1)))
    point, draws = bootstrap_means({k: np.array(v) for k, v in columns.items()}, section["bootstrap"], section["bootstrap_seed"])
    result = with_intervals(dict(point), draws, section["ci_level"])
    return {"prompts": len(chosen), "draws_per_prompt": len(reps), "eligible_prompts": len(complete), "spread": result}


def analyze(run_dir, cfg) -> dict:
    main_cfg, section = cfg["main_grid"], cfg[SECTION]
    bases, teachers, seeds = list(main_cfg["bases"]), list(main_cfg["teachers"]), list(main_cfg["seeds"])
    texts, index, prompts, _ = h1.build_texts(run_dir, cfg)
    matrix = embed_cached(run_dir, texts, index, section["embedder"])
    students = [(b, t, s) for b, t, s in itertools.product(bases, teachers, seeds)]
    teacher_keys = [h1.teacher_key(t) for t in teachers]
    columns = defaultdict(list)
    for pid in prompts:
        for key, value in prompt_stats(matrix, index, pid, students, teacher_keys).items():
            columns[column_name(key)].append(value)
    point, draws = bootstrap_means({k: np.array(v) for k, v in columns.items()}, section["bootstrap"], section["bootstrap_seed"])
    estimate = derive(point, bases, teachers, seeds)
    sampled = [derive(d, bases, teachers, seeds) for d in draws]
    return {"prompts_used": len(prompts), "embedder": section["embedder"]} | with_intervals(estimate, sampled, section["ci_level"])


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        "--train-spread", action="store_true",
        help="Run only the training-answer spread, in its own process so one embedder at a time holds GPU memory",
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for name in ("main_grid", h1.SECTION, SECTION):
        verify_lock(cfg, args.run_dir, name)
    folder = results_dir(args.run_dir, MAIN, "teacher_match")
    if args.train_spread:
        spread = train_spread(args.run_dir, cfg)
        write_json(folder / "train_spread.json", spread)
        print("train spread", {t: round(v["estimate"], 4) for t, v in spread["spread"].items()}, f"on {spread['prompts']} prompts")
        return
    result = analyze(args.run_dir, cfg)
    result["style"] = style_table(args.run_dir, cfg["main_grid"])
    path = folder / "teacher_match.json"
    write_json(path, result)
    teachers = list(cfg["main_grid"]["teachers"])
    for name in ("energy", "centroid", "direction"):
        print(name, "(rows: students of; columns: test answers of)")
        for a in teachers:
            print(f"  {a:9s}", "  ".join(f"{result[name][a][b]['estimate']:.4f}" for b in teachers), f"  own {result[name + ' own'][a]['estimate']:+.4f} [{result[name + ' own'][a]['ci_low']:+.4f}, {result[name + ' own'][a]['ci_high']:+.4f}]")
    print("spread teacher ", {t: round(v["estimate"], 4) for t, v in result["spread teacher"].items()})
    print("spread students", {t: round(v["estimate"], 4) for t, v in result["spread students"].items()})
    print(path)


if __name__ == "__main__":
    main()
