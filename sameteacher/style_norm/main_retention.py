"""H2 on the main grid: how much of the shared-teacher effect remains in plain prose?

The rewriter turns the first usable answers of every student and prompt into plain, neutral
prose with the pilot's instruction. The H1 quantities are computed twice on the same answers,
as written and rewritten, with one word budget per prompt across both stages:
    R = S rewritten / S as written
A rewrite must stay closer to its own source than to the other answers of the same student
on the same prompt (source match), or the rewriter changed the content. The decision rule is
in the locked `main_style_norm` config section.
"""
import argparse
import asyncio
import itertools
from collections import defaultdict

import numpy as np

from sameteacher.analysis.main_shared_teacher import prompt_table, quantities, student_columns
from sameteacher.core.config import add_common_args, load_yaml, read_jsonl, stable_seed, write_json
from sameteacher.core.grid import main_cells
from sameteacher.core.llm import Budget, Chat, load_models, run_jobs
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, results_dir, students_path, style_norm_dir
from sameteacher.core.text import LIST_MARKER, MARKDOWN, clean, common_word_budget, first_words, is_answer
from sameteacher.eval.embed import embed_texts, rows_dense
from sameteacher.eval.stats import GO, STOP, bootstrap_means, percentile_interval, three_way
from sameteacher.style_norm.pilot_retention import NEUTRAL_INSTRUCTION, STAGES, source_hits

SECTION = "main_style_norm"
STATES = {GO: "SUPPORTED", STOP: "VOICE"}


def rewrites_path(run_dir):
    return style_norm_dir(run_dir) / "main_students.jsonl"


def grid(main_cfg) -> tuple[list, list, list, list]:
    bases, teachers, seeds = list(main_cfg["bases"]), list(main_cfg["teachers"]), list(main_cfg["seeds"])
    return bases, teachers, seeds, list(itertools.product(bases, teachers, seeds))


def select_sources(run_dir, main_cfg, per_cell: int) -> list[dict]:
    """The first usable answers by sample index, per main-grid student and prompt."""
    students = [(MAIN, c["student"], (c["base"], c["teacher"], c["seed"])) for c in main_cells(main_cfg)]
    return select_student_sources(run_dir, students, per_cell)


def select_student_sources(run_dir, students, per_cell: int) -> list[dict]:
    """The same selection for any students, given as (stage, name, cell key)."""
    rows = []
    for stage, name, cell in students:
        taken = defaultdict(int)
        for row in sorted(read_jsonl(students_path(run_dir, stage, name)), key=lambda r: (r["prompt_id"], r["sample_idx"])):
            if is_answer(row["text"]) and taken[row["prompt_id"]] < per_cell:
                taken[row["prompt_id"]] += 1
                key = f"{name}|{row['prompt_id']}|{row['sample_idx']}"
                rows.append({"key": key, "cell": cell, **row})
    return rows


async def rewrite(args, cfg, sources) -> dict:
    return await rewrite_sources(args, cfg, sources, rewrites_path(args.run_dir))


async def rewrite_sources(args, cfg, sources, out_path) -> dict:
    """Rewrite answers into plain prose with the main_style_norm rewriter, skipping finished ones."""
    sampling = cfg["sampling"]
    spec = load_models(args.models)[cfg[SECTION]["rewriter"]]
    chat = Chat(Budget(args.max_usd))

    async def handler(job):
        result = await chat.complete(
            spec,
            [{"role": "user", "content": NEUTRAL_INSTRUCTION + job["text"]}],
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            max_tokens=sampling["max_tokens"],
            seed=stable_seed(job["key"]),
            metadata={"task": "paraphrase", "source": job["text"]},
        )
        return {"student": job["student"], "prompt_id": job["prompt_id"], "sample_idx": job["sample_idx"], "text": result.text, "finish_reason": result.finish_reason}

    try:
        stats = await run_jobs(sources, handler, out_path, args.concurrency or sampling["concurrency"])
    finally:
        await chat.aclose()
    print(f"rewrites: {stats}, spent about ${chat.budget.spent:.2f} at the configured prices, {chat.throttled} throttled and {chat.malformed} malformed-response retries")
    return stats


def matched_pairs(run_dir, cells, sources, min_pairs: int, rewrite_files=None) -> tuple[dict, list[str]]:
    """(source, rewrite) pairs per cell and prompt, keeping rewrites that are complete answers."""
    rewrites = {}
    for path in rewrite_files or [rewrites_path(run_dir)]:
        rewrites |= {f"{r['student']}|{r['prompt_id']}|{r['sample_idx']}": r for r in read_jsonl(path)}
    pairs = defaultdict(lambda: defaultdict(list))
    for source in sources:
        rewritten = rewrites.get(source["key"])
        if rewritten and rewritten["finish_reason"] != "length" and is_answer(rewritten["text"]):
            pairs[source["cell"]][source["prompt_id"]].append((source["text"], rewritten["text"]))
    prompts = set.intersection(*[{p for p, v in pairs[c].items() if len(v) >= min_pairs} for c in cells])
    return pairs, sorted(prompts)


def style_stats(pairs, cells, teachers, prompts) -> dict:
    """Surface style of each teacher's students, as written and rewritten."""
    out = {}
    for teacher, (stage_index, stage) in itertools.product(teachers, enumerate(STAGES)):
        texts = [pair[stage_index] for cell in cells if cell[1] == teacher for p in prompts for pair in pairs[cell][p]]
        out[f"{teacher} {stage}"] = {
            "mean_words": float(np.mean([len(t.split()) for t in texts])),
            "formatted_share": float(np.mean([bool(MARKDOWN.search(t) or LIST_MARKER.search(t) or "\n" in t.strip()) for t in texts])),
        }
    return out


def stage_view(m: dict, stage: str) -> dict:
    prefix = stage + " "
    return {k[len(prefix):]: v for k, v in m.items() if k.startswith(prefix)}


def retention(m: dict, bases, teachers) -> dict:
    """Per group (pooled, teacher pair, base pair): S and carryover in both stages, R, and the teacher-effect share."""
    written, rewritten = (quantities(stage_view(m, s), bases, teachers) for s in STAGES)
    out = {}
    for group in written:
        w, r = written[group], rewritten[group]
        out[group] = {
            "R": r["S"] / w["S"] if w["S"] != 0 else float("nan"),
            "S written": w["S"],
            "S rewritten": r["S"],
            "carryover written": w["carryover"],
            "carryover rewritten": r["carryover"],
            "teacher_effect_share": r["net"] / w["net"] if w["net"] > 0 else float("nan"),
        }
    return out


def analyze(run_dir, cfg, sources, embedder=None, source_match: bool = True) -> dict:
    """The locked H2 analysis; `embedder` replaces the section's embedder (main_robustness), and
    `source_match` False skips that mpnet check of the rewriter and the state's instrument test."""
    main_cfg, section = cfg["main_grid"], cfg[SECTION]
    embedder = embedder or section["embedder"]
    bases, teachers, seeds, cells = grid(main_cfg)
    pairs, prompts = matched_pairs(run_dir, cells, sources, section["min_pairs"])
    if not prompts:
        raise SystemExit("No prompt has enough usable rewrites for every student. Check the rewrites.")

    texts, index = [], {}
    for pid in prompts:
        groups = {(stage, cell): [clean(pair[stage]) for pair in pairs[cell][pid]] for stage in range(len(STAGES)) for cell in cells}
        # One budget for both stages, so a shorter rewrite cannot shrink the distances by length alone.
        budget = common_word_budget(list(groups.values()), section["min_words"])
        for key, values in groups.items():
            start = len(texts)
            texts.extend(first_words(t, budget) for t in values)
            index[(pid, key)] = list(range(start, len(texts)))
    matrix = embed_texts(texts, embedder)

    columns, hits = defaultdict(list), []
    for pid in prompts:
        for stage_index, stage in enumerate(STAGES):
            keys = [(stage_index, cell) for cell in cells]
            table = prompt_table(matrix, index, pid, keys)

            def dist(first, second, table=table, s=stage_index):
                return table[((s, first), (s, second))]

            for k, v in student_columns(dist, bases, teachers, seeds).items():
                columns[f"{stage} {k}"].append(v)
        for cell in cells if source_match else ():
            sources_text = [clean(pair[0]) for pair in pairs[cell][pid]]
            hits += source_hits(rows_dense(matrix, index[(pid, (0, cell))]), rows_dense(matrix, index[(pid, (1, cell))]), sources_text)

    point, draws = bootstrap_means({k: np.array(v) for k, v in columns.items()}, section["bootstrap"], section["bootstrap_seed"])
    estimate = retention(point, bases, teachers)
    sampled = [retention(d, bases, teachers) for d in draws]
    summary = {}
    for group, values in estimate.items():
        summary[group] = {}
        for name, value in values.items():
            low, high = percentile_interval([s[group][name] for s in sampled], section["ci_level"])
            summary[group][name] = {"estimate": value, "ci_low": low, "ci_high": high}

    pooled = summary["pooled"]
    match = float(np.mean(hits)) if source_match else None
    if source_match and match < section["min_source_match"]:
        state = "INSTRUMENT_INVALID"
    elif pooled["S written"]["ci_low"] <= 0:
        state = "NO_SHARED_EFFECT"
    else:
        state = STATES.get(three_way(pooled["R"]["ci_low"], pooled["R"]["ci_high"], section["most_low"], section["voice_high"]), "MIXED_OR_UNCLEAR")

    return {
        "state": state,
        "rewriter": section["rewriter"],
        "embedder": embedder,
        "prompts_used": len(prompts),
        "answers_used": sum(len(pairs[c][p]) for c in cells for p in prompts),
        "source_match": match,
        **summary,
        "style": style_stats(pairs, cells, teachers, prompts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--max-usd", type=float, required=True, help="Budget for the rewrites at the configured prices; finished ones are skipped")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--skip-rewrite", action="store_true", help="Analyze the rewrites that exist")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    verify_lock(cfg, args.run_dir, "main_grid")
    verify_lock(cfg, args.run_dir, SECTION)
    sources = select_sources(args.run_dir, cfg["main_grid"], cfg[SECTION]["samples_per_cell"])
    print(f"{len(sources)} source answers")
    if not args.skip_rewrite and asyncio.run(rewrite(args, cfg, sources))["stopped_on_budget"]:
        raise SystemExit("Rewrites stopped on budget. Rerun with a larger --max-usd; finished rewrites are kept.")
    result = analyze(args.run_dir, cfg, sources)
    path = results_dir(args.run_dir, MAIN, "style_norm") / "h2.json"
    write_json(path, result)
    print(f"main-grid style normalization: {result['state']}")
    for group in [k for k in result if k == "pooled" or k.startswith(("teacher ", "base "))]:
        row = result[group]["R"]
        print(f"{group:22s} R {row['estimate']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]")
    print(f"source_match {result['source_match']:.3f} over {result['answers_used']} answers on {result['prompts_used']} prompts")
    print(path)


if __name__ == "__main__":
    main()
