"""Rewrite the answers of the H3, extra or stretch students into plain prose, as H2 did for the main grid.

  h3       the students in runs/train_data/manifest_h3.json, into runs/style_norm/h3_students.jsonl
  extra    the students in runs/train_data/manifest_extra.json, into runs/style_norm/extra_students.jsonl
  stretch  the students in runs/train_data/manifest_stretch.json, into runs/style_norm/stretch_students.jsonl

The first `main_style_norm.samples_per_cell` usable answers per student and prompt are rewritten
by the main_style_norm rewriter with its instruction and seeds. Students whose answers are not all
sampled yet are skipped and named, so the step can run again as more students finish.
"""
import argparse
import asyncio

from sameteacher.core.config import add_common_args, load_yaml, read_json, read_jsonl
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import students_path, style_norm_dir
from sameteacher.data import extra_students, h3_data, stretch_students
from sameteacher.style_norm import main_retention as h2

# Lock sections and run list for each set of students.
SETS = {
    "h3": (("h3_cross_style", "h3_analysis"), h3_data.manifest_path),
    "extra": (("extra_students", "extra_analysis"), extra_students.manifest_path),
    "stretch": (("extra_students", "stretch_students", "stretch_analysis"), stretch_students.manifest_path),
}


def rewrites_path(run_dir, which: str):
    return style_norm_dir(run_dir) / f"{which}_students.jsonl"


def students(run_dir, cfg, which: str) -> tuple[list, list]:
    """(stage, name, name) for every student with all its answers, and the names still missing."""
    main_cfg = cfg["main_grid"]
    ready, missing = [], []
    for run in read_json(SETS[which][1](run_dir))["runs"]:
        expected = stretch_students.expected_rows(run_dir, main_cfg, run)
        complete = len(read_jsonl(students_path(run_dir, run["stage"], run["student"]))) >= expected
        (ready if complete else missing).append((run["stage"], run["student"], run["student"]))
    return ready, [name for _, name, _ in missing]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("which", choices=sorted(SETS))
    parser.add_argument("--max-usd", type=float, required=True, help="Budget at the configured prices; finished rewrites are skipped")
    parser.add_argument("--concurrency", type=int)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for name in ("main_grid", h2.SECTION, *SETS[args.which][0]):
        verify_lock(cfg, args.run_dir, name)
    ready, missing = students(args.run_dir, cfg, args.which)
    if missing:
        print(f"not sampled yet, skipped: {', '.join(missing)}")
    sources = h2.select_student_sources(args.run_dir, ready, cfg[h2.SECTION]["samples_per_cell"])
    print(f"{len(ready)} students, {len(sources)} source answers")
    if asyncio.run(h2.rewrite_sources(args, cfg, sources, rewrites_path(args.run_dir, args.which)))["stopped_on_budget"]:
        raise SystemExit("Rewrites stopped on budget. Rerun with a larger --max-usd; finished rewrites are kept.")


if __name__ == "__main__":
    main()
