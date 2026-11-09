"""The energy-distance analyses repeated with the robustness embedders (Proposal, Evaluation).

H1 is repeated by main_shared_teacher. This module reruns H2, H3 and the extra_analysis
baselines with each embedder of the locked `main_robustness` section. Each analysis keeps its
own locked section's answers, prompts, word budgets and bootstrap draws; only the embedder
changes, and the mpnet source-match check of the rewriter is not repeated.

Usage: python3 -m sameteacher.analysis.robustness [--embedders tfidf] [--analyses h2 h3]
Results go to runs/results/main/robustness/{analysis}__{embedder}.json.
"""
import argparse

from sameteacher.analysis import extra_baselines, h3_analysis
from sameteacher.core.config import add_common_args, load_yaml, write_json
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, results_dir
from sameteacher.data import extra_students, h3_data
from sameteacher.style_norm import main_retention

SECTION = "main_robustness"
NOTE = "robustness embedder (main_robustness); reported with no state, and source match is not repeated"

# Quantities printed after each run; every quantity is in the result file.
HEADLINE = {
    "h2": ["R", "S written", "S rewritten", "carryover written", "carryover rewritten"],
    "h3": ["normalized cross pooled", "written cross pooled", "normalized cross deepseek in gemma",
           "normalized cross gemma in deepseek", "normalized self pooled", "written self pooled"],
    "scaling": ["written S n500", "written S main", "written carryover n500", "written carryover main", "R n500", "R main"],
    "human": ["written G human", "written G teachers", "written G human - teachers",
              "written carryover human pooled", "written carryover teacher pairs", "R human pooled", "R teacher pairs"],
    "untuned": ["written U - G same", "rewritten U - G same", "written U - G different", "rewritten U - G different"],
}


def locks(analyses) -> list[str]:
    names = ["main_grid", main_retention.SECTION]
    if "h3" in analyses:
        names += [h3_data.SECTION, h3_analysis.SECTION]
    if set(analyses) & set(extra_baselines.FAMILIES):
        names += [extra_students.SECTION, extra_baselines.SECTION]
    return names + [SECTION]


def run_one(run_dir, cfg, analysis: str, embedder: str) -> dict:
    if analysis == "h2":
        sources = main_retention.select_sources(run_dir, cfg["main_grid"], cfg[main_retention.SECTION]["samples_per_cell"])
        return main_retention.analyze(run_dir, cfg, sources, embedder=embedder, source_match=False)
    if analysis == "h3":
        return h3_analysis.analyze(run_dir, cfg, embedder=embedder, source_match=False)
    return extra_baselines.analyze(run_dir, cfg, analysis, embedder=embedder, source_match=False)


def headline(analysis: str, result: dict) -> list[str]:
    values = result["pooled"] if analysis == "h2" else result
    lines = []
    for name in HEADLINE[analysis]:
        v = values.get(name)
        if isinstance(v, dict) and "ci_low" in v:
            lines.append(f"  {name:38s} {v['estimate']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]")
    return lines


def output_path(run_dir, analysis: str, embedder: str):
    return results_dir(run_dir, MAIN, "robustness") / f"{analysis}__{embedder.split('/')[-1]}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--embedders", nargs="+", help="A subset of the section's embedders")
    parser.add_argument("--analyses", nargs="+", help="A subset of the section's analyses")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    section = cfg[SECTION]
    embedders, analyses = args.embedders or section["embedders"], args.analyses or section["analyses"]
    for chosen, allowed, kind in ((embedders, section["embedders"], "embedders"), (analyses, section["analyses"], "analyses")):
        unknown = [n for n in chosen if n not in allowed]
        if unknown:
            raise SystemExit(f"{unknown} are not configured {kind}: {allowed}")
    for name in locks(analyses):
        verify_lock(cfg, args.run_dir, name)

    for embedder in embedders:
        for analysis in analyses:
            result = {"analysis": analysis, "note": NOTE, **run_one(args.run_dir, cfg, analysis, embedder)}
            path = output_path(args.run_dir, analysis, embedder)
            write_json(path, result)
            state = f" reading {result['state']}" if "state" in result else ""
            print(f"[{embedder}] {analysis}{state}: {result['answers_used']} answers on {result['prompts_used']} prompts -> {path}", flush=True)
            print("\n".join(headline(analysis, result)), flush=True)


if __name__ == "__main__":
    main()
