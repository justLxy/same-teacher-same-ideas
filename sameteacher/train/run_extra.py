"""Train and sample the students beyond the main grid, skipping work already done.

Reads runs/train_data/manifest_extra.json (from `sameteacher.data.extra_students`), with
--manifest h3 runs/train_data/manifest_h3.json (from `sameteacher.data.h3_data`), or with
--manifest stretch runs/train_data/manifest_stretch.json (from `sameteacher.data.stretch_students`).
Runs with a training file are trained with the main-grid recipe and then sampled; a run with an
initial adapter continues training that main-grid student's adapter. Untuned runs are sampled
from the base model with the few-shot prefix, and released chat models with their own chat
template and the run's sampling settings. Answers go to runs/students/<stage>/. Safe to rerun.
"""
import argparse

from sameteacher.core.config import add_common_args, load_yaml, read_json, read_jsonl
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import model_dir, prompts_path, students_path
from sameteacher.data import extra_students, h3_data, stretch_students
from sameteacher.train.run_main import check_reused
from sameteacher.train.run_pilot import call


# Lock sections, folder of the run files, and manifest path for each run list.
MANIFESTS = {
    "extra": ((extra_students.SECTION,), extra_students.extra_dir, extra_students.manifest_path),
    "h3": ((h3_data.SECTION, h3_data.TRAINING_SECTION), h3_data.h3_dir, h3_data.manifest_path),
    "stretch": ((extra_students.SECTION, stretch_students.SECTION), stretch_students.stretch_dir, stretch_students.manifest_path),
}


def train(run, run_dir, main_cfg, folder) -> None:
    out = model_dir(run_dir, run["student"])
    data = folder / run["train_file"]
    if (out / "train_meta.json").exists():
        check_reused(out / "train_meta.json", data, run["model"])
        return
    print("training", run["student"], flush=True)
    start = ["--init-adapter", model_dir(run_dir, run["init_adapter"]) / "adapter"] if run.get("init_adapter") else []
    if start and not start[1].exists():
        raise SystemExit(f"{run['student']} continues {start[1]}, which is not on this machine.")
    call(
        "sameteacher.train.lora",
        *start,
        "--base-model", run["model"],
        "--train-file", data,
        "--output", out,
        "--seed", run["seed"],
        "--max-steps", main_cfg["max_steps"],
        "--batch-size", main_cfg["batch_size"],
        "--grad-accum", main_cfg["grad_accum"],
        "--learning-rate", main_cfg["learning_rate"],
        "--lora-r", main_cfg["lora_r"],
        "--lora-alpha", main_cfg["lora_alpha"],
        "--max-length", main_cfg["max_length"],
    )


def sampler_args(run, run_dir, cfg, folder) -> list:
    """Model input and sampling settings of one run: main-grid settings unless the run sets its own."""
    main_cfg = cfg["main_grid"]
    if run.get("chat"):
        extra = ["--chat", *(["--thinking"] if run.get("thinking") else []), "--answer-words", run["answer_words"]]
    elif run.get("prefix_file"):
        extra = ["--prefix-file", folder / run["prefix_file"]]
    else:
        extra = ["--adapter", model_dir(run_dir, run["student"]) / "adapter"]
    return [
        *extra,
        "--samples", run.get("samples", main_cfg["samples_per_prompt"]),
        "--gen-batch", run.get("gen_batch", main_cfg["gen_batch"]),
        "--temperature", run.get("temperature", cfg["sampling"]["temperature"]),
        "--top-p", run.get("top_p", cfg["sampling"]["top_p"]),
        "--max-new-tokens", run.get("max_new_tokens", cfg["sampling"]["max_tokens"]),
    ]


def sample(run, run_dir, cfg, folder) -> None:
    main_cfg = cfg["main_grid"]
    samples = students_path(run_dir, run["stage"], run["student"])
    samples.parent.mkdir(parents=True, exist_ok=True)
    if len(read_jsonl(samples)) >= stretch_students.expected_rows(run_dir, main_cfg, run):
        return
    print("sampling", run["student"], flush=True)
    call(
        "sameteacher.train.sample_students",
        "--base-model", run["model"],
        "--prompts", prompts_path(run_dir, stretch_students.prompts_split(run, main_cfg)),
        "--output", samples,
        "--student", run["student"],
        *sampler_args(run, run_dir, cfg, folder),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--manifest", choices=sorted(MANIFESTS), default="extra")
    parser.add_argument("--stages", nargs="*", help="Limit the run to these stages (scaling, human, untuned, h3, persist, released)")
    parser.add_argument("--students", nargs="*", help="Limit the run to these students")
    parser.add_argument("--only-train", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    sections, folder_of, manifest_of = MANIFESTS[args.manifest]
    for name in ("main_grid", *sections):
        verify_lock(cfg, args.run_dir, name)
    folder = folder_of(args.run_dir)
    main_cfg = cfg["main_grid"]
    runs = read_json(manifest_of(args.run_dir))["runs"]
    if args.stages:
        runs = [r for r in runs if r["stage"] in set(args.stages)]
    if args.students:
        runs = [r for r in runs if r["student"] in set(args.students)]
    # Untuned runs need no training, so they go first and the GPU starts sampling at once.
    runs.sort(key=lambda r: r["stage"] != "untuned")
    for run in runs:
        if run.get("train_file"):
            train(run, args.run_dir, main_cfg, folder)
        if not args.only_train:
            sample(run, args.run_dir, cfg, folder)
    print(f"all {args.manifest} runs finished")


if __name__ == "__main__":
    main()
