"""Run list for the untuned bases of the released models (locked `released_untuned` section).

Each base is sampled from the base model with the untuned few-shot prefix of extra_students,
copied here unchanged, on the released models' prompts and with their sampling settings, and its
answers are cut like theirs. Writes runs/train_data/released_untuned/untuned_prefix.txt and
runs/train_data/manifest_released_untuned.json, which
`sameteacher.train.run_extra --manifest released_untuned` reads.

Usage: python3 -m sameteacher.data.released_untuned
"""
import argparse
import shutil

from sameteacher.core.config import add_common_args, load_yaml, write_json
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import prompts_path, train_data_dir
from sameteacher.data import extra_students, stretch_students

SECTION = "released_untuned"
PREFIX = "untuned_prefix.txt"


def folder(run_dir):
    return train_data_dir(run_dir) / "released_untuned"


def manifest_path(run_dir):
    return train_data_dir(run_dir) / "manifest_released_untuned.json"


def untuned_name(base) -> str:
    return f"released__{base}__untuned"


def build(run_dir, cfg) -> list[dict]:
    section, released = cfg[SECTION], cfg[stretch_students.SECTION]["released"]
    source = extra_students.extra_dir(run_dir) / PREFIX
    prompts = prompts_path(run_dir, stretch_students.RELEASED_SPLIT)
    for path in (source, prompts):
        if not path.exists():
            raise SystemExit(f"{path} is missing; it comes from the extra_students and stretch_students run lists.")
    folder(run_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, folder(run_dir) / PREFIX)
    runs = [
        {"student": untuned_name(base), "stage": "released", "base": base, "source": "untuned", "model": model, "seed": 0,
         "prefix_file": PREFIX, "prompts_split": stretch_students.RELEASED_SPLIT,
         "samples": released["samples_per_prompt"], "gen_batch": section["gen_batch"],
         "temperature": released["temperature"], "top_p": released["top_p"],
         "max_new_tokens": section["max_new_tokens"], "answer_words": released["answer_words"]}
        for base, model in section["bases"].items()
    ]
    write_json(manifest_path(run_dir), {"section": SECTION, "runs": runs, "prefix_sha256": stretch_students.sha256(folder(run_dir) / PREFIX)})
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for name in ("main_grid", extra_students.SECTION, stretch_students.SECTION, SECTION):
        verify_lock(cfg, args.run_dir, name)
    runs = build(args.run_dir, cfg)
    print(f"{len(runs)} runs -> {manifest_path(args.run_dir)}")


if __name__ == "__main__":
    main()
