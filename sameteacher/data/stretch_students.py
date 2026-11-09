"""Run list for the stretch goals: a second SFT stage on human answers, and released distilled models.

  persistence  main-grid students that continue training their adapter on the human file
  released     released chat models: DeepSeek-R1 distills on two base families, and models of the
               same families tuned by other developers; nothing is trained

Writes runs/train_data/stretch/ (a copy of the extra_students human file), the prompt subset
runs/prompts/stretch_released.jsonl, and runs/train_data/manifest_stretch.json, which
`sameteacher.train.run_extra --manifest stretch` reads. The design is in the locked
`stretch_students` section.
"""
import argparse
import hashlib
import shutil

from sameteacher.core.config import add_common_args, load_yaml, read_jsonl, write_json, write_jsonl
from sameteacher.core.grid import main_cells
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import prompts_path, train_data_dir
from sameteacher.data import extra_students

SECTION = "stretch_students"
STAGES = ("persist", "released")
RELEASED_SPLIT = "stretch_released"
HUMAN_FILE = "human.jsonl"


def stretch_dir(run_dir):
    return train_data_dir(run_dir) / "stretch"


def manifest_path(run_dir):
    return train_data_dir(run_dir) / "manifest_stretch.json"


def persist_name(base, teacher, seed) -> str:
    return f"{base}__{teacher}__s{seed}__then_human"


def released_name(family, source) -> str:
    return f"released__{family}__{source}"


def prompts_split(run, main_cfg) -> str:
    """The prompt split a run is sampled on."""
    return run.get("prompts_split", main_cfg["test_split"])


def expected_rows(run_dir, main_cfg, run) -> int:
    """Answers a run has once it is fully sampled."""
    prompts = read_jsonl(prompts_path(run_dir, prompts_split(run, main_cfg)))
    return len(prompts) * run.get("samples", main_cfg["samples_per_prompt"])


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(run_dir, cfg) -> list[dict]:
    main_cfg, section = cfg["main_grid"], cfg[SECTION]
    folder = stretch_dir(run_dir)
    folder.mkdir(parents=True, exist_ok=True)
    runs = []

    human = extra_students.extra_dir(run_dir) / HUMAN_FILE
    if not human.exists():
        raise SystemExit(f"{human} is missing; build the extra_students files first.")
    copy = folder / HUMAN_FILE
    shutil.copyfile(human, copy)
    cells = {(c["base"], c["teacher"], c["seed"]): c for c in main_cells(main_cfg)}
    spec = section["persistence"]
    for base in spec["bases"]:
        for teacher in spec["teachers"]:
            for seed in spec["seeds"]:
                parent = cells[(base, teacher, seed)]
                runs.append({"student": persist_name(base, teacher, seed), "stage": "persist", "base": base, "model": parent["model"],
                             "teacher": teacher, "seed": seed, "init_adapter": parent["student"], "examples": len(read_jsonl(copy)),
                             "train_file": copy.name})

    released = section["released"]
    test = read_jsonl(prompts_path(run_dir, main_cfg["test_split"]))
    if len(test) < released["prompts"]:
        raise SystemExit(f"The test split has {len(test)} prompts, fewer than {released['prompts']}.")
    write_jsonl(prompts_path(run_dir, RELEASED_SPLIT), test[: released["prompts"]])
    for family, models in released["models"].items():
        for source, model in models.items():
            runs.append({"student": released_name(family, source), "stage": "released", "base": family, "source": source, "model": model,
                         "seed": 0, "chat": True, "thinking": source in released["reasoning"], "prompts_split": RELEASED_SPLIT,
                         "samples": released["samples_per_prompt"], "gen_batch": released["gen_batch"],
                         "temperature": released["temperature"], "top_p": released["top_p"],
                         "max_new_tokens": released["max_new_tokens"], "answer_words": released["answer_words"]})

    write_json(manifest_path(run_dir), {"section": SECTION, "runs": runs, "human_sha256": sha256(copy),
                                        "released_prompts": released["prompts"]})
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for name in ("main_grid", extra_students.SECTION, SECTION):
        verify_lock(cfg, args.run_dir, name)
    runs = build(args.run_dir, cfg)
    for stage in STAGES:
        print(stage, sum(r["stage"] == stage for r in runs), "runs")
    print(manifest_path(args.run_dir))


if __name__ == "__main__":
    main()
