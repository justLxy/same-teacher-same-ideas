"""Untuned bases of the released models: the run list samples each base with the untuned prefix and
the released models' settings; an untuned answer is cut like a released one; a released pair that
shares a source comes out closer than its bases, and a base with no usable answers leaves its pair
out while the other pair is still analyzed. The mock rewriter keeps only words that start with
"concept", so ideas survive rewriting and voices do not."""
import json
import random

from sameteacher.core.config import read_jsonl, write_jsonl
from sameteacher.data.released_untuned import untuned_name
from sameteacher.data.stretch_students import released_name
from sameteacher.train.run_extra import sampler_args
from sameteacher.train.sample_students import prefix_answer
from tests.test_stretch import RELEASED_PROMPTS, RELEASED_SAMPLES, configure, human_runs, lock_all, released_rows

BASES = {"llama": "m-llama-base", "qwen": "m-qwen-base", "qwen_math": "m-qwen-math-base"}


def setup(ws):
    configure(ws)
    ws.cfg["released_untuned"].update({"bases": BASES, "min_prompts": 5, "embedder": "tfidf", "bootstrap": 300})
    ws.save_config()
    test_split = read_jsonl(ws.run_dir / "prompts" / f"{ws.cfg['main_grid']['test_split']}.jsonl")
    write_jsonl(ws.run_dir / "prompts" / "stretch_released.jsonl", test_split[:RELEASED_PROMPTS])
    prefix = ws.run_dir / "train_data" / "extra" / "untuned_prefix.txt"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.write_text("### Instruction:\nWhat is a balloon?\n\n### Response:\nA bag of air.\n\n", encoding="utf-8")
    return prefix


def untuned_rows(name, idea_of, usable=True):
    rng = random.Random(name)
    rows = []
    for prompt in range(RELEASED_PROMPTS):
        for i in range(RELEASED_SAMPLES):
            idea = idea_of(rng)
            words = " ".join(f"concept{prompt}x{idea}w{k}" for k in range(16))
            text = f"{('voice' + name.replace('_', '') + ' ') * 4}{words} concept{prompt}d{i}" if usable else ""
            rows.append({"student": name, "prompt_id": f"p{prompt}", "sample_idx": i, "text": text, "answer_words": len(text.split())})
    return rows


def analyzed(workspace, math_usable=True):
    ws = workspace(separation=1.0)
    setup(ws)
    runs = []
    for family in ("qwen", "llama"):
        for source in ("r1", "other"):
            name = released_name(family, source)
            runs.append({"student": name, "stage": "released", "base": family, "source": source, "seed": 0,
                         "prompts_split": "stretch_released", "samples": RELEASED_SAMPLES})
            idea = (lambda rng: f"rone{rng.randrange(3)}") if source == "r1" else (lambda rng, f=family: f"{f}own{rng.randrange(3)}")
            write_jsonl(ws.run_dir / "students" / "released" / f"{name}.jsonl", released_rows(name, idea, name, source == "r1"))
    lock_all(ws, human_runs(ws), runs)
    ws.run("sameteacher.core.lock", "--section", "released_untuned")
    ws.run("sameteacher.data.released_untuned")
    for base in BASES:
        name = untuned_name(base)
        rows = untuned_rows(name, lambda rng, b=base: f"{b.replace('_', '')}base{rng.randrange(3)}", usable=math_usable or base != "qwen_math")
        write_jsonl(ws.run_dir / "students" / "released" / f"{name}.jsonl", rows)
    ws.run("sameteacher.style_norm.rewrite_students", "stretch", "--max-usd", "5")
    ws.run("sameteacher.style_norm.rewrite_students", "released_untuned", "--max-usd", "5")
    ws.run("sameteacher.analysis.released_untuned")
    return json.loads((ws.run_dir / "results" / "main" / "stretch" / "released_untuned.json").read_text())


def test_run_list_samples_each_base_with_the_untuned_prefix_and_the_released_settings(workspace):
    ws = workspace(separation=1.0)
    prefix = setup(ws)
    for name in ("main_grid", "extra_students", "stretch_students", "released_untuned"):
        ws.run("sameteacher.core.lock", "--section", name)
    ws.run("sameteacher.data.released_untuned")
    manifest = json.loads((ws.run_dir / "train_data" / "manifest_released_untuned.json").read_text())
    runs = manifest["runs"]
    assert {r["model"]: r["student"] for r in runs} == {m: untuned_name(b) for b, m in BASES.items()}
    folder = ws.run_dir / "train_data" / "released_untuned"
    assert (folder / "untuned_prefix.txt").read_text() == prefix.read_text()
    args = sampler_args(runs[0], ws.run_dir, ws.cfg, folder)
    assert args[:4] == ["--prefix-file", folder / "untuned_prefix.txt", "--answer-words", 150]
    for flag, value in (("--temperature", 0.6), ("--top-p", 0.95), ("--samples", RELEASED_SAMPLES), ("--max-new-tokens", 256)):
        assert args[args.index(flag) + 1] == value


def test_an_untuned_answer_stops_at_the_next_header_and_is_cut_like_a_released_one():
    assert prefix_answer("A bag of air that floats.\n\n### Instruction:\nNext", 3) == {"text": "A bag of", "answer_words": 6}
    assert prefix_answer("A bag of air.\n### Response", None) == "A bag of air."


def test_released_models_that_share_a_source_come_out_closer_than_their_bases(workspace):
    out = analyzed(workspace)
    assert out["pairs_analyzed"] == ["r1", "other"] and not out["r1_left_out"]
    assert out["prompts_used"] == RELEASED_PROMPTS
    assert out["written change r1"]["ci_high"] < 0 and out["rewritten change r1"]["ci_high"] < 0
    assert out["written change r1 - other"]["ci_high"] < 0
    assert {"written U other", "rewritten d r1 - other", "written U r1 - other"} <= set(out)
    assert out["usable"]["qwen_math untuned"]["answer_share"] == 1.0 and out["usable"]["qwen r1"]["answer_share"] < 1.0
    assert out["source_match"] > 0.9


def test_a_base_without_usable_answers_leaves_its_pair_out(workspace):
    out = analyzed(workspace, math_usable=False)
    assert out["pairs_analyzed"] == ["other"] and out["r1_left_out"]
    assert out["usable"]["qwen_math untuned"]["answer_share"] == 0.0
    assert out["usable"]["qwen_math untuned"]["prompts_with_min_pairs"] == 0
    assert out["prompts_used"] == RELEASED_PROMPTS and "written change r1" not in out
