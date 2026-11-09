"""Stretch goals: a teacher effect that a second stage keeps is read as surviving and one it removes
as erased; released models that share a source are found closer than models that do not; the run
list continues the right adapters and samples the released models with their own settings; and
a reasoning model's answer is the text after its reasoning. The mock rewriter keeps only words
that start with "concept", so ideas survive rewriting and voices do not."""
import json
import random

from sameteacher.core.config import read_jsonl, write_jsonl
from sameteacher.data.stretch_students import persist_name, released_name
from sameteacher.train.run_extra import sampler_args
from sameteacher.train.sample_students import chat_answer, cut_words
from tests.test_extra_baselines import PROMPTS, SAMPLES, answers, prepare, write_main, write_student

RELEASED_PROMPTS, RELEASED_SAMPLES = 8, 14


def configure(ws):
    prepare(ws)
    ws.cfg["stretch_students"]["persistence"].update({"bases": ["X", "Y"], "teachers": ["t_a", "t_b"], "seeds": [0, 1]})
    ws.cfg["stretch_students"]["released"].update({
        "models": {"qwen": {"r1": "m-qwen-r1", "other": "m-qwen-other"}, "llama": {"r1": "m-llama-r1", "other": "m-llama-other"}},
        "prompts": RELEASED_PROMPTS, "samples_per_prompt": RELEASED_SAMPLES, "gen_batch": RELEASED_SAMPLES,
    })
    ws.cfg["stretch_analysis"].update({"embedder": "tfidf", "bootstrap": 300})
    ws.save_config()


def human_runs(ws):
    runs = []
    for base in ("X", "Y", "Z"):
        for seed in (0, 1):
            name = f"{base}__human__s{seed}"
            runs.append({"student": name, "stage": "human", "base": base, "seed": seed})
            write_student(ws, "human", name, lambda rng: f"human{rng.randrange(3)}")
    return runs


def lock_all(ws, extra_runs, stretch_runs):
    folder = ws.run_dir / "train_data"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest_extra.json").write_text(json.dumps({"section": "extra_students", "runs": extra_runs}))
    (folder / "manifest_stretch.json").write_text(json.dumps({"section": "stretch_students", "runs": stretch_runs}))
    for name in ("main_grid", "main_style_norm", "extra_students", "extra_analysis", "stretch_students", "stretch_analysis"):
        ws.run("sameteacher.core.lock", "--section", name)


def analyzed(ws, family):
    ws.run("sameteacher.style_norm.main_retention", "--max-usd", "5")
    ws.run("sameteacher.style_norm.rewrite_students", "extra", "--max-usd", "5")
    ws.run("sameteacher.style_norm.rewrite_students", "stretch", "--max-usd", "5")
    ws.run("sameteacher.analysis.stretch", family)
    return json.loads((ws.run_dir / "results" / "main" / "stretch" / f"{family}.json").read_text())


def persistence_case(workspace, keep: float):
    ws = workspace(separation=1.0)
    configure(ws)
    write_main(ws, lambda base, teacher, rng: f"{teacher}{rng.randrange(3)}")
    runs = []
    for base in ("X", "Y"):
        for teacher in ("t_a", "t_b"):
            for seed in (0, 1):
                name = persist_name(base, teacher, seed)
                runs.append({"student": name, "stage": "persist", "base": base, "teacher": teacher, "seed": seed})
                write_student(ws, "persist", name, lambda rng, t=teacher: f"{t}{rng.randrange(3)}" if rng.random() < keep else f"human{rng.randrange(3)}")
    lock_all(ws, human_runs(ws), runs)
    return analyzed(ws, "persistence")


def test_a_teacher_effect_kept_by_the_second_stage_survives(workspace):
    out = persistence_case(workspace, keep=0.95)
    assert out["state"] == "SURVIVES" and out["reading rewritten"] == "SURVIVES"
    assert out["written S after"]["ci_low"] > 0 and out["written P"]["ci_low"] >= 0.5
    assert out["source_match"] > 0.9
    assert {"written carryover after - before", "rewritten G after - human", "written H after / before", "R after", "R before"} <= set(out)
    assert {"t_a then human written", "human rewritten", "t_b written"} <= set(out["style"])


def test_a_teacher_effect_removed_by_the_second_stage_is_erased(workspace):
    out = persistence_case(workspace, keep=0.0)
    assert out["state"] == "ERASED"
    assert out["written S before"]["ci_low"] > 0
    assert out["written S after - before"]["ci_high"] < 0
    # The second-stage students moved toward the human-data students.
    assert out["written H after / before"]["ci_high"] < 1


def released_rows(name, idea_of, rng_seed, reasoning):
    rng = random.Random(rng_seed)
    rows = []
    for prompt in range(RELEASED_PROMPTS):
        for i in range(RELEASED_SAMPLES):
            closed = not reasoning or i % 7 != 6
            idea = idea_of(rng)
            words = " ".join(f"concept{prompt}x{idea}w{k}" for k in range(16))
            text = f"{('voice' + name.replace('_', '') + ' ') * 4}{words} concept{prompt}d{i}" if closed else ""
            rows.append({"student": name, "prompt_id": f"p{prompt}", "sample_idx": i, "text": text, "answer_closed": closed,
                         "reasoning_words": 50 if reasoning else None, "answer_words": len(text.split())})
    return rows


def test_released_models_that_share_a_source_are_closer(workspace):
    ws = workspace(separation=1.0)
    configure(ws)
    write_main(ws, lambda base, teacher, rng: f"{teacher}{rng.randrange(3)}")
    test_split = read_jsonl(ws.run_dir / "prompts" / f"{ws.cfg['main_grid']['test_split']}.jsonl")
    write_jsonl(ws.run_dir / "prompts" / "stretch_released.jsonl", test_split[:RELEASED_PROMPTS])
    runs = []
    for family in ("qwen", "llama"):
        for source in ("r1", "other"):
            name = released_name(family, source)
            runs.append({"student": name, "stage": "released", "base": family, "source": source, "seed": 0,
                         "prompts_split": "stretch_released", "samples": RELEASED_SAMPLES})
            idea = (lambda rng: f"rone{rng.randrange(3)}") if source == "r1" else (lambda rng, f=family: f"{f}own{rng.randrange(3)}")
            write_jsonl(ws.run_dir / "students" / "released" / f"{name}.jsonl", released_rows(name, idea, name, source == "r1"))
    lock_all(ws, human_runs(ws), runs)
    out = analyzed(ws, "released")
    assert out["state"] == "SUPPORTED" and out["R reading"] == "CONTENT"
    assert out["written S"]["ci_low"] > 0 and out["written shared minus other"]["ci_low"] > 0
    assert out["prompts_used"] == RELEASED_PROMPTS
    assert 0.8 < out["answers"]["qwen r1"]["answer_share"] < 0.9 and out["answers"]["llama other"]["answer_share"] == 1.0
    assert out["answers"]["llama r1"]["mean_reasoning_words"] == 50 and out["answers"]["qwen other"]["mean_reasoning_words"] is None
    assert {"written S/T", "rewritten T qwen", "written d mixed"} <= set(out)


def test_run_list_continues_main_grid_adapters_and_samples_released_models_on_their_own_settings(workspace):
    ws = workspace(separation=1.0)
    configure(ws)
    human = [{"prompt": f"train prompt {i}", "response": f"human answer {i}"} for i in range(5)]
    write_jsonl(ws.run_dir / "train_data" / "extra" / "human.jsonl", human)
    for name in ("main_grid", "extra_students", "stretch_students"):
        ws.run("sameteacher.core.lock", "--section", name)
    ws.run("sameteacher.data.stretch_students")
    manifest = json.loads((ws.run_dir / "train_data" / "manifest_stretch.json").read_text())
    persist = [r for r in manifest["runs"] if r["stage"] == "persist"]
    released = [r for r in manifest["runs"] if r["stage"] == "released"]
    assert len(persist) == 8 and len(released) == 4
    assert {r["init_adapter"] for r in persist} == {f"{b}__{t}__s{s}" for b in ("X", "Y") for t in ("t_a", "t_b") for s in (0, 1)}
    assert all(r["student"] == persist_name(r["base"], r["teacher"], r["seed"]) for r in persist)
    assert read_jsonl(ws.run_dir / "train_data" / "stretch" / "human.jsonl") == human
    assert {r["model"]: r["thinking"] for r in released} == {"m-qwen-r1": True, "m-qwen-other": False, "m-llama-r1": True, "m-llama-other": False}
    test_split = read_jsonl(ws.run_dir / "prompts" / f"{ws.cfg['main_grid']['test_split']}.jsonl")
    assert read_jsonl(ws.run_dir / "prompts" / "stretch_released.jsonl") == test_split[:RELEASED_PROMPTS]

    cfg = ws.cfg
    chat = sampler_args(next(r for r in released if r["source"] == "r1"), ws.run_dir, cfg, ws.run_dir)
    assert chat[:4] == ["--chat", "--thinking", "--answer-words", 150]
    assert chat[chat.index("--temperature") + 1] == 0.6 and chat[chat.index("--samples") + 1] == RELEASED_SAMPLES
    tuned = sampler_args(persist[0], ws.run_dir, cfg, ws.run_dir)
    assert tuned[0] == "--adapter" and tuned[tuned.index("--temperature") + 1] == cfg["sampling"]["temperature"]


def test_a_reasoning_models_answer_is_the_text_after_its_reasoning():
    out = chat_answer("Okay, the user wants a fruit.\n</think>\n\nApple.\n\n- red and sweet", True, 3)
    assert out == {"text": "Apple.\n\n- red", "answer_closed": True, "reasoning_words": 6, "answer_words": 5}
    assert chat_answer("Okay, the user wants a fruit and I keep thinking", True, 3)["text"] == ""
    assert chat_answer("Banana, yellow.", False, None) == {"text": "Banana, yellow.", "answer_closed": True, "reasoning_words": None, "answer_words": 2}
    assert cut_words("one two\nthree four", 3) == "one two\nthree"
    assert len(answers("s", lambda rng: "i")) == PROMPTS * SAMPLES
