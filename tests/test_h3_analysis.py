"""H3 analysis: students whose ideas follow the content teacher and whose voice and format follow
the style teacher are SUPPORTED; students with the reverse pattern are NOT_SUPPORTED. The mock
rewriter keeps only words that start with "concept", so ideas survive rewriting and voice,
length and list formatting do not."""
import json
import random

import yaml

from sameteacher.core.config import write_jsonl
from sameteacher.core.grid import main_cells
from sameteacher.data import h3_data

SAMPLES, PROMPTS = 12, 12


def text(prompt, idea, voice, i) -> str:
    words = [f"concept{prompt}x{idea}w{k}" for k in range(16)]
    if voice == "t_a":
        # A long, formatted voice: an opener and a list.
        lines = [" ".join(words[j:j + 4]) for j in range(0, 16, 4)]
        return "honestly speaking overall indeed truly really quite so plainly\n" + "\n".join(f"- {line}" for line in lines) + f" concept{prompt}d{i}"
    return " ".join(words) + f" concept{prompt}d{i}"


def write(ws, stage, name, idea_teacher, voice):
    rng = random.Random(name)
    rows = [{"student": name, "prompt_id": f"p{p}", "sample_idx": i, "text": text(p, f"{idea_teacher}{rng.randrange(3)}".replace("_", ""), voice, i)}
            for p in range(PROMPTS) for i in range(SAMPLES)]
    write_jsonl(ws.run_dir / "students" / stage / f"{name}.jsonl", rows)


def run(workspace, reverse: bool) -> dict:
    ws = analyzed(workspace, reverse)
    return json.loads((ws.run_dir / "results" / "main" / "h3" / "h3.json").read_text())


def analyzed(workspace, reverse: bool):
    """A workspace after H2 and the H3 analysis have run."""
    ws = workspace(separation=1.0)
    models = yaml.safe_load(ws.models.read_text())
    models["models"]["rw"] = {"kind": "mock", "mock": {"separation": 0.0, "style": "plain", "family": "rw"}}
    ws.models.write_text(yaml.safe_dump(models))
    ws.cfg["main_grid"]["samples_per_prompt"] = SAMPLES
    ws.cfg["main_style_norm"].update({"rewriter": "rw", "embedder": "tfidf", "bootstrap": 300, "samples_per_cell": 10, "min_pairs": 6})
    ws.cfg[h3_data.SECTION].update({"teachers": ["t_a", "t_b"], "embedder": "tfidf"})
    ws.cfg["h3_analysis"].update({"embedder": "tfidf", "bootstrap": 300})
    ws.save_config()
    write_jsonl(ws.run_dir / "prompts" / f"{ws.cfg['main_grid']['test_split']}.jsonl", [{"prompt_id": f"p{p}", "prompt": f"prompt {p}"} for p in range(PROMPTS)])

    for cell in main_cells(ws.cfg["main_grid"]):
        write(ws, "main", cell["student"], cell["teacher"], cell["teacher"])
    runs = [{**f, "stage": "h3"} for f in h3_data.files(ws.cfg)]
    for f in runs:
        ideas, voice = (f["style"], f["content"]) if reverse else (f["content"], f["style"])
        write(ws, "h3", f["student"], ideas, voice)
    (ws.run_dir / "train_data").mkdir(parents=True, exist_ok=True)
    h3_data.manifest_path(ws.run_dir).write_text(json.dumps({"section": h3_data.SECTION, "runs": runs}))

    for name in ("main_grid", "main_style_norm", h3_data.SECTION, "h3_analysis"):
        ws.run("sameteacher.core.lock", "--section", name)
    ws.run("sameteacher.style_norm.main_retention", "--max-usd", "5")
    ws.run("sameteacher.style_norm.rewrite_students", "h3", "--max-usd", "5")
    ws.run("sameteacher.analysis.h3_analysis")
    return ws


def test_content_following_the_content_teacher_is_supported(workspace):
    out = run(workspace, reverse=False)
    assert out["state"] == "SUPPORTED", {k: out[k] for k in out if "pooled" in k}
    assert out["normalized cross pooled"]["estimate"] > 0.5
    assert out["length cross pooled"]["estimate"] < -0.5 and out["formatting cross pooled"]["estimate"] < -0.5
    assert out["normalized self pooled"]["estimate"] > 0.5
    assert set(out["references"]["X__t_a_in_t_b__s0"]["content"]) == {"X__t_a__s1", "X__t_a__s2"}
    assert {"normalized cross t_a in t_b", "written cross base Y", "length self t_b"} <= set(out)


def test_content_following_the_style_teacher_is_not_supported(workspace):
    out = run(workspace, reverse=True)
    assert out["state"] == "NOT_SUPPORTED", {k: out[k] for k in out if "pooled" in k}
    assert out["normalized cross pooled"]["estimate"] < -0.5
    assert out["length cross pooled"]["estimate"] > 0.5
