"""Robustness embedders: rerunning a locked analysis through main_robustness with the analysis's
own embedder reproduces its result, so only the embedder changes; source match is not repeated."""
import json

import test_extra_baselines as extra
from test_h3_analysis import analyzed


def robustness(ws, analyses):
    ws.cfg["main_robustness"].update({"embedders": ["tfidf"], "analyses": analyses})
    ws.save_config()
    ws.run("sameteacher.core.lock", "--section", "main_robustness")
    ws.run("sameteacher.analysis.robustness")
    return {a: json.loads((ws.run_dir / "results" / "main" / "robustness" / f"{a}__tfidf.json").read_text()) for a in analyses}


def same(first, second, path=""):
    if isinstance(first, dict):
        for k, v in first.items():
            same(v, second[k], f"{path}/{k}")
    elif isinstance(first, float):
        assert abs(first - second) < 1e-9, path
    else:
        assert first == second, path


def test_h2_and_h3_reproduce_with_their_own_embedder(workspace):
    ws = analyzed(workspace, reverse=False)
    out = robustness(ws, ["h2", "h3"])
    for analysis, folder, name in (("h2", "style_norm", "h2"), ("h3", "h3", "h3")):
        primary = json.loads((ws.run_dir / "results" / "main" / folder / f"{name}.json").read_text())
        result = out[analysis]
        assert result["source_match"] is None and result["embedder"] == "tfidf" and "main_robustness" in result["note"]
        kept = {k: v for k, v in primary.items() if not k.startswith("source_match")}
        same(kept, {k: result[k] for k in kept})
    assert out["h3"]["state"] == "SUPPORTED" and out["h2"]["state"] == "SUPPORTED"


def test_baselines_reproduce_with_their_own_embedder(workspace):
    ws = workspace(separation=1.0)
    extra.prepare(ws)
    extra.write_main(ws, lambda base, teacher, rng: f"{teacher}{rng.randrange(3)}" if rng.random() < 0.5 else f"{base}{rng.randrange(3)}")
    runs = []
    for base in ("X", "Y", "Z"):
        for seed in (0, 1):
            name = f"{base}__human__s{seed}"
            runs.append({"student": name, "stage": "human", "base": base, "seed": seed})
            extra.write_student(ws, "human", name, lambda rng: f"human{rng.randrange(3)}")
    primary = extra.finish(ws, runs, "human")
    result = robustness(ws, ["human"])["human"]
    assert result["source_match"] is None
    kept = {k: v for k, v in primary.items() if k not in ("source_match", "embedder")}
    same(kept, {k: result[k] for k in kept})
