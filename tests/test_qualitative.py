"""Per-category breakdown and examples: categories below the size floor are pooled, the category
means of S follow the per-prompt values, the size-weighted category S equals H2's pooled S, the top
prompts are the largest S, and students sit nearest their own teacher's centroid."""
import json
import random

from test_main_retention import prepare, write_students

from sameteacher.core.config import write_json, write_jsonl

PROMPTS = 12
CATEGORY = {p: "A" if p < 6 else "B" if p < 10 else "C" for p in range(PROMPTS)}


def test_categories_and_examples(workspace):
    ws = workspace(separation=1.0)
    prepare(ws)
    ws.cfg["main_qualitative"].update({"min_prompts": 4, "top_k": 3, "examples_per_group": 2, "embedder": "tfidf", "bootstrap": 300})
    ws.save_config()
    write_students(ws, lambda base, teacher, prompt, rng: f"{teacher}{rng.randrange(3)}")
    write_jsonl(ws.run_dir / "prompts" / "main_test.jsonl",
                [{"prompt_id": f"p{p}", "prompt": f"prompt {p}", "category": CATEGORY[p]} for p in range(PROMPTS)])
    rng = random.Random(0)
    teachers = [{"teacher": t, "variant": "baseline", "prompt_id": f"p{p}", "sample_idx": i, "finish_reason": "stop",
                 "text": " ".join(f"concept{p}x{t.replace('_', '')}{rng.randrange(3)}w{k}" for k in range(16))}
                for t in ("t_a", "t_b", "t_c") for p in range(PROMPTS) for i in range(12)]
    write_jsonl(ws.run_dir / "teachers" / "main_test.jsonl", teachers)
    planted = {f"p{p}": {"S": 0.01 * (p + 1)} for p in range(PROMPTS)}
    write_json(ws.run_dir / "results" / "main" / "shared_teacher" / "h1.json", {"per_prompt": planted})

    for name in ("main_grid", "main_style_norm", "main_shared_teacher", "main_qualitative"):
        ws.run("sameteacher.core.lock", "--section", name)
    ws.run("sameteacher.style_norm.main_retention", "--max-usd", "5")
    ws.run("sameteacher.analysis.qualitative")
    folder = ws.run_dir / "results" / "main" / "qualitative"
    cats = json.loads((folder / "categories.json").read_text())["categories"]
    h2 = json.loads((ws.run_dir / "results" / "main" / "style_norm" / "h2.json").read_text())

    assert list(cats) == ["A", "B", "other"] and cats["other"]["members"] == ["C"]
    assert [cats[c]["h1_prompts"] for c in cats] == [6, 4, 2]
    assert abs(cats["A"]["S"]["estimate"] - 0.035) < 1e-12 and abs(cats["other"]["S"]["estimate"] - 0.115) < 1e-12
    n = sum(c["h2_prompts"] for c in cats.values())
    weighted = sum(c["h2_prompts"] * c["S written"]["estimate"] for c in cats.values()) / n
    assert n == h2["prompts_used"] and abs(weighted - h2["pooled"]["S written"]["estimate"]) < 1e-9
    assert cats["A"]["R"]["ci_low"] <= cats["A"]["R"]["estimate"] <= cats["A"]["R"]["ci_high"]

    ex = json.loads((folder / "examples.json").read_text())["prompts"]
    assert [e["prompt_id"] for e in ex] == ["p11", "p10", "p9"] and ex[0]["category"] == "C"
    group = ex[0]["groups"]["X students of t_b"]
    assert group["nearest teacher share"]["t_b"] > 0.9 and len(group["central answers"]) == 2
    assert {"teacher t_a", "Z students of t_c"} <= set(ex[0]["groups"])
