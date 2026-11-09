"""Follow-up to the human evaluation: on a grid with a teacher effect, the embedding sees a
same-teacher advantage on the labeled pairs and over the grid; labels that follow the cosine give
a positive slope and a positive implied difference; the grid distribution weighs cells, seed pairs
and answer pairs as the labeled pairs were drawn; the logistic fit and the power formula are right."""
import csv
import itertools
import json
import random

import numpy as np
from test_main_analysis import write_grid

from sameteacher.core.config import write_jsonl
from sameteacher.core.text import clean
from sameteacher.eval import human_eval as he
from sameteacher.eval import human_eval_followup as fu
from sameteacher.eval.embed import embed_texts


def setup(tmp_path, prompts=20, per_kind=9):
    cfg = write_grid(tmp_path, teacher_weight=0.4, prompts=prompts)
    cfg[he.SECTION] = {"pairs_per_kind": per_kind, "annotators": 3, "seed": 7, "embedder": "tfidf", "bootstrap": 300, "bootstrap_seed": 0, "ci_level": 0.95}
    cfg[fu.SECTION] = {"embedders": ["tfidf"], "link_embedder": "tfidf", "bin_width": 0.01, "power": 0.8, "alpha": 0.05,
                       "bootstrap": 200, "bootstrap_seed": 0, "ci_level": 0.95}
    write_jsonl(tmp_path / "prompts" / "main_test.jsonl", [{"prompt_id": f"p{p}", "prompt": f"request number {p}"} for p in range(prompts)])
    return cfg


def fill_sheets(folder, label_of):
    for a in (1, 2, 3):
        path = folder / f"sheet_a{a}.csv"
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["same_idea"] = label_of(r)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def shared_ideas(row) -> int:
    ideas = lambda text: {w for w in text.split() if w.startswith("idea")}
    return len(ideas(row["answer_a"]) & ideas(row["answer_b"]))


def test_labels_that_follow_shared_ideas_give_a_positive_slope_and_implied_difference(tmp_path):
    cfg = setup(tmp_path)
    he.build(tmp_path, cfg)
    rng = random.Random(3)
    # People say Y mostly when the answers share ideas, with some noise, so the fit does not separate.
    fill_sheets(he.check_dir(tmp_path), lambda r: "Y" if (rng.random() < 0.8) == (shared_ideas(r) > 0) else "N")
    out = fu.analyze(tmp_path, cfg)
    assert out["pairs tfidf same minus different"]["estimate"] > 0
    assert out["grid cosine same minus different"]["ci_low"] > 0
    assert out["link slope"]["estimate"] > 0
    assert out["grid implied same minus different"]["estimate"] > 0
    assert out["grid implied same"]["estimate"] > out["grid implied different"]["estimate"]
    assert isinstance(out["pairs per kind needed"]["at estimate"], int)
    assert out["labeled pairs"] == {"same": 9, "different": 9} and out["prompts"] == 20


def test_grid_distribution_weighs_cells_seed_pairs_and_answers_as_the_pairs_were_drawn(tmp_path):
    cfg = setup(tmp_path, prompts=4)
    main_cfg = cfg["main_grid"]
    prompts = ["p0", "p1", "p2", "p3"]
    edges = np.linspace(-1, 1, 201)
    mean, hist = fu.grid_distribution(tmp_path, cfg, prompts, "tfidf", edges)
    assert np.allclose(hist.sum(-1), 1)

    answers = he.usable_answers(tmp_path, main_cfg)
    keys = [(g, p) for g in sorted(answers) for p in prompts]
    texts = [clean(t) for g, p in keys for _, t in answers[g][p]]
    vectors = embed_texts(texts, "tfidf")
    start, rows = 0, {}
    for g, p in keys:
        rows[(g, p)] = vectors[start:start + len(answers[g][p])]
        start += len(answers[g][p])

    def cell_mean(p, left, right):
        return np.mean([np.asarray((rows[((left[0], left[1], i), p)] @ rows[((right[0], right[1], j), p)].T).todense()).mean()
                        for i in main_cfg["seeds"] for j in main_cfg["seeds"]])

    base_pairs = list(itertools.combinations(sorted(main_cfg["bases"]), 2))
    teachers = main_cfg["teachers"]
    same = np.mean([cell_mean("p2", (x, t), (y, t)) for x, y in base_pairs for t in teachers])
    different = np.mean([(cell_mean("p2", (x, a), (y, b)) + cell_mean("p2", (x, b), (y, a))) / 2
                         for x, y in base_pairs for a, b in itertools.combinations(teachers, 2)])
    assert np.isclose(mean[2, 0], same) and np.isclose(mean[2, 1], different)


def test_an_answer_with_no_tfidf_terms_has_cosine_zero():
    texts = {"p1": ("the cat sat on the mat", "0 0 0 0 0"), "p2": ("the cat sat on the mat", "the cat sat on a mat")}
    c = fu.pair_cosines(["p1", "p2"], texts, "tfidf")
    assert c[0] == 0 and 0.5 < c[1] <= 1


def test_logistic_fit_and_pairs_needed():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 20000)
    y = (rng.uniform(size=x.size) < 1 / (1 + np.exp(-(-2 + 4 * x)))).astype(float)
    a, b = fu.fit_logistic(x, y)
    assert abs(a + 2) < 0.15 and abs(b - 4) < 0.25
    assert fu.pairs_needed(0.1, 0.5, 0.05, 0.8) == 393
    assert fu.pairs_needed(-0.01, 0.5, 0.05, 0.8) is None
    assert json.dumps({"n": fu.pairs_needed(0.05, 0.58, 0.05, 0.8)})
