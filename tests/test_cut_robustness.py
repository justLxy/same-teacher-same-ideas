"""Cut check of H1 and H2: the sentence cut keeps whole sentences within the budget, the budget comes
from the uncut texts, and the `words` cut reproduces the locked H1 and H2 results exactly."""
import json

from sameteacher.analysis import cut_robustness as cr
from sameteacher.analysis import main_shared_teacher as h1
from sameteacher.eval.embed import embed_texts
from tests.test_main_analysis import write_grid
from tests.test_main_retention import VOICES, h2, prepare, write_students

CUTS = ["words", "sentences", "full"]


def test_the_sentence_cut_keeps_whole_sentences_within_the_budget():
    text = "One two three. Four five six seven. Eight nine."
    assert cr.sentence_cut(text, 7) == ("One two three. Four five six seven.", False)
    assert cr.sentence_cut(text, 6) == ("One two three.", False)
    assert cr.sentence_cut(text, 2) == ("One two three.", True)
    assert cr.sentence_cut(text, 100) == (text, False)
    assert cr.cut_text(text, 5, "words") == ("One two three. Four five", False)
    assert cr.cut_text(text, 5, "full") == (text, False)


def test_the_budget_comes_from_the_uncut_texts_and_is_shared_by_every_cut():
    groups = {"p": {"a": ["w " * 40, "w " * 40], "b": ["x. " * 60, "x. " * 60]}}
    for cut in CUTS:
        texts, index, stats = cr.cut_groups(groups, 30, cut)
        lengths = {key: words for (_, key), (words, _) in stats.items()}
        expected = {"words": 40, "sentences": 40, "full": 60}[cut]
        assert lengths["b"] == [expected, expected], cut
        assert lengths["a"] == [40, 40]
        assert index[("p", "b")] == [2, 3]


def test_the_words_cut_reproduces_h1(tmp_path):
    cfg = write_grid(tmp_path, teacher_weight=0.4)
    texts, index, prompts, _ = h1.build_texts(tmp_path, cfg)
    primary = h1.analyze_embedding(embed_texts(texts, "tfidf"), index, prompts, cfg, primary=True)
    out = cr.h1_part(tmp_path, cfg, CUTS)
    assert out["prompts_used"] == primary["prompts_used"]
    for group in ("pooled", "teacher t_a-t_b", "base X-Y"):
        for name in ("S", "carryover"):
            assert abs(out[group][f"{name} words"]["estimate"] - primary[group][name]["estimate"]) < 1e-12
            assert abs(out[group][f"{name} words"]["ci_low"] - primary[group][name]["ci_low"]) < 1e-12
    assert out["readings"]["words"]["S lower bound above 0"]


def test_the_words_cut_reproduces_h2(workspace):
    ws = workspace(separation=1.0)
    prepare(ws)
    write_students(ws, lambda base, teacher, prompt, rng: f"{teacher}{rng.randrange(3)}", voice_of=VOICES.get)
    primary = h2(ws)
    ws.run("sameteacher.core.lock", "--section", cr.SECTION)
    ws.run("sameteacher.analysis.cut_robustness", "--parts", "h2")
    out = json.loads((ws.run_dir / "results" / "main" / "robustness" / "cut.json").read_text())["h2"]
    assert out["prompts_used"] == primary["prompts_used"]
    for group in ("pooled", "teacher t_a-t_b", "base X-Y"):
        for name in ("R", "S written", "S rewritten"):
            assert abs(out[group][f"{name} words"]["estimate"] - primary[group][name]["estimate"]) < 1e-9
    assert set(out["readings"]) == set(CUTS)
