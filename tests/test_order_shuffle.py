"""Sentence-order check of H2: a shuffle keeps every sentence and changes only their order, and with
single-sentence rewrites (nothing to reorder) the check reproduces the locked H2 result exactly."""
import json

from sameteacher.analysis.order_shuffle import sentences, shuffle_sentences
from tests.test_main_retention import VOICES, h2, prepare, write_students


def test_a_shuffle_keeps_the_sentences_and_changes_their_order():
    text = "First point here. Second point follows! Third one ends? Fourth closes."
    out = shuffle_sentences(text, 3)
    assert out != text and sorted(sentences(out)) == sorted(sentences(text))
    assert sorted(out.split()) == sorted(text.split())
    assert shuffle_sentences(text, 3) == out
    assert shuffle_sentences("Only one sentence here.", 3) == "Only one sentence here."
    assert shuffle_sentences("Two parts. Swapped here.", 0) == "Swapped here. Two parts."


def test_single_sentence_rewrites_reproduce_the_locked_h2_result(workspace):
    ws = workspace(separation=1.0)
    prepare(ws)
    write_students(ws, lambda base, teacher, prompt, rng: f"{teacher}{rng.randrange(3)}", voice_of=VOICES.get)
    primary = h2(ws)
    ws.run("sameteacher.core.lock", "--section", "main_order_shuffle")
    ws.run("sameteacher.analysis.order_shuffle")
    out = json.loads((ws.run_dir / "results" / "main" / "style_norm" / "order_shuffle.json").read_text())
    assert out["prompts_used"] == primary["prompts_used"]
    assert out["rewrites_with_two_or_more_sentences"] == 0.0
    for group in ("pooled", "teacher t_a-t_b", "base X-Y"):
        assert abs(out[group]["R"]["estimate"] - primary[group]["R"]["estimate"]) < 1e-9
        assert abs(out[group]["R shuffled - R"]["estimate"]) < 1e-12
        assert abs(out[group]["S rewritten"]["estimate"] - primary[group]["S rewritten"]["estimate"]) < 1e-9
