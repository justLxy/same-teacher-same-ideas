"""Train-test prompt overlap: a test prompt copied from training has cosine and Jaccard 1 to it, the
reference uses the nearest other test prompt, and dropping the most similar prompts recomputes S."""
import numpy as np

from sameteacher.analysis import prompt_overlap as po

SECTION = {"embedder": "tfidf", "drop_top_share": 0.25, "near_duplicate_cosine": 0.8, "bootstrap": 200,
           "bootstrap_seed": 0, "ci_level": 0.95}


def test_overlap_finds_a_copied_prompt_and_drops_it():
    train = ["write a poem about the sea", "plan a trip to rome", "explain how vaccines work",
             "list ways to save money", "describe a medieval castle"]
    test = [{"prompt_id": "a", "prompt": "write a poem about the sea"},
            {"prompt_id": "b", "prompt": "what if dragons ran a bakery"},
            {"prompt_id": "c", "prompt": "invent a new board game"},
            {"prompt_id": "d", "prompt": "argue for a four day week"}]
    s = {"a": 0.9, "b": 0.1, "c": 0.2, "d": 0.3}
    out = po.analyze(test, train, s, SECTION)
    assert np.isclose(out["nearest training cosine"]["max"], 1.0)
    assert np.isclose(out["nearest training jaccard"]["max"], 1.0)
    assert out["most similar"][0]["test"] == "write a poem about the sea"
    assert out["S without the most similar"]["dropped"] == 1
    assert np.isclose(out["S without the most similar"]["estimate"], 0.2)
    assert np.isclose(out["S all prompts"]["estimate"], 0.375)
    assert out["share with nearest training cosine above threshold"]["share"] == 0.25


def test_jaccard_and_reference_skip_the_prompt_itself():
    pool = [po.words("a b c"), po.words("a b d")]
    assert po.max_jaccard(po.words("a b c"), pool, skip=0) == 0.5
    vecs = np.eye(3)
    cos, idx = po.nearest(vecs, vecs, exclude_self=True)
    assert np.allclose(cos, 0)
