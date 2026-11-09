"""Follow-up to the human evaluation of H1, in the locked `main_human_eval_followup` section.

  pairs  does each embedder see a same-teacher advantage on the 150 labeled pairs?
  link   a logistic fit of the majority label on the cosine of the two answers
  grid   the same-teacher advantage in cosine over every cross-base answer pair of the main grid,
         and the difference in the human label that it implies for one answer pair

  python3 -m sameteacher.eval.human_eval_followup  ->  runs/results/main/human_eval/followup.json
"""
import argparse

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from sameteacher.analysis import main_shared_teacher as h1
from sameteacher.analysis.teacher_match import embed_cached
from sameteacher.core.config import add_common_args, load_yaml, read_json, write_json
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, results_dir
from sameteacher.core.text import clean
from sameteacher.eval import human_eval as he
from sameteacher.eval.embed import embed_texts, rows_dense
from sameteacher.eval.rewrite_check import majority
from sameteacher.eval.stats import percentile_interval

SECTION = "main_human_eval_followup"
KINDS = ("same", "different")
CACHE = "human_followup_grid"


def unit(vectors) -> np.ndarray:
    """Rows scaled to length 1. A row of zeros (an answer with no term TF-IDF keeps, such as a string
    of zeros) stays zero, so its cosine with any answer is 0."""
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def labeled_pairs(run_dir, cfg) -> tuple[list, dict, np.ndarray, np.ndarray]:
    """Item ids, their two texts, the majority label (1 = Y) and the kind of each pair."""
    folder = he.check_dir(run_dir)
    key = read_json(folder / "key_DO_NOT_OPEN.json")
    labels, texts = he.read_sheets(folder, cfg[he.SECTION]["annotators"])
    ids = sorted(labels)
    y = np.array([majority(list(labels[i].values())) == "Y" for i in ids], dtype=float)
    return ids, texts, y, np.array([key[i]["kind"] for i in ids])


def pair_cosines(ids, texts, embedder) -> np.ndarray:
    """Cosine of the two answers of each pair; TF-IDF is fit on these answers only."""
    flat = [clean(t) for i in ids for t in texts[i]]
    vectors = unit(rows_dense(embed_texts(flat, embedder), list(range(len(flat)))))
    return np.einsum("ij,ij->i", vectors[0::2], vectors[1::2])


def fit_logistic(x, y, iterations: int = 100) -> np.ndarray:
    """Intercept and slope of P(y = 1) = expit(a + b x), by Newton's method."""
    design = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(iterations):
        p = expit(design @ w)
        hessian = design.T @ (design * (p * (1 - p))[:, None])
        step = np.linalg.solve(hessian + 1e-9 * np.eye(2), design.T @ (y - p))
        w = w + step
        if np.abs(step).max() < 1e-10:
            break
    return w


def grid_cells(main_cfg) -> dict:
    """Per kind, the cells as lists of (weight, (base, teacher), (base, teacher)) assignments."""
    out = {}
    for kind, cells in he.cells(main_cfg).items():
        out[kind] = []
        for (x, y), (a, b) in cells:
            assignments = [((x, a), (y, b))] if a == b else [((x, a), (y, b)), ((x, b), (y, a))]
            out[kind].append([(1 / len(assignments), left, right) for left, right in assignments])
    return out


def grid_distribution(run_dir, cfg, prompts, embedder, edges) -> tuple[np.ndarray, np.ndarray]:
    """Per prompt and kind, the mean cosine and the weighted histogram of cosines of the answer pairs
    the labeled pairs were drawn from."""
    main_cfg = cfg["main_grid"]
    answers = he.usable_answers(run_dir, main_cfg)
    texts, index = [], {}
    for group in sorted(answers):
        for pid in prompts:
            rows = answers[group].get(pid, [])
            index[(pid, group)] = [len(texts), len(texts) + len(rows)]
            texts += [clean(t) for _, t in rows]
    if embedder == "tfidf":
        vectors = embed_texts(texts, embedder)
        block = lambda s, e: vectors[s:e]
        cosine = lambda u, v: np.asarray((u @ v.T).todense())
    else:
        vectors = unit(embed_cached(run_dir, texts, index, embedder, name=CACHE,
                                    note="Rows are the usable main-grid answers on the H1 prompts, full and cleaned. Keys: prompt|base|teacher|seed."))
        block = lambda s, e: vectors[s:e]
        cosine = lambda u, v: u @ v.T
    seeds, cells = main_cfg["seeds"], grid_cells(main_cfg)
    mean = np.zeros((len(prompts), len(KINDS)))
    hist = np.zeros((len(prompts), len(KINDS), len(edges) - 1))
    for n, pid in enumerate(prompts):
        for k, kind in enumerate(KINDS):
            for cell in cells[kind]:
                for share, (bx, tx), (by, ty) in cell:
                    for i in seeds:
                        for j in seeds:
                            c = np.clip(cosine(block(*index[(pid, (bx, tx, i))]), block(*index[(pid, (by, ty, j))])), -1, 1)
                            w = share / (len(cells[kind]) * len(seeds) ** 2 * c.size)
                            mean[n, k] += c.sum() * w
                            hist[n, k] += np.histogram(c, bins=edges)[0] * w
    return mean, hist


def pairs_needed(diff: float, rate: float, alpha: float, power: float):
    """Pairs per kind for a two-sided test of two proportions (normal approximation)."""
    if not np.isfinite(diff) or diff <= 0:
        return None
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil(z * z * 2 * rate * (1 - rate) / diff ** 2))


def analyze(run_dir, cfg) -> dict:
    section = cfg[SECTION]
    ids, texts, y, kind = labeled_pairs(run_dir, cfg)
    groups = {k: np.flatnonzero(kind == k) for k in KINDS}
    cosines = {e: pair_cosines(ids, texts, e) for e in section["embedders"]}
    link = section["link_embedder"]
    if link not in cosines:
        cosines[link] = pair_cosines(ids, texts, link)
    prompts = sorted(h1.build_texts(run_dir, cfg)[2])
    bins = int(round(2 / section["bin_width"]))
    edges = np.linspace(-1, 1, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    grid_mean, grid_hist = grid_distribution(run_dir, cfg, prompts, link, edges)

    def stats(same, different, chosen) -> dict:
        out = {}
        for e in section["embedders"]:
            c = cosines[e]
            out[f"pairs {e} same"] = c[same].mean()
            out[f"pairs {e} different"] = c[different].mean()
            out[f"pairs {e} same minus different"] = c[same].mean() - c[different].mean()
        both = np.concatenate([same, different])
        a, b = fit_logistic(cosines[link][both], y[both])
        fitted = expit(a + b * cosines[link])
        out["link intercept"], out["link slope"] = a, b
        out["link implied, labeled pairs, same minus different"] = fitted[same].mean() - fitted[different].mean()
        g = grid_mean[chosen]
        out["grid cosine same"], out["grid cosine different"] = g[:, 0].mean(), g[:, 1].mean()
        out["grid cosine same minus different"] = (g[:, 0] - g[:, 1]).mean()
        implied = grid_hist[chosen] @ expit(a + b * centers)
        out["grid implied same"], out["grid implied different"] = implied[:, 0].mean(), implied[:, 1].mean()
        out["grid implied same minus different"] = (implied[:, 0] - implied[:, 1]).mean()
        return out

    point = stats(groups["same"], groups["different"], np.arange(len(prompts)))
    rng = np.random.default_rng(section["bootstrap_seed"])
    draws = []
    for _ in range(section["bootstrap"]):
        same = rng.choice(groups["same"], groups["same"].size)
        different = rng.choice(groups["different"], groups["different"].size)
        draws.append(stats(same, different, rng.integers(0, len(prompts), len(prompts))))
    out = {}
    for name, value in point.items():
        low, high = percentile_interval([d[name] for d in draws], section["ci_level"])
        out[name] = {"estimate": float(value), "ci_low": low, "ci_high": high}

    rate = float(y.mean())
    implied = out["grid implied same minus different"]
    out["pairs per kind needed"] = {
        "power": section["power"], "alpha": section["alpha"], "observed same-idea rate": rate,
        "at estimate": pairs_needed(implied["estimate"], rate, section["alpha"], section["power"]),
        "at ci_high": pairs_needed(implied["ci_high"], rate, section["alpha"], section["power"]),
        "at ci_low": pairs_needed(implied["ci_low"], rate, section["alpha"], section["power"]),
    }
    out["labeled pairs"] = {k: int(v.size) for k, v in groups.items()}
    out["prompts"] = len(prompts)
    out["link embedder"] = link
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for name in ("main_grid", h1.SECTION, he.SECTION, SECTION):
        verify_lock(cfg, args.run_dir, name)
    result = analyze(args.run_dir, cfg)
    path = results_dir(args.run_dir, MAIN, "human_eval") / "followup.json"
    write_json(path, result)
    for name in ("grid cosine same minus different", "grid implied same minus different", "link slope"):
        print(name, result[name])
    print(path)


if __name__ == "__main__":
    main()
