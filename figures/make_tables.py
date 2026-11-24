"""LaTeX rows of the paper's tables, read from the result summaries so that no number is typed by hand.

  python3 -m figures.make_tables      ->  paper/tables/{h1_h2,h3,markers}.tex
"""
import json

from figures.style import RESULTS, ROOT

OUT = ROOT / "paper" / "tables"


def load(*parts):
    return json.loads(RESULTS.joinpath(*parts).read_text())


def num(x, digits=2):
    s = f"{x:.{digits}f}"
    return s.replace("-", "$-$") if x < 0 else s


def est_ci(e, scale=1.0, digits=2):
    return (f"{num(e['estimate'] * scale, digits)} {{\\scriptsize[{num(e['ci_low'] * scale, digits)}, "
            f"{num(e['ci_high'] * scale, digits)}]}}")


ROWS = [("pooled", "All pairs"), ("teacher deepseek-gemma", "DeepSeek--Gemma"), ("teacher deepseek-glm", "DeepSeek--GLM"),
        ("teacher gemma-glm", "Gemma--GLM"), ("base X-Y", "Qwen--Olmo"), ("base X-Z", "Qwen--SmolLM3"),
        ("base Y-Z", "Olmo--SmolLM3")]


def h1_h2():
    h1 = load("shared_teacher", "h1.json")
    h2 = load("style_norm", "h2.json")
    l2 = load("style_norm", "h2_second_rewriter.json")
    lines = []
    for key, label in ROWS:
        if key == "teacher deepseek-gemma":
            lines.append("\\midrule\n\\multicolumn{6}{l}{\\textit{Teacher pair}} \\\\")
        if key == "base X-Y":
            lines.append("\\midrule\n\\multicolumn{6}{l}{\\textit{Base pair}} \\\\")
        pair = key.split(" ", 1)[1] if key.startswith("teacher") else None
        dist = num(h1["teacher_reference"][pair]["estimate"]) if pair else "--"
        cells = [label, dist, est_ci(h1[key]["S"], 100), est_ci(h1[key]["carryover"]),
                 est_ci(h2[key]["R"]), est_ci(l2[key]["R second"])]
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def robustness():
    b1, t1 = (load("shared_teacher", f) for f in ("robustness__bge-small-en-v1.5.json", "robustness__tfidf.json"))
    b2, t2 = (load("robustness", f) for f in ("h2__bge-small-en-v1.5.json", "h2__tfidf.json"))
    lines = []
    for key, label in ROWS:
        if key in ("teacher deepseek-gemma", "base X-Y"):
            lines.append("\\midrule")
        cells = [label] + [num(d[key][q]["estimate"]) for d, q in ((b1, "carryover"), (t1, "carryover"),
                                                                    (b2, "R"), (t2, "R"))]
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def position(e):
    """The H3 index i, +1 at the source teacher's students and -1 at the rewriting teacher's, as the position
    (1 - i) / 2 between them: 0 at the source teacher's students, 1 at the rewriting teacher's."""
    return {"estimate": (1 - e["estimate"]) / 2, "ci_low": (1 - e["ci_high"]) / 2, "ci_high": (1 - e["ci_low"]) / 2}


def stacked(e, digits=2):
    """Estimate over its interval, for narrow columns."""
    return (f"\\makecell{{{num(e['estimate'], digits)}\\\\[-1pt]{{\\scriptsize[{num(e['ci_low'], digits)}, "
            f"{num(e['ci_high'], digits)}]}}}}")


def h3():
    h = load("h3", "h3.json")
    rows = [("cross deepseek in gemma", "\\makecell[l]{DeepSeek's answers,\\\\rewritten by Gemma}"),
            ("cross gemma in deepseek", "\\makecell[l]{Gemma's answers,\\\\rewritten by DeepSeek}"),
            ("cross pooled", "\\makecell[l]{Both directions}"),
            ("self pooled", "\\makecell[l]{Each teacher's own\\\\answers, rewritten by it}")]
    lines = []
    for key, label in rows:
        if key in ("cross pooled", "self pooled"):
            lines.append("\\midrule")
        cells = [label] + [stacked(position(h[f"{m} {key}"])) for m in ("normalized", "length", "formatting")]
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def markers():
    m = load("qualitative", "markers.json")["markers"]
    lines = []
    order = [("frames the island as an ecosystem", "gemma"), ("says a sword would transform Tracer", "gemma"),
             ("funds the island through tourism", "gemma"), ("blames unexpanded glob patterns", "deepseek")]
    labels = {"frames the island as an ecosystem": "Dog island as an ecosystem",
              "says a sword would transform Tracer": "A sword transforms Tracer",
              "funds the island through tourism": "Island funded by tourism",
              "blames unexpanded glob patterns": "Unexpanded glob patterns"}
    names = {"gemma": "Gemma", "deepseek": "DeepSeek"}
    for i, (name, teacher) in enumerate(order):
        if i == 2:
            lines.append("\\midrule\n\\multicolumn{4}{l}{\\textit{A fact in the teacher's own answers}} \\\\")
        s = m[name]["shares"]
        others = [t for t in ("deepseek", "gemma", "glm") if t != teacher]
        other_students = max(s[f"{b} {t}"] for b in "XYZ" for t in others)
        pct = lambda v: f"{100 * v:.0f}"
        cells = [f"{labels[name]} ({names[teacher]})", pct(s[f"teacher {teacher}"]),
                 " / ".join(pct(s[f"{b} {teacher}"]) for b in "XYZ"), f"$\\leq${pct(other_students)}"]
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def categories():
    c = load("qualitative", "categories.json")["categories"]
    lines = []
    for name, e in c.items():
        short = {"Creative Content Generation": "Creative writing", "Analytical and Interpretive Questions": "Analytical",
                 "Speculative and Hypothetical Scenarios": "Speculative", "Concept Explanations": "Concept explanation",
                 "Recommendations": "Recommendations", "Problem Solving": "Problem solving", "other": "Other (11 kinds)"}
        label = short[name]
        lines.append(" & ".join([label, str(e["h1_prompts"]), est_ci(e["S"], 100), est_ci(e["R"])]) + " \\\\")
    return "\n".join(lines) + "\n"


def untuned():
    u = load("extra", "untuned.json")
    lines = []
    for pair, label in (("X-Y", "Qwen--Olmo"), ("X-Z", "Qwen--SmolLM3"), ("Y-Z", "Olmo--SmolLM3")):
        cells = [label]
        for stage in ("written", "rewritten"):
            cells += [num(u[f"{stage} U {pair}"]["estimate"], 3), num(u[f"{stage} G same {pair}"]["estimate"], 3),
                      est_ci(u[f"{stage} U - G same {pair}"], digits=3)]
        lines.append(" & ".join(cells) + " \\\\")
    cells = ["All pairs"]
    for stage in ("written", "rewritten"):
        cells += [num(u[f"{stage} U"]["estimate"], 3), num(u[f"{stage} G same"]["estimate"], 3),
                  est_ci(u[f"{stage} U - G same"], digits=3)]
    lines.append("\\midrule\n" + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("h1_h2", h1_h2), ("robustness", robustness), ("h3", h3), ("markers", markers),
                     ("categories", categories), ("untuned", untuned)):
        (OUT / f"{name}.tex").write_text(f"% generated by figures/make_tables.py; do not edit\n{fn()}")
        print(OUT / f"{name}.tex")


if __name__ == "__main__":
    main()
