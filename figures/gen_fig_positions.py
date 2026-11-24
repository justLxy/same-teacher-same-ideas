"""Figure 4: where students trained on rewritten teacher answers sit between the two teachers' students.

From runs/results/main/h3/h3.json (Qwen and Olmo, two seeds, 187 prompts). The analysis stores the index i,
+1 at the main-grid students of the source teacher and -1 at those of the rewriting teacher; the figure shows the
position (1 - i) / 2, which is 0 at the source teacher's students and 1 at the rewriting teacher's (the same
values as paper/tables/h3.tex). Content is measured on answers rewritten into plain prose by gpt-oss-120b, length
and formatting on the written answers. Rows of each lane: the two directions, then both directions pooled and
the students trained on each teacher's rewrites of its own answers (for them, 1 is the other teacher's students).
The direction marks carry the colour of the rewriting teacher. Error bars are 95% prompt-bootstrap intervals.
"""
import matplotlib.pyplot as plt

from figures.make_tables import position
from figures.style import COLUMN, INK, MID, PALE, TEACHER, TICK, VALUE, ci, intervals, result, save, setup

LANES = [("normalized", "Content", "(plain prose)"), ("length", "Length", "(words)"),
         ("formatting", "Formatting", "(share)")]
SERIES = [("cross deepseek in gemma", 0.29, dict(color=TEACHER["gemma"], marker=">", face=TEACHER["gemma"]),
           "DeepSeek’s answers, rewritten by Gemma"),
          ("cross gemma in deepseek", 0.0, dict(color=TEACHER["deepseek"], marker=">", face=TEACHER["deepseek"]),
           "Gemma’s answers, rewritten by DeepSeek"),
          ("cross pooled", -0.29, dict(color=INK, marker="o", face=INK), "both directions"),
          ("self pooled", -0.29, dict(color=MID, marker="D", face="white"), "a teacher’s rewrites of its own answers")]
W, H = COLUMN, 1.76
AX = (0.74, 0.46, 2.19, 0.90)      # left, bottom, width, height in inches


def main():
    setup()
    h = result("h3", "h3.json")
    fig = plt.figure(figsize=(W, H))
    c = fig.add_axes([0, 0, 1, 1])
    c.set_xlim(0, W)
    c.set_ylim(0, H)
    c.axis("off")
    ax = fig.add_axes([AX[0] / W, AX[1] / H, AX[2] / W, AX[3] / H])
    lo, hi = -0.5, len(LANES) - 0.5
    for i, (measure, name, unit) in enumerate(LANES):
        yc = len(LANES) - 1 - i
        for key, dy, style, _ in SERIES:
            e = ci(position(h[f"{measure} {key}"]))
            intervals(ax, [e], [yc + dy], style, ms=3.8 if style["marker"] != "o" else 4.0)
            ax.text(e[2] + 0.014, yc + dy, f"{e[0]:.2f}", fontsize=VALUE, color=style["color"], va="center",
                    zorder=2, bbox=dict(fc="white", ec="none", pad=0.6))
        if i:
            ax.axhline(yc + 0.5, color=PALE, lw=0.5, zorder=1)
        yi = AX[1] + (yc - lo) / (hi - lo) * AX[3]
        c.text(AX[0] - 0.07, yi + 0.055, name, ha="right", va="center", fontsize=TICK)
        c.text(AX[0] - 0.07, yi - 0.065, unit, ha="right", va="center", fontsize=VALUE, color=MID)
    ax.axvline(0.5, color=MID, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(lo, hi)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    c.text(AX[0] - 0.04, 0.115, "source teacher\u2019s\nstudents", fontsize=VALUE, ha="left", va="center",
           linespacing=1.05)
    c.text(AX[0] + AX[2] + 0.04, 0.115, "rewriting teacher\u2019s\nstudents", fontsize=VALUE, ha="right", va="center",
           linespacing=1.05)
    c.text(AX[0] + AX[2] / 2, 0.115, "midpoint", fontsize=VALUE, ha="center", va="center", color=MID)
    # key, above the axes
    y0 = H - 0.075
    rows = [(0.10, y0, 0), (0.10, y0 - 0.13, 1), (0.10, y0 - 0.26, 2), (1.18, y0 - 0.26, 3)]
    for x, y, k in rows:
        _, _, style, label = SERIES[k]
        c.plot([x], [y], style["marker"], ms=3.8, mfc=style["face"], mec=style["color"], mew=0.9)
        c.text(x + 0.085, y, label, fontsize=VALUE, va="center")
    return save(fig, "positions", COLUMN)


if __name__ == "__main__":
    print(main())
