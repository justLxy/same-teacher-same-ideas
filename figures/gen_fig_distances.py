"""Figure 1: overview. (a) the controlled design and the measurement, (b) the distances, (c) the two shares.

Three sets of answers enter the figure, and each mark is computed within one of them from unrounded values:

  main analysis        runs/results/main/shared_teacher/h1.json. 188 prompts, 30 answers per student, as written.
                       Panel (c), first bar: S, T - N ("net") and the carryover S / (T - N) with its interval.
  rewrite comparison   runs/results/main/style_norm/h2.json. 187 prompts, the first 15 usable answers per student
                       and prompt, the same answers as written and rewritten by gpt-oss-120b. Panel (c), second
                       bar: S written, S rewritten and their ratio R with its interval.
  with untuned bases   runs/results/main/extra/untuned.json. The 187 prompts and first 15 answers of the rewrite
                       comparison, with the three untuned bases added (their answers enter the word budgets, so
                       S here is 0.0317 as written and 0.0250 rewritten, against 0.0316 and 0.0249 above).
                       Panel (b): raw energy distances, that is, with the seed noise still in the student rows:
                         N          same base and teacher, another seed
                         same       different bases, same teacher, every seed pair
                         U          the three base models prompted with three examples, no fine-tuning
                         different  different bases, different teachers
                       The two brackets are S on this set: different - same, as written and rewritten.

(a) The grid of 27 students (three seeds per cell). In one block of two bases and two teachers (grey) the six
    pairs are drawn: same teacher on two bases (solid), different teachers on two bases (dashed), and different
    teachers on one base (dotted, T). The block is an example: the grid has 9 such blocks (3 pairs of bases times
    3 pairs of teachers), and every distance and S average over all of them and over the seed pairs; the key
    says so. The icons before the rows of (b) repeat these pairs.
Error bars are 95% prompt-bootstrap intervals.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from figures.style import (BLOCK, INK, LABEL, MID, NOTE, PALE, REWRITTEN, STAGE, TEACHER, TEACHER_NAME, TEXT, TICK,
                           TITLE, VALUE, WRITTEN, ci, intervals, panel_title, result, save, setup)

W, H = TEXT, 2.56
# rows of panel (b) from top to bottom, with their height in the axes: two references, then the two kinds of
# student pairs whose difference is S
ROWS = [("N", "same base, same teacher\n(seed noise $N$)", 3.3),
        ("U", "different bases,\nno fine-tuning", 2.3),
        ("same", "different bases,\nsame teacher", 1.0),
        ("different", "different bases,\ndifferent teachers", 0.0)]
BASES = ["Qwen3-4B", "Olmo-3-7B", "SmolLM3-3B"]
SOLID, DASHED, DOTTED = "-", (0, (3.2, 2.0)), (0, (0.9, 1.7))
COLS, ROWS_Y, HW, HH = [0.995, 1.545, 2.095], [1.78, 1.46, 1.14], 0.165, 0.08     # the grid of panel (a), inches
AX = (3.90, 1.13, 2.31, 1.20)                       # the axes of panel (b): left, bottom, width, height (inches)
LABEL_X, ICON_X = 3.56, 3.73                        # right edge of the row labels, centre of the pair icons


def box(c, x, y, w, h, **kw):
    style = dict(boxstyle="round,pad=0,rounding_size=0.035", lw=0.5, ec=PALE, fc="white", zorder=2)
    style.update(kw)
    c.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, **style))


def dots(c, xs, ys, color, ms=3.3, hollow=False, zorder=4):
    c.plot(xs, ys, "o", ms=ms, mfc="white" if hollow else color, mec=color, mew=0.7 if hollow else 0.0,
           zorder=zorder, clip_on=False)


def line(c, p, q, ls, lw=1.0, color=INK, zorder=3):
    c.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=zorder,
           clip_on=False)


def design(c):
    """Panel (a): training grid, pair key and measurement, drawn in inches on the canvas `c`."""
    c.text(0.03, H - 0.035, "(a)", fontsize=TITLE, weight="bold", va="top")
    c.text(0.03 + 13.5 / 72, H - 0.035, "Design: only the teacher differs", fontsize=TITLE, va="top")
    teachers = list(TEACHER)
    # training: teachers answer the prompts, each base is fine-tuned on each teacher's answers
    c.text(0.70, 2.215, "Teachers answer", fontsize=VALUE, ha="right", va="center", color=INK)
    c.text(0.70, 2.095, "4,938 prompts", fontsize=VALUE, ha="right", va="center", color=INK)
    c.text(0.70, 1.945, "LoRA, one recipe", fontsize=NOTE, ha="right", va="center", color=MID, style="italic")
    for t, x in zip(teachers, COLS):
        box(c, x, 2.155, 0.47, 0.21, fc=TEACHER[t], ec="none")
        c.text(x, 2.15, TEACHER_NAME[t], ha="center", va="center", fontsize=TICK, color="white", weight="bold",
               zorder=3)
        c.add_patch(FancyArrowPatch((x, 2.045), (x, ROWS_Y[0] + HH + 0.02), arrowstyle="-|>", mutation_scale=6,
                                    lw=0.8, color=TEACHER[t], shrinkA=0, shrinkB=0, zorder=3))
    # one block of two bases and two teachers
    c.add_patch(FancyBboxPatch((COLS[0] - HW - 0.07, ROWS_Y[1] - HH - 0.06), COLS[1] - COLS[0] + 2 * HW + 0.14,
                               ROWS_Y[0] - ROWS_Y[1] + 2 * HH + 0.12, boxstyle="round,pad=0,rounding_size=0.06",
                               fc=BLOCK, ec="none", zorder=1))
    for base, y in zip(BASES, ROWS_Y):
        c.text(0.77, y, base, ha="right", va="center", fontsize=TICK)
        for t, x in zip(teachers, COLS):
            box(c, x, y, 2 * HW, 2 * HH)
            dots(c, [x - 0.09, x, x + 0.09], [y] * 3, TEACHER[t])
    (xa, xb), (y1, y2) = COLS[:2], ROWS_Y[:2]
    for x in (xa, xb):                                               # same teacher, two bases
        line(c, (x, y1 - HH), (x, y2 + HH), SOLID, lw=1.1)
    line(c, (xa + HW, y1 - HH), (xb - HW, y2 + HH), DASHED)          # different teachers, two bases
    line(c, (xb - HW, y1 - HH), (xa + HW, y2 + HH), DASHED)
    for y in (y1, y2):                                               # different teachers, one base
        line(c, (xa + HW, y), (xb - HW, y), DOTTED, lw=1.1)
    # key of the pairs and of the grey block: the block is one example, every block enters the averages
    yk = 0.885
    line(c, (0.06, yk), (0.28, yk), SOLID, lw=1.1)
    c.text(0.33, yk, "same teacher", fontsize=VALUE, va="center")
    line(c, (1.08, yk), (1.30, yk), DASHED)
    c.text(1.35, yk, "different teachers", fontsize=VALUE, va="center")
    yk = 0.755
    line(c, (0.06, yk), (0.28, yk), DOTTED, lw=1.1)
    c.text(0.33, yk, "different teachers on one base ($T$)", fontsize=VALUE, va="center")
    yk = 0.625
    c.add_patch(FancyBboxPatch((0.06, yk - 0.045), 0.22, 0.09, boxstyle="round,pad=0,rounding_size=0.03", fc=BLOCK,
                               ec="none", zorder=1))
    c.text(0.33, yk, "one of 9 blocks (2 bases $\\times$ 2 teachers), all averaged", fontsize=VALUE, va="center")
    # measurement
    c.plot([0.06, 2.26], [0.525, 0.525], color=PALE, lw=0.5)
    c.text(0.06, 0.405, "Test: 188 unseen prompts, 30 answers per student.", fontsize=VALUE, va="center")
    c.text(0.06, 0.27, "Energy distance between two students\u2019 answers,", fontsize=VALUE, va="center")
    y = 0.135
    c.plot([0.105], [y], STAGE["written"]["marker"], ms=4.2, mfc=WRITTEN, mec=WRITTEN, mew=0.9)
    c.text(0.185, y, "as written", fontsize=VALUE, va="center", color=WRITTEN)
    c.text(0.70, y, "and", fontsize=VALUE, va="center")
    c.plot([0.98], [y], STAGE["rewritten"]["marker"], ms=3.9, mfc="white", mec=REWRITTEN, mew=0.9)
    c.text(1.06, y, "rewritten into plain prose.", fontsize=VALUE, va="center", color=REWRITTEN)


def pair_icon(c, key, x, y):
    """The pair of a row of (b), drawn as in the grid: columns are teachers, rows are bases."""
    a, b = TEACHER["deepseek"], TEACHER["gemma"]
    dx, dy = 0.05, 0.05
    if key == "N":
        box(c, x, y, 0.19, 0.11, zorder=2)
        dots(c, [x - 0.04, x + 0.04], [y, y], a, ms=3.0)
    elif key == "same":
        line(c, (x, y + dy), (x, y - dy), SOLID, lw=1.1)
        dots(c, [x, x], [y + dy, y - dy], a, ms=3.4)
    elif key == "U":
        dots(c, [x, x], [y + dy, y - dy], MID, ms=3.2, hollow=True)
    elif key == "different":
        line(c, (x - dx, y + dy), (x + dx, y - dy), DASHED)
        dots(c, [x - dx], [y + dy], a, ms=3.4)
        dots(c, [x + dx], [y - dy], b, ms=3.4)
    elif key == "T":
        line(c, (x - 0.06, y), (x + 0.06, y), DOTTED, lw=1.1)
        dots(c, [x - 0.06], [y], a, ms=3.4)
        dots(c, [x + 0.06], [y], b, ms=3.4)


def distances(fig, c, u):
    """Panel (b): raw distances on the set with untuned bases, and S as the gap between two adjacent rows."""
    ax = fig.add_axes([AX[0] / W, AX[1] / H, AX[2] / W, AX[3] / H])
    y = {key: yy for key, _, yy in ROWS}
    lo, hi = -0.82, 3.82
    off = {"written": 0.18, "rewritten": -0.18}          # in each row: as written above, rewritten below
    left = {"same"}                                      # rows whose values are printed left of the interval
    for stage in ("written", "rewritten"):
        color = STAGE[stage]["color"]
        vals = {key: ci(u[f"{stage} {key}"]) for key in y}
        intervals(ax, list(vals.values()), [y[k] + off[stage] for k in vals], stage,
                  ms=4.2 if stage == "written" else 3.9)
        for key, (e, low, top) in vals.items():
            if key in left:
                ax.text(low - 0.0022, y[key] + off[stage], f"{e:.3f}", ha="right", va="center", fontsize=VALUE,
                        color=color)
            else:
                ax.text(top + 0.0022, y[key] + off[stage], f"{e:.3f}", ha="left", va="center", fontsize=VALUE,
                        color=color)
    # S on this set, from unrounded estimates: different - same. The bracket of the written answers stands above
    # the two rows and the bracket of the rewritten answers hangs below them, so that no leader crosses a mark.
    for stage, far, near, sign in (("written", "different", "same", +1), ("rewritten", "same", "different", -1)):
        color = STAGE[stage]["color"]
        same, diff = u[f"{stage} same"]["estimate"], u[f"{stage} different"]["estimate"]
        yb = y[near] + off[stage] + sign * 0.37
        for key in (far, near):
            x = u[f"{stage} {key}"]["estimate"]
            ax.plot([x, x], [y[key] + off[stage] + sign * 0.2, yb], color=color, lw=0.6, ls=(0, (1, 1.6)), zorder=1)
        ax.annotate("", xy=(same, yb), xytext=(diff, yb),
                    arrowprops=dict(arrowstyle="<->", lw=0.8, color=color, shrinkA=0, shrinkB=0, mutation_scale=6))
        ax.text(same - 0.003, yb, f"$S$ = {diff - same:.3f}", ha="right", va="center", fontsize=VALUE, color=color)
    ax.set_yticks(list(y.values()))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(lo, hi)
    ax.set_xlim(0, 0.132)
    ax.set_xticks([0, 0.025, 0.05, 0.075, 0.1, 0.125])
    ax.set_xlabel("Energy distance between two students")
    handles = [plt.Line2D([], [], ls="none", marker=STAGE[s]["marker"], mfc=STAGE[s]["face"], mec=STAGE[s]["color"],
                          mew=0.9, ms=4.2 if s == "written" else 3.9) for s in ("written", "rewritten")]
    ax.text(0.1315, 3.56, "raw distances (seed noise included)\n187 prompts, first 15 answers", fontsize=NOTE,
            color=MID, style="italic", ha="right", va="center", linespacing=1.1)
    ax.legend(handles, ["as written", "rewritten"], loc="lower left", bbox_to_anchor=(0.012, 0.115),
              handletextpad=0.2, borderaxespad=0.0, labelspacing=0.25, handlelength=1.2)
    for key, label, yy in ROWS:
        yi = AX[1] + (yy - lo) / (hi - lo) * AX[3]
        c.text(LABEL_X, yi, label, ha="right", va="center", fontsize=TICK, linespacing=1.05)
        pair_icon(c, key, ICON_X, yi)
    c.text(2.50, H - 0.035, "(b)", fontsize=TITLE, weight="bold", va="top")
    c.text(2.50 + 13.5 / 72, H - 0.035, "Students on different bases are closer when they share a teacher", fontsize=TITLE,
           va="top")
    return ax


def shares(c, h1, h2):
    """Panel (c): the carryover (main analysis) and the retention R (rewrite comparison) as shares of a bar."""
    c.text(2.50, 0.685, "(c)", fontsize=TITLE, weight="bold", va="center")
    c.text(2.50 + 13.5 / 72, 0.685, "How much carries over, and how much of that survives rewriting", fontsize=TITLE,
           va="center")
    x0, wt, hb = AX[0], 1.62, 0.105
    p1, p2 = h1["pooled"], h2["pooled"]
    rows = [(0.49, "Carryover\n$S\\,/\\,(T-N)$", p1["carryover"], WRITTEN, PALE,
             f"$S$ = {p1['S']['estimate']:.3f} of $T-N$ = {p1['net']['estimate']:.3f}",
             "as written, 188 prompts, 30 answers"),
            (0.17, "Retention $R$\n$S$ rewritten$\\,/\\,S$ written", p2["R"], REWRITTEN, "#BBD6E8",
             f"$S$ = {p2['S rewritten']['estimate']:.3f} of {p2['S written']['estimate']:.3f}",
             "the same first 15 answers, 187 prompts")]
    for yc, label, share, color, track, parts, basis in rows:
        est, low, high = ci(share)
        c.text(LABEL_X, yc - 0.035, label, ha="right", va="center", fontsize=TICK, linespacing=1.05)
        c.add_patch(Rectangle((x0, yc - hb / 2), wt, hb, fc=track, ec="none", zorder=2))
        c.add_patch(Rectangle((x0, yc - hb / 2), wt * est, hb, fc=color, ec="none", zorder=3))
        c.plot([x0 + wt * low, x0 + wt * high], [yc, yc], color=INK, lw=0.9, zorder=4, solid_capstyle="butt")
        for v in (low, high):
            c.plot([x0 + wt * v] * 2, [yc - 0.028, yc + 0.028], color=INK, lw=0.9, zorder=4)
        c.text(x0 + wt + 0.07, yc, f"{100 * est:.0f}%", fontsize=10.5, weight="bold", color=color, va="center")
        c.text(x0 + wt + 0.41, yc - 0.006, f"[{100 * low:.0f}, {100 * high:.0f}]", fontsize=VALUE, color=INK,
               va="center")
        c.text(x0, yc - 0.125, parts, fontsize=NOTE, color=INK, va="center")
        c.text(x0 + wt + 0.70, yc - 0.125, basis, fontsize=NOTE, color=MID, va="center", ha="right",
               style="italic")


def main():
    setup()
    u = result("extra", "untuned.json")
    h1 = result("shared_teacher", "h1.json")
    h2 = result("style_norm", "h2.json")
    assert u["prompts_used"] == 187 and h2["prompts_used"] == 187 and h1["prompts_used"] == 188
    fig = plt.figure(figsize=(W, H))
    c = fig.add_axes([0, 0, 1, 1])
    c.set_xlim(0, W)
    c.set_ylim(0, H)
    c.axis("off")
    design(c)
    distances(fig, c, u)
    shares(c, h1, h2)
    return save(fig, "distances", TEXT)


if __name__ == "__main__":
    print(main())
