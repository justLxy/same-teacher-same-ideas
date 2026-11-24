"""Figure 4: how the amount of training data and a later fine-tuning stage change the shared-teacher effect.

(a), (b) From runs/results/main/extra/scaling.json: students of Qwen3-4B-Base and Olmo-3-7B with DeepSeek or Gemma,
    trained for 600 steps on 500, 1,000, 2,500 or all 4,938 distinct examples (one seed below 4,938, the three
    main-grid seeds at 4,938), 181 prompts, written and rewritten. (a) S, with the carryover S / (T - N) of the
    written answers printed above each size. (b) T, with seed noise N dashed.
(c) From runs/results/main/stretch/persistence.json: the eight students of Qwen and Olmo with DeepSeek or Gemma
    (seeds 0 and 1) after the teacher stage and after 600 further steps on human-written answers, as written,
    188 prompts. S is the shared-teacher effect; T the distance between the two teachers' students on one base;
    G the distance between one teacher's students on the two bases after seed noise is removed; H the distance
    between a student and the human-data students on its base. An arrow joins the two stages of each quantity.
    The row under G gives G of the students trained only on the human answers ("written G human").
Error bars are 95% prompt-bootstrap intervals.
"""
import matplotlib.pyplot as plt
import numpy as np

from figures.style import (INK, MID, NOTE, STAGE, TEXT, TICK, TITLE, VALUE, ci, intervals, panel_title, result, save,
                           setup)

SIZES = [("n500", 500), ("n1000", 1000), ("n2500", 2500), ("main", 4938)]
STAGE_ROWS = [("S", "shared-teacher effect $S$", 4.0), ("T", "two teachers, one base $T$", 3.0),
              ("G", "one teacher, two bases $G$", 2.0), ("H", "to the human-data students $H$", 0.0)]
HUMAN_ROW = ("$G$ of human-data students", 1.0)
BEFORE = dict(color=INK, marker="o", face=INK, label="after the teacher stage")
AFTER = dict(color=INK, marker="D", face="white", label="after a further stage on human answers")
HUMAN = dict(color=MID, marker="D", face="white", label=None)
W, H = TEXT, 1.74
BOTTOM = 0.40
AX_A, AX_B = (0.44, BOTTOM, 1.17, 1.12), (2.02, BOTTOM, 1.17, 1.12)
AX_C = (4.86, BOTTOM, 1.38, 0.98)


def axes(fig, rect):
    return fig.add_axes([rect[0] / W, rect[1] / H, rect[2] / W, rect[3] / H])


def scaling_panel(ax, s, quantity):
    n = np.array([size for _, size in SIZES])
    for stage, shift in (("written", 0.965), ("rewritten", 1.035)):
        vals = [ci(s[f"{stage} {quantity} {key}"]) for key, _ in SIZES]
        st = dict(STAGE[stage], label="as written" if stage == "written" else "rewritten")
        intervals(ax, vals, n * shift, st, horizontal=False, ms=3.9 if stage == "written" else 3.6, ls="-", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlim(360, 6700)
    ax.set_xticks(n)
    ax.set_xticklabels(["500", "1k", "2.5k", "4.9k"])
    ax.minorticks_off()
    ax.set_xlabel("Distinct training examples")


def stage_panel(ax, c, p):
    rows = {key: yy for key, _, yy in STAGE_ROWS}
    before = {key: ci(p[f"written {key} before"]) for key in rows}
    after = {key: ci(p[f"written {key} after"]) for key in rows}
    for key, yy in rows.items():
        ax.annotate("", xy=(after[key][2] + 0.006, yy), xytext=(before[key][1] - 0.004, yy),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color=MID, shrinkA=0, shrinkB=0, mutation_scale=5),
                    zorder=2)
    intervals(ax, list(before.values()), list(rows.values()), BEFORE, ms=4.0)
    intervals(ax, list(after.values()), list(rows.values()), AFTER, ms=3.5)
    human = ci(p["written G human"])
    intervals(ax, [human], [HUMAN_ROW[1]], HUMAN, ms=3.5)
    ax.text(human[2] + 0.006, HUMAN_ROW[1], f"{human[0]:.3f}", fontsize=VALUE, color=MID, va="center")
    ax.text(after["G"][0], rows["G"] + 0.40, f"{after['G'][0]:.3f}", fontsize=VALUE, color=INK, va="center",
            ha="center")
    lo, hi = -0.6, 4.6
    ax.set_ylim(lo, hi)
    ax.set_yticks([])
    ax.set_xlim(-0.006, 0.162)
    ax.set_xticks([0, 0.05, 0.1, 0.15])
    ax.axvline(0, color=INK, lw=0.5, zorder=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Energy distance")
    for label, yy, color, style in [(lab, yy, INK, "normal") for _, lab, yy in STAGE_ROWS] + \
                                   [(HUMAN_ROW[0], HUMAN_ROW[1], MID, "italic")]:
        yi = AX_C[1] + (yy - lo) / (hi - lo) * AX_C[3]
        c.text(AX_C[0] - 0.07, yi, label, ha="right", va="center", fontsize=TICK, color=color, style=style)
    # key of the two stages, on one line under the title
    yk = AX_C[1] + AX_C[3] + 0.105
    x = 3.42
    c.plot([x + 0.04], [yk], BEFORE["marker"], ms=4.0, mfc=INK, mec=INK, mew=0.9)
    c.text(x + 0.12, yk, "after the teacher stage", fontsize=VALUE, va="center")
    c.annotate("", xy=(x + 1.31, yk), xytext=(x + 1.13, yk),
               arrowprops=dict(arrowstyle="-|>", lw=0.6, color=MID, shrinkA=0, shrinkB=0, mutation_scale=5))
    c.plot([x + 1.43], [yk], AFTER["marker"], ms=3.5, mfc="white", mec=INK, mew=0.9)
    c.text(x + 1.51, yk, "after the human-data stage", fontsize=VALUE, va="center")


def main():
    setup()
    s = result("extra", "scaling.json")
    p = result("stretch", "persistence.json")
    fig = plt.figure(figsize=(W, H))
    c = fig.add_axes([0, 0, 1, 1])
    c.set_xlim(0, W)
    c.set_ylim(0, H)
    c.axis("off")
    ax_a, ax_b, ax_c = axes(fig, AX_A), axes(fig, AX_B), axes(fig, AX_C)
    scaling_panel(ax_a, s, "S")
    scaling_panel(ax_b, s, "T")
    ax_a.set_ylim(0, 0.092)
    ax_a.set_yticks([0, 0.02, 0.04, 0.06])
    ax_a.set_ylabel("Energy distance")
    # the carryover of the written answers at each size, which relates (a) to (b)
    for key, size in SIZES:
        ax_a.text(size, 0.0795, f"{s[f'written carryover {key}']['estimate']:.2f}", fontsize=VALUE, ha="center",
                  va="center", color=INK)
    ax_a.text(0.02, 0.985, "carryover $S\\,/\\,(T-N)$, as written:", transform=ax_a.transAxes, fontsize=NOTE,
              color=MID, style="italic", ha="left", va="top")
    ax_b.set_ylim(0, 0.44)
    ax_b.axhline(s["written N"]["estimate"], ls=(0, (3, 2)), lw=0.7, color=MID, zorder=1)
    ax_b.text(520, s["written N"]["estimate"] + 0.014, "seed noise $N$", fontsize=VALUE, color=MID)
    ax_b.legend(loc="upper right", handlelength=1.5, handletextpad=0.3, borderaxespad=0.0, labelspacing=0.25)
    stage_panel(ax_c, c, p)
    top = H - 0.035
    for x, tag, text in ((0.03, "a", "Shared-teacher effect $S$"), (1.63, "b", "Teacher gap on one base $T$"),
                         (3.42, "c", "A second stage on human answers")):
        c.text(x, top, f"({tag})", fontsize=TITLE, weight="bold", va="top")
        c.text(x + 13.5 / 72, top, text, fontsize=TITLE, va="top")
    return save(fig, "data_training", TEXT)


if __name__ == "__main__":
    print(main())
