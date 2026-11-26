"""Figure 2: released models distilled from one teacher answer more alike, and stay as close as their bases.

(a) From runs/results/main/stretch/released.json (98 prompts, 20 samples per model, the answer after the
reasoning, first 150 words): energy distance between DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B
(same distillation source) and between Qwen2.5-7B-Instruct and Llama-3.1-Tulu-3-8B (other developers), as written and
rewritten by gpt-oss-120b.
(b) From runs/results/main/stretch/released_untuned.json (96 prompts, one common word budget per prompt, median 58
words): each released pair (marker) and the untuned bases it starts from (grey tick), with an arrow where the pair
moved; the number is the change d - U. Error bars are 95% prompt-bootstrap intervals.
"""
import matplotlib.pyplot as plt

from figures.style import COLUMN, MID, NOTE, STAGE, TICK, VALUE, ci, intervals, panel_title, result, save, setup

ROWS = [("d shared", "the two R1 distills"), ("d other pair", "Qwen2.5-Instruct\nand Tülu 3")]
PAIRS = [("r1", "the two R1 distills"), ("other", "Qwen2.5-Instruct\nand Tülu 3")]
W, H = COLUMN, 1.90
AX_A = (1.42, 1.25, 1.54, 0.47)    # left, bottom, width, height in inches
AX_B = (1.42, 0.37, 1.54, 0.47)
XMAX, DY = 0.34, 0.17
MS = {"written": 4.2, "rewritten": 3.9}


def axes(fig, box):
    return fig.add_axes([box[0] / W, box[1] / H, box[2] / W, box[3] / H])


def frame(ax, y, labels):
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=TICK, linespacing=1.05)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylim(min(y) - 0.5, max(y) + 0.5)
    ax.set_xlim(0, XMAX)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.grid(axis="y", visible=False)


def stage_handles():
    return [plt.Line2D([], [], ls="none", marker=STAGE[s]["marker"], mfc=STAGE[s]["face"], mec=STAGE[s]["color"],
                       mew=0.9, ms=MS[s]) for s in ("written", "rewritten")]


def panel_a(ax, rel):
    y = [1.0, 0.0]
    for stage, dy in (("written", DY), ("rewritten", -DY)):
        vals = [ci(rel[f"{stage} {key}"]) for key, _ in ROWS]
        intervals(ax, vals, [yy + dy for yy in y], stage, ms=MS[stage])
        for (e, _, top), yy in zip(vals, y):
            ax.text(top + 0.006, yy + dy, f"{e:.3f}", ha="left", va="center", fontsize=VALUE,
                    color=STAGE[stage]["color"])
    frame(ax, y, [lab for _, lab in ROWS])
    ax.legend(stage_handles(), ["as written", "rewritten"], loc="lower left", bbox_to_anchor=(0.03, 0.03),
              handletextpad=0.2, borderaxespad=0.0, labelspacing=0.25, handlelength=1.2)
    panel_title(ax, "a", "answers cut to 150 words", x=-0.9)


def panel_b(ax, base):
    y = [1.0, 0.0]
    grey = dict(color=MID, marker="|", face=MID, label=None)
    for stage, dy in (("written", DY), ("rewritten", -DY)):
        for (name, _), yy in zip(PAIRS, y):
            u, d = ci(base[f"{stage} U {name}"]), ci(base[f"{stage} d {name}"])
            change = base[f"{stage} change {name}"]["estimate"]
            intervals(ax, [u], [yy + dy], grey, ms=6.5, mew=1.1, zorder=2)
            if abs(d[0] - u[0]) > 0.02:
                ax.annotate("", xy=(d[0], yy + dy), xytext=(u[0], yy + dy), zorder=2,
                            arrowprops=dict(arrowstyle="-|>", color=STAGE[stage]["color"], lw=0.7,
                                            shrinkA=2.5, shrinkB=3.2, mutation_scale=5.5))
            intervals(ax, [d], [yy + dy], stage, ms=MS[stage])
            ax.text(max(u[2], d[2]) + 0.006, yy + dy, f"{change:+.3f}".replace("-", "\u2212"), ha="left", va="center", fontsize=VALUE,
                    color=STAGE[stage]["color"])
    frame(ax, y, [lab for _, lab in PAIRS])
    ax.set_xlabel("Energy distance between two models")
    # Name the grey ticks once, on the pair whose bases sit apart from its released models.
    u = base["written U other"]["estimate"]
    ax.text(u, y[1] + DY + 0.2, "untuned bases", ha="center", va="bottom", fontsize=NOTE, color=MID)
    panel_title(ax, "b", "change from their untuned bases, at 58 words", x=-0.9)


def main():
    setup()
    fig = plt.figure(figsize=(W, H))
    panel_a(axes(fig, AX_A), result("stretch", "released.json"))
    panel_b(axes(fig, AX_B), result("stretch", "released_untuned.json"))
    return save(fig, "released", COLUMN)


if __name__ == "__main__":
    print(main())
