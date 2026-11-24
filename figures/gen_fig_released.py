"""Figure 2: released models distilled from one teacher answer more alike.

From runs/results/main/stretch/released.json (98 prompts, 20 samples per model, the answer after the reasoning,
first 150 words): energy distance between DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B (same
distillation source), between Qwen2.5-7B-Instruct and Llama-3.1-Tulu-3-8B (other developers), and the mean of the
two mixed pairs, as written and rewritten by gpt-oss-120b. Error bars are 95% prompt-bootstrap intervals.
"""
import matplotlib.pyplot as plt

from figures.style import COLUMN, STAGE, TICK, VALUE, ci, intervals, result, save, setup

ROWS = [("d shared", "the two R1 distills\n(Qwen and Llama)"),
        ("d other pair", "Qwen2.5-Instruct\nand Tülu 3"),
        ("d mixed", "an R1 distill and the other\nmodel of the other family")]
W, H = COLUMN, 1.38
AX = (1.42, 0.37, 1.54, 0.98)      # left, bottom, width, height in inches


def main():
    setup()
    rel = result("stretch", "released.json")
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([AX[0] / W, AX[1] / H, AX[2] / W, AX[3] / H])
    y = [2.0, 1.0, 0.0]
    for stage, dy in (("written", 0.17), ("rewritten", -0.17)):
        vals = [ci(rel[f"{stage} {key}"]) for key, _ in ROWS]
        intervals(ax, vals, [yy + dy for yy in y], stage, ms=4.2 if stage == "written" else 3.9)
        for (e, _, top), yy in zip(vals, y):
            ax.text(top + 0.006, yy + dy, f"{e:.3f}", ha="left", va="center", fontsize=VALUE,
                    color=STAGE[stage]["color"])
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in ROWS], fontsize=TICK, linespacing=1.05)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlim(0, 0.32)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel("Energy distance between two models")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], ls="none", marker=STAGE[s]["marker"], mfc=STAGE[s]["face"], mec=STAGE[s]["color"],
                          mew=0.9, ms=4.2 if s == "written" else 3.9) for s in ("written", "rewritten")]
    ax.legend(handles, ["as written", "rewritten"], loc="lower left", bbox_to_anchor=(0.03, 0.03),
              handletextpad=0.2, borderaxespad=0.0, labelspacing=0.25, handlelength=1.2)
    return save(fig, "released", COLUMN)


if __name__ == "__main__":
    print(main())
