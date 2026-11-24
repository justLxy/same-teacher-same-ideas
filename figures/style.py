"""Shared style of the paper figures: one visual specification at final print size.

Every figure is drawn at the exact size it has in the paper (acl.sty: column 7.7 cm, text width 16 cm) and saved
without cropping, so `\\includegraphics[width=\\columnwidth]` or `[width=\\textwidth]` places it at scale 1 and
the sizes below are the sizes on the page. `save` checks both: the page size of the PDF, and that no artist
reaches outside the canvas.

Type scale (pt on the page):  panel titles 8.5 · axis labels and row labels 8 · ticks, legends 7.5 ·
value labels and annotations 7 · notes 6.5 (the smallest size used).
Marks: points 4.2 pt with a 0.9 pt edge, interval bars 0.9 pt with 1.6 pt caps, axes 0.6 pt, grid 0.4 pt.

Colour has two meanings, the same in every figure:
  how an answer is compared   as written = blue, filled circle; rewritten into plain prose = vermillion, hollow
                              square (WRITTEN, REWRITTEN)
  whose answers a model saw   DeepSeek = green, Gemma = purple, GLM = ochre (TEACHER); none of them is blue or
                              red, so a teacher is never mistaken for a comparison
Everything else is neutral: dark grey for a single series, mid grey for references.

Every figure script reads result summaries under runs/results/main/ only and writes fig_<name>.pdf (vector,
TrueType fonts) and fig_<name>.png (300 dpi) into figures/. Run one with `python3 -m figures.gen_fig_<name>`
from the repository root, or all with `python3 -m figures.make_all`.
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
RESULTS = ROOT / "runs" / "results" / "main"
COLUMN, TEXT = 219.08 / 72.27, 455.24 / 72.27     # inches: \columnwidth and \textwidth of acl.sty (TeX points)

# type scale, pt on the page
TITLE, LABEL, TICK, VALUE, NOTE = 8.5, 8.0, 7.5, 7.0, 6.5
# marks
MS, MEW, ELW, CAP = 4.2, 0.9, 0.9, 1.6

OKABE = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
         "vermillion": "#D55E00", "pink": "#CC79A7", "black": "#000000", "grey": "#8C8C8C", "light": "#D9D9D9"}
WRITTEN, REWRITTEN = OKABE["blue"], OKABE["vermillion"]
TEACHER = {"deepseek": "#009E73", "gemma": "#A8559B", "glm": "#B8930A"}
TEACHER_NAME = {"deepseek": "DeepSeek", "gemma": "Gemma", "glm": "GLM"}
INK, MID, PALE, BLOCK = "#333333", "#8C8C8C", "#D9D9D9", "#EDEDED"
STAGE = {"written": dict(color=WRITTEN, marker="o", face=WRITTEN, label="as written"),
         "rewritten": dict(color=REWRITTEN, marker="s", face="white", label="rewritten into plain prose")}


def result(*parts):
    return json.loads((RESULTS.joinpath(*parts)).read_text())


def ci(entry):
    """(estimate, low, high) of a result entry."""
    return entry["estimate"], entry["ci_low"], entry["ci_high"]


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": LABEL, "axes.titlesize": TITLE, "axes.labelsize": LABEL, "xtick.labelsize": TICK,
        "ytick.labelsize": TICK, "legend.fontsize": TICK, "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "axes.labelpad": 3, "axes.grid": True, "grid.color": "#B0B0B0", "grid.alpha": 0.35, "grid.linewidth": 0.4,
        "axes.axisbelow": True, "lines.linewidth": 1.0, "lines.markersize": MS, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": None, "figure.dpi": 150})


def panel_title(ax, tag, text, x=0.0, y=1.0, pad=4):
    """A bold panel tag and a plain title, left-aligned at (x, y) in axes coordinates."""
    ax.annotate(f"({tag})", (x, y), xycoords="axes fraction", xytext=(0, pad), textcoords="offset points",
                ha="left", va="bottom", fontsize=TITLE, weight="bold", annotation_clip=False)
    ax.annotate(text, (x, y), xycoords="axes fraction", xytext=(13.5, pad), textcoords="offset points",
                ha="left", va="bottom", fontsize=TITLE, annotation_clip=False)


def intervals(ax, entries, y, stage, horizontal=True, ms=MS, **kw):
    """Points with 95% intervals in the style of one stage ('written' or 'rewritten') or a custom style."""
    st = STAGE[stage] if isinstance(stage, str) else stage
    est = [e[0] for e in entries]
    err = [[e[0] - e[1] for e in entries], [e[2] - e[0] for e in entries]]
    args = dict(fmt=st["marker"], color=st["color"], mfc=st["face"], mec=st["color"], mew=MEW, ms=ms,
                elinewidth=ELW, capsize=CAP, capthick=ELW, label=st.get("label"), zorder=3)
    args.update(kw)
    if horizontal:
        return ax.errorbar(est, y, xerr=err, **args)
    return ax.errorbar(y, est, yerr=err, **args)


def save(fig, name, width):
    """Write the PDF and PNG at the drawn size and check that the page is `width` inches wide and that nothing
    is cut off."""
    fig.canvas.draw()
    w, h = fig.get_size_inches()
    assert abs(w - width) < 1e-6, f"{name}: drawn {w:.3f} in wide, expected {width:.3f}"
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    inside = tight.x0 > -0.005 and tight.y0 > -0.005 and tight.x1 < w + 0.005 and tight.y1 < h + 0.005
    message = (f"{name}: content leaves the canvas ({tight.x0:.3f}, {tight.y0:.3f}, {tight.x1:.3f}, "
               f"{tight.y1:.3f}) of {w:.3f} x {h:.3f}")
    if os.environ.get("FIG_DRAFT"):          # while laying out a figure: report and still write the file
        print("ok" if inside else message, f"| used ({tight.x0:.3f}, {tight.y0:.3f}, {tight.x1:.3f}, {tight.y1:.3f})")
    else:
        assert inside, message
    # no creation date in the PDF, so redrawing unchanged data gives byte-identical files
    fig.savefig(OUT / f"fig_{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(OUT / f"fig_{name}.png")
    plt.close(fig)
    return OUT / f"fig_{name}.pdf"
