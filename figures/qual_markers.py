"""Table 3 (exploratory): how often a teacher's framing or facts appear in its own answers and its students'.

The marker phrases were chosen after reading the answers to the prompts with the largest shared-teacher effect
(runs/results/main/qualitative/examples.json), so the counts illustrate the effect and do not test it. Shares
are over usable answers (sameteacher.core.text.is_answer): 30 per teacher, 90 per base and teacher (three
seeds). Writes runs/results/main/qualitative/markers.json.

  python3 -m figures.qual_markers
"""
import json
import re
from collections import defaultdict

from figures.style import RESULTS, ROOT
from sameteacher.core.text import is_answer

PROMPTS = {"dog island": "248eb190b86e2ec0dd2146eb1e324306", "tracer": "abe6b1b2823c2fdf268446127a1e74f7",
           "ketos": "3ac2a64bafaf4eb188d99a891651b0eb"}
MARKERS = {  # name: (prompt, regex, kind)
    "frames the island as an ecosystem": ("dog island", r"ecosystem", "framing"),
    "says a sword would transform Tracer": ("tracer", r"\btransform", "framing"),
    "funds the island through tourism": ("dog island", r"tourism", "fact"),
    "blames unexpanded glob patterns": ("ketos", r"glob", "fact"),
}
BASES = {"X": "Qwen", "Y": "Olmo", "Z": "SmolLM3"}
TEACHERS = ("deepseek", "gemma", "glm")


def load():
    answers = defaultdict(list)          # (group, prompt name) -> texts
    wanted = {v: k for k, v in PROMPTS.items()}
    for line in open(ROOT / "runs" / "teachers" / "main_test.jsonl"):
        r = json.loads(line)
        if r["prompt_id"] in wanted and r.get("variant", "baseline") == "baseline" and is_answer(r["text"]):
            answers[(f"teacher {r['teacher']}", wanted[r["prompt_id"]])].append(r["text"])
    for base in BASES:
        for teacher in TEACHERS:
            for seed in (0, 1, 2):
                for line in open(ROOT / "runs" / "students" / "main" / f"{base}__{teacher}__s{seed}.jsonl"):
                    r = json.loads(line)
                    if r["prompt_id"] in wanted and is_answer(r["text"]):
                        answers[(f"{base} {teacher}", wanted[r["prompt_id"]])].append(r["text"])
    return answers


def main():
    answers = load()
    out = {}
    for name, (prompt, pattern, kind) in MARKERS.items():
        rx = re.compile(pattern, re.IGNORECASE)
        row = {"prompt": prompt, "kind": kind, "pattern": pattern, "shares": {}, "n": {}}
        groups = [f"teacher {t}" for t in TEACHERS] + [f"{b} {t}" for t in TEACHERS for b in BASES]
        for g in groups:
            texts = answers[(g, prompt)]
            row["shares"][g] = sum(bool(rx.search(t)) for t in texts) / len(texts)
            row["n"][g] = len(texts)
        out[name] = row
    path = RESULTS / "qualitative" / "markers.json"
    path.write_text(json.dumps({"note": __doc__.strip().splitlines()[0], "markers": out}, indent=1))
    for name, row in out.items():
        s = row["shares"]
        print(f"{name:30s}", " ".join(f"{g}={s[g]:.2f}" for g in s))
    return path


if __name__ == "__main__":
    print(main())
