# Same Teacher, Same Ideas

When base models from different families are fine-tuned on synthetic data from the same teacher LLM, do they converge in the content of their answers to open-ended prompts, beyond surface style? This repository holds the code, configuration, tests and result summaries for that question.

## Layout

```
configs/
  experiment.yaml     run parameters and analysis thresholds, one section per study part
  models.yaml         hosted model backends
sameteacher/
  core/               config and I/O, model client, mock backend, text cleaning, run paths,
                      config locks, and the experimental grid
  data/               prompt splits, teacher sampling, quality checks, training data
  train/              LoRA training, student sampling, drivers for the pilot and the main grid
  eval/               embeddings, energy distance and bootstrap, blinded human-evaluation sheets
  style_norm/         neutral-style rewriting of student answers and normalized retention
  analysis/           teacher distance and the shared-teacher effect
tests/                offline tests on a mock backend (no network, no cost)
runs/                 outputs; only result summaries, prompt splits and locks are tracked
```

`sameteacher/core/paths.py` documents the layout of `runs/`.

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
```

Hosted teachers and rewriters are called through Vertex AI. Authenticate with `gcloud auth login` and set the default project; `python3 -m sameteacher.core.check_models` confirms that each configured model responds and that sampling varies. Prices in `configs/models.yaml` are placeholders used only by the budget guard.

## Pipeline

All commands run from the repository root. Steps that call hosted models resume where they stopped and halt at `--max-usd`.

| Step | Module | Where |
|---|---|---|
| Draw the pilot prompt splits from Infinity-Chat | `sameteacher.data.make_splits` | local |
| Draw the main test split from the untouched pool | `sameteacher.data.make_test_split` | local |
| Sample teachers (and calibration controls) on a split | `sameteacher.data.sample_teachers` | local, hosted models |
| Quality check of teacher answers | `sameteacher.data.quality_check` | local |
| Teacher distance and its secondary checks | `sameteacher.analysis.teacher_distance`, `sameteacher.analysis.teacher_distance_checks` | local |
| Blinded teacher-pair annotation sheets | `sameteacher.eval.blind_review` | local |
| Pilot training data from Dolly prompts | `sameteacher.data.pilot_training_data` | local, hosted models |
| Teacher draws for the main grid | `sameteacher.data.training_draws` | local, hosted models |
| Train and sample the pilot students | `sameteacher.train.run_pilot` | GPU machine |
| Train and sample the main-grid students | `sameteacher.train.run_main` | GPU machine |
| Pilot shared-teacher effect and carryover | `sameteacher.analysis.pilot_shared_teacher` | local or GPU |
| Pilot style normalization and retention | `sameteacher.style_norm.pilot_retention` | local, hosted models |

Analysis parameters in `configs/experiment.yaml` are frozen with `python3 -m sameteacher.core.lock --section <name>` before results are seen. Analysis scripts refuse to run if a locked section changes.

## Results tracked in git

- `runs/results/pilot/teacher_distance/`: teacher distances on the screen and confirm splits, robustness and quality summaries
- `runs/results/pilot/shared_teacher/`: pilot shared-teacher effect, embedder robustness, per-cell distance matrix
- `runs/results/pilot/style_norm/`: pilot style normalization
- `runs/results/pilot/qualitative/`: labels for the qualitative example
- `runs/models_meta/`: training metadata for each pilot student
- `runs/prompts/`: the prompt splits
- `runs/locks/`: frozen config sections
