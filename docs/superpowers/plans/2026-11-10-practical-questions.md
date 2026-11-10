# Practical HAR Questions Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task. Execution is authorized in the current workspace; do not change frozen results or commit automatically.

**Goal:** Answer three practical HAR questions with an executed notebook and traceable evidence.

**Architecture:** One self-contained notebook reuses the established CNN-B design while varying channels. Archived test predictions supply descriptive analyses; new training accesses official train only. Outputs are exclusive, separate, and hashed.

**Tech Stack:** Python 3.12, TensorFlow, NumPy, pandas, Matplotlib, nbformat/nbclient, unittest.

**Spec:** `docs/research/practical-questions.md` (approved scope in conversation).

## Global Constraints

- Preserve `results/` and all files hashed by its lock.
- Five input groups, seeds 42/123/2026; same subject split and training settings as protocol 1.0.
- No new test inference. Label post-test exploration and validation selection limits explicitly.
- Work in current checkout under the existing sandbox fallback; no Git commits.

### Task 1: Notebook safety and scientific calculations

**Files:** Create `tests/test_practical_notebook.py`, `notebooks/03_practical_questions.ipynb`.

- [x] Write tests exercising channel selection (including order), train-only scaling, six-class metrics, archived-prediction validation, and exclusive output paths.
- [x] Run `.venv/Scripts/python.exe -m unittest discover -s tests -p test_practical_notebook.py -v` and confirm failure because notebook is absent.
- [x] Implement tagged helper cells; re-run tests until passing.

### Task 2: Controlled sensor study and practical explanations

**Files:** Complete notebook; create `practical_results/sensor_study_v1/`.

- [x] Implement direct `model.fit()` loop, configuration and source provenance, best-checkpoint reload and per-run CSV/JSON saving.
- [x] Implement archived test analyses, deterministic signal examples, all 15 validation metrics and paired comparisons, plots and `answers.md` with limitations.
- [x] Execute smoke through nbclient in a fresh kernel; verify reload and derived outputs.
- [x] Execute full 15-run study, save executed notebook, set inspect as safe default.
- [x] Execute inspect in a fresh kernel, verify artifact hashes and recompute metrics from predictions without mutation.

### Task 3: Review and integration

**Files:** Modify `README.md`, `.gitignore`; update plan checkboxes.

- [x] Document notebook modes, outputs and interpretation. Ignore bulky model files only, retain lightweight evidence.
- [x] Request independent code/scientific review and resolve important findings.
- [x] Run all unit tests, original `python -m har_project verify`, fresh notebook inspect, `git diff --check`; inspect figure layout and link targets.
- [x] Report actual findings and limits in Vietnamese, link notebook and answers.

## Verification evidence

- Full study completed: 15 runs, five groups, three seeds. Six figures and eight analysis CSVs, plus per-run evidence.
- Fresh-kernel inspect completed all code cells, recomputed metrics and matched generated answers; all 114 artifact hashes valid. All artifact sizes and modification timestamps unchanged by inspect.
- 33 unit tests passed. Original nine-run archive verified unchanged by `har_project verify`.
- Independent source review resolved smoke inspection, per-person Macro-F1 SD and directional error reporting. Final independent review verified all 15 prediction CSV metrics, summaries, paired deltas and 114 hashes; no remaining Important issues.
- 28 local links validated; figure layout inspected; `git diff --check` passed.
- Added `docs/research/practical-presentation.md` with evidence-based speaking notes and counterexamples. No report, slides, merge, commit or new test inference performed.
