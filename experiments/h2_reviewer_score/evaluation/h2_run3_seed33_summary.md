# H2 — Run 3 (Seed 33): Reviewer-Score vs. Code Correctness

## Setup

- **Prompt:** Identical to Runs 1 and 2, with "random seed 33".
- **Source run:** `out_2026-08-20T17-44-17`
- **Model:** `gpt-5.4-mini`
- **10 nodes total** (2 drafts, 5 tree-search iterations, 3 refinement
  iterations including the final node)

## Requirements Assessment (Dimension A)

Both requirement sets are unambiguous, internally consistent, and specify
seed 33 explicitly. **Requirements category: adequate** — the third
consecutive run with this assessment. Across all three runs (30 requirement
files total: 2 per run × ... actually 2 sets per run × 3 runs = 6 files),
requirements quality has never varied. This strengthens the conclusion that
requirements quality is not a driver of the defects and score inconsistencies
documented below.

## Results Overview

| Node | Stage | Reviewer score | Buggy? | Code category | Notes |
|---|---|---:|---|---|---|
| `3ab42888...` | Draft 1 | 25.0% | Yes | Broken | MakeImplicit missing + wrong metric (HR) |
| `c868c7d3...` | Draft 2 | 75.0% | No | Correct | Prototype-stage conflict (2 algorithms) |
| `5939076a...` | Iteration 1 | 12.5% | Yes | Broken | MakeImplicit missing + Precision omitted |
| `34f72300...` | Iteration 2 | 37.5% | Yes | Broken | MakeImplicit missing |
| `ca8f1af8...` | Iteration 3 | 37.5% | Yes | Broken | MakeImplicit missing |
| `a112de11...` | Iteration 4 | 62.5% | No | Correct | MakeImplicit restored; fully correct |
| `3964d81c...` | Iteration 5 | 37.5% | Yes | Broken (new type) | Post-processing defect, OmniRec usage itself correct |
| `..._seed` | Refinement 1 | 85.71% | No | Correct | Identical code to Draft 2 |
| `..._iteration1` | Refinement 2 | 71.43% | No | Correct | **Only difference from previous node: a cosmetic string** |
| `..._iteration2` | Refinement 3 (final) | 85.71% | No | Correct | Genuine minor improvement (extraction robustness) |

## Key Findings

### F6 — Fifth and sixth confirmations of the `MakeImplicit`-omission defect, within a single run

Draft 1, Iteration 1, Iteration 2, and Iteration 3 all omit `MakeImplicit`
from the preprocessing pipeline, causing the same `KeyError: 'rank'` crash
documented repeatedly across this project. This run is notable because
Iteration 4 finally restores `MakeImplicit` — the only one of the three H2
runs in which this specific recurring defect was successfully self-corrected
within the run, rather than persisting to the final node (contrast with Run
2, where it was never fixed).

Across all three runs, the `MakeImplicit`-omission defect has now been
independently observed **nine times** (Run 1: 1, Run 2: 7, Run 3: 6 — note
some counts include repeated occurrences of the same underlying defect
across debug iterations within a run). This is now one of the best-supported
findings in the entire project, spanning H1's verification run, the initial
smoke test, and all three H2 runs.

### F7 — A new, qualitatively distinct defect type: correct OmniRec usage, broken post-processing

Iteration 5 is the first node across all three runs where the OmniRec
pipeline itself — preprocessing, split, both algorithms, both metrics —
executed completely successfully, with correct values confirmed in the raw
log output, matching the run's final reported results exactly. The failure
occurred entirely in author-generated post-processing code that made an
incorrect assumption about the structure of `run_omnirec()`'s return value.

This finding matters methodologically: it shows that "buggy" nodes in
AutoRecLab's tree search conflate at least two distinct failure modes —
(a) incorrect usage of the OmniRec framework itself, and (b) ordinary
Python programming errors in generated glue/formatting code unrelated to
OmniRec. The reviewer's binary buggy flag does not distinguish between
these, even though they have different implications for what "the code is
wrong" means. A seventh evaluation criterion (B7: result-extraction
correctness) was introduced to capture this distinction going forward.

### F8 — The strongest divergence finding in the project: a cosmetic, functionally inert change coincides with a 14.3-point score drop

The clearest and most concerning finding across all three H2 runs. The
`_seed` node (85.71%) and the `_iteration1` node (71.43%) contain code that
is identical in every respect **except a single string literal** — the
`ExperimentPlan`'s `plan_name` argument, changed from
`'ml100k_itemknn_popularity_prototype'` to
`'ml100k_itemknn_popularity_final'`. This label has no effect on preprocessing,
algorithm execution, metric computation, or output; it is not read or
compared anywhere in the surrounding code.

All seven of this report's evaluation criteria (B1–B7) are satisfied
identically by both nodes. Both are in the refinement stage, so the
stage-transition explanation used for F3 (Run 1) and the analogous finding
in this run's `_seed` node does not apply here — there is no stage change
between these two nodes.

No code-level, task-level, or stage-level explanation was found for this
score drop during manual review. This is the strongest available evidence in
this project that the reviewer's scoring carries variance not attributable to
code correctness, task conformance, or evaluation-stage differences — i.e.,
noise in the reviewer mechanism itself, at least in the specific sense of the
score's sensitivity to functionally irrelevant code differences.

### F9 — Recurrence of prototype-stage/task-conflict inconsistency

Draft 2 in this run reproduces the same defect pattern as Run 1's Draft 2 —
both algorithms present during the prototype stage despite an "exactly one
algorithm" instruction — but the two nodes were scored differently for a
structurally equivalent situation (87.5% in Run 1 vs. 75.0% here). Combined
with the `_seed`-node score changes in both runs, this defect pattern now
has multiple, inconsistent scoring outcomes across independent runs.

## Evidence Level

F6 (MakeImplicit omission) is now Evidence Level 4: independently observed
across the smoke test, the H1 E6 run, and all three H2 runs, with a directly
identified and verified code-level cause each time. F7 and F8 are Evidence
Level 3: directly observed and documented within a single run, not yet
cross-checked for recurrence, but each supported by a clear, code-verified
mechanism (or, in F8's case, verified absence of any mechanism). F9 is
Evidence Level 2–3, based on two data points across two runs.

## Source Data

- Full per-node ratings: `h2_run3_seed33_evaluations.csv` (this directory)
- Raw run output: to be archived at
  `experiments/h2_reviewer_score/runs/seed_33/out_2026-08-20T17-44-17/`
