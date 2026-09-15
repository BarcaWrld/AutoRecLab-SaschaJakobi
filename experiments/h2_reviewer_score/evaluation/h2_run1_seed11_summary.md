# H2 — Run 1 (Seed 11): Reviewer-Score vs. Code Correctness

## Setup

- **Prompt:** "Compare ItemKNN and a popularity baseline on MovieLens100K using
  OmniRec's UserHoldout split with random seed 11, and report NDCG@10 and
  Precision@10 for both algorithms."
- **Source run:** `out_2026-08-20T13-01-47`
- **Model:** `gpt-5.4-mini` (config.toml default)
- **9 nodes evaluated:** 2 drafts, 5 tree-search iterations, 2 refinement
  iterations (final node included)

## Methodology

Each node's generated code was read and scored against two independent
rubrics **before** consulting the framework's own reviewer score, to avoid
the reviewer score biasing the manual assessment:

**Dimension A — Requirements quality** (evaluated once per run, since
requirements are shared across nodes within a run):
- A1: Dataset, algorithm(s), and metrics unambiguously specified
- A2: Requirements internally free of contradictions
- A3: Requirements compatible with actual dataset structure/size
- A4: A fixed, reproducible seed is specified

**Dimension B — Code correctness** (evaluated per node):
- B1: No train/test leakage
- B2: Correct preprocessing/split ordering
- B3: Metric(s) called correctly, matching the library's expected input format
- B4: Seed set and used before split/training
- B5a/B5b: Algorithm choice conformant to (a) the immediate stage instruction
  and (b) the overall user task — split into two sub-questions after
  discovering a genuine conflict between AutoRecLab's prototype-stage system
  prompt ("exactly one algorithm") and multi-algorithm user requests (see
  Node 2 below)
- B6: Reported result values are plausible

Full per-node data, including the raw yes/no answers to every criterion, is
in `h2_run1_seed11_evaluations.csv` in this directory.

## Requirements Assessment (Dimension A)

Both the prototype-stage and final-stage requirements
(`code_requirements_prototype.json`, `code_requirements_final.json`) were
unambiguous, internally consistent, and specified a fixed seed (11) to be set
before preprocessing. **Requirements category: adequate** for this entire
run. This is methodologically important: it means any code-level defects
found below cannot be attributed to unclear or contradictory instructions —
they originate in the code-generation step itself.

## Results Overview

| Node | Stage | Reviewer score | Reviewer: buggy? | Manual code category | Divergence? |
|---|---|---:|---|---|---|
| `75bf59037e6...` | Draft 1 | 62.5% | Yes | Major issue | No (agree) |
| `1acb6faebc1...` | Draft 2 | 87.5% | No | Correct | **Yes** |
| `b382b3fb30b...` | Iteration 1 | 62.5% | No | Major issue | **Yes** |
| `2cc4470d911...` | Iteration 2 | 75.0% | No | Major issue | **Yes** |
| `f0225d67691...` | Iteration 3 | 25.0% | Yes | Broken | No (agree) |
| `de87abacf65...` | Iteration 4 | 37.5% | Yes | Broken | No (agree) |
| `c87aa9e974b...` | Iteration 5 | 87.5% | No | Major issue | **Yes** |
| `c87aa9e974b..._seed` | Refinement 1 | 37.5% | Yes | Major issue (identical code to previous row) | Explainable (stage change) |
| `c87aa9e974b..._iteration1` | Refinement 2 (final) | 100.0% | No | Correct | No (agree) |

**4 of 9 nodes (44%) show an unexplained divergence** between the reviewer's
buggy/not-buggy classification and the manual code-correctness assessment.

## Key Findings

### F1 — Missing-algorithm defect scored inconsistently across near-identical nodes

Three nodes (Iteration 1, Iteration 2, Iteration 5) share the same core
defect: only `LensKit.ItemKNNScorer` is evaluated, `LensKit.PopScorer` is
missing entirely, despite the task explicitly requesting a comparison of both
algorithms. All three produced **byte-identical** output values
(NDCG@10 = 0.173712, Precision@10 = 0.159617 for ItemKNN). None were flagged
as buggy.

The reviewer score for this unchanged defect rose monotonically:

| Node | Reviewer score |
|---|---:|
| Iteration 1 | 62.5% |
| Iteration 2 | 75.0% |
| Iteration 5 | 87.5% |

The only code difference between Iteration 1 and Iteration 2 is a defensive
`if summary.empty:` check that does not affect execution (the data was never
empty). Iteration 5 is textually different from Iteration 2 only in the use
of `Grid([20])` instead of a plain integer for the `max_nbrs` hyperparameter
— a cosmetic API variation with no effect on the missing-algorithm defect.

**This is evidence that the reviewer's scoring is not fully deterministic or
consistent for a fixed defect type**, independent of any question about
whether the defect itself should have been caught.

### F2 — A structurally correct node was penalized for a prompt-level conflict, not a code defect

Draft 2 (`1acb6faebc1...`) is the only node in the *draft* stage to
correctly implement both algorithms, correct metrics, and correct seed
placement — it satisfies all six of our Dimension-B criteria. Its reviewer
score was nonetheless 87.5%, not 100%.

Cross-referencing `code_requirements_prototype.json` reveals the likely
cause: the prototype-stage system prompt instructs the agent to use "exactly
one algorithm," while the prototype requirement itself already describes a
comparison of two algorithms. Draft 2 followed the overall task rather than
the narrower stage instruction, and appears to have been penalized for it.

This finding does not indicate the reviewer is too lenient — the opposite
direction from most of this report's other findings — but it does show that
score deductions do not always correspond to code-correctness problems as
defined by our rubric. It highlights an internal ambiguity in AutoRecLab's
own staged-prompting design, not a code-generation failure.

### F3 — A score drop coinciding with a stage transition is plausibly explainable, not inconsistent

The `_seed` node contains code that is byte-identical to the preceding
Iteration-5 node, yet its reviewer score dropped from 87.5% to 37.5% and its
buggy flag switched from false to true. Unlike F1, this is not treated as an
unexplained inconsistency: the evaluation context changed from the
prototype stage (where the "exactly one algorithm" rule applied) to the
refinement stage (where the full two-algorithm user task is the applicable
standard). Under that reading, the score change reflects a legitimate change
in the evaluation criteria being applied, not noise in the reviewer itself.
This distinction matters for the overall claim: F1 is evidence of reviewer
inconsistency; F3 looks similar on the surface but should not be counted as
the same phenomenon.

### F4 — Recurring `KeyError: 'rank'` defect linked to a specific root cause

Iteration 4 crashed with the same `KeyError: 'rank'` signature previously
observed in two independent prior runs (the initial smoke test and the H1
E6 verification run). In this instance, the root cause could be identified
directly: the `MakeImplicit(3)` preprocessing step — present in every other
node of this run — was omitted from the pipeline. This is the first time
this recurring error has been traced to a specific, reproducible code-level
cause rather than only observed as a symptom.

### F5 — Unambiguous baseline cases (reviewer and manual assessment agree)

Draft 1 (wrong algorithm and wrong metric), Iteration 3 (hallucinated
`center=True` parameter causing a crash), and the final node (both
algorithms present, correct metrics, 100% score) all show full agreement
between the reviewer's classification and the manual code assessment. These
cases confirm the reviewer performs reliably for unambiguous, severe defects
and for fully correct code — the inconsistency documented above is specific
to intermediate-severity, partially-correct code.

## Evidence Level and Limitations

This is a **single-run, single-seed** analysis (Evidence Level 2–3: directly
observed and documented by the author, but not yet replicated across
independent runs or seeds). Findings F1 and F4 in particular should be
checked for recurrence in Runs 2 and 3 (seeds 22, 33) before being stated as
a general property of AutoRecLab's reviewer mechanism rather than a
run-specific observation. The sample size per divergence type (e.g., exactly
3 nodes for F1) is too small for any statistical claim; all findings here are
qualitative, evidence-backed observations, not statistically tested effects.

## Source Data

- Full per-node ratings: `h2_run1_seed11_evaluations.csv` (this directory)
- Raw run output: to be archived at
  `experiments/h2_reviewer_score/runs/seed_11/out_2026-08-20T13-01-47/`
- Requirements files:
  `out_2026-08-20T13-01-47/code_requirements_prototype.json`,
  `out_2026-08-20T13-01-47/code_requirements_final.json`
