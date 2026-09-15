# H2 — Run 2 (Seed 22): Reviewer-Score vs. Code Correctness

## Setup

- **Prompt:** Identical to Run 1, with "random seed 22" substituted for
  "random seed 11".
- **Source run:** `out_2026-08-20T15-39-09`
- **Model:** `gpt-5.4-mini`
- **10 nodes total** (2 drafts, 5 tree-search iterations, 3 refinement
  iterations including the final node)

## Requirements Assessment (Dimension A)

Both `code_requirements_prototype.json` and `code_requirements_final.json`
are unambiguous, internally consistent, and specify seed 22 explicitly.
**Requirements category: adequate**, identical assessment to Run 1. This is
an important control: the requirements quality did not differ between a run
that succeeded (Run 1) and this run, which did not.

## Headline Result: Complete Run Failure

**No node in this run produced a valid NDCG@10 or Precision@10 result for
either algorithm.** All 10 nodes were flagged as buggy by the reviewer. The
final node (`adfa9c7c39b24bc69836895f96d4365f_iteration2`) scored only 11.1%
(1/9) and, per `summary.md`, reported `N/A` for both metrics on both
algorithms. This is a stark contrast to Run 1 (Seed 11), which reached a
fully correct, 100%-scored final result.

This is itself a significant finding for the broader project (see
Interpretation below), independent of the reviewer-score-consistency
question this hypothesis targets.

## Root Cause: An Unaddressed Defect Persisting Across the Entire Run

Seven of the ten nodes failed with `KeyError: 'rank'` inside
`omnirec/metrics/ranking.py`'s `make_topk_dict` — the same error signature
observed independently in the initial smoke test, the H1 E6 verification
run, and Run 1's Iteration 4. In all seven cases here, the cause was
verified directly: **the `MakeImplicit(3)` preprocessing step is missing
from the pipeline.**

Critically, this defect was **never fixed** across six consecutive debug
iterations (Iteration 1 through the final refinement node), despite:
- The algorithm-count defect (only one algorithm present) being correctly
  diagnosed and fixed by the refinement stage
- Substantial code restructuring occurring between iterations (more robust
  results-table extraction, improved plotting logic)
- The reviewer's bug descriptions consistently naming the `KeyError: 'rank'`
  symptom in every affected node

The debugging process addressed surface-level code robustness (defensive
extraction functions, plotting fallbacks) without ever tracing the error
back to the missing preprocessing call. This is evidence that AutoRecLab's
debug loop can iterate extensively on a wrong diagnosis without converging
on the actual defect — a finding about the framework's self-correction
capability, not only about reviewer scoring accuracy.

## Reviewer Consistency in This Run

Unlike Run 1, this run shows **no false-negative divergence** (no case
where the reviewer marked broken code as not-buggy). All 10 nodes were
correctly flagged as buggy given their actual broken state. This is
consistent with Run 1's finding (F5) that the reviewer performs reliably
for unambiguous, severe defects.

One score-variance pattern was observed, however:

| Node | Defect | Reviewer score |
|---|---|---:|
| Iteration 1 | Missing MakeImplicit | 37.5% |
| Iteration 2 | Missing MakeImplicit (identical) | 37.5% |
| Iteration 3 | Missing MakeImplicit (identical) | 37.5% |
| Iteration 4 | Missing MakeImplicit (identical) | 37.5% |
| Iteration 5 | Missing MakeImplicit (identical) | **25.0%** |

Four consecutive nodes with the identical underlying defect scored
identically (37.5%), then the fifth dropped to 25.0% without an identifiable
code-level cause for the drop (verified: `MakeImplicit` is absent in all
five in the same way). This is a smaller-magnitude analogue of Run 1's F1
finding (unexplained score drift for an unchanged defect), but in the
opposite direction (score fell rather than rose) and against a "broken"
rather than "major issue" baseline. It is noted here as a secondary,
weaker instance of the same phenomenon type, not a new independent finding.

The `_seed` → `_iteration1` transition (Refinement 1 to Refinement 2) shows
a score change (0.0% → 22.2%) coinciding with the algorithm-count defect
being fixed — this is a **legitimate, explainable score change** (the code
genuinely improved in one respect), not an inconsistency.

## Evidence Level and Limitations

Evidence Level 2–3, single run. The `MakeImplicit`-omission finding is now
supported by **two independent runs** (this run and Run 1's Iteration 4),
which raises its evidence level for the general claim ("this specific
preprocessing omission causes `KeyError: 'rank'`") to Level 3–4. The
weaker score-drift observation (37.5% → 25.0%) remains Level 2 and should
not be over-stated; the sample is a single instance.

## Source Data

- Full per-node ratings: `h2_run2_seed22_evaluations.csv` (this directory)
- Raw run output: to be archived at
  `experiments/h2_reviewer_score/runs/seed_22/out_2026-08-20T15-39-09/`
