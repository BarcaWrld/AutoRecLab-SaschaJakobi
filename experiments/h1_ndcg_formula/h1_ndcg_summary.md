# H1: Reliability of the NDCG Normalization in OmniRec

## Hypothesis

OmniRec's implementation of NDCG@k deviates from the normalization convention
described in the foundational cumulated-gain literature (Järvelin & Kekäläinen,
2002) when a user has fewer relevant test items than the cutoff *k*. This
deviation is hypothesized to be systematic rather than incidental, and
potentially large enough to affect reported metric values on real-world
datasets.

## Reference Definition

Järvelin and Kekäläinen (2002) define the ideal gain vector for a given query
(user) as one constructed from *that query's own relevant document set*: gain
values are placed at the top of the ranking according to the number of
relevant documents actually available, and the remaining positions are filled
with a gain of zero (Järvelin & Kekäläinen, 2002, Eq. 4). Because zero-gain
positions do not contribute to a discounted sum, the ideal DCG (IDCG) at cutoff
*k* is, in effect, computed over `min(|Rel(u)|, k)` terms, where `|Rel(u)|` is
the number of relevant items for user *u*. The normalized score for user *u*
is then

```
NDCG@k(u) = DCG@k(u) / IDCG@k(u),  with IDCG@k(u) based on min(|Rel(u)|, k)
```

This convention is explicitly motivated in the source paper as a design
advantage over related measures: *"[...] our nCG is based on the recall base
of the search topic, the first ranks of the ideal vector are not affected at
all by extension of the evaluation to further ranks"* (Järvelin & Kekäläinen,
2002, Sec. 2.4).

## Observed Implementation (OmniRec 1.0.0)

Inspection of `omnirec/metrics/ranking.py` (class `NDCG`, `calculate()`) shows
that the ideal discounted gain is computed as a fixed cumulative sum over
exactly `k` positions, independent of how many relevant items the user
actually has:

```python
ideal_discounted_gain_per_k = [
    discounted_gain_per_k[: ind + 1].sum()
    for ind in range(len(discounted_gain_per_k))
]
```

For any user with `|Rel(u)| < k`, this implementation therefore normalizes
against an ideal score that assumes `k` achievable relevant items, even when
fewer than `k` exist. This is a deviation from the reference definition above.

## Evidence

All values below were computed independently by the author and are fully
reproducible from the scripts referenced in each subsection
(`experiments/h1_ndcg_formula/`).

### E1 — Control Case (`|Rel(u)| ≥ k`)

For a user with more relevant items than the cutoff (`|Rel(u)| = 12`, `k =
10`), both the reference definition and OmniRec's implementation reduce to
summing the same `k` discount terms, and produce identical results:

| Metric | Reference | OmniRec |
|---|---:|---:|
| IDCG@10 | 4.5436 | 4.5436 |
| NDCG@10 | 1.0000 | 1.0000 |

This confirms that the deviation is conditional on `|Rel(u)| < k`, not a
general implementation error.

*Script: `ndcg_analysis.py` (control block).*

### E2 — Boundary Series

Holding `k = 10` fixed and varying `|Rel(u)|`, the ratio of OmniRec's fixed
IDCG to the reference IDCG (`min(|Rel(u)|, k)`-based) was computed
analytically:

| `\|Rel(u)\|` | IDCG (reference) | IDCG (OmniRec) | Ratio (OmniRec / reference) |
|---:|---:|---:|---:|
| 1  | 1.0000 | 4.5436 | 4.5436 |
| 2  | 1.6309 | 4.5436 | 2.7859 |
| 3  | 2.1309 | 4.5436 | 2.1322 |
| 5  | 2.9485 | 4.5436 | 1.5410 |
| 7  | 3.6380 | 4.5436 | 1.2489 |
| 9  | 4.2545 | 4.5436 | 1.0679 |
| 10 | 4.5436 | 4.5436 | 1.0000 |
| 11 | 4.5436 | 4.5436 | 1.0000 |

The deviation is monotonically decreasing in `|Rel(u)|` and converges exactly
to 1.0 at `|Rel(u)| = k`, consistent with the mechanism identified above.

*Script: `experiments/h1_ndcg_formula/ndcg_analysis.py`.*

### E3 — Empirical Prevalence on MovieLens100K

To assess whether this is a marginal edge case or a practically relevant
issue, the actual test-set relevance counts per user were measured using
OmniRec's own preprocessing pipeline (`RecSysDataSet.use_dataloader` with
`UserHoldout(validation_size=0.1, test_size=0.1)`, called via the public
`process()` API).

- Users in test set: 943
- Median relevant test items per user: 7
- Mean: 11.07 (SD: 10.09)

| Cutoff *k* | Users with `\|Rel(u)\| < k` | Share |
|---:|---:|---:|
| 5  | 305 / 943 | 32.3% |
| 10 | 560 / 943 | 59.4% |
| 20 | 784 / 943 | 83.1% |

At `k = 10` — the cutoff most commonly used in the reproduced AutoRecLab
experiments — a majority of users (59.4%) fall into the affected range. The
result was reproduced across two independent runs (using the internal
`_process()` hook and the public `process()` entry point respectively) with
identical distributions, indicating a fixed random seed in OmniRec's splitting
routine (`get_random_state()`).

*Script: `experiments/h1_ndcg_formula/empirical_distribution.py`.*

### E4 — Independent Reference Implementation

An independent NDCG implementation following the reference definition
(`min(|Rel(u)|, k)`-based IDCG) was written and validated on a minimal test
case (2 relevant items, both ranked first, `k = 10`), producing the expected
value of 1.0.

*Script: `experiments/h1_ndcg_formula/reference_ndcg.py`.*

### E5 — Direct Comparison Against OmniRec's Implementation

Using identical synthetic prediction/test inputs, OmniRec's `NDCG` class
(`omnirec.metrics.ranking.NDCG`) was called directly and compared against the
independent reference implementation (E4) across the same `|Rel(u)|` sweep as
E2:

| `\|Rel(u)\|` | NDCG (OmniRec) | NDCG (reference) | Difference |
|---:|---:|---:|---:|
| 1  | 0.2201 | 1.0000 | -0.7799 |
| 2  | 0.3590 | 1.0000 | -0.6410 |
| 3  | 0.4690 | 1.0000 | -0.5310 |
| 5  | 0.6489 | 1.0000 | -0.3511 |
| 7  | 0.8007 | 1.0000 | -0.1993 |
| 9  | 0.9364 | 1.0000 | -0.0636 |
| 10 | 1.0000 | 1.0000 |  0.0000 |
| 15 | 1.0000 | 1.0000 |  0.0000 |

This result independently confirms E1 and E2 using the actual OmniRec code
path (not a manual re-derivation), closing the loop between the analytical
argument and the library's real behavior.

### E6 — Verification on a Real AutoRecLab Run

To confirm that the deviation is not an artifact of synthetic test cases,
a dedicated AutoRecLab run was executed
(prompt: *"Build a simple ItemKNN recommender on the MovieLens100K dataset
using OmniRec's UserHoldout split, and report NDCG@10."*), producing a
`LensKit.ItemKNNScorer` result with `MakeImplicit(3)` and
`UserHoldout(validation_size=0.15, test_size=0.15)` (fixed seed 42), reported
as NDCG@10 = 0.172938288508597 in the run's `summary.md`.

The exact same preprocessing pipeline (same seed, same split configuration)
was independently rebuilt, and the run's real predictions
(`predictions.json`, 1,366,317 rows across 943 users, columns
`[user, item, score, rank]`) were loaded directly from the run's checkpoint
directory — no synthetic data was used for this verification.

| | Value |
|---|---:|
| NDCG@10 reported in `summary.md` | 0.172938288508597 |
| NDCG@10 recomputed with OmniRec's own code, same data | 0.172938288508597 |
| NDCG@10 with the independent reference implementation, same data | 0.210225356333718 |
| Absolute difference (OmniRec − reference) | −0.037287 |
| Relative difference | 21.56% |

The exact match between the reported and recomputed OmniRec value (agreement
to 15 decimal places) confirms that the original run's evaluation pipeline
was correctly reconstructed. The 21.56% relative difference between OmniRec's
reported score and the reference implementation, on real predictions from a
real end-to-end AutoRecLab run, is consistent in direction and magnitude with
the synthetic analysis (E1–E2) and the empirical prevalence finding (E3):
MovieLens100K's user-level sparsity, combined with `k=10`, places the
majority of test users in the regime where the deviation applies.

*Script: `experiments/h1_ndcg_formula/real_run_comparison.py`. Source run:
`out_2026-08-19T16-27-35`, final node
`f612857fa00949189e2f4149fb34eb1e_iteration2`.*

## Interpretation

The five evidence pieces above are mutually consistent and were obtained
through independent methods (source code inspection, manual derivation,
independent reimplementation, and direct invocation of the library code). This
convergence is what supports the claim, not any single computation in
isolation.

**What is established (Evidence Level 4 — directly verified by the author
against library source code, an independent reimplementation, and a real
end-to-end AutoRecLab run):**
OmniRec's NDCG@k systematically overestimates the difficulty of achieving a
perfect score for users with fewer than `k` relevant test items, because it
normalizes against a fixed-length ideal ranking rather than one bounded by the
user's actual relevant-item count. On MovieLens100K with `k = 10`, this
affects the majority of users (59.4%), and produces a 21.56% relative
difference between OmniRec's reported metric and the reference definition on
real predictions from an actual reproduced AutoRecLab experiment (E6).

**What remains open (Evidence Level 1 — not yet verified):** Whether this
divergence from the reference definition is an unintentional implementation
error or a deliberate (if undocumented) design choice by the OmniRec
maintainers has not been established. No OmniRec issue tracker or commit
history review has been performed as part of this analysis. This distinction
should be investigated before characterizing the finding as a "bug" rather
than a "deviation from a specific literature convention."

**Practical implication for reproduced AutoRecLab experiments:** Because the
deviation is not random noise but a directional bias correlated with
`|Rel(u)|`, it could differentially affect algorithms or datasets with
different user-interaction sparsity, independent of actual ranking quality.
This is a caveat for interpreting any NDCG@10 comparison across datasets with
different sparsity profiles, but does not by itself indicate which direction
(if any) it would bias an algorithm comparison within a single dataset.

## References

Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR
techniques. *ACM Transactions on Information Systems*, 20(4), 422–446.

## Reproducibility

All scripts referenced above are located in `experiments/h1_ndcg_formula/`
within the author's fork. Environment: `uv run python <script>`, using
`omnirec==1.0.0` as installed via `uv sync` (see `uv.lock`).
