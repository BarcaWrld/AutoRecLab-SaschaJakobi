# Experiment Summary

## User Request
Compare ItemKNN and a popularity baseline on MovieLens100K using OmniRec's UserHoldout split with random seed 11, and report NDCG@10 and Precision@10 for both algorithms.

## What Was Run
The experiment:
- Set the random seed to `11`
- Loaded `MovieLens100K`
- Converted ratings to implicit feedback with threshold `3`
- Applied `UserHoldout(validation_size=0.15, test_size=0.15)`
- Evaluated two algorithms:
  - `LensKit.ItemKNNScorer` with `max_nbrs=20`, `min_nbrs=5`
  - `LensKit.PopScorer` as the popularity baseline
- Measured:
  - `NDCG@10`
  - `Precision@10`

## Key Results

| Algorithm | NDCG@10 | Precision@10 |
|---|---:|---:|
| ItemKNN | 0.173712 | 0.159617 |
| Popularity baseline | 0.116415 | 0.106575 |

ItemKNN outperformed the popularity baseline on both metrics.

## Limitations
- The output reports algorithm names with internal run identifiers, but the underlying algorithms are clearly identifiable as `ItemKNNScorer` and `PopScorer`.
- The output does not provide additional per-user or statistical significance analysis.

## Conclusion
On MovieLens100K with OmniRec UserHoldout split and random seed 11, ItemKNN achieved higher `NDCG@10` and `Precision@10` than the popularity baseline:
- `NDCG@10`: `0.173712` vs `0.116415`
- `Precision@10`: `0.159617` vs `0.106575`