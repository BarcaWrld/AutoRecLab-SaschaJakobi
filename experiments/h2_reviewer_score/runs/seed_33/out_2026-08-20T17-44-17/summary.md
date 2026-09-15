# Experiment Summary

## User Request
Compare ItemKNN and a popularity baseline on MovieLens100K using OmniRec's UserHoldout split with random seed 33, and report NDCG@10 and Precision@10 for both algorithms.

## What Was Run
The experiment:
- Loaded `MovieLens100K`
- Converted ratings to implicit feedback with threshold `3`
- Applied `UserHoldout(0.15, 0.15)`
- Set random seed to `33`
- Evaluated two LensKit algorithms:
  - `ItemKNNScorer` with `max_nbrs=20` and `min_nbrs=5`
  - `PopScorer` as the popularity baseline
- Computed ranking metrics:
  - `NDCG@10`
  - `Precision@10`

## Key Results

| Algorithm | NDCG@10 | Precision@10 |
|---|---:|---:|
| ItemKNN | 0.17885636252846182 | 0.1616365568544102 |
| Popularity baseline | 0.11605320487869693 | 0.10455991516436905 |

ItemKNN outperformed the popularity baseline on both metrics in this run.

## Limitations
The output does not provide per-user results, confidence intervals, or statistical significance tests. The exact internal algorithm labels in the output include run-specific suffixes, but they correspond to `LensKit.ItemKNNScorer` and `LensKit.PopScorer` as configured in the code.

## Conclusion
On MovieLens100K with OmniRec `UserHoldout` split and random seed 33, ItemKNN achieved higher `NDCG@10` and `Precision@10` than the popularity baseline:
- ItemKNN: `NDCG@10 = 0.1789`, `Precision@10 = 0.1616`
- Popularity baseline: `NDCG@10 = 0.1161`, `Precision@10 = 0.1046`