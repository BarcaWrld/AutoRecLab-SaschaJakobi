# Experiment Summary

## User Request
Build a simple ItemKNN recommender on the MovieLens100K dataset using OmniRec's UserHoldout split, and report NDCG@10.

## What Was Run
- Dataset: `MovieLens100K` via `RecSysDataSet.use_dataloader(DataSet.MovieLens100K)`
- Preprocessing:
  - `MakeImplicit(3)` converted ratings to implicit feedback using threshold 3
  - `UserHoldout(validation_size=0.15, test_size=0.15)` split the data
- Algorithm:
  - `LensKit.ItemKNNScorer`
  - Parameter: `max_nbrs=20`
- Evaluation:
  - `Evaluator(NDCG([10]))`

## Key Results

| Dataset | Algorithm | Split | Metric | Value |
|---|---|---|---|---:|
| MovieLens100K | LensKit.ItemKNNScorer | UserHoldout(0.15, 0.15) | NDCG@10 | 0.172938288508597 |

## Limitations
- The output reports a single evaluation result for the configured run; no additional folds or confidence intervals are provided.
- The experiment output does not include any comparison against other algorithms or baselines.
- The printed table in the output shows `fold = None`, so there is no fold-level breakdown to summarize.

## Conclusion
A simple ItemKNN recommender was successfully run on MovieLens100K with OmniRec’s `UserHoldout(0.15, 0.15)` split. The reported NDCG@10 is **0.172938288508597**.