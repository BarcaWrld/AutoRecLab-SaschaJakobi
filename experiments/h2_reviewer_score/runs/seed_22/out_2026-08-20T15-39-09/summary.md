# Experiment Summary

## User Request

Compare ItemKNN and a popularity baseline on MovieLens100K using OmniRec's UserHoldout split with random seed 22, and report NDCG@10 and Precision@10 for both algorithms.

## What Was Run

The code:

- Set the random seed to `22`.
- Loaded `MovieLens100K`.
- Applied `UserHoldout(validation_size=0.15, test_size=0.15)`.
- Ran two algorithms:
  - `LensKit.ItemKNNScorer` with `max_nbrs=40`, `min_nbrs=1`
  - `LensKit.PopScorer`
- Evaluated with:
  - `NDCG([10])`
  - `Precision([10])`

## Key Results

The run did not produce usable metric values because evaluation failed for both algorithms with the same error: `KeyError: 'rank'`. The saved results DataFrame was empty.

| Algorithm | NDCG@10 | Precision@10 |
|---|---:|---:|
| ItemKNN | N/A | N/A |
| Popularity baseline | N/A | N/A |

## Limitations

- No valid `NDCG@10` or `Precision@10` values were reported in the output.
- The evaluator crashed during metric calculation because the predictions data lacked a required `rank` column.
- Because of this failure, there is no factual basis in the provided output to compare the two algorithms numerically.

## Conclusion

This experiment attempted to compare ItemKNN and a popularity baseline on MovieLens100K with OmniRec UserHoldout seed 22, but it failed during evaluation. As a result, `NDCG@10` and `Precision@10` are not available for either algorithm from the provided output.