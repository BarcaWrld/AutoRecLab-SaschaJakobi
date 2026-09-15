# Experiment Summary

## User Request

Build a tiny popularity baseline on a small MovieLens sample and report Recall@10.

## What Was Run

- Loaded `MovieLensLatestSmall` via `RecSysDataSet.use_dataloader(DataSet.MovieLensLatestSmall)`.
- Subsampled the dataset to `5%` of interactions.
- Converted ratings to implicit feedback using a threshold of `3`.
- Split the data with `RandomHoldout(validation_size=0.1, test_size=0.1)`.
- Ran a single popularity baseline algorithm: `LensKit.PopScorer`.
- Evaluated with `Recall([10])`.

## Key Results

| Algorithm | Metric | k | Value |
|---|---:|---:|---:|
| LensKit.PopScorer-f4aa5539-42 | Recall | 10 | 0.03174603174603175 |

## Limitations

- The output reports only one evaluated algorithm, so this is a single-baseline result.
- The exact number of users/items in the post-processed split is not shown, so no further dataset-level interpretation can be made from the provided output.
- No comparison against other baselines is available in the output.

## Conclusion

The tiny popularity baseline (`LensKit.PopScorer`) on the sampled MovieLens dataset achieved **Recall@10 = 0.03174603174603175**.