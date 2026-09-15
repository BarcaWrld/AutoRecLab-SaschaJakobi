"""
E6: Comparison of OmniRec's reported NDCG@10 against the independent
reference implementation, using real predictions and a real test split
produced by an actual AutoRecLab run (not synthetic data).

Source run: out_2026-08-19T16-27-35 (final node: f612857fa00949189e2f4149fb34eb1e_iteration2)
AutoRecLab-reported NDCG@10 (from summary.md): 0.172938288508597
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from reference_ndcg import ndcg_reference

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.util.util import set_random_state
from omnirec.metrics.ranking import NDCG as OmniRecNDCG

# --- Step 1: Rebuild the exact same test split the original run used ---
set_random_state(42)

dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
dataset = Pipe(
    MakeImplicit(3),
    UserHoldout(validation_size=0.15, test_size=0.15),
).process(dataset)

test_df = dataset._data.test[["user", "item"]].copy()

print("Rebuilt test set — users:", test_df["user"].nunique(), "| interactions:", len(test_df))

# --- Step 2: Load the real predictions produced by the AutoRecLab run ---
predictions_path = (
    r"out_2026-08-19T16-27-35\checkpoint\f612857fa00949189e2f4149fb34eb1e_iteration2"
    r"\generated\checkpoints\MovieLens100K-78811d22\LensKit.ItemKNNScorer-44df4040-42\predictions.json"
)
with open(predictions_path) as f:
    raw = json.load(f)

predictions_df = pd.DataFrame(raw)
print("Loaded predictions — rows:", len(predictions_df), "| users:", predictions_df["user"].nunique())

# --- Step 3: Compute NDCG@10 with OmniRec's own code, on the real data ---
omnirec_metric = OmniRecNDCG(k=10)
omnirec_result = omnirec_metric.calculate(predictions_df, test_df)
ndcg_omnirec_recomputed = omnirec_result.result[10]

# --- Step 4: Compute NDCG@10 with the independent reference implementation ---
ndcg_reference_value = ndcg_reference(predictions_df, test_df, k_list=[10])[10]

# --- Step 5: Report ---
print("\n--- Results ---")
print(f"NDCG@10 reported in summary.md:          0.172938288508597")
print(f"NDCG@10 recomputed with OmniRec's code:  {ndcg_omnirec_recomputed:.15f}")
print(f"NDCG@10 with independent reference impl: {ndcg_reference_value:.15f}")
print(f"Difference (OmniRec vs. reference):      {ndcg_omnirec_recomputed - ndcg_reference_value:.6f}")
print(f"Relative difference:                     {100 * (ndcg_reference_value / ndcg_omnirec_recomputed - 1):.2f}%")