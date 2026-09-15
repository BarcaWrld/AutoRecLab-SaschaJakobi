"""
Direkter Vergleich: OmniRecs NDCG-Implementierung vs. eigene Referenzimplementierung
(nach Jarvelin & Kekalainen 2002), auf identischen synthetischen Eingabedaten.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from reference_ndcg import ndcg_reference

from omnirec.metrics.ranking import NDCG as OmniRecNDCG


def build_test_case(n_relevant: int, k: int, all_hits: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Baut einen synthetischen Testfall: 1 User, n_relevant relevante Items,
    die (falls all_hits=True) an den besten Positionen 1..n_relevant liegen.
    """
    n_pred = max(k, n_relevant)
    predictions = pd.DataFrame({
        "user": [1] * n_pred,
        "item": list(range(100, 100 + n_pred)),
        "score": list(range(n_pred, 0, -1)),
        "rank": list(range(1, n_pred + 1)),
    })
    relevant_item_ids = list(range(100, 100 + n_relevant)) if all_hits else []
    test = pd.DataFrame({
        "user": [1] * n_relevant,
        "item": relevant_item_ids,
    })
    return predictions, test


k = 10
print(f"{'n_relevant':>10} | {'NDCG_omnirec':>13} | {'NDCG_reference':>15} | {'diff':>8}")
for n_relevant in [1, 2, 3, 5, 7, 9, 10, 11, 15, 20]:
    predictions, test = build_test_case(n_relevant, k)

    omnirec_metric = OmniRecNDCG(k=k)
    omnirec_result = omnirec_metric.calculate(predictions, test)
    ndcg_omnirec = omnirec_result.result[k]

    ndcg_ref = ndcg_reference(predictions, test, k_list=[k])[k]

    diff = ndcg_omnirec - ndcg_ref
    print(f"{n_relevant:>10} | {ndcg_omnirec:>13.4f} | {ndcg_ref:>15.4f} | {diff:>8.4f}")