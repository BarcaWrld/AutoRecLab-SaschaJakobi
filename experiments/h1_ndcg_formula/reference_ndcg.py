"""
Eigene Referenzimplementierung von NDCG@k nach Jarvelin & Kekalainen (2002),
basierend auf der Definition der idealen Rangliste als Funktion der
tatsaechlich vorhandenen relevanten Items pro User (nicht fix bei k).

Vergleichsziel: omnirec.metrics.ranking.NDCG
"""

import numpy as np
import pandas as pd


def dcg_at_k(hits: np.ndarray, k: int) -> float:
    """hits: boolesches Array, hits[i] = True wenn Position i+1 relevant ist."""
    positions = np.arange(1, len(hits) + 1)
    gains = np.where(hits, 1.0 / np.log2(positions + 1), 0.0)
    return float(gains[:k].sum())


def idcg_at_k(n_relevant: int, k: int) -> float:
    """Ideale DCG nach Paper-Definition: min(n_relevant, k) Treffer ganz oben."""
    n_ideal_hits = min(n_relevant, k)
    if n_ideal_hits == 0:
        return 0.0
    positions = np.arange(1, n_ideal_hits + 1)
    return float((1.0 / np.log2(positions + 1)).sum())


def ndcg_reference(predictions: pd.DataFrame, test: pd.DataFrame, k_list: list[int]) -> dict[int, float]:
    """
    Eigene, unabhaengige NDCG-Implementierung.

    predictions: DataFrame mit Spalten [user, item, score, rank]
    test: DataFrame mit Spalten [user, item] (Ground-Truth relevante Items)
    k_list: Liste der k-Werte

    Rueckgabe: {k: durchschnittlicher NDCG@k ueber alle User}
    """
    max_k = max(k_list)
    ndcg_per_k: dict[int, list[float]] = {k: [] for k in k_list}

    for user, group in predictions.sort_values("rank").groupby("user"):
        pred_items = group["item"].to_numpy()[:max_k]
        relevant_items = test.loc[test["user"] == user, "item"].to_numpy()
        n_relevant = len(relevant_items)

        hits = np.isin(pred_items, relevant_items)

        for k in k_list:
            dcg = dcg_at_k(hits, k)
            idcg = idcg_at_k(n_relevant, k)
            user_ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_per_k[k].append(user_ndcg)

    return {k: float(np.mean(scores)) for k, scores in ndcg_per_k.items()}


if __name__ == "__main__":
    # Minimaler Testfall: 1 User, 2 relevante Items, k=10
    predictions = pd.DataFrame({
        "user": [1] * 10,
        "item": list(range(100, 110)),
        "score": list(range(10, 0, -1)),
        "rank": list(range(1, 11)),
    })
    test = pd.DataFrame({
        "user": [1, 1],
        "item": [100, 101],  # beide relevanten Items an Position 1 und 2
    })

    result = ndcg_reference(predictions, test, k_list=[10])
    print("Eigene Referenzimplementierung, NDCG@10:", result[10])
    print("Erwartungswert (aus Baustein 1/2 Handrechnung): 1.0")