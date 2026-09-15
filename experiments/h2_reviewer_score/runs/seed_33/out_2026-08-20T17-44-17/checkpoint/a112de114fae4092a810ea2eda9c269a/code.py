import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def _get_results_df(results):
    if isinstance(results, pd.DataFrame):
        return results.copy()
    return pd.DataFrame(results)


def _normalize_results_df(df):
    out = df.copy()
    if 'algorithm' not in out.columns:
        out = out.reset_index().rename(columns={'index': 'algorithm'})
    return out


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(33)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    dataset = Pipe(
        MakeImplicit(3),
        UserHoldout(validation_size=0.15, test_size=0.15),
    ).process(dataset)

    try:
        train_df = dataset._data.get('train')
        val_df = dataset._data.get('val')
        test_df = dataset._data.get('test')
        print(f'Train/Val/Test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}')
    except Exception:
        print('SplitData loaded successfully.')

    plan = ExperimentPlan('ml100k_itemknn_prototype')
    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {'max_nbrs': 20, 'min_nbrs': 5},
    )

    evaluator = Evaluator(NDCG([10]), Precision([10]))
    results = run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
    print(results)

    results_df = _normalize_results_df(_get_results_df(results))
    metric_cols = [c for c in results_df.columns if 'NDCG@10' in str(c) or 'Precision@10' in str(c)]
    if not metric_cols:
        metric_cols = [c for c in results_df.columns if 'ndcg' in str(c).lower() or 'precision' in str(c).lower()]

    if metric_cols:
        plot_df = results_df[['algorithm'] + metric_cols].melt(id_vars='algorithm', var_name='metric', value_name='value')
        pivot = plot_df.pivot(index='algorithm', columns='metric', values='value')
        ax = pivot.plot(kind='bar', figsize=(7, 4))
        ax.set_title('MovieLens100K ItemKNN Prototype')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Score')
        plt.tight_layout()
        plot_path = Path(working_dir) / 'prototype_itemknn_metrics.png'
        plt.savefig(plot_path, dpi=150)
        print(f'Saved plot to {plot_path}')


if __name__ == '__main__':
    main()
