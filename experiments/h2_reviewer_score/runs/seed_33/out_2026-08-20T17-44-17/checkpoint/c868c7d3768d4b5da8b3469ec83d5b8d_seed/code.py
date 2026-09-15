import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(33)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    pipeline = Pipe(
        MakeImplicit(3),
        UserHoldout(0.15, 0.15),
    )
    dataset = pipeline.process(dataset)

    plan = ExperimentPlan(plan_name='ml100k_itemknn_popularity_prototype')
    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {'max_nbrs': 20, 'min_nbrs': 5}
    )
    plan.add_algorithm(LensKit.PopScorer)

    evaluator = Evaluator(NDCG([10]), Precision([10]))

    results = run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
    print(results)

    if isinstance(results, pd.DataFrame):
        plot_df = results.copy()
    else:
        plot_df = pd.DataFrame(results)

    if 'algorithm' not in plot_df.columns:
        plot_df = plot_df.reset_index().rename(columns={'index': 'algorithm'})

    metric_cols = [c for c in plot_df.columns if 'NDCG@10' in str(c) or 'Precision@10' in str(c)]
    if not metric_cols:
        metric_cols = [c for c in plot_df.columns if 'ndcg' in str(c).lower() or 'precision' in str(c).lower()]

    if metric_cols:
        long_df = plot_df[['algorithm'] + metric_cols].melt(id_vars=['algorithm'], var_name='metric', value_name='value')
        pivot = long_df.pivot(index='algorithm', columns='metric', values='value')
        ax = pivot.plot(kind='bar', figsize=(8, 4))
        ax.set_title('MovieLens100K: ItemKNN vs Popularity')
        ax.set_ylabel('Score')
        ax.set_xlabel('Algorithm')
        plt.tight_layout()
        plot_path = Path(working_dir) / 'prototype_metrics.png'
        plt.savefig(plot_path, dpi=150)
        print(f'Saved plot to {plot_path}')


if __name__ == '__main__':
    main()
