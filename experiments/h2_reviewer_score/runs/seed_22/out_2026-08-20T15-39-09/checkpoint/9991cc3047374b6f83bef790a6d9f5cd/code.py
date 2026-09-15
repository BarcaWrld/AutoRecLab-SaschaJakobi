import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import RecSysDataSet, NDCG
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.plan_components import Grid
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def _extract_results_table(result_obj):
    if isinstance(result_obj, pd.DataFrame):
        return result_obj
    if isinstance(result_obj, dict):
        return pd.DataFrame(result_obj)
    if hasattr(result_obj, "to_dataframe"):
        return result_obj.to_dataframe()
    return pd.DataFrame(result_obj)


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(22)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    pipeline = Pipe(
        MakeImplicit(3),
        UserHoldout(validation_size=0.15, test_size=0.15),
    )
    dataset = pipeline.process(dataset)

    plan = ExperimentPlan(plan_name='prototype_movielens100k_itemknn_pop')
    plan.add_algorithm(LensKit.ItemKNNScorer, {
        'max_nbrs': Grid([20]),
        'min_nbrs': 5,
        'center': True,
    })
    plan.add_algorithm(LensKit.PopScorer, {})

    evaluator = Evaluator(NDCG([10]),)

    results = run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
    results_df = _extract_results_table(results)

    print('\nEvaluation results:')
    print(results_df)

    if not results_df.empty:
        metric_cols = [c for c in results_df.columns if 'NDCG@10' in str(c) or 'Precision@10' in str(c)]
        print('\nSelected metric columns:')
        print(results_df[[c for c in metric_cols if c in results_df.columns]])

    plot_path = Path(working_dir) / 'prototype_metrics.png'
    fig, ax = plt.subplots(figsize=(8, 4))
    if not results_df.empty:
        label_col = next((c for c in results_df.columns if str(c).lower() in {'algorithm', 'algo', 'name'}), results_df.columns[0])
        metric_cols = [c for c in results_df.columns if 'NDCG@10' in str(c) or 'Precision@10' in str(c)]
        if metric_cols:
            plot_df = results_df[[label_col] + metric_cols].copy()
            plot_df = plot_df.set_index(label_col)
            plot_df.plot(kind='bar', ax=ax)
            ax.set_ylabel('Score')
            ax.set_title('MovieLens100K Prototype: ItemKNN vs PopScorer')
            ax.legend(title='Metric')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f'Plot saved to: {plot_path}')


if __name__ == '__main__':
    main()
