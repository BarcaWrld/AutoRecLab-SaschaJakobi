import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import NDCG, RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.metrics.ranking import Precision
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def results_to_dataframe(results):
    if isinstance(results, pd.DataFrame):
        return results.copy()
    return pd.DataFrame(results)


def normalize_results(df):
    out = df.copy()
    if 'algorithm' not in out.columns:
        if 'index' in out.columns:
            out = out.rename(columns={'index': 'algorithm'})
        else:
            out = out.reset_index().rename(columns={'index': 'algorithm'})
    return out


def find_metric_column(columns, metric_name):
    metric_name_lower = metric_name.lower()
    for col in columns:
        col_lower = str(col).lower()
        if metric_name_lower in col_lower and '@10' in col_lower:
            return col
    for col in columns:
        col_lower = str(col).lower()
        if metric_name_lower in col_lower:
            return col
    return None


def extract_metric_table(results_df):
    df = normalize_results(results_to_dataframe(results_df))
    ndcg_col = find_metric_column(df.columns, 'ndcg')
    precision_col = find_metric_column(df.columns, 'precision')
    metric_cols = [c for c in [ndcg_col, precision_col] if c is not None]
    if not metric_cols:
        raise ValueError(f'Could not find NDCG/Precision metric columns in results: {list(df.columns)}')
    table = df[['algorithm'] + metric_cols].copy()
    rename_map = {}
    if ndcg_col is not None:
        rename_map[ndcg_col] = 'NDCG@10'
    if precision_col is not None:
        rename_map[precision_col] = 'Precision@10'
    table = table.rename(columns=rename_map)
    return table


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(33)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    pipeline = Pipe(
        MakeImplicit(3),
        UserHoldout(validation_size=0.15, test_size=0.15),
    )
    dataset = pipeline.process(dataset)

    plan = ExperimentPlan('ml100k_itemknn_popularity_prototype')
    plan.add_algorithm(LensKit.ItemKNNScorer, {'max_nbrs': 20, 'min_nbrs': 5})
    plan.add_algorithm(LensKit.PopScorer, {})

    evaluator = Evaluator(NDCG([10]), Precision([10]))

    results = run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
    print(results)

    results_df = results_to_dataframe(results)
    metric_table = extract_metric_table(results_df)
    print('\nPrototype comparison (MovieLens100K, UserHoldout seed=33):')
    print(metric_table.to_string(index=False))

    plot_df = metric_table.set_index('algorithm')
    ax = plot_df.plot(kind='bar', figsize=(7, 4))
    ax.set_title('MovieLens100K Prototype: ItemKNN vs Popularity')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Score')
    ax.legend(title='Metric')
    plt.tight_layout()
    plot_path = Path(working_dir) / 'ml100k_itemknn_popularity_prototype.png'
    plt.savefig(plot_path, dpi=150)
    print(f'Saved plot to {plot_path}')


if __name__ == '__main__':
    main()
