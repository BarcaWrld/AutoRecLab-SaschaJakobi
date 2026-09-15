import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.metrics.ranking import NDCG
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(42)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    dataset = Pipe(
        MakeImplicit(3),
        UserHoldout(validation_size=0.15, test_size=0.15),
    ).process(dataset)

    plan = ExperimentPlan('prototype_movielens100k_itemknn')
    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {
            'max_nbrs': 20,
        },
    )

    evaluator = Evaluator(NDCG([10]))

    print('Running experiment...')
    run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)

    results = evaluator.get_results()
    if not results:
        raise RuntimeError('No results were produced by the evaluator.')

    dataset_id, df = next(iter(results.items()))
    print(f'\nDataset: {dataset_id}')
    print(df)

    if not isinstance(df, pd.DataFrame):
        raise TypeError('Expected evaluator results to be a pandas DataFrame.')

    plot_df = df.copy()
    if 'algorithm' in plot_df.columns:
        algo_col = 'algorithm'
    elif 'name' in plot_df.columns:
        algo_col = 'name'
    else:
        raise RuntimeError('Could not find an algorithm column in the results.')

    value_col = 'value' if 'value' in plot_df.columns else None
    if value_col is None:
        metric_cols = [c for c in plot_df.columns if c not in {algo_col, 'k', 'metric'}]
        if len(metric_cols) == 1:
            value_col = metric_cols[0]
        else:
            raise RuntimeError('Could not infer metric value column for plotting.')

    if 'k' in plot_df.columns:
        plot_df = plot_df[plot_df['k'] == 10]
    if 'name' in plot_df.columns:
        plot_df = plot_df[plot_df['name'].astype(str).str.upper() == 'NDCG']
    if plot_df.empty:
        raise RuntimeError('NDCG@10 results not found for plotting.')

    plt.figure(figsize=(6, 4))
    plt.bar(plot_df[algo_col].astype(str), plot_df[value_col])
    plt.ylabel('NDCG@10')
    plt.title('Prototype ItemKNN on MovieLens100K')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()

    plot_path = Path(working_dir) / 'ndcg10_itemknn.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f'Saved plot to: {plot_path}')
    print(
        'Prototype complete: '
        'dataset=MovieLens100K, '
        'algorithm=ItemKNNScorer, '
        'split=UserHoldout(0.15, 0.15), '
        'metric=NDCG@10, '
        f'plot={plot_path}'
    )


if __name__ == '__main__':
    main()
