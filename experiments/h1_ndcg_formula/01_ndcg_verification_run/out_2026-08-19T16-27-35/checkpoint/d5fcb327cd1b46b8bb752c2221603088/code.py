import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(42)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    dataset = Pipe(
        UserHoldout(validation_size=0.15, test_size=0.15)
    ).process(dataset)

    plan = ExperimentPlan('prototype_movielens100k_itemknn')
    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {
            'max_nbrs': 20,
            'min_nbrs': 5,
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

    plot_df = df[(df['name'] == 'NDCG') & (df['k'] == 10)].copy()
    if plot_df.empty:
        raise RuntimeError('NDCG@10 results not found for plotting.')

    plt.figure(figsize=(6, 4))
    plt.bar(plot_df['algorithm'], plot_df['value'])
    plt.ylabel('NDCG@10')
    plt.title('Prototype ItemKNN on MovieLens100K')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()

    plot_path = Path(working_dir) / 'ndcg10_itemknn.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f'Saved plot to: {plot_path}')


if __name__ == '__main__':
    main()
