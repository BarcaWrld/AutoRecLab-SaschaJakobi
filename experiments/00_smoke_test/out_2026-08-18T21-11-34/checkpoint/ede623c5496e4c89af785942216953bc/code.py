import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import Recall, RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import RandomHoldout
from omnirec.preprocess.subsample import Subsample
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    set_random_state(42)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLensLatestSmall)
    print(dataset)
    print(f'Interactions: {dataset.num_interactions()}')
    try:
        print(f'Rating range: {dataset.min_rating()} to {dataset.max_rating()}')
    except Exception:
        pass

    pipeline = Pipe(
        Subsample(0.05),
        MakeImplicit(3),
        RandomHoldout(validation_size=0.1, test_size=0.1),
    )
    dataset = pipeline.process(dataset)

    plan = ExperimentPlan(plan_name='prototype_popularity_movielens_small')
    plan.add_algorithm(LensKit.PopScorer)

    evaluator = Evaluator(Recall([10]))

    run_omnirec(dataset, plan, evaluator)

    results = evaluator.get_results()
    if not results:
        raise RuntimeError('No evaluation results were produced.')

    dataset_id, df = next(iter(results.items()))
    print('\nEvaluation results:')
    print(df)

    out_csv = Path(working_dir) / 'recall_at_10_results.csv'
    df.to_csv(out_csv, index=False)
    print(f'Saved results to {out_csv}')

    metric_df = df[(df['name'] == 'Recall') & (df['k'] == 10)].copy()
    if metric_df.empty:
        raise RuntimeError('Recall@10 results were not found.')

    plt.figure(figsize=(7, 4))
    plt.bar(metric_df['algorithm'].astype(str), metric_df['value'])
    plt.ylabel('Recall@10')
    plt.title('Popularity baseline on MovieLensLatestSmall')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plot_path = Path(working_dir) / 'recall_at_10_plot.png'
    plt.savefig(plot_path, dpi=150)
    print(f'Saved plot to {plot_path}')

    print('\nRecall@10 summary:')
    print(metric_df[['algorithm', 'value']])


if __name__ == '__main__':
    main()
