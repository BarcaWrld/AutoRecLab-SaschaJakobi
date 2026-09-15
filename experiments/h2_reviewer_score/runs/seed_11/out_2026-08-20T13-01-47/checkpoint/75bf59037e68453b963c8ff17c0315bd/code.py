import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import RecSysDataSet, NDCG, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.algos import LensKit
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def build_dataset():
    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    pipe = Pipe(
        MakeImplicit(3),
        UserHoldout(validation_size=0.15, test_size=0.15),
    )
    return pipe.process(dataset)


def build_plan():
    plan = ExperimentPlan(plan_name='prototype_movielens100k')
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def plot_results(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df[df['name'].isin(['NDCG', 'Recall']) & (df['k'] == 10)].copy()
    if plot_df.empty:
        return
    pivot = plot_df.pivot_table(index='algorithm', columns='name', values='value', aggfunc='mean')
    ax = pivot.plot(kind='bar', figsize=(8, 4))
    ax.set_title('MovieLens100K Prototype Results')
    ax.set_ylabel('Metric value')
    ax.set_xlabel('Algorithm')
    ax.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    set_random_state(11)

    dataset = build_dataset()
    plan = build_plan()
    evaluator = Evaluator(NDCG([10]), Recall([10]))

    run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)

    results = evaluator.get_results()
    for dataset_id, df in results.items():
        print(f'=== Results for {dataset_id} ===')
        print(df.sort_values(['algorithm', 'name', 'k']).to_string(index=False))
        out_csv = Path(working_dir) / 'prototype_results.csv'
        df.to_csv(out_csv, index=False)
        plot_path = Path(working_dir) / 'prototype_results.png'
        plot_results(df, plot_path)
        print(f'Saved results to: {out_csv}')
        print(f'Saved plot to: {plot_path}')


if __name__ == '__main__':
    main()
