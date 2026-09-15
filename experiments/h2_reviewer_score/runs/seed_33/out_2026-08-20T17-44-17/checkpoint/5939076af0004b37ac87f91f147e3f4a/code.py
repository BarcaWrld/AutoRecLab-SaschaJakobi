import os
import pandas as pd
import matplotlib.pyplot as plt

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG
from omnirec.runner.algos import LensKit
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def _extract_results_table(results):
    if results is None:
        return pd.DataFrame()
    if isinstance(results, pd.DataFrame):
        return results.copy()
    if isinstance(results, dict):
        return pd.DataFrame(results)
    try:
        return pd.DataFrame(results)
    except Exception:
        return pd.DataFrame()


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(33)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    print('Loaded dataset:', dataset)
    print('Num interactions:', dataset.num_interactions())

    pipe = Pipe(
        UserHoldout(validation_size=0.15, test_size=0.15)
    )
    dataset = pipe.process(dataset)

    plan = ExperimentPlan(plan_name='MovieLens100K_Prototype_ItemKNN')
    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {'max_nbrs': 20, 'min_nbrs': 5}
    )

    evaluator = Evaluator(NDCG([10]))

    results = run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
    print('run_omnirec returned:', type(results))
    print(results)

    df = _extract_results_table(results)
    if not df.empty:
        metrics_path = os.path.join(working_dir, 'prototype_metrics.csv')
        df.to_csv(metrics_path, index=False)
        print('Saved metrics to', metrics_path)

        plot_cols = [c for c in df.columns if 'NDCG' in str(c)]
        if plot_cols:
            plot_df = df[plot_cols].copy()
            ax = plot_df.plot(kind='bar', legend=True, figsize=(8, 4))
            ax.set_title('MovieLens100K Prototype: ItemKNN NDCG@10')
            ax.set_ylabel('NDCG@10')
            plt.tight_layout()
            plot_path = os.path.join(working_dir, 'prototype_metrics.png')
            plt.savefig(plot_path, dpi=150)
            print('Saved plot to', plot_path)
            plt.close()


if __name__ == '__main__':
    main()
