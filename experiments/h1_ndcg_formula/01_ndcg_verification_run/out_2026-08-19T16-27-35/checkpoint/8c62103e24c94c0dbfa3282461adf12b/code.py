import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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


def extract_metric_value(results_obj, metric_name="NDCG", cutoff=10):
    if isinstance(results_obj, pd.DataFrame):
        cols = list(results_obj.columns)
        target_cols = [c for c in cols if metric_name.lower() in str(c).lower() and str(cutoff) in str(c)]
        if target_cols:
            return float(results_obj[target_cols[0]].iloc[0])
        if metric_name in results_obj.index:
            row = results_obj.loc[metric_name]
            for c in row.index:
                if str(cutoff) in str(c):
                    return float(row[c])
    if isinstance(results_obj, dict):
        for k, v in results_obj.items():
            if metric_name.lower() in str(k).lower() and str(cutoff) in str(k):
                return float(v)
    raise ValueError(f"Could not extract {metric_name}@{cutoff} from results type {type(results_obj)}")


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(42)

    print('Loading MovieLens100K...')
    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    print(f'Loaded dataset: {dataset}')
    print(f'Interactions: {dataset.num_interactions()}')
    if hasattr(dataset, 'min_rating'):
        try:
            print(f'Min rating: {dataset.min_rating()}')
            print(f'Max rating: {dataset.max_rating()}')
        except Exception:
            pass

    print('Applying UserHoldout split...')
    pipeline = Pipe(UserHoldout(validation_size=0.15, test_size=0.15))
    dataset = pipeline.process(dataset)
    print('Split complete.')

    plan = ExperimentPlan('MovieLens100K_ItemKNN_Prototype')
    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {
            'max_nbrs': 50,
            'min_nbrs': 1,
            'feedback': 'explicit',
        },
    )

    evaluator = Evaluator(NDCG([10]))

    print('Running OmniRec experiment...')
    results = run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
    print('Experiment finished.')
    print(results)

    ndcg10 = extract_metric_value(results, 'NDCG', 10)
    print(f'NDCG@10: {ndcg10:.6f}')

    plot_path = Path(working_dir) / 'ndcg10_itemknn.png'
    plt.figure(figsize=(5, 4))
    plt.bar(['ItemKNN'], [ndcg10], color=['steelblue'])
    plt.ylabel('NDCG@10')
    plt.title('MovieLens100K ItemKNN Prototype')
    plt.ylim(0, max(1.0, ndcg10 * 1.2 + 1e-9))
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f'Saved plot to: {plot_path}')


if __name__ == '__main__':
    main()
