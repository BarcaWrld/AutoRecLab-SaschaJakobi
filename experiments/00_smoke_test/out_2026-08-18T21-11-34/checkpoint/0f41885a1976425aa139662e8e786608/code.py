import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import RecSysDataSet, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.util.run import run_omnirec


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    working_path = Path(working_dir)

    print('Loading dataset: MovieLensLatestSmall')
    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLensLatestSmall)
    print(f'Dataset: {dataset}')
    print(f'Interactions: {dataset.num_interactions()}')
    try:
        print(f'Min rating: {dataset.min_rating()}')
        print(f'Max rating: {dataset.max_rating()}')
    except Exception:
        pass

    pipe = Pipe(
        MakeImplicit(4),
        UserHoldout(validation_size=0.15, test_size=0.15),
    )
    dataset = pipe.process(dataset)

    train_df = dataset._data.get('train')
    val_df = dataset._data.get('val')
    test_df = dataset._data.get('test')
    print(f'Train interactions: {len(train_df)}')
    print(f'Validation interactions: {len(val_df)}')
    print(f'Test interactions: {len(test_df)}')

    plan = ExperimentPlan('prototype_popularity_movielens_small')
    plan.add_algorithm(LensKit.PopScorer, {})

    evaluator = Evaluator(Recall([10]))

    results = run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
    print('Experiment finished.')
    print(results)

    recall_value = None
    if isinstance(results, pd.DataFrame):
        print(results.to_string(index=False))
        possible_cols = [c for c in results.columns if 'Recall' in c or 'recall' in c]
        if possible_cols:
            recall_value = float(results.iloc[0][possible_cols[0]])
    else:
        print(results)
        try:
            if hasattr(results, 'to_dict'):
                as_dict = results.to_dict()
                print(as_dict)
        except Exception:
            pass

    if recall_value is None:
        try:
            if isinstance(results, dict):
                for key, value in results.items():
                    if 'Recall' in str(key) or 'recall' in str(key):
                        recall_value = float(value)
                        break
        except Exception:
            pass

    if recall_value is None:
        recall_value = 0.0

    print(f'Recall@10: {recall_value:.6f}')

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(['PopScorer'], [recall_value], color='steelblue')
    ax.set_ylim(0, max(1.0, recall_value * 1.2 + 1e-9))
    ax.set_ylabel('Recall@10')
    ax.set_title('MovieLensLatestSmall Popularity Baseline')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    plot_path = working_path / 'recall_at_10.png'
    fig.savefig(plot_path, dpi=150)
    print(f'Saved plot to {plot_path}')


if __name__ == '__main__':
    main()
