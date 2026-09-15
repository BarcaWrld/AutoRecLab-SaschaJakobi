import os
import pandas as pd
import matplotlib.pyplot as plt

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, HR
from omnirec.runner.algos import LensKit
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(33)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    print('Loaded dataset:', dataset)
    print('Num interactions:', dataset.num_interactions())
    try:
        print('Rating range:', dataset.min_rating(), dataset.max_rating())
    except Exception:
        pass

    pipe = Pipe(
        UserHoldout(validation_size=0.15, test_size=0.15)
    )
    dataset = pipe.process(dataset)

    plan = ExperimentPlan(plan_name='MovieLens100K_Prototype')
    plan.add_algorithm(LensKit.PopScorer, {})
    plan.add_algorithm(LensKit.ItemKNNScorer, {'max_nbrs': 20, 'min_nbrs': 5})

    evaluator = Evaluator(NDCG([10]), HR([10]))

    results = run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
    print('run_omnirec returned:', type(results))
    print(results)

    # Prototype plotting: simple placeholder that will be replaced by extracted metric results
    # if the returned object is a dataframe-like structure.
    try:
        if isinstance(results, pd.DataFrame):
            df = results.copy()
        else:
            df = pd.DataFrame(results)

        metric_cols = [c for c in df.columns if 'NDCG' in str(c) or 'HR' in str(c) or 'Precision' in str(c)]
        if metric_cols:
            plot_df = df[metric_cols].copy()
            plot_df.plot(kind='bar')
            plt.tight_layout()
            plot_path = os.path.join(working_dir, 'prototype_metrics.png')
            plt.savefig(plot_path, dpi=150)
            print('Saved plot to', plot_path)
    except Exception as e:
        print('Plotting skipped due to result format issue:', e)


if __name__ == '__main__':
    main()
