import os
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def extract_metric_table(results_obj):
    if isinstance(results_obj, pd.DataFrame):
        return results_obj.copy()
    if isinstance(results_obj, dict):
        return pd.DataFrame(results_obj)
    return pd.DataFrame(results_obj)


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

    plan = ExperimentPlan(plan_name='prototype_movielens100k_knn_pop')
    plan.add_algorithm(LensKit.ItemKNNScorer, {
        'max_nbrs': 40,
        'min_nbrs': 1,
        'center': True,
    })
    plan.add_algorithm(LensKit.PopScorer, {})

    evaluator = Evaluator(NDCG([10]), Precision([10]))

    results = run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)

    results_path = os.path.join(working_dir, 'prototype_results.json')
    try:
        results_df = extract_metric_table(results)
        results_df.to_json(results_path, orient='records', indent=2)
    except Exception:
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(str(results), f, indent=2)
        results_df = pd.DataFrame()

    print('Run completed. Results saved to:', results_path)
    print(results)

    # Basic plot: if structured results are available, plot the metric comparison.
    plot_path = os.path.join(working_dir, 'prototype_metrics_plot.png')
    try:
        if not results_df.empty:
            df = results_df.copy()
            cols = [c for c in df.columns if 'NDCG' in str(c) or 'Precision' in str(c)]
            if cols:
                plot_df = df[cols].copy()
                plot_df = plot_df.T
                plot_df.columns = [str(c) for c in range(plot_df.shape[1])]
                ax = plot_df.plot(kind='bar', figsize=(8, 4))
                ax.set_title('Prototype: NDCG@10 and Precision@10')
                ax.set_ylabel('Score')
                ax.set_xlabel('Metric')
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150)
                plt.close()
            else:
                plt.figure(figsize=(6, 4))
                plt.bar(['ItemKNN', 'Pop'], [0, 0])
                plt.title('Prototype metric plot placeholder')
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150)
                plt.close()
        else:
            plt.figure(figsize=(6, 4))
            plt.bar(['ItemKNN', 'Pop'], [0, 0])
            plt.title('Prototype metric plot placeholder')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
    except Exception as e:
        plt.figure(figsize=(6, 4))
        plt.bar(['ItemKNN', 'Pop'], [0, 0])
        plt.title(f'Plot fallback due to error: {e}')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    print('Plot saved to:', plot_path)


if __name__ == '__main__':
    main()
