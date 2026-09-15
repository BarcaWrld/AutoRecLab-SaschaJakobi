import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def _results_to_df(results_obj):
    if isinstance(results_obj, pd.DataFrame):
        return results_obj.copy()
    if isinstance(results_obj, dict):
        frames = []
        for value in results_obj.values():
            if isinstance(value, pd.DataFrame):
                frames.append(value.copy())
        if frames:
            return pd.concat(frames, ignore_index=True)
    try:
        return pd.DataFrame(results_obj)
    except Exception:
        return pd.DataFrame()


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(22)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    dataset = Pipe(UserHoldout(validation_size=0.15, test_size=0.15)).process(dataset)

    plan = ExperimentPlan(plan_name='prototype_movielens100k_itemknn')
    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {
            'max_nbrs': 40,
            'min_nbrs': 1,
        },
    )

    evaluator = Evaluator(NDCG([10]), Precision([10]))

    run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)

    results_map = evaluator.get_results()
    results_df = _results_to_df(next(iter(results_map.values()))) if results_map else pd.DataFrame()

    results_path = os.path.join(working_dir, 'prototype_itemknn_results.json')
    if not results_df.empty:
        results_df.to_json(results_path, orient='records', indent=2)
    else:
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write(str(results_map))

    print('Run completed. Results saved to:', results_path)
    print(results_df)

    plot_path = os.path.join(working_dir, 'prototype_itemknn_metrics_plot.png')
    plt.figure(figsize=(7, 4))
    if not results_df.empty and {'name', 'k', 'value'}.issubset(results_df.columns):
        plot_df = results_df.copy()
        plot_df['metric'] = plot_df.apply(
            lambda r: f"{r['name']}@{int(r['k'])}" if pd.notna(r['k']) else str(r['name']),
            axis=1,
        )
        pivot = plot_df.pivot_table(index='metric', values='value', aggfunc='mean')
        pivot.plot(kind='bar', legend=False, ax=plt.gca())
        plt.ylabel('Score')
        plt.title('Prototype: ItemKNN NDCG@10 and Precision@10')
        plt.tight_layout()
    else:
        plt.bar(['ItemKNN'], [0.0])
        plt.ylabel('Score')
        plt.title('Prototype: ItemKNN metrics unavailable')
        plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print('Plot saved to:', plot_path)


if __name__ == '__main__':
    main()
