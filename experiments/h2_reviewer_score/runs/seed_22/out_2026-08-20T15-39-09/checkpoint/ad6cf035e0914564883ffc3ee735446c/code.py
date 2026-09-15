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


def extract_results_table(results_obj):
    if isinstance(results_obj, pd.DataFrame):
        return results_obj.copy()
    if isinstance(results_obj, dict):
        parts = []
        for _, value in results_obj.items():
            if isinstance(value, pd.DataFrame):
                parts.append(value.copy())
        if parts:
            return pd.concat(parts, ignore_index=True)
        return pd.DataFrame(results_obj)
    try:
        return pd.DataFrame(results_obj)
    except Exception:
        return pd.DataFrame()


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(22)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    pipeline = Pipe(UserHoldout(0.15, 0.15))
    dataset = pipeline.process(dataset)

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
    results_df = pd.DataFrame()
    if isinstance(results_map, dict) and results_map:
        results_df = extract_results_table(next(iter(results_map.values())))

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
        plot_df['metric'] = plot_df.apply(lambda r: f"{r['name']}@{int(r['k'])}" if pd.notna(r['k']) else str(r['name']), axis=1)
        if 'algorithm' in plot_df.columns:
            grouped = plot_df.groupby(['algorithm', 'metric'], as_index=False)['value'].mean()
            pivot = grouped.pivot(index='metric', columns='algorithm', values='value')
        else:
            pivot = plot_df.pivot_table(index='metric', values='value', aggfunc='mean')
        pivot.plot(kind='bar', ax=plt.gca())
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
