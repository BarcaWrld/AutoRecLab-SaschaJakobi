import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.algos import LensKit
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def results_to_dataframe(evaluator):
    results = evaluator.get_results()
    if not results:
        return pd.DataFrame()
    frames = []
    for dataset_id, df in results.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            tmp = df.copy()
            tmp["dataset_id"] = dataset_id
            frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(33)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    print("Loaded dataset:", dataset)
    print("Num interactions:", dataset.num_interactions())

    pipeline = Pipe(UserHoldout(validation_size=0.15, test_size=0.15))
    dataset = pipeline.process(dataset)

    plan = ExperimentPlan(plan_name="MovieLens100K_Prototype_ItemKNN")
    plan.add_algorithm(LensKit.ItemKNNScorer, {"max_nbrs": 20, "min_nbrs": 5})

    evaluator = Evaluator(NDCG(10), Precision(10))

    run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)

    df = results_to_dataframe(evaluator)
    print(df)

    metrics_path = Path(working_dir) / "prototype_metrics.csv"
    df.to_csv(metrics_path, index=False)
    print("Saved metrics to", metrics_path)

    plot_path = Path(working_dir) / "prototype_metrics.png"
    if not df.empty:
        metric_cols = [c for c in df.columns if c not in {"algorithm", "dataset_id"}]
        plot_df = df[["algorithm"] + metric_cols].copy()
        value_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(plot_df[c])]
        if value_cols:
            melted = plot_df.melt(id_vars=["algorithm"], value_vars=value_cols, var_name="metric", value_name="value")
            ax = melted.plot(x="metric", y="value", kind="bar", legend=False, figsize=(8, 4))
            ax.set_title("MovieLens100K Prototype: ItemKNN Metrics")
            ax.set_ylabel("Metric value")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print("Saved plot to", plot_path)
        else:
            print("No numeric metric columns found for plotting. Available columns:", list(df.columns))
    else:
        print("No evaluation results returned; plot not generated.")


if __name__ == "__main__":
    main()
