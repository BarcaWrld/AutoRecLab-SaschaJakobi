import os
from pathlib import Path

import matplotlib.pyplot as plt

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.plan_components import Grid
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(11)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    pipe = Pipe(
        MakeImplicit(3),
        UserHoldout(validation_size=0.15, test_size=0.15),
    )
    dataset = pipe.process(dataset)

    plan = ExperimentPlan(plan_name='movielens100k_itemknn_prototype')
    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {
            'max_nbrs': Grid([20]),
            'min_nbrs': 5,
        },
    )

    evaluator = Evaluator(NDCG([10]), Precision([10]))

    run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)

    results = evaluator.get_results()
    if not results:
        print('No results were returned by the evaluator.')
        return

    dataset_id, df = next(iter(results.items()))
    summary = (
        df[df['name'].isin(['NDCG', 'Precision']) & (df['k'] == 10)]
        .groupby(['algorithm', 'name', 'k'], as_index=False)['value']
        .mean()
    )
    print('\nSummary for:', dataset_id)
    print(summary.to_string(index=False))

    pivot = summary.pivot(index='algorithm', columns='name', values='value').reset_index()
    pivot.columns.name = None

    ax = pivot.plot(kind='bar', x='algorithm', figsize=(6, 4), rot=0)
    ax.set_ylabel('Score')
    ax.set_title('MovieLens100K: ItemKNN @10')
    ax.legend(title='Metric')
    plt.tight_layout()

    plot_path = Path(working_dir) / 'movielens100k_itemknn_metrics.png'
    plt.savefig(plot_path, dpi=150)
    print(f'Plot saved to: {plot_path}')


if __name__ == '__main__':
    main()
