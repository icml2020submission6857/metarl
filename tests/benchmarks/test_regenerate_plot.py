import pytest

from tests import benchmark_helper


class TestRegenerateBenchmark:
    def test_regenerate_benchmark(self):
        baselines_csvs, metarl_tf_csvs = None
        plt_file = ""
        env_id = ""

        benchmark_helper.plot_average_over_trials_with_x(
            [baselines_csvs, metarl_tf_csvs],
            [
                'eprewmean', 'Evaluation/AverageReturn'
            ],
            plt_file=plt_file,
            env_id=env_id,
            x_label='Iteration',
            y_label='Evaluation/AverageReturn',
            names=['baseline', 'metarl-TensorFlow'],
        )
