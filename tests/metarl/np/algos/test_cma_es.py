from metarl.np.algos import CMAES
from metarl.np.baselines import LinearFeatureBaseline
from metarl.sampler import OnPolicyVectorizedSampler
from metarl.tf.envs import TfEnv
from metarl.tf.experiment import LocalTFRunner
from metarl.tf.policies import CategoricalMLPPolicy
from tests.fixtures import snapshot_config, TfGraphTestCase


class TestCMAES(TfGraphTestCase):

    def test_cma_es_cartpole(self):
        """Test CMAES with Cartpole-v1 environment."""
        with LocalTFRunner(snapshot_config) as runner:
            env = TfEnv(env_name='CartPole-v1')

            policy = CategoricalMLPPolicy(name='policy',
                                          env_spec=env.spec,
                                          hidden_sizes=(32, 32))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            n_samples = 20

            algo = CMAES(env_spec=env.spec,
                         policy=policy,
                         baseline=baseline,
                         max_path_length=100,
                         n_samples=n_samples)

            runner.setup(algo, env, sampler_cls=OnPolicyVectorizedSampler)
            runner.train(n_epochs=1, batch_size=1000)
            # No assertion on return because CMAES is not stable.

            env.close()
