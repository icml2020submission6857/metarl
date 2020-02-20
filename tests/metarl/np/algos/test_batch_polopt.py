import gym

from metarl.envs import normalize
from metarl.np.baselines import LinearFeatureBaseline
from metarl.sampler import OnPolicyVectorizedSampler
from metarl.tf.envs import TfEnv
from metarl.tf.samplers import BatchSampler
from tests.fixtures.algos import DummyAlgo
from tests.fixtures.policies import DummyPolicy, DummyPolicyWithoutVectorized


class TestBatchPolopt:

    def setup_method(self):
        self.env = TfEnv(normalize(gym.make('CartPole-v1')))
        self.baseline = LinearFeatureBaseline(env_spec=self.env.spec)

    def test_default_sampler_cls(self):
        policy = DummyPolicy(env_spec=self.env.spec)
        algo = DummyAlgo(env_spec=self.env.spec,
                         policy=policy,
                         baseline=self.baseline)
        sampler = algo.sampler_cls(algo, self.env, dict())
        assert isinstance(sampler, OnPolicyVectorizedSampler)

        policy = DummyPolicyWithoutVectorized(env_spec=self.env.spec)
        algo = DummyAlgo(env_spec=self.env.spec,
                         policy=policy,
                         baseline=self.baseline)
        sampler = algo.sampler_cls(algo, self.env, dict())
        assert isinstance(sampler, BatchSampler)
