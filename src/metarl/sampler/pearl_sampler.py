# pylint: disable=unnecessary-pass
"""A sampler for PEARL."""
import numpy as np

from metarl.sampler.utils import rollout_pearl
from metarl.sampler.base import BaseSampler


class PEARLSampler(BaseSampler):
    """A sampler used in meta-RL algorithms involving context.

    It stores context and resample belief in the policy every step.

    Args:
        env (metarl.envs.MetaRLEnv): An environement instance.
        policy (metarl.policies.Policy): Policy used for sampling.
        max_path_length (int): Maximum length of path for each step

    """

    def __init__(self, env, policy, max_path_length):
        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length

    def start_worker(self):
        """Start worker."""
        pass

    def shutdown_worker(self):
        """Terminate worker."""
        pass

    def obtain_samples(self,
                       max_samples=np.inf,
                       max_trajs=np.inf,
                       deterministic=False,
                       accum_context=True,
                       resample_rate=1):
        """Obtain samples in the environment up to max_samples or max_trajs.

        Args:
            max_samples (int): Maximum number of samples.
            max_trajs (int):  Maximum number of trajectories.
            deterministic (bool): Whether or not policy is deterministic.
            accum_context (bool): Whether or not to accumulate the collected
                context.
            resample_rate (int): How often (in trajectories) to resample context.

        Returns:
            list: A list of paths.
            int: Total number of steps.

        """
        assert max_samples < np.inf or max_trajs < np.inf
        paths = []
        n_steps_total = 0
        n_trajs = 0

        while n_steps_total < max_samples and n_trajs < max_trajs:
            path = rollout_pearl(self.env,
                                 self.policy,
                                 max_path_length=self.max_path_length,
                                 deterministic=deterministic,
                                 accum_context=accum_context)

            # save the latent context that generated this trajectory
            path['context'] = self.policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1

            # resample z every transition
            if n_trajs % resample_rate == 0:
                self.policy.sample_from_belief()
        return paths, n_steps_total
