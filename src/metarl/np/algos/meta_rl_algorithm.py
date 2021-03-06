"""Interface of Meta-RL ALgorithms."""
import abc

from metarl.np.algos import RLAlgorithm


class MetaRLAlgorithm(RLAlgorithm, abc.ABC):
    """Base class for Meta-RL Algorithms."""

    @abc.abstractmethod
    def get_exploration_policy(self):
        """Return a policy used before adaptation to a specific task.

        Each time it is retrieved, this policy should only be evaluated in one
        task.

        Returns:
            metarl.Policy: The policy used to obtain samples that are later
                used for meta-RL adaptation.

        """

    @abc.abstractmethod
    def adapt_policy(self, exploration_policy, exploration_trajectories):
        """Produce a policy adapted for a task.

        Args:
            exploration_policy (metarl.Policy): A policy which was returned
                from get_exploration_policy(), and which generated
                exploration_trajectories by interacting with an environment.
                The caller may not use this object after passing it into this
                method.
            exploration_trajectories (metarl.TrajectoryBatch): Trajectories to
                adapt to, generated by exploration_policy exploring the
                environment.

        Returns:
            metarl.Policy: A policy adapted to the task represented by the
                exploration_trajectories.

        """
