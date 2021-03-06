"""Model-Agnostic Meta-Learning (MAML) algorithm applied to TRPO."""
import torch

from metarl.torch.algos import _Default, VPG
from metarl.torch.algos.maml import MAML
from metarl.torch.optimizers import ConjugateGradientOptimizer


class MAMLTRPO(MAML):
    """Model-Agnostic Meta-Learning (MAML) applied to TRPO.

    Args:
        env (metarl.envs.MetaRLEnv): A multi-task environment.
        policy (metarl.torch.policies.base.Policy): Policy.
        baseline (metarl.np.baselines.Baseline): The baseline.
        inner_lr (float): Adaptation learning rate.
        outer_lr (float): Meta policy learning rate.
        max_kl_step (float): The maximum KL divergence between old and new
            policies.
        max_path_length (int): Maximum length of a single rollout.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
        meta_batch_size (int): Number of tasks sampled per batch.
        num_grad_updates (int): Number of adaptation gradient steps.

    """

    def __init__(self,
                 env,
                 policy,
                 baseline,
                 inner_lr=_Default(1e-2),
                 outer_lr=1e-3,
                 max_kl_step=0.01,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy',
                 meta_batch_size=40,
                 num_grad_updates=1):
        inner_algo = VPG(env.spec,
                         policy,
                         baseline,
                         optimizer=torch.optim.Adam,
                         policy_lr=inner_lr,
                         max_path_length=max_path_length,
                         num_train_per_epoch=1,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method)

        meta_optimizer = (ConjugateGradientOptimizer,
                          dict(max_constraint_value=max_kl_step))

        super().__init__(inner_algo=inner_algo,
                         env=env,
                         policy=policy,
                         baseline=baseline,
                         meta_optimizer=meta_optimizer,
                         meta_batch_size=meta_batch_size,
                         inner_lr=inner_lr,
                         outer_lr=outer_lr,
                         num_grad_updates=num_grad_updates)
