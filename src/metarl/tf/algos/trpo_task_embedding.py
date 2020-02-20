"""Trust Region Policy Optimization."""
from metarl.tf.algos.npo_task_embedding import NPOTaskEmbedding
from metarl.tf.optimizers import ConjugateGradientOptimizer
from metarl.tf.optimizers import PenaltyLbfgsOptimizer


class TRPOTaskEmbedding(NPOTaskEmbedding):
    """Trust Region Policy Optimization with Task Embedding."""

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 scope=None,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=0.98,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 pg_loss='surrogate',
                 lr_clip_range=0.01,
                 max_kl_step=0.01,
                 optimizer=None,
                 optimizer_args=None,
                 policy_ent_coeff=0.0,
                 embedding_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 use_neg_logli_entropy=False,
                 stop_entropy_gradient=False,
                 stop_ce_gradient=False,
                 entropy_method='no_entropy',
                 flatten_input=True,
                 inference=None,
                 inference_optimizer=None,
                 inference_optimizer_args=None,
                 inference_ce_coeff=0.0,
                 kl_constraint='hard',
                 name='TRPOTaskEmbedding'):
        self._kl_constraint = kl_constraint

        optimizer, optimizer_args = self._build_optimizer(
            optimizer, optimizer_args)
        inference_optimizer, inference_optimizer_args = self._build_optimizer(
            inference_optimizer, inference_optimizer_args)

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         scope=scope,
                         max_path_length=max_path_length,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         fixed_horizon=fixed_horizon,
                         pg_loss=pg_loss,
                         lr_clip_range=lr_clip_range,
                         max_kl_step=max_kl_step,
                         optimizer=optimizer,
                         optimizer_args=optimizer_args,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         use_neg_logli_entropy=use_neg_logli_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         stop_ce_gradient=stop_ce_gradient,
                         entropy_method=entropy_method,
                         flatten_input=flatten_input,
                         inference=inference,
                         inference_optimizer=inference_optimizer,
                         inference_optimizer_args=inference_optimizer_args,
                         inference_ce_coeff=inference_ce_coeff,
                         name=name)

    def _build_optimizer(self, optimizer, optimizer_args):
        """Build up optimizer."""
        if not optimizer:
            if self._kl_constraint == 'hard':
                optimizer = ConjugateGradientOptimizer
            elif self._kl_constraint == 'soft':
                optimizer = PenaltyLbfgsOptimizer
            else:
                raise ValueError('Invalid kl_constraint')
        return optimizer, optimizer_args
