"""Natural Policy Gradient Optimization."""
from dowel import logger, tabular
import numpy as np
import copy

import tensorflow as tf

from metarl.misc import tensor_utils as np_tensor_utils
from metarl.tf.algos import NPO


class RL2NPO(NPO):
    """Natural Policy Gradient Optimization.

    This is specific for RL^2
    (https://arxiv.org/pdf/1611.02779.pdf).

    NPO is used as an inner algorithm for RL^2. When sampling for RL^2,
    there are more than one environments to be sampled from. In the original
    implementation, within each task/environment, all rollouts sampled will be
    concatenated into one single rollout, and fed to the inner algorithm.
    Thus, returns and advantages are calculated across the rollout.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 scope=None,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 pg_loss='surrogate',
                 lr_clip_range=0.01,
                 max_kl_step=0.01,
                 optimizer=None,
                 optimizer_args=None,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 use_neg_logli_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy',
                 flatten_input=True,
                 fit_baseline='before',
                 name='NPO'):
        self._fit_baseline = fit_baseline
        if fit_baseline != 'before':
            raise ValueError('RL2NPO only supports fit baseline before')
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
                         entropy_method=entropy_method,
                         flatten_input=flatten_input)

    def optimize_policy(self, itr, samples_data):
        """Optimize policy.

        Args:
            itr (int): Iteration number.
            samples_data (dict): Processed sample data.
                See process_samples() for details.

        """
        if self._fit_baseline == 'before':
            self._fit_baseline_with_data(samples_data)
            samples_data['baselines'] = self._get_baseline_prediction(
                samples_data)

        policy_opt_input_values = self._policy_opt_input_values(samples_data)
        # Train policy network
        logger.log('Computing loss before')
        loss_before = self._optimizer.loss(policy_opt_input_values)
        logger.log('Computing KL before')
        policy_kl_before = self._f_policy_kl(*policy_opt_input_values)
        logger.log('Optimizing')
        self._optimizer.optimize(policy_opt_input_values)
        logger.log('Computing KL after')
        policy_kl = self._f_policy_kl(*policy_opt_input_values)
        logger.log('Computing loss after')
        loss_after = self._optimizer.loss(policy_opt_input_values)
        tabular.record('{}/LossBefore'.format(self.policy.name), loss_before)
        tabular.record('{}/LossAfter'.format(self.policy.name), loss_after)
        tabular.record('{}/dLoss'.format(self.policy.name),
                       loss_before - loss_after)
        tabular.record('{}/KLBefore'.format(self.policy.name),
                       policy_kl_before)
        tabular.record('{}/KL'.format(self.policy.name), policy_kl)
        pol_ent = self._f_policy_entropy(*policy_opt_input_values)
        tabular.record('{}/Entropy'.format(self.policy.name), np.mean(pol_ent))

        if self._fit_baseline == 'after':
            self._fit_baseline_with_data(samples_data)

        ev = np_tensor_utils.explained_variance_1d(samples_data['baselines'],
                                                   samples_data['returns'],
                                                   samples_data['valids'])
        tabular.record('{}/ExplainedVariance'.format(self.baseline.name), ev)

    def _get_baseline_prediction(self, samples_data):
        paths = samples_data['paths']
        baselines = [self.baseline.predict(path) for path in paths]
        return np_tensor_utils.pad_tensor_n(baselines, self.max_path_length)
