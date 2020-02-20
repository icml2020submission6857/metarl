"""GaussianGRUPolicy with GaussianGRUModel."""
import akro
import numpy as np
import tensorflow as tf

from metarl.tf.models import GaussianGRUModel
from metarl.tf.policies import StochasticPolicy


class GaussianGRUPolicy(StochasticPolicy):
    """Models the action distribution using a Gaussian parameterized by a GRU.

    Args:
        env_spec (metarl.envs.env_spec.EnvSpec): Environment specification.
        name (str): Model name, also the variable scope.
        hidden_dim (int): Hidden dimension for GRU cell for mean.
        hidden_nonlinearity (Callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (Callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (Callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        recurrent_nonlinearity (Callable): Activation function for recurrent
            layers. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        recurrent_w_init (Callable): Initializer function for the weight
            of recurrent layer(s). The function should return a
            tf.Tensor.
        output_nonlinearity (Callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (Callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor.
        output_b_init (Callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor.
        hidden_state_init (Callable): Initializer function for the
            initial hidden state. The functino should return a tf.Tensor.
        hidden_state_init_trainable (bool): Bool for whether the initial
            hidden state is trainable.
        learn_std (bool): Is std trainable.
        std_share_network (bool): Boolean for whether mean and std share
            the same network.
        init_std (float): Initial value for std.
        layer_normalization (bool): Bool for using layer normalization or not.
        state_include_action (bool): Whether the state includes action.
            If True, input dimension will be
            (observation dimension + action dimension).

    """

    def __init__(self,
                 env_spec,
                 hidden_dims=[32],
                 name='GaussianGRUPolicy',
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 recurrent_nonlinearity=tf.nn.sigmoid,
                 recurrent_w_init=tf.glorot_uniform_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.glorot_uniform_initializer(),
                 output_b_init=tf.zeros_initializer(),
                 hidden_state_init=tf.zeros_initializer(),
                 hidden_state_init_trainable=False,
                 learn_std=True,
                 std_share_network=False,
                 init_std=1.0,
                 layer_normalization=False,
                 state_include_action=True):
        if not isinstance(env_spec.action_space, akro.Box):
            raise ValueError('GaussianGRUPolicy only works with '
                             'akro.Box action space, but not {}'.format(
                                 env_spec.action_space))
        super().__init__(name, env_spec)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._hidden_dims = hidden_dims
        self._state_include_action = state_include_action

        if state_include_action:
            self._input_dim = self._obs_dim + self._action_dim
        else:
            self._input_dim = self._obs_dim

        self.model = GaussianGRUModel(
            output_dim=self._action_dim,
            hidden_dims=hidden_dims,
            name='GaussianGRUModel',
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            recurrent_nonlinearity=recurrent_nonlinearity,
            recurrent_w_init=recurrent_w_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            hidden_state_init=hidden_state_init,
            hidden_state_init_trainable=hidden_state_init_trainable,
            layer_normalization=layer_normalization,
            learn_std=learn_std,
            std_share_network=std_share_network,
            init_std=init_std)

        self._prev_actions = None
        self._prev_hiddens = None
        self._initialize()

    def _initialize(self):
        obs_ph = tf.compat.v1.placeholder(tf.float32,
                                          shape=(None, None, self._input_dim))
        step_input_var = tf.compat.v1.placeholder(shape=(None,
                                                         self._input_dim),
                                                  name='step_input',
                                                  dtype=tf.float32)
        step_hidden_var = tf.compat.v1.placeholder(shape=(None,
                                                          self._hidden_dims[0]),
                                                   name='step_hidden_input',
                                                   dtype=tf.float32)

        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            self.model.build(obs_ph, step_input_var, step_hidden_var)

        self._f_step_mean_std = (
            tf.compat.v1.get_default_session().make_callable(
                [
                    self.model.networks['default'].step_mean,
                    self.model.networks['default'].step_log_std,
                    self.model.networks['default'].step_hidden
                ],
                feed_list=[step_input_var, step_hidden_var]))

    @property
    def vectorized(self):
        """bool: Whether the policy is vectorized or not."""
        return True

    def dist_info_sym(self, obs_var, state_info_vars, name=None):
        """Build a symbolic graph of the distribution parameters.

        Args:
            obs_var (tf.Tensor): Tensor input for symbolic graph.
            state_info_vars (dict): Extra state information, e.g.
                previous action.
            name (str): Name for symbolic graph.

        Returns:
            dict[tf.Tensor]: Outputs of the symbolic graph of distribution
                parameters.

        """
        if self._state_include_action:
            prev_action_var = state_info_vars['prev_action']
            prev_action_var = tf.cast(prev_action_var, tf.float32)
            all_input_var = tf.concat(axis=2,
                                      values=[obs_var, prev_action_var])
        else:
            all_input_var = obs_var

        with tf.compat.v1.variable_scope(self._variable_scope):
            mean_var, _, log_std_var, _, _, _, _ = self.model.build(
                all_input_var,
                self.model.networks['default'].step_input,
                self.model.networks['default'].step_hidden_input,
                name=name)

        return dict(mean=mean_var, log_std=log_std_var)

    def reset(self, dones=None):
        """Reset the policy.

        Note:
            If `dones` is None, it will be by default `np.array([True])` which
            implies the policy will not be "vectorized", i.e. number of
            parallel environments for training data sampling = 1.

        Args:
            dones (numpy.ndarray): Bool that indicates terminal state(s).

        """
        if dones is None:
            dones = np.array([True])
        if self._prev_actions is None or len(dones) != len(self._prev_actions):
            self._prev_actions = np.zeros(
                (len(dones), self.action_space.flat_dim))
            self._prev_hiddens = np.zeros((len(dones), self._hidden_dims[0]))

        self._prev_actions[dones] = 0.
        self._prev_hiddens[dones] = self.model.networks[
            'default'].init_hidden.eval()

    def get_action(self, observation):
        """Get a single action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from environment.

        Returns:
            tuple[numpy.ndarray, dict]: Predicted action and agent info.

                action (numpy.ndarray): Predicted action.
                agent_info (dict): Distribution obtained after observing the
                    given observation, with keys
                    * mean: (numpy.ndarray)
                    * log_std: (numpy.ndarray)
                    * prev_action: (numpy.ndarray), only present if
                        self._state_include_action is True.

        """
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        """Get multiple actions from this policy for the input observations.

        Args:
            observations (numpy.ndarray): Observations from environment.

        Returns:
            tuple[numpy.ndarray, dict]: Prediction actions and agent infos.

                actions (numpy.ndarray): Predicted actions.
                agent_infos (dict): Distribution obtained after observing the
                    given observation, with keys
                    * mean: (numpy.ndarray)
                    * log_std: (numpy.ndarray)
                    * prev_action: (numpy.ndarray), only present if
                        self._state_include_action is True.

        """
        # flat_obs = self.observation_space.flatten_n(observations)
        if self._state_include_action:
            assert self._prev_actions is not None
            all_input = np.concatenate([observations, self._prev_actions], axis=-1)
        else:
            all_input = observations
        means, log_stds, hidden_vec = self._f_step_mean_std(
            all_input, self._prev_hiddens)
        rnd = np.random.normal(size=means.shape)
        samples = rnd * np.exp(log_stds) + means
        # samples = self.action_space.unflatten_n(samples)
        prev_actions = self._prev_actions
        self._prev_actions = samples
        self._prev_hiddens = hidden_vec
        agent_infos = dict(mean=means, log_std=log_stds)
        if self._state_include_action:
            agent_infos['prev_action'] = np.copy(prev_actions)
        return samples, agent_infos

    @property
    def recurrent(self):
        """bool: Whether this policy is recurrent or not."""
        return True

    @property
    def distribution(self):
        """metarl.tf.distributions.DiagonalGaussian: Policy distribution."""
        return self.model.networks['default'].dist

    @property
    def state_info_specs(self):
        """list: State info specification."""
        if self._state_include_action:
            return [
                ('prev_action', (self._action_dim, )),
            ]

        return []

    def __getstate__(self):
        """See `Object.__getstate__`."""
        new_dict = super().__getstate__()
        del new_dict['_f_step_mean_std']
        return new_dict

    def __setstate__(self, state):
        """See `Object.__setstate__`."""
        super().__setstate__(state)
        self._initialize()
    