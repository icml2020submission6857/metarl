import tensorflow as tf

from metarl.tf.models import Model


class SimpleMLPMergeModel(Model):
    """Simple MLPMergeModel for testing."""

    def __init__(self, output_dim, name=None, *args, **kwargs):
        super().__init__(name)
        self.output_dim = output_dim

    def network_input_spec(self):
        """Network input spec."""
        return ['input_var1', 'input_var2']

    def _build(self, obs_input, act_input, name=None):
        return_var = tf.compat.v1.get_variable(
            'return_var', (), initializer=tf.constant_initializer(0.5))
        return tf.fill((tf.shape(obs_input)[0], self.output_dim), return_var)