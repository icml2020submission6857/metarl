[![Docs](https://readthedocs.org/projects/metarl/badge)](http://metarl.readthedocs.org/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/icml2020submission6857/metarl/blob/master/LICENSE)

# Notes for reviewers:

This is a toolkit for developing and evaluating (Meta/Multi-Task) reinforcement learning algorithms, and an accompanying library of state-of-the-art implementations built using that toolkit.

The launchers for running the experiments presented in the paper are located in [`icml_launchers/`](https://github.com/icml2020submission6857/metarl/tree/master/icml_launchers).

Algorithm 1 in the paper is located in the [`MetaEvaluator`](https://github.com/icml2020submission6857/metarl/tree/master/src/metarl/experiment/meta_evaluator.py#L65).
RL Algorithm implementations are located in [`metarl.torch.algos`](https://github.com/icml2020submission6857/metarl/tree/master/src/metarl/torch/algos) and [`metarl.tf.algos`](https://github.com/icml2020submission6857/metarl/tree/master/src/metarl/tf/algos).


### Contributions
We have a well defined process for accepting contributions, but it has been removed from the README to preserve anonymity.

### Testing

The live version of this repository has roughly a 90% test coverage.
We have removed references to our CI to preserve anonymity.
Furthermore, there are some test failures in the anonymized code which are not
present in the original code.

### Installation
In order to install everything needed to reproduce the paper results:

```
pip install -e '.[dev,all]'
```

## Algorithms
The table below summarizes the algorithms available in metarl.

| Algorithm              | Framework(s)        |
| ---------------------- | ------------------- |
| CEM                    | numpy               |
| CMA-ES                 | numpy               |
| REINFORCE (a.k.a. VPG) | PyTorch, TensorFlow |
| DDPG                   | PyTorch, TensorFlow |
| DQN                    | TensorFlow          |
| DDQN                   | TensorFlow          |
| ERWR                   | TensorFlow          |
| NPO                    | TensorFlow          |
| PPO                    | PyTorch, TensorFlow |
| REPS                   | TensorFlow          |
| TD3                    | TensorFlow          |
| TNPG                   | TensorFlow          |
| TRPO                   | PyTorch, TensorFlow |
| SAC                    | PyTorch             |
| MAML                   | PyTorch             |
| RL^2                   | TensorFlow          |
| PEARL                  | PyTorch             |
| MT-SAC+SA              | PyTorch             |
| TE                     | TensorFlow          |

## Supported Tools and Frameworks
We support Python 3.5+.

We currently support [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/) for implementing the neural network portions of RL algorithms, and additions of new framework support are always welcome. PyTorch modules can be found in the package [`metarl.torch`](https://github.com/icml2020submission6857/metarl/tree/master/src/metarl/torch) and TensorFlow modules can be found in the package [`metarl.tf`](https://github.com/icml2020submission6857/metarl/tree/master/src/metarl/tf). Algorithms which do not require neural networks are found in the package [`metarl.np`](https://github.com/icml2020submission6857/metarl/tree/master/src/metarl/np).

The package is available for download on PyPI, and we ensure that it installs successfully into environments defined using [conda](https://docs.conda.io/en/latest/), [Pipenv](https://pipenv.readthedocs.io/en/latest/), and [virtualenv](https://virtualenv.pypa.io/en/latest/).

All components use the popular [`gym.Env`](https://github.com/openai/gym) interface for RL environments.

The toolkit provides wide range of modular tools for implementing RL algorithms, including:
* Composable neural network models
* Replay buffers
* High-performance samplers
* An expressive experiment definition interface
* Tools for reproducibility (e.g. set a global random seed which all components respect)
* Logging to many outputs, including TensorBoard
* Reliable experiment checkpointing and resuming
* Environment interfaces for many popular benchmark suites
* Supporting for running metarl in diverse environments, including always up-to-date Docker containers
