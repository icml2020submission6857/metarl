"""PyTorch algorithms."""
from metarl.torch.algos._utils import _Default  # noqa: F401
from metarl.torch.algos._utils import compute_advantages  # noqa: F401
from metarl.torch.algos._utils import filter_valids  # noqa: F401
from metarl.torch.algos._utils import make_optimizer  # noqa: F401
from metarl.torch.algos._utils import pad_to_last  # noqa: F401
from metarl.torch.algos.ddpg import DDPG
# VPG has to been import first because it is depended by PPO and TRPO.
from metarl.torch.algos.vpg import VPG
from metarl.torch.algos.pearl_sac import PEARLSAC
from metarl.torch.algos.ppo import PPO  # noqa: I100
from metarl.torch.algos.sac import SAC
from metarl.torch.algos.trpo import TRPO
from metarl.torch.algos.multi_task_sac import MTSAC
from metarl.torch.algos.maml_ppo import MAMLPPO  # noqa: I100
from metarl.torch.algos.maml_trpo import MAMLTRPO
from metarl.torch.algos.maml_vpg import MAMLVPG

__all__ = ['DDPG', 'VPG', 'PPO', 'TRPO', 'MAMLPPO', 'MAMLTRPO', 'MAMLVPG',
           'MTSAC', 'PEARLSAC']
