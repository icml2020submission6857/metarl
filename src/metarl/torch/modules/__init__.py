"""Pytorch modules."""

from metarl.torch.modules.gaussian_mlp_module import \
    GaussianMLPIndependentStdModule, GaussianMLPModule, \
    GaussianMLPTwoHeadedModule
from metarl.torch.modules.mlp_module import MLPModule
from metarl.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule
from metarl.torch.modules.tanh_gaussian_mlp_module import TanhGaussianMLPTwoHeadedModule
from metarl.torch.modules.tanh_gaussian_mlp_module_2 import TanhGaussianMLPTwoHeadedModule2

__all__ = [
    'MLPModule', 'MultiHeadedMLPModule', 'GaussianMLPModule',
    'GaussianMLPIndependentStdModule', 'GaussianMLPTwoHeadedModule',
    'TanhGaussianMLPTwoHeadedModule', 'TanhGaussianMLPTwoHeadedModule2',
]
