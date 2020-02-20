"""Baselines (value functions) which use NumPy as a numerical backend."""
from metarl.np.baselines.base import Baseline
from metarl.np.baselines.linear_feature_baseline import LinearFeatureBaseline
from metarl.np.baselines.multi_task_linear_feature_baseline import MultiTaskLinearFeatureBaseline
from metarl.np.baselines.zero_baseline import ZeroBaseline

__all__ = ['Baseline', 'LinearFeatureBaseline', 'ZeroBaseline', 'MultiTaskLinearFeatureBaseline']
