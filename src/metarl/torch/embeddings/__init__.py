"""PyTorch embedding modules for meta-learning algorithms."""
from metarl.torch.embeddings.mlp_encoder import MLPEncoder
from metarl.torch.embeddings.recurrent_encoder import RecurrentEncoder

__all__ = ['MLPEncoder', 'RecurrentEncoder']
