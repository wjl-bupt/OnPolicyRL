"""Default network components -- **optional defaults, not a required path**.

Any `nn.Module` satisfying the `oprl.Policy` protocol works directly (DESIGN.md §4.5,
level 3); importing `oprl.nets` is optional.
"""

from .actor_critic import ActorCritic
from .encoders import CNNEncoder, MLPEncoder, orthogonal_init

__all__ = ["ActorCritic", "MLPEncoder", "CNNEncoder", "orthogonal_init"]
