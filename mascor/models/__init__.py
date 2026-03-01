from .agent import actor, critic
from .gan import generator_1dcnn_24_v2 as generator 
from .gan import discriminator_1dcnn_24_v2 as discriminator
__all__ = ['actor', 'critic', 'generator', 'discriminator']