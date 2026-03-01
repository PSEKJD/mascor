import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ray.rllib.core.models.base import ENCODER_OUT

class policy(torch.nn.Module):

    def __init__(self,
                 actor_encoder: torch.nn.Module,
                 pi: torch.nn.Module,
                 distribution: torch.distributions.Distribution,):
        super().__init__()
        self.actor_encoder = actor_encoder
        self.pi = pi
        self.distribution = distribution
        
    def forward(self, state:torch.Tensor) -> torch.distributions.Distribution:
        
        #Convert state to dict for forward
        batch = {}
        batch['obs'] = state

        # Actor 
        actor_encoder_output = self.actor_encoder(batch)[ENCODER_OUT]
        action_logit = self.pi(actor_encoder_output)
        mean = action_logit[:,:4]
        log_std = action_logit[:,4:]
        std = torch.exp(log_std)
        distribution = self.distribution(loc = mean,
                                         scale = std)
                
        return distribution