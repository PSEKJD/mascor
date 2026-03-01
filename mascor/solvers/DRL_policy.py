import os
from gymnasium import spaces
import numpy as np
import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec as RLModuleSpec
from ray.rllib.policy.policy import Policy
from gymnasium import spaces
from mascor.solvers import bc_policy
from pathlib import Path

class policy():
    def __init__(self, args, mode = 'drl'):
        self.args = args
        REPO_ROOT = Path(__file__).resolve().parents[2]
        DATA_DIR = str(REPO_ROOT / "dataset")
        self.data_path = os.path.join(DATA_DIR, '{country}/{region}/oracle_dataset_{option}_sample_{sample}'.format(country = args.target_country,
                                                                                                                  region = args.region,
                                                                                                                  option = args.design_option,
                                                                                                                  sample = args.sample_size))
        self.device = args.device
        
        # Pre-trained model loading
        self.mode = mode # DRL or BC
        if mode == 'drl':
            self.actor = Policy.from_checkpoint(os.path.join(self.data_path, 'checkpoint_drl/policy_flatten_{}_bc_{}'.format(args.flatten, args.bc_support)))
        else:
            self.actor = None
            
        # Define obs & action space, and bc_policy
        self.state_flatten = args.flatten
        self.obs_length = args.obs_length
        
        if self.state_flatten:
            space_low = np.zeros(shape = (self.obs_length*2 + 2 + 4),dtype = np.float32) 
            space_high = np.zeros(shape = (self.obs_length*2 + 2 + 4),dtype = np.float32)
        else:
            space_low = np.zeros(shape = (2, self.obs_length + 2 + 4, 1), dtype = np.float32)
            space_high = np.zeros(shape = (2, self.obs_length + 2 + 4, 1), dtype = np.float32)
        space_high[:] = 1
        self.observation_space = spaces.Box(low = space_low, high = space_high, shape=(space_high.shape), dtype=np.float32)
        
        # Action = ESS power & AWE power & Stored H2 util & H2 to market
        if self.mode == 'drl':
            if args.bc_support:
                # Actions = epsilon for balancing exploration
                self.raw_action_space = spaces.Box(low=np.array([-1, 0, 0, 0],dtype = np.float32),
                                           high=np.array([1, 1,  1, 1],dtype = np.float32), shape=(4,),dtype=np.float32) #real action
                self.action_space = spaces.Box(low=np.array([0, 0, 0, 0],dtype = np.float32),
                                               high=np.array([1, 1, 1, 1],dtype = np.float32), shape=(4,),dtype=np.float32) #epsilon
                self.loading_bc_policy(flatten=self.state_flatten)
            else:
                self.action_space = spaces.Box(low=np.array([-1, 0, 0, 0],dtype = np.float32),
                                           high=np.array([1, 1, 1, 0],dtype = np.float32), shape=(4,),dtype=np.float32)
                self.bc_policy = None
        else:
            self.raw_action_space = spaces.Box(low=np.array([-1, 0, 0, 0],dtype = np.float32),
                                       high=np.array([1, 1,  1, 1],dtype = np.float32), shape=(4,),dtype=np.float32) #real action
            self.loading_bc_policy(flatten=self.state_flatten)
            self.actor = self.bc_policy

    @torch.no_grad()      
    def compute_actions(self, s):
        if self.mode == 'drl':
            action, _, _ = self.actor.compute_actions(s)
            if self.bc_policy is not None:
                action_rescale = self.bc_compute_single_action(epsilon=action, state = s)
                return action_rescale
            else:
                action = np.clip(action, self.action_space.low, self.action_space.high)
                return action
        else:
            action_dist =  self.actor(torch.from_numpy(s).to(self.device))
            action_sampled = action_dist.sample()
            action_sampled = action_sampled.detach().cpu().numpy()
            action_sampled = np.clip(action_sampled, self.raw_action_space.low, self.raw_action_space.high)
            return action_sampled
        
    @torch.no_grad() 
    def bc_compute_single_action(self, epsilon, state):
        action_dist= self.bc_policy(torch.from_numpy(state).to(self.device))
        action_mean, action_std = action_dist.mean,action_dist.stddev
        epsilon = np.clip(epsilon, 0, 1)
        epsilon = torch.tensor(epsilon).to(self.device)
        action_std = action_std*epsilon + 1e-5
        distribution = torch.distributions.Normal(loc = action_mean, scale = action_std)
        action_sampled = distribution.sample()
        del distribution, action_mean, action_std, epsilon
        action_sampled = action_sampled.detach().cpu().numpy()
        action_sampled = np.clip(action_sampled, self.raw_action_space.low, self.raw_action_space.high)
        
        return action_sampled
    
    def loading_bc_policy(self, flatten = True):
        if flatten:
            self.policy_config = {"fcnet_hiddens":[512, 512, 512, 256, 256, 256],
                                  "fcnet_activation":"relu",
                                  "post_fcnet_hiddens": [256, 128, 64],
                                  "post_fcnet_activation":"relu",
                                  "vf_share_layers": False,}
        else:
            self.policy_config = {"conv_filters": [[4, [3,4], 2], #[out_channels, kernel, stride] 
                                                       [8, [2,4], 2],
                                                       [16, [1,4], 1]],
                                  "conv_activation": "relu",
                                  "post_fcnet_hiddens": [256, 256, 256, 128],
                                  "post_fcnet_activation":"relu",
                                  "vf_share_layers": False,}
        
        module_spec = RLModuleSpec(module_class=PPOTorchRLModule,
                    observation_space=self.observation_space,
                    action_space = self.raw_action_space,
                    model_config_dict= self.policy_config,
                    catalog_class=PPOCatalog,)
        module = module_spec.build()
        self.bc_policy = bc_policy(module.encoder.actor_encoder, module.pi,
                                 torch.distributions.Normal)
        REPO_ROOT = Path(__file__).resolve().parents[2]
        DATA_DIR = str(REPO_ROOT / "dataset")
        save_dir = os.path.join(DATA_DIR, '{country}/{region}/oracle_dataset_{option}_sample_{sample}'.format(country = self.args.target_country, 
                                                                                                                                region = self.args.region,
                                                                                                                                option = self.args.design_option, 
                                                                                                                                sample = self.args.sample_size))
        checkpoint_path = os.path.join(save_dir, 'checkpoint_bc/policy_flatten_{flatten}/policy_epoch_{epoch}.pt'.format(flatten = flatten, epoch = 60))
        self.device = torch.device("cpu")
        self.bc_policy.load_state_dict(torch.load(checkpoint_path, map_location = self.device)['actor'])
        print("//"*100)
        print("Succeeding pre-trained BC Actor loading")

