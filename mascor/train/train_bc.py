import os
import time
import argparse
from gymnasium import spaces
import numpy as np
import torch
from torch.utils.data import DataLoader
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec as RLModuleSpec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mascor.utils.bc_data_loader import *
from mascor.solvers import bc_policy 
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--target-country", type=str, default="France", help="target country")
parser.add_argument("--region", type=str, default="Dunkirk", help="target country")
parser.add_argument("--design-option", type=str, default="c_fax_fix", help="oracle dataset option")
parser.add_argument("--sample-size", type=int, default=50000, help="oracle dataset size")

parser.add_argument("--stop-iters", type=int, default=100, help="Number of iterations to train.")
parser.add_argument("--device", type=str, default="cuda:0", help="device for training.")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size for actor training")

#Custom env configuration
parser.add_argument("--obs-length", type = int, default= 24, help="Observation length of profile data",)
parser.add_argument("--op-period", type = int, default= 720, help="Planning period",)
parser.add_argument("--flatten", action="store_true", help="Flatten the dimension of renewable and grid profile",)
# %%
if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device(args.device)
    REPO_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = str(REPO_ROOT / "dataset")
    data_path = os.path.join(DATA_DIR, '{country}/{region}/oracle_dataset_{option}_sample_{sample}'.format(country = args.target_country, 
                                                                                                              region = args.region,
                                                                                                              option = args.design_option,
                                                                                                              sample = args.sample_size))
    # %% Loading actor network
    dataset = Dataset_global_solution(data_path, obs_length=args.obs_length, flatten = args.flatten) 
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    del dataset
    #%% Actor network config
    if args.flatten:               
        model_config = dict(
             fcnet_hiddens=[512, 512, 512, 256, 256, 256],
             fcnet_activation="relu",
             post_fcnet_hiddens = [256, 128, 64],
             post_fcnet_activation = "relu",
             vf_share_layers =  False)
        space_low = np.zeros(shape = (args.obs_length*2 + 2 + 4),dtype = np.float32) 
        space_high = np.zeros(shape = (args.obs_length*2 + 2 + 4),dtype = np.float32)
    else:
        model_config = dict(
             conv_filters=[[4, [3,4], 2], #[out_channels, kernel, stride] 
                            [8, [2,4], 2],
                            [16, [1,4], 1]],
             conv_activation = 'relu',
             post_fcnet_hiddens=[256, 256, 256, 128],
             post_fcnet_activation="relu",
             vf_share_layers =  False)
        space_low = np.zeros(shape = (2, args.obs_length + 2 + 4, 1), dtype = np.float32)
        space_high = np.zeros(shape = (2, args.obs_length + 2 + 4, 1), dtype = np.float32)
    space_high[:] = 1
    observation_space = spaces.Box(low = space_low, high = space_high, shape=(space_high.shape), dtype=np.float32)
    
    action_low = np.array([-1, 0, 0, 0],dtype = np.float32)
    action_high = np.array([1, 1, 1, 1],dtype = np.float32)
    action_space = spaces.Box(low=action_low, high= action_high, shape=(4,),dtype=np.float32)
   
    module_spec = RLModuleSpec(module_class=PPOTorchRLModule,
                               observation_space=observation_space,
                               action_space = action_space,
                               model_config_dict=model_config,
                               catalog_class=PPOCatalog,)
    module = module_spec.build()
    actor = bc_policy(module.encoder.actor_encoder, module.pi,
                             torch.distributions.Normal).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5, weight_decay = 1e-4)
    action_loss = torch.nn.MSELoss()
    log_temperature = torch.tensor(np.log(0.1))
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(actor_optimizer,lambda steps: min((steps+1)/10000, 1))
    # %% Actor training loop
    actor.train()
    num_steps = args.stop_iters
    actor_losses = {}
    actor_losses['loss'] = []
    actor_losses['nll'] = []
    actor_losses['entropy'] = []
    
    #Loading pre-trained state
    save_path = os.path.join(DATA_DIR, '{country}/{region}/oracle_dataset_{option}_sample_{sample}/checkpoint_bc/policy_flatten_{flatten}'.format(country = args.target_country, 
                                                                                                                                                      region = args.region,
                                                                                                                                                      option = args.design_option,
                                                                                                                                                      sample = args.sample_size,
                                                                                                                                                      flatten = args.flatten))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    for epoch in range(0, num_steps):
        start = time.time()
        for i, (s, a) in enumerate(data_loader):
            action_dist = actor(s.to(device))
            action_log_likelihood = torch.mean(action_dist.log_prob(a.to(device)))
            action_entropy = action_dist.entropy().mean()
            temperature = log_temperature.exp().detach()
            actor_loss = -(action_log_likelihood + temperature*action_entropy)
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.25)
            actor_optimizer.step()
            actor_scheduler.step()
            
            if i%100 == 0:
                end = time.time()
                epooch_run_time = (end-start)*len(data_loader)/100/3600
                print(f"Action Loss: {actor_loss.item():.4f}, action-nll: {-action_log_likelihood.item():.4f}, Epoch time: {epooch_run_time:.4f}hr, Epoch {epoch}")
                actor_losses['loss'].append(actor_loss.item())
                actor_losses['nll'].append(-action_log_likelihood.item())
                actor_losses['entropy'].append(action_entropy.item())
                start = time.time()
        
        if epoch%10 == 0 or epoch == num_steps-1:
            torch.save({'actor' : actor.state_dict(),},
                os.path.join(save_path, 'policy_epoch_{epoch}.pt'.format(epoch = epoch)))
            with open(os.path.join(save_path, 'loss_epoch_{epoch}.pkl'.format(epoch = epoch)), 'wb') as f:
                pickle.dump(actor_loss, f)
                
            # %% Evaluation step
            if not os.path.isdir(os.path.join(save_path, 'figure')):
                os.makedirs(os.path.join(save_path, 'figure'))
            
            fig_path = os.path.join(save_path, 'figure')
            val_idx = torch.randint(len(data_loader), size=(100,))
            val_s, val_a = [], []
            for idx in val_idx:
                s, a = data_loader.dataset[idx]
                val_s.append(torch.from_numpy(s))
                val_a.append(torch.from_numpy(a))
            val_s = torch.stack(val_s)
            val_a = torch.stack(val_a)
            
            with torch.no_grad():
                action_dist = actor(val_s.to(device))
                mean, std = action_dist.mean, action_dist.stddev
                lower, upper = mean-1.645*std, mean+1.645*std
                lower = lower.detach().cpu().numpy()
                upper = upper.detach().cpu().numpy()
                mean = mean.detach().cpu().numpy()
                val_a = val_a.detach().cpu().numpy()
                
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            action_label = ['ESS', 'PEM', 'H2-util', 'H2-market']
            t = np.arange(0, 100)
            for k, ax in enumerate(axes.flat):
                ax.plot(mean[:, k], label='Pred-mean')
                ax.plot(val_a[:, k], label='Ground-truth', alpha = 0.6)
                ax.fill_between(t, lower[:,k], upper[:,k], alpha=0.2, label="90% Confidence Interval")
                #ax.legend()
                ax.set_title(action_label[k])
                if k ==0:
                    ax.set_ylim(-1.5, 1.5)
                else:
                    ax.set_ylim(-0.5, 1.5)
                
                if k ==1:
                    ax.legend()
            fig.suptitle("epoch_{epoch}_val_{val_idx}_result".format(epoch = epoch, val_idx = i), fontsize=18)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_path, 'epoch_{epoch}_val_{val_idx}_result.jpg').format(epoch = epoch, val_idx = i), dpi = 100)
            plt.show()
            plt.close()