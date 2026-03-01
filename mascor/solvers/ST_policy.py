import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from mascor.models import actor, critic 
import numpy as np
import re
import torch
from pathlib import Path
class policy():
    def __init__(self, args, pt_epoch = None):
        # Rollout buffer: ctg, rtg scaler, state and action dim info
        REPO_ROOT = Path(__file__).resolve().parents[2]
        DATASET_DIR = str(REPO_ROOT / "dataset")
        self.data_path = os.path.join(DATASET_DIR, '{country}/{region}/oracle_dataset_{option}_sample_{sample}'.format(country = args.target_country,
                                                                                                                  region = args.region,
                                                                                                                  option = args.design_option,
                                                                                                                  sample = args.sample_size))
        # Pre-trained model loading
        self.actor = actor(
                    state_dim = 4,
                    act_dim = 4,
                    des_dim = 4,
                    z_dim = 205,
                    max_length = 24,
                    ep_length = 576,
                    hidden_size = 256,
                    n_layer = 4,
                    n_head = 2,
                    n_inner = 4*256,
                    activation_function = 'relu',
                    n_positions = 1024,
                    resid_pdrop = 0.1,
                    attn_pdrop = 0.1,
                    des_token = args.des_token,
                    z_token = False,
                )
        if args.critic is None:
            self.critic = None
        else:
            self.critic = critic(
                    state_dim = 4,
                    act_dim = 4,
                    des_dim = 4,
                    z_dim = 205,
                    max_length = 24,
                    ep_length = 576,
                    hidden_size = 256,
                    n_layer = 4,
                    n_head = 2,
                    n_inner = 4*256,
                    activation_function = 'relu',
                    n_positions = 1024,
                    resid_pdrop = 0.1,
                    attn_pdrop = 0.1,
                    des_token = args.des_token,
                    z_token = args.z_token,  
                )

        self.actor = self.actor.to(device=args.device)
        if self.critic is None:
            print("//"*50)
            print("This solver has no Critic!!!")
            pass
        else:
            self.critic = self.critic.to(device=args.device)
        save_path = os.path.join(DATASET_DIR,'{country}/{region}/oracle_dataset_{option}_sample_{sample}/checkpoint_transformer_des_{des}_z_{z}_z_type_{z_type}'.format(
                                     country=args.target_country,
                                     region=args.region,
                                     option=args.design_option,
                                     sample=args.sample_size,
                                     des=args.des_token,
                                     z=args.z_token, z_type=args.z_type))
        file_history = os.listdir(save_path)
        file_history = [s for s in file_history if "state" in s]
        if file_history:
            extract_numbers = np.vectorize(lambda s: int(re.search(r'\d+', s).group()) if re.search(r'\d+', s) else None)
            num_arr = extract_numbers(file_history)
            max_epoch = max(num_arr)
        else:
            assert False, "Error: pre-trained state-dict is not existed"
        if pt_epoch:    
            checkpoint_path = os.path.join(save_path, 'state_dict_{pre_epoch}'.format(pre_epoch = args.pre_epoch))
        else:
            checkpoint_path = os.path.join(save_path, 'state_dict_{pre_epoch}'.format(pre_epoch = max_epoch))
        state_dict = torch.load(checkpoint_path, map_location = args.device)
        self.actor.load_state_dict(state_dict['actor'])
        self.actor.eval()
        if self.critic is None:
            pass
        else:
            print("//" * 50)
            print("This solver has Critic!!!")
            self.critic.load_state_dict(state_dict['critic'])
            self.critic.eval()
        del file_history, save_path, checkpoint_path, state_dict
        
        #action lb & ub
        self.lb = torch.tensor([-1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=args.device)
        self.ub = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=args.device)
        self.device = args.device

    @torch.no_grad()
    def compute_actions(self, des, z, ctg, rtg, s, a, t, mask, mode = "mu"):
        mu_action_preds, std_action_preds = self.actor.forward(des, z, ctg, rtg, s,
                                                                         a, t, attention_mask = mask)
        if mode == "mu":
            return torch.clamp(mu_action_preds[:, -1, :], self.lb, self.ub)
        elif mode == "random":
            action_dist = torch.distributions.Normal(loc=mu_action_preds[:, -1, :],
                                                     scale=torch.exp(std_action_preds[:, -1, :]))
            action = action_dist.sample().clamp(
                mu_action_preds[:, -1, :] - 1.645 * torch.exp(std_action_preds[:, -1, :]),
                mu_action_preds[:, -1, :] + 1.645 * torch.exp(std_action_preds[:, -1, :]))
            return action.clamp(self.lb, self.ub)
        else:
            action_dist = torch.distributions.Normal(loc=mu_action_preds[:, -1, :],
                                                     scale=torch.exp(std_action_preds[:, -1, :]))
            action = action_dist.sample().clamp(
                mu_action_preds[:, -1, :] - 1.645 * torch.exp(std_action_preds[:, -1, :]),
                mu_action_preds[:, -1, :] + 1.645 * torch.exp(std_action_preds[:, -1, :]))
            return mu_action_preds[:, -1, :], torch.exp(std_action_preds[:, -1, :]), action.clamp(self.lb, self.ub)
     
    @torch.no_grad()
    def compute_goals(self, des, z, s, a, co2, r, t, mask, limit, buffer, step, simcase = "uq"):
        ctg_preds, rtg_preds = self.critic.forward(des, z, s, a, co2, 
                                                   r, t, attention_mask = mask)
        mu_ctg_preds, log_std_ctg_preds = ctg_preds[:, -1, 0].unsqueeze(-1), ctg_preds[:, -1, 1].unsqueeze(-1)
        mu_rtg_preds, log_std_rtg_preds = rtg_preds[:, -1, 0].unsqueeze(-1), rtg_preds[:, -1, 1].unsqueeze(-1)

        if simcase == "uq":
            return mu_rtg_preds, mu_ctg_preds
        elif simcase == "online":
            rtg_dist = torch.distributions.Normal(loc = mu_rtg_preds, scale = torch.exp(log_std_rtg_preds))
            rtg = rtg_dist.sample().clamp(mu_rtg_preds - 1.645 * torch.exp(log_std_rtg_preds),
                                          mu_rtg_preds + 1.645 * torch.exp(log_std_rtg_preds))
            ctg_dist = torch.distributions.Normal(loc=mu_ctg_preds, scale=torch.exp(log_std_ctg_preds))
            ctg = ctg_dist.sample().clamp(mu_ctg_preds - 1.645 * torch.exp(log_std_ctg_preds),
                                          mu_ctg_preds + 1.645 * torch.exp(log_std_ctg_preds))
            return (mu_rtg_preds, torch.exp(log_std_rtg_preds),
                    mu_ctg_preds, torch.exp(log_std_ctg_preds),
                    rtg, ctg)