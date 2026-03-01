import argparse
import numpy as np
import time
import os
import pickle
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mascor.models import actor, critic
from torch.utils.data import DataLoader
from mascor.utils.pt_data_loader import *
import torch
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    '--target-country', type=str, default="France", help="target country",
)

parser.add_argument(
    "--region", type=str, default="Dunkirk", help="target region"
)

parser.add_argument(
    "--sample-size", type = int, default= 50000, help="oracle dataset size",
)

parser.add_argument(
    "--design-option", type = str, default= 'c_fax_fix', help="whether fixing c-tax or not",
)

parser.add_argument(
    '--data-type', type=str, default="wind", help="datatype",
)

parser.add_argument(
    "--device", type=str, default="cuda:0", help="device"
)

parser.add_argument(
    "--epoch", type = int, default= 100, help="Epoch",
)

parser.add_argument(
    "--pre-train-loading", action="store_true", help="Enable pre-training loading"
)

parser.add_argument(
    "--batch-size", type = int, default= 128, help="Batch size",
)

parser.add_argument(
    "--lr", type = float, default= 1e-4, help="learning rate",
)

parser.add_argument(
    "--z-type", type = str, default= "mv", help="z_type for ctg & rtg prediction",
)

#token including part (default is vanilla decision transformer)
parser.add_argument("--des-token", action="store_true", help="adding des token",)
parser.add_argument("--z-token", action="store_true", help="adding z (noise) token",)

# decision transformer parsing
parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
parser.add_argument('--K', type=int, default=24, help = "maximum sequence length")
parser.add_argument('--pct_traj', type=float, default=1., help = "top pc episode used")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
parser.add_argument('--embed_dim', type=int, default=256)
parser.add_argument('--n_layer', type=int, default=4)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--activation_function', type=str, default='relu')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
parser.add_argument('--warmup_steps', type=int, default=10000)
parser.add_argument('--num_eval_episodes', type=int, default=100)
parser.add_argument('--max_iters', type=int, default=500)
parser.add_argument('--num_steps_per_iter', type=int, default=10000)
parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
# %% 
if __name__ == "__main__":
    args = parser.parse_args()
    REPO_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = str(REPO_ROOT / "dataset")
    data_path = os.path.join(DATA_DIR, '{country}/{region}/oracle_dataset_{option}_sample_{sample}'.format(country = args.target_country, 
                                                                                                              region = args.region,
                                                                                                              option = args.design_option,
                                                                                                              sample = args.sample_size))
    dataset = Dataset_global_solution(data_path, max_seq = 24, z_type = args.z_type)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Model & optimizer instantiate
    device = args.device
    state_dim = dataset.state_dim
    act_dim = dataset.action_dim
    des_dim = dataset.des_dim
    z_dim = dataset.noise_dim
    ep_length = dataset.state.shape[1]
    max_seq_len = 24
    embed_dim = args.embed_dim
    n_layer = args.n_layer
    n_head = args.n_head
    activation_function = args.activation_function
    dropout = args.dropout
    del dataset
    
    actor = actor(
                state_dim = state_dim,
                act_dim = act_dim,
                des_dim = des_dim,
                z_dim = z_dim,
                max_length = max_seq_len,
                ep_length = ep_length,
                hidden_size = embed_dim,
                n_layer = n_layer,
                n_head = n_head,
                n_inner = 4*embed_dim,
                activation_function = activation_function,
                n_positions = 1024,
                resid_pdrop = dropout,
                attn_pdrop = dropout,
                des_token = args.des_token, #attribute for des token layer
                z_token = False
            )
    critic = critic(
                state_dim = state_dim,
                act_dim = act_dim,
                des_dim = des_dim,
                z_dim = z_dim,
                max_length = max_seq_len,
                ep_length = ep_length,
                hidden_size = embed_dim,
                n_layer = n_layer,
                n_head = n_head,
                n_inner = 4*embed_dim,
                activation_function = activation_function,
                n_positions = 1024,
                resid_pdrop = dropout,
                attn_pdrop = dropout,
                des_token = args.des_token, #attribute for des token layer
                z_token = args.z_token, #attribute for z token layer
            )
    
    actor = actor.to(device=device)
    critic = critic.to(device=device)
    
    warmup_steps = args.warmup_steps    
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr = args.learning_rate,
        weight_decay = args.weight_decay,
    )
    log_temperature = torch.tensor(np.log(args.temperature))
    #log_temperature.requires_grad = True
    #log_temperature_optimizer = torch.optim.Adam([log_temperature],
    #            lr=1e-4,betas=[0.9, 0.999],)
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr = args.learning_rate,
        weight_decay = args.weight_decay,
    )
    
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
        actor_optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
        critic_optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )
    
    state_loss_fn = torch.nn.MSELoss()
    
    # Training loop
    train_losses = []
    actor.train()
    critic.train()
    num_steps = args.max_iters

    #loss list
    action_loss_list = []
    state_loss_list = []
    rtg_loss_list = []
    ctg_loss_list = []
    
    #Loading pre-trained state
    save_path = os.path.join(DATA_DIR, '{country}/{region}/oracle_dataset_{option}_sample_{sample}/checkpoint_transformer_des_{des}_z_{z}_z_type_{z_type}'.format(country = args.target_country, 
                                                                                                                                                  region = args.region,
                                                                                                                                                  option = args.design_option,
                                                                                                                                                  sample = args.sample_size,
                                                                                                                                                  des = args.des_token,
                                                                                                                                                  z = args.z_token, z_type = args.z_type))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    file_history = os.listdir(save_path)
    file_history = [s for s in file_history if "state" in s]
    
    if file_history:
        extract_numbers = np.vectorize(lambda s: int(re.search(r'\d+', s).group()) if re.search(r'\d+', s) else None)
        num_arr = extract_numbers(file_history)
        pre_epoch = max(num_arr)
    else:
        pre_epoch = None
    
    if args.pre_train_loading:
        if pre_epoch:
            checkpoint_path = os.path.join(save_path, 'state_dict_{pre_epoch}'.format(pre_epoch = pre_epoch))
            state_dict = torch.load(checkpoint_path)
            actor.load_state_dict(state_dict['actor'])
            critic.load_state_dict(state_dict['critic'])
            print("Pre-trained weight loaded:", checkpoint_path)
            #loss history data
            with open(os.path.join(save_path, 'loss_{pre_epoch}.pkl'.format(pre_epoch = pre_epoch)), 'rb') as f:
                loss = pickle.load(f) 
        else:
            assert False, "Error: pre_trained state_dict is not existed"
            pass
    else:
        pre_epoch = 0
        loss = {}
        loss['action'] = []
        loss['rtg'] = []
        loss['ctg'] = []
    #%% Training step
    for epoch in range(pre_epoch, args.epoch):
        start = time.time()
        for i, (states, actions, rewards, co2, rtg, ctg, timestamp, des, z, mask) in enumerate(data_loader):
            states = states.to(dtype = torch.float32, device = device)
            actions = actions.to(dtype = torch.float32, device = device)
            rewards = rewards.to(dtype = torch.float32, device = device)
            co2 = co2.to(dtype = torch.float32, device = device)
            rtg = rtg.to(dtype = torch.float32, device = device)
            ctg = ctg.to(dtype = torch.float32, device = device)
            timestamp = timestamp.to(dtype = torch.long, device = device)
            des = des.to(dtype = torch.float32, device = device)
            z = z.to(dtype = torch.float32, device = device)
            mask = mask.to(dtype = torch.long, device = device)
            action_target = torch.clone(actions)
            ctg_target = torch.clone(ctg)
            rtg_target = torch.clone(rtg)
            
            #Model forward
            mu_action_preds, std_action_preds = actor.forward(des, z, ctg, rtg, states, actions, timestamp, attention_mask = mask)
            ctg_preds, rtg_preds = critic.forward(des, z, states, actions, co2, rewards, timestamp, attention_mask = mask)
            
            # Loss function
            # action loss: negative log-likelihood + entropy term
            act_dim = mu_action_preds.shape[2]
            mu_action_preds = mu_action_preds.reshape(-1, act_dim)[mask.reshape(-1) > 0]
            std_action_preds = std_action_preds.reshape(-1, act_dim)[mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[mask.reshape(-1) > 0]            
            action_dist = torch.distributions.Normal(loc = mu_action_preds, scale = torch.exp(std_action_preds))
            action_log_likelihood = torch.mean(action_dist.log_prob(action_target)) 
            action_entropy = action_dist.entropy().mean()
            temperature = log_temperature.exp().detach()
            action_loss = -(action_log_likelihood + temperature*action_entropy)
            
            #rtg loss: negative log-likelihood
            mu_rtg_preds, log_std_rtg_pres = rtg_preds[:,:,0].unsqueeze(-1), rtg_preds[:,:,1].unsqueeze(-1)
            mu_rtg_preds = mu_rtg_preds.reshape(-1, 1)[mask.reshape(-1) > 0]
            log_std_rtg_pres = log_std_rtg_pres.reshape(-1, 1)[mask.reshape(-1) > 0]
            rtg_target = rtg_target.reshape(-1, 1)[mask.reshape(-1) > 0]   
            rtg_dist = torch.distributions.Normal(loc = mu_rtg_preds, scale = torch.exp(log_std_rtg_pres))
            rtg_loss = -torch.mean(rtg_dist.log_prob(rtg_target))
            
            #ctg loss: negative log-likelihood
            mu_ctg_preds, log_std_ctg_pres = ctg_preds[:,:,0].unsqueeze(-1), ctg_preds[:,:,1].unsqueeze(-1)
            mu_ctg_preds = mu_ctg_preds.reshape(-1, 1)[mask.reshape(-1) > 0]
            log_std_ctg_pres = log_std_ctg_pres.reshape(-1, 1)[mask.reshape(-1) > 0]
            ctg_target = ctg_target.reshape(-1, 1)[mask.reshape(-1) > 0]   
            ctg_dist = torch.distributions.Normal(loc = mu_ctg_preds, scale = torch.exp(log_std_ctg_pres))
            ctg_loss = -torch.mean(ctg_dist.log_prob(ctg_target))
            
            # Back-propagation
            # Actor
            actor_loss = action_loss
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.25)
            actor_optimizer.step()
            actor_scheduler.step()
            
            # Critic
            critic_loss = rtg_loss + ctg_loss
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.25)
            critic_optimizer.step()
            critic_scheduler.step()
            
            if i%100 == 0 and i>0:
                end = time.time()
                epooch_run_time = (end-start)*len(data_loader)/100/3600
                print(f"Action Loss: {action_loss.item():.4f}, RTG Loss: {rtg_loss.item():.4f}, CTG Loss: {ctg_loss.item():.4f}, Epoch: {epoch}, Epoch time: {epooch_run_time:.4f}hr")
                loss['action'].append(action_loss.item())
                loss['rtg'].append(rtg_loss.item())
                loss['ctg'].append(ctg_loss.item())
                start = time.time()
        
        if (epoch%10 == 0 and epoch>0) or (epoch == args.epoch-1):
            torch.save({
                'actor' : actor.state_dict(),
                'critic' : critic.state_dict()},
                os.path.join(save_path, 'state_dict_{epoch}'.format(epoch = epoch)))
            with open(os.path.join(save_path, 'loss_{epoch}.pkl'.format(epoch = epoch)), 'wb') as f:
                pickle.dump(loss, f)
            
            # Evaluation step
            if not os.path.isdir(os.path.join(save_path, 'figure')):
                os.makedirs(os.path.join(save_path, 'figure'))
            fig_path = os.path.join(save_path, 'figure')
            val_size = 5
            ep_s, ep_a, ep_r, ep_c, ep_rtg, ep_ctg, ep_t, ep_des, ep_z, ep_mask = data_loader.dataset.valid_set(val_size, device = device)
            with torch.no_grad():
                ep_mu_action, ep_std_action = actor.forward(ep_des, ep_z, ep_ctg, ep_rtg, ep_s, ep_a, ep_t, attention_mask = ep_mask)
                ep_ctg_pred, ep_rtg_pred = critic.forward(ep_des, ep_z, ep_s, ep_a, ep_c, ep_r, ep_t, attention_mask = ep_mask)
                action_list = ['ESS', 'PEM', 'H2-util', 'H2-market']   
                ep_mu_action = ep_mu_action.detach().cpu().numpy().reshape(val_size, ep_length, act_dim)
                ep_std_action = ep_std_action.detach().cpu().numpy().reshape(val_size, ep_length, act_dim)
                ep_a = ep_a.detach().cpu().numpy().reshape(val_size, ep_length, act_dim)
                a_lower_bound = ep_mu_action - np.exp(ep_std_action)*1.645
                a_uppder_bound = ep_mu_action + np.exp(ep_std_action)*1.645
                
                ep_rtg_pred = ep_rtg_pred.detach().cpu().numpy().reshape(val_size, ep_length, 2)
                ep_rtg = ep_rtg.detach().cpu().numpy().reshape(val_size, ep_length, 1)
                rtg_lower_bound = ep_rtg_pred[:,:,0]-np.exp(ep_rtg_pred[:,:,1])*1.645
                rtg_uppder_bound = ep_rtg_pred[:,:,0]+np.exp(ep_rtg_pred[:,:,1])*1.645
                ep_ctg_pred = ep_ctg_pred.detach().cpu().numpy().reshape(val_size, ep_length, 2)
                ep_ctg = ep_ctg.detach().cpu().numpy().reshape(val_size, ep_length, 1)
                ctg_lower_bound = ep_ctg_pred[:,:,0]-np.exp(ep_ctg_pred[:,:,1])*1.645
                ctg_uppder_bound = ep_ctg_pred[:,:,0]+np.exp(ep_ctg_pred[:,:,1])*1.645
                t = np.arange(0, 576)
            for j in range(val_size):
                fig, axes = plt.subplots(3, 2, figsize=(10, 8))
                for k, ax in enumerate(axes.flat):
                    if k<4: #action figure
                        ax.plot(ep_mu_action[j][:, k], label='Pred-mean')
                        ax.plot(ep_a[j][:, k], label='Ground-truth', alpha = 0.6)
                        ax.fill_between(t, a_lower_bound[j][:,k], a_uppder_bound[j][:,k], alpha=0.2, label="90% Confidence Interval")
                        #ax.legend()
                        ax.set_title(action_list[k])
                        if k ==0:
                            ax.set_ylim(-1.5, 1.5)
                        else:
                            ax.set_ylim(-0.5, 1.5)
                    else:
                        if k == 4:
                            ax.plot(t, ep_rtg_pred[j][:, 0], label='Pred-mean')
                            ax.fill_between(t, rtg_lower_bound[j], rtg_uppder_bound[j], alpha=0.2, label="90% Confidence Interval")
                            ax.plot(ep_rtg[j][:, 0], label='Ground-truth', alpha = 0.6)
                            #ax.legend()
                            ax.set_title('RTG')
                        else:
                            ax.plot(ep_ctg_pred[j][:, 0], label='Pred-mean')
                            ax.fill_between(t, ctg_lower_bound[j], ctg_uppder_bound[j], alpha=0.2, label="90% Confidence Interval")
                            ax.plot(ep_ctg[j][:, 0], label='Ground-truth', alpha = 0.6)
                            ax.legend()
                            ax.set_title('CTG')
                fig.suptitle("epoch_{epoch}_val_{val_idx}_result".format(epoch = epoch, val_idx = j), fontsize=18)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_path, 'epoch_{epoch}_val_{val_idx}_result.jpg').format(epoch = epoch, val_idx = j), dpi = 100)
                plt.show()
                plt.close()