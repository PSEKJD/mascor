import numpy as np
import torch
import torch.nn as nn
import transformers
from .model import TrajectoryModel
from .trajectory_gpt2 import GPT2Model

class DecisionTransformer_actor(TrajectoryModel):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    def __init__(
            self,
            state_dim,
            act_dim,
            des_dim,
            z_dim,
            ep_length, 
            hidden_size,
            max_length=None,
            action_tanh=True,
            des_token = True,
            z_token = True, 
            **kwargs
    ):
        super().__init__(state_dim, act_dim, des_dim, z_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        
        # trajectory = {des, z, ctg, rtg, s, a}
        self.embed_timestep = nn.Embedding(ep_length, hidden_size)
        
        if des_token:
            self.embed_des = torch.nn.Linear(self.des_dim, hidden_size)
        else:
            self.embed_des = None
        
        if z_token:
            self.embed_noise = torch.nn.Sequential(torch.nn.Linear(self.z_dim, hidden_size),nn.LeakyReLU(negative_slope=0.01),
                                               torch.nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(negative_slope=0.01),
                                               torch.nn.Linear(hidden_size, hidden_size))
        else:
            self.embed_noise = None
            
        self.embed_ctg = torch.nn.Linear(1, hidden_size)
        self.embed_rtg = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_action_mu = nn.Sequential(
            *([nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(negative_slope=0.01),
               nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(negative_slope=0.01),
               nn.Linear(hidden_size, self.act_dim)] + ([] if action_tanh else []))
        )
        self.predict_action_std = nn.Sequential(
            *([nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(negative_slope=0.01),
               nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(negative_slope=0.01),
               nn.Linear(hidden_size, self.act_dim)] + ([] if action_tanh else []))
        )

    def forward(self, des, z, ctg, rtg, states ,actions, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        if self.embed_des:    
            des_embeddings = self.embed_des(des)
        if self.embed_noise:
            z_embeddings = self.embed_noise(z)
            
        co2_embeddings = self.embed_ctg(ctg)
        returns_embeddings = self.embed_rtg(rtg)
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)
        
        # time embeddings are treated similar to positional embeddings
        if self.embed_des:
            des_embeddings = des_embeddings + time_embeddings
        if self.embed_noise:
            z_embeddings = z_embeddings + time_embeddings
            
        co2_embeddings = co2_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        
        # this makes the sequence look like (des, noise, ctg, rtg, state, action)
        # which works nice in an autoregressive sense since states predict actions
        embedding_list = []
        attention_mask_list = []
        
        if self.embed_des:
            embedding_list.append(des_embeddings)
            attention_mask_list.append(attention_mask)
        if self.embed_noise:
            embedding_list.append(z_embeddings)
            attention_mask_list.append(attention_mask)
        
        embedding_list.extend([co2_embeddings, returns_embeddings, 
                               state_embeddings, action_embeddings])
        stacked_inputs = torch.stack(embedding_list, dim=1).permute(0, 2, 1, 3).reshape(batch_size, len(embedding_list)*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        attention_mask_list.extend([attention_mask, attention_mask, 
                                    attention_mask, attention_mask])
        stacked_attention_mask = torch.stack(attention_mask_list, dim=1).permute(0, 2, 1).reshape(batch_size, len(embedding_list)*seq_length).to(device = stacked_inputs.device)
        
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        x = x.reshape(batch_size, seq_length, len(embedding_list), self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        mu_action_preds = self.predict_action_mu(x[:,-2])  # predict next action given state
        log_std_action_preds = self.predict_action_std(x[:,-2])  # predict next action given state
        
        return mu_action_preds, log_std_action_preds

class DecisionTransformer_critic(TrajectoryModel):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    def __init__(
            self,
           state_dim,
           act_dim,
           des_dim,
           z_dim,
           ep_length, 
           hidden_size,
           max_length=None,
           action_tanh=True,
           des_token = True,
           z_token = True, 
           **kwargs
    ):
        super().__init__(state_dim, act_dim, des_dim, z_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        
        # trajectory = {des, z, s, a, co2, reward}
        self.embed_timestep = nn.Embedding(ep_length, hidden_size)
        if des_token:
            self.embed_des = torch.nn.Linear(self.des_dim, hidden_size) #newly added
        else:
            self.embed_des = None
        if z_token:    
            self.embed_noise = torch.nn.Sequential(torch.nn.Linear(self.z_dim, hidden_size),nn.LeakyReLU(negative_slope=0.01),
                                               torch.nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(negative_slope=0.01),
                                               torch.nn.Linear(hidden_size, hidden_size))
        else:
            self.embed_noise = None
            
        self.embed_c = torch.nn.Linear(1, hidden_size)
        self.embed_r = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_return = torch.nn.Linear(hidden_size, 2) #mu and std
        self.predict_co2 = torch.nn.Linear(hidden_size, 2) #mu and std

    def forward(self, des, z, states ,actions, co2, reward, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        if self.embed_des:    
            des_embeddings = self.embed_des(des)
        if self.embed_noise:
            z_embeddings = self.embed_noise(z)
            
        co2_embeddings = self.embed_c(co2)
        returns_embeddings = self.embed_r(reward)
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        if self.embed_des:
            des_embeddings = des_embeddings + time_embeddings
        if self.embed_noise:
            z_embeddings = z_embeddings + time_embeddings
            
        co2_embeddings = co2_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        # this makes the sequence look like (des, noise, state, action, co2, reward)
        # which works nice in an autoregressive sense since states predict actions
        embedding_list = []
        attention_mask_list = []
        
        if self.embed_des:
            embedding_list.append(des_embeddings)
            attention_mask_list.append(attention_mask)
        if self.embed_noise:
            embedding_list.append(z_embeddings)
            attention_mask_list.append(attention_mask)
        
        embedding_list.extend([state_embeddings, action_embeddings, 
                               co2_embeddings, returns_embeddings])
        stacked_inputs = torch.stack(embedding_list, dim=1).permute(0, 2, 1, 3).reshape(batch_size, len(embedding_list)*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # to make the attention mask fit the stacked inputs, have to stack it as well
        attention_mask_list.extend([attention_mask, attention_mask, 
                                    attention_mask, attention_mask])
        stacked_attention_mask = torch.stack(attention_mask_list, dim=1).permute(0, 2, 1).reshape(batch_size, len(embedding_list)*seq_length).to(device = stacked_inputs.device)
        
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        x = x.reshape(batch_size, seq_length, len(embedding_list), self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        ctg_pred = self.predict_co2(x[:,-3])  # predict next ctg given action and state
        rtg_pred = self.predict_return(x[:,-3])  # predict next ctg given action and state

        return ctg_pred, rtg_pred
