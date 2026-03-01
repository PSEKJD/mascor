import os
import numpy as np
from gymnasium import spaces
import argparse
from mascor.utils.gan_data_loader import Dataset
from mascor.utils.env import env_rl_train 
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_torch
from ray import tune, train
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.schedulers import PopulationBasedTraining
import pprint
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec as RLModuleSpec
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
import shutil
from pathlib import Path

torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--target-country", type=str, default="France", help="target country")
parser.add_argument("--region", type=str, default="Dunkirk", help="target country")
parser.add_argument("--design-option", type=str, default="c_fax_fix", help="oracle dataset option")
parser.add_argument("--sample-size", type=int, default=50000, help="oracle dataset size")

parser.add_argument("--run", type=str, default="PPO", help="The RLlib-registered algorithm to use.")
parser.add_argument("--stop-iters", type=int, default=500, help="Number of iterations to train.")
parser.add_argument("--num-samples", type=int, default=5, help="Number of samples in populations.")

#Custom env configuration
parser.add_argument("--obs-length", type = int, default= 24, help="Observation length of profile data",)
parser.add_argument("--op-period", type = int, default= 576, help="Planning period",)
parser.add_argument("--flatten", action="store_true", help="Flatten the dimension of renewable and grid profile",)
parser.add_argument("--bc-support", action="store_true", help="restore checkpoint of behavior policy",)

def register_env(env_name, env_config={}):
    # env = create_env(env_name)
    tune.register_env(env_name,
                      lambda env_config: env_rl_train(config=env_config))
def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config
# %%
if __name__ == "__main__":
    args = parser.parse_args()
    print('//' * 100)
    
    # Register env
    env_name = 'ptx_env'
    env_config = {}
    
    env_config = {}
    env_config['scale'] = 50000 #50MW renewable power plant
    env_config['country'] = args.target_country
    env_config['region'] = args.region
    env_config['co2-option'] = 'strict'
    c_tax_list = {'France':47.96, 'Denmark':28.10, 'Germany': 48.39, 'Norway':107.78}
    env_config['c-tax'] = c_tax_list[args.target_country]
    dataset = Dataset(args.target_country, args.region, uni_seq = 24, max_seq = 24*24, data_type = 'wind-ele', flag='train')
    env_config['price-max'] = dataset.price_scale.data_max_[0]
    env_config['price-min'] = dataset.price_scale.data_min_[0]
    del dataset
    
    env_config['bc-support'] = args.bc_support
    env_config['flatten'] = args.flatten
    env_config['obs-length'] = args.obs_length
    env_config['op-period'] = args.op_period
    REPO_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = str(REPO_ROOT / "dataset")
    env_config['data-dir'] = os.path.join(DATA_DIR, '{country}/{region}/oracle_dataset_{option}_sample_{sample}'.format(country = args.target_country, 
                                                                                                                            region = args.region,
                                                                                                                            option = args.design_option,
                                                                                                                            sample = args.sample_size))
    #%% 
    register_env(env_name, env_config)

    hyperparam_mutations = {
            "lambda": lambda: random.uniform(0.5, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": lambda: random.uniform(1e-6, 1e-4),
            "train_batch_size": lambda: random.randint(1000, 8000),
            "num_sgd_iter": lambda: random.randint(1, 20),
            "sgd_minibatch_size": lambda: random.randint(100, 1000),
            "kl_coeff":  lambda: random.uniform(0.01, 0.5),
            "entropy_coeff": lambda: random.uniform(0, 0.2),
        }
    pbt = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=50,
            resample_probability=0.25,
            # Specifies the mutations of these hyperparams
            hyperparam_mutations=hyperparam_mutations,
            custom_explore_fn=explore
        )
        
    stopping_criteria = {"training_iteration": args.stop_iters}
                 
    if args.flatten:               
        model_config = dict(
             fcnet_hiddens=[512, 512, 512, 256, 256, 256],
             fcnet_activation="relu",
             post_fcnet_hiddens = [256, 128, 64],
             post_fcnet_activation = "relu")
        space_low = np.zeros(shape = (args.obs_length*2 + 2 + 4),dtype = np.float32) 
        space_high = np.zeros(shape = (args.obs_length*2 + 2 + 4),dtype = np.float32)
    else:
        model_config = dict(
             conv_filters=[[4, [3,4], 2], #[out_channels, kernel, stride] 
                            [8, [2,4], 2],
                            [16, [1,4], 1]],
             conv_activation = 'relu',
             post_fcnet_hiddens=[256, 256, 256, 128],
             post_fcnet_activation="relu")
        space_low = np.zeros(shape = (2, args.obs_length + 2 + 4, 1), dtype = np.float32)
        space_high = np.zeros(shape = (2, args.obs_length + 2 + 4, 1), dtype = np.float32)
    space_high[:] = 1
    observation_space = spaces.Box(low = space_low, high = space_high, shape=(space_high.shape), dtype=np.float32)

    if args.bc_support:
         action_space = spaces.Box(low=np.array([0, 0, 0, 0],dtype = np.float32),
                                  high=np.array([1, 1,  1, 1],dtype = np.float32), shape=(4,),dtype=np.float32)
    else:
         action_space = spaces.Box(low=np.array([-1, 0, 0, 0],dtype = np.float32),
                                  high=np.array([1, 1,  1, 1],dtype = np.float32), shape=(4,),dtype=np.float32)

    print('Policy model: ')
    pprint.pprint(model_config)  
    model_config['vf_share_layers'] = False

    module_spec = RLModuleSpec(module_class=PPOTorchRLModule,
                               observation_space=observation_space,
                               action_space = action_space,
                               model_config_dict= model_config,
                               catalog_class=PPOCatalog)
    config = (PPOConfig()
              .environment(env = env_name, env_config=env_config)
              .rl_module(rl_module_spec=module_spec)
              .training(
                lr=1e-5,
                gamma=0.99,
                clip_param = 0.2,
                kl_coeff = 0.5,
                train_batch_size= tune.choice([1000, 2000, 4000]),
                entropy_coeff = 0.01,
                model = model_config,))
    param_space = config.to_dict()
    param_space['disable_env_checking'] = True
    param_space['num_workers'] = 8
    param_space['num_cpus'] = 1
    param_space['num_gpus'] = 0

    tuner = tune.Tuner(args.run,
                       tune_config=tune.TuneConfig(
                            metric="episode_reward_mean",
                            mode="max",
                            scheduler=pbt,
                            num_samples= args.num_samples,),
                        param_space= param_space,
                        run_config=train.RunConfig(stop=stopping_criteria, 
                                                   name = '{country}_{region}_{flatten}_{bc_support}_{stop_iters}'.format(country = args.target_country,
                                                                                                                          region = args.region,
                                                                                                                          flatten = args.flatten,
                                                                                                                          bc_support = args.bc_support,
                                                                                                                          stop_iters = args.stop_iters),))
    results = tuner.fit()
    #%% Transfer best checkpoint to ./datatset/../checkpoint dir
    best_result = results.get_best_result()
    save_dir = os.path.join(DATA_DIR, '{country}/{region}/oracle_dataset_{option}_sample_{sample}/checkpoint_drl'.format(country = args.target_country, region = args.region,
                                                                                                                             option = args.design_option, sample = args.sample_size))
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    checkpoint_path = best_result.checkpoint.path
    progress_path = os.path.join(os.path.dirname(checkpoint_path), "progress.csv".format(args.flatten))
    shutil.copytree(os.path.join(checkpoint_path, "policies/default_policy"), os.path.join(save_dir, 'policy_flatten_{}_bc_{}'.format(args.flatten, args.bc_support)), dirs_exist_ok = True)
    shutil.copy2(progress_path, os.path.join(save_dir, "progress_flatten_{}_bc_{}.csv".format(args.flatten, args.bc_support)))