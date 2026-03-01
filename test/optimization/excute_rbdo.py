import argparse
import time
import warnings
import os
import pickle
import numpy as np
import torch
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch import fit_gpytorch_mll
from mascor.models import generator
from mascor.utils.gan_data_loader import Dataset
from mascor.optimization import generate_initial_data, initialize_model, optimize_qnehvi_and_get_observation
from mascor.optimization import rbdo_problem 
from mascor.solvers import pt_policy
from mascor.utils.env import env_stack
from pathlib import Path
# %% 
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
    "--op-period", type = int, default= 576, help="oracle dataset size",
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
    "--prob-fail", type = float, default= 0.5, help="failure probability",
)

parser.add_argument(
    "--solver", type = str, default="PT", help="solver-type",
)

parser.add_argument(
    "--scenario-size", type = int, default= 1000, help="scenario size",
)

parser.add_argument(
    "--optim-iter", type = int, default= 100, help="MOBO iteration",
)

parser.add_argument(
    "--pre-loading", action="store_true", help="Pre-loading optim history"
)

parser.add_argument(
    "--infer_action", type = str, default="mu", help="action-inference based on mu value of normal dist",
)
# %%
if __name__ == "__main__":
    args = parser.parse_args()
    dataset = Dataset(args.target_country, args.region, uni_seq = 24, max_seq = 24*24, data_type = 'wind-ele', flag='train')
    #env-cofing setting
    env_config = {}
    env_config['scale'] = 50000 #50MW
    env_config['op-period'] = args.op_period
    env_config['max-SMP'] = dataset.price_scale.data_max_[0]
    env_config['min-SMP'] = dataset.price_scale.data_min_[0]
    env_config['max-c-tax'] = 132.12
    env_config['min-c-tax'] = 0.10
    env_config['flatten'] = True
    env_config['renew_split'] = False
    env_config['fw'] = 1
    env_config['n-worker'] = args.scenario_size
    c_tax_list = {'France':47.96, 'Denmark':28.10, 'Germany': 48.39, 'Norway':107.78}
    env_config['c-tax'] = c_tax_list[args.target_country]
    args.env_config = env_config

    # GAN loading
    if "ele" in args.data_type:
        ch_dim = 2
    else:
        ch_dim = 1

    save_epoch = {}
    save_epoch['France/Dunkirk'] = 15000
    save_epoch['France/Alpes-de-Haute-Provence'] = 15000
    save_epoch['Denmark/Skive'] = 15000
    save_epoch['Denmark/Fredericia'] = 15000
    save_epoch['Germany/Wunsiedel'] = 15000
    save_epoch['Germany/Weener'] = 15000
    save_epoch['Norway/Porsgrunn'] = 15000

    REPO_ROOT = Path(__file__).resolve().parents[2]
    DATASET_DIR = str(REPO_ROOT / "dataset")
    save_path = os.path.join(DATASET_DIR, '{country}/{region}/checkpoint_gan/{data_type}_{gp}'.format(country = args.target_country, 
                                                                                                          region = args.region, data_type = args.data_type,
                                                                                                          gp = 20.0))
    checkpoint_path = os.path.join(save_path, 'model_mmd_True_epoch_{epoch}'.format(epoch = save_epoch[args.target_country+'/'+ args.region]))
    state_dict = torch.load(checkpoint_path, map_location=args.device)   
    netG = generator(ch_dim = ch_dim, nz = 205).to(args.device)
    netG.load_state_dict(state_dict['netG'])
    netG.eval()    
    del save_path, checkpoint_path, state_dict
    
    args.critic = True
    args.des_token = True
    args.z_token = True
    args.z_type = 'mv'
    policy = pt_policy(args)
    problem = rbdo_problem(args, env_stack, policy, dataset)
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    N_BATCH = args.optim_iter
    MC_SAMPLES = 128 
    BATCH_SIZE = 10
    verbose = True
    hvs_qnehvi = []

    BASE_DIR = Path(__file__).resolve().parent 
    log_dir = BASE_DIR / "result" / f"{args.target_country}/{args.region}"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(log_dir, 'iter_{iteration}_history_pfss_{pfss}_sample_size_{size}.pkl'.format(iteration =args.optim_iter,
                                                                                                           pfss = args.prob_fail,
                                                                                                           size = args.scenario_size))
    if args.pre_loading:
        with open(save_path,"rb") as f:
            history = pickle.load(f)
        train_x_qnehvi = history['step-0']['des']
        mu_lcox = np.array([history['step-0']['mu-lcox[$/kg]']]).reshape(-1,1)
        mu_ctg = np.array([history['step-0']['mu-ctg[ton/month]']]).reshape(-1,1)
        train_obj_qnehvi = np.concatenate((-mu_lcox, -mu_ctg/100), axis = 1)
        train_con_qnehvi = np.array([history['step-0']['pfss']]).reshape(-1,1)-args.prob_fail
        train_x_qnehvi, train_obj_qnehvi, train_con_qnehvi = torch.tensor(train_x_qnehvi, dtype = torch.float64), torch.tensor(train_obj_qnehvi, dtype = torch.float64), torch.tensor(train_con_qnehvi, dtype = torch.float64)
    else:
        train_x_qnehvi, train_obj_qnehvi, train_con_qnehvi = generate_initial_data(problem, 2 * (len(problem.lb) + 1), netG, dataset, save_path=save_path)
    
    mll_qnehvi, model_qnehvi = initialize_model(problem, train_x_qnehvi, train_obj_qnehvi, train_con_qnehvi, args.device)  
    problem.ref_point = torch.tensor([torch.min(train_obj_qnehvi[:,0]), torch.min(train_obj_qnehvi[:,1])])
    hv = Hypervolume(ref_point=problem.ref_point) 
    # compute pareto front
    is_feas = (train_con_qnehvi <= 0).all(dim=-1)
    feas_train_obj = train_obj_qnehvi[is_feas]
    if feas_train_obj.shape[0] > 0:
        #ref-point update
        print('------------------ref-point update in feas-point------------------')
        problem.ref_point = torch.min(feas_train_obj, dim = 0).values
    else:
        naive = (train_con_qnehvi <= 0.1).all(dim=-1)
        naive_train_obj = train_obj_qnehvi[naive]
        if naive_train_obj.shape[0] > 0:
            print('------------------ref-point update in naive-point------------------')
            problem.ref_point = torch.min(naive_train_obj, dim =0).values
        else:
            problem.ref_point = torch.tensor([torch.min(train_obj_qnehvi[:,0]), torch.min(train_obj_qnehvi[:,1])])

    # compute hypervolume   
    hv = Hypervolume(ref_point=problem.ref_point) 
    if feas_train_obj.shape[0] > 0:
        pareto_mask = is_non_dominated(feas_train_obj)
        pareto_y = feas_train_obj[pareto_mask]
        volume = hv.compute(pareto_y/abs(problem.ref_point)) #normalize
    else:
        volume = 0
    hvs_qnehvi.append(volume)
    #%% run N_BATCH rounds of RBDO
    MAX_ATTEMPTS = 10
    for iteration in range(1, N_BATCH + 1):
        t0 = time.monotonic()
        # fit the models
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                fit_gpytorch_mll(mll_qnehvi)
                break
            except Exception as e:
                print(f"[Attempt {attempt}] GP fitting failed: {e}")
        if args.pre_loading:
            history_iter =  [int(k.split('-')[1]) for k in history.keys()]
        else:
            history_iter = None
        if args.pre_loading and iteration in history_iter:
            new_x_qnehvi = history['step-{}'.format(iteration)]['des']
            mu_lcox = np.array([history['step-{}'.format(iteration)]['mu-lcox[$/kg]']]).reshape(-1,1)
            mu_ctg = np.array([history['step-{}'.format(iteration)]['mu-ctg[ton/month]']]).reshape(-1,1)
            new_obj_qnehvi = np.concatenate((-mu_lcox, -mu_ctg/100), axis = 1)
            new_con_qnehvi = np.array([history['step-{}'.format(iteration)]['pfss']]).reshape(-1,1)-args.prob_fail
            new_x_qnehvi = torch.tensor(new_x_qnehvi, dtype = torch.float64)
            new_obj_qnehvi = torch.tensor(new_obj_qnehvi, dtype = torch.float64)
            new_con_qnehvi = torch.tensor(new_con_qnehvi, dtype = torch.float64)   
        else:
            # define the qParEGO and qNEHVI acquisition modules using a QMC sampler
            qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
            new_x_qnehvi, new_obj_qnehvi, new_con_qnehvi = optimize_qnehvi_and_get_observation(problem, model_qnehvi, train_x_qnehvi, 
                                                                                           qnehvi_sampler,  BATCH_SIZE, 
                                                                                           netG, dataset, iteration, args.device, save_path=save_path)
        train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
        train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])
        train_con_qnehvi = torch.cat([train_con_qnehvi, new_con_qnehvi])
        
        _, unique_mask = torch.unique(train_x_qnehvi, dim=0, return_inverse=True)
        train_x_qnehvi = train_x_qnehvi[unique_mask]
        train_obj_qnehvi = train_obj_qnehvi[unique_mask]
        train_con_qnehvi = train_con_qnehvi[unique_mask]
        noise_std = 1e-3
        train_obj_qnehvi += noise_std * torch.randn_like(train_obj_qnehvi)
        
        # compute pareto front
        is_feas = (train_con_qnehvi <= 0).all(dim=-1)
        feas_train_obj = train_obj_qnehvi[is_feas]
        if feas_train_obj.shape[0] > 0:
            pareto_mask = is_non_dominated(feas_train_obj)
            pareto_y = feas_train_obj[pareto_mask]
            print('pareto points: ', pareto_y.shape[0])
            
            #ref-point update
            if iteration%4 == 0:
                if pareto_y.shape[0]>1:
                    print('------------------ref-point update in pareto-point------------------')
                    problem.ref_point = (torch.min(pareto_y, dim = 0).values*0.9 + problem.ref_point*0.1)
                else:
                    print('------------------ref-point update in feas-point------------------') 
                    problem.ref_point = (torch.min(feas_train_obj, dim = 0).values*0.9 + problem.ref_point*0.1)
                hv = Hypervolume(ref_point=problem.ref_point) 
                
            # compute feasible hypervolume
            volume = hv.compute(pareto_y/abs(problem.ref_point)) #normalize
        else:
            volume = 0.0
        hvs_qnehvi.append(volume)
        
        # reinitialize the models so they are ready for fitting on next iteration
        # Note: we find improved performance from not warm starting the model hyperparameters
        # using the hyperparameters from the previous iteration
        mll_qnehvi, model_qnehvi = initialize_model(problem, train_x_qnehvi, train_obj_qnehvi, train_con_qnehvi, args.device)
        
        t1 = time.monotonic()
        if verbose:
            print(f"\nBatch {iteration:>2}: Hypervolume ( qNEHVI) = "
                  f"({hvs_qnehvi[-1]:>4.2f}), "
                  f"time = {t1-t0:>4.2f}.",end="",)
        else:
            print(".", end="")