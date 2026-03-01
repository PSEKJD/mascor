import argparse
import time
import warnings
import os
import copy
import json
import numpy as np
import torch
from botorch.exceptions import BadInitialCandidatesWarning
from mascor.models import generator, discriminator
from mascor.utils.gan_data_loader import Dataset
from mascor.utils.helper import select_pareto_and_dominated_min
from mascor.optimization import uq_problem
from mascor.solvers import pt_policy
from mascor.utils.env import env_stack 
from pathlib import Path 

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--target-country', type=str, default="France", help="target country", )
parser.add_argument("--region", type=str, default="Dunkirk", help="target region")
parser.add_argument("--sample-size", type=int, default=50000, help="oracle dataset size", )
parser.add_argument("--op-period", type=int, default=576, help="oracle dataset size", )
parser.add_argument("--design-option", type=str, default='c_fax_fix', help="whether fixing c-tax or not", )
parser.add_argument('--data-type', type=str, default="wind", help="datatype", )
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument("--prob-fail", type=float, default=0.5, help="failure probability", )
parser.add_argument("--solver", type=str, default="PT", help="solver-type", )
parser.add_argument("--scenario-size", type=int, default=1000, help="scenario size", )
parser.add_argument("--optim-iter", type=int, default=100, help="MOBO iteration", )
parser.add_argument("--pre-loading", action="store_true", help="Pre-loading optim history")
parser.add_argument("--infer_action", type=str, default="mu",
                    help="action-inference based on mu value of normal dist", )
parser.add_argument("--validation-type", type=str, default="test",)
# %%
if __name__ == "__main__":
    args = parser.parse_args()
    train_dataset = Dataset(args.target_country, args.region, uni_seq=24, max_seq=24 * 24, data_type='wind-ele',
                            flag='train')
    # env-cofing setting
    env_config = {}
    env_config['scale'] = 50000  # 50MW
    env_config['op-period'] = args.op_period
    env_config['max-SMP'] = train_dataset.price_scale.data_max_[0]
    env_config['min-SMP'] = train_dataset.price_scale.data_min_[0]
    env_config['max-c-tax'] = 132.12
    env_config['min-c-tax'] = 0.10
    env_config['flatten'] = True
    env_config['renew_split'] = False
    env_config['fw'] = 1
    env_config['n-worker'] = args.scenario_size
    c_tax_list = {'France': 47.96, 'Denmark': 28.10, 'Germany': 48.39, 'Norway': 107.78}
    env_config['c-tax'] = c_tax_list[args.target_country]
    args.env_config = env_config

    # GAN loading
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
    save_path = os.path.join(DATASET_DIR, '{country}/{region}/checkpoint_gan/{data_type}_{gp}'.format(country=args.target_country,
                                                                                         region=args.region,
                                                                                         data_type=args.data_type,
                                                                                         gp=20.0))
    checkpoint_path = os.path.join(save_path, 'model_mmd_True_epoch_{epoch}'.format(epoch=save_epoch[args.target_country + '/' + args.region]))
    state_dict = torch.load(checkpoint_path, map_location=args.device)
    netG = generator(ch_dim=ch_dim, nz=205).to(args.device)
    netG.load_state_dict(state_dict['netG'])
    netG.eval()
    del save_path, checkpoint_path, state_dict

    args.critic = True
    args.des_token = True
    args.z_token = True
    args.z_type = 'mv'
    policy = pt_policy(args)
    problem = uq_problem(args, env_stack, policy, train_dataset)
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # %% Optimal & dominant data loading
    BASE_DIR = Path(__file__).resolve().parent 
    log_dir = BASE_DIR / "result" / f"{args.target_country}/{args.region}"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(log_dir, 'iter_{iteration}_history_pfss_{pfss}_sample_size_{size}.pkl'.format(
                                 country=args.target_country,
                                 region=args.region,
                                 iteration=args.optim_iter,
                                 pfss=args.prob_fail,
                                 size=1000))
    pareto_settings = {"Dunkirk": [0.05, 30, 0.5], "Skive": [0.05, 30, 0.5], "Fredericia": [0.05, 30, 0.5], "Weener": [0.05, 30, 0.5]}
    settings = pareto_settings[args.region]
    (pareto_obj, pareto_con, pareto_des, pareto_obj_error, pareto_con_error,
     dom_obj, dom_con, dom_des, dom_obj_error, dom_con_error) = select_pareto_and_dominated_min(save_path,
                                                                                                country = args.target_country,
                                                                                                region = args.region,
                                                                                                min_diff=settings[0],
                                                                                                dominated_k=settings[1],
                                                                                                front_gap=settings[2])
    asc_idx = torch.argsort(-pareto_obj[:, 0], dim=0)  # ascending
    pareto_obj = pareto_obj[asc_idx]
    pareto_con = pareto_con[asc_idx]
    pareto_des = pareto_des[asc_idx]
    del save_path

    error_tensors = [pareto_obj_error, pareto_con_error, dom_obj_error, dom_con_error, ]
    for err in error_tensors:
        if (err > 0).any():
            raise RuntimeError(f"Error detected in {err.shape}: {err[err > 0]}")

    test_dataset = Dataset(args.target_country, args.region, uni_seq=24, max_seq=24 * 24, data_type='wind-ele',
                           flag="test")
    # %% result-data-package
    uq_result = {"train": {}, "test": {}}  # train: gan, test: 2023-2025 data
    uq_result["train"] = {"pareto": {}, "dominant": {}}
    uq_result["test"] = {"pareto": {}, "dominant": {}}
    template = {"des": [], "LCOX": [], "CO2": [],
                "mu-LCOX(s-10000)[$/kg]": [], "mu-ctg(s-10000)[ton/month]": [], "pfss (s-10000)": [],
                "mu-LCOX(s-1000)[$/kg]": [], "mu-ctg(s-1000)[ton/month]": [], "pfss (s-1000)": []}
    for idx, des in enumerate(pareto_des):
        uq_result["train"]["pareto"][f"des-{idx}"] = copy.deepcopy(template)
        uq_result["train"]["pareto"][f"des-{idx}"]["mu-LCOX(s-1000)[$/kg]"] = -pareto_obj[idx, 0].detach().cpu().numpy()
        uq_result["train"]["pareto"][f"des-{idx}"]["mu-ctg(s-1000)[ton/month]"] = -pareto_obj[idx, 1].detach().cpu().numpy()
        uq_result["train"]["pareto"][f"des-{idx}"]["pfss"] = pareto_con[idx].detach().cpu().numpy() + args.prob_fail

        uq_result["test"]["pareto"][f"des-{idx}"] = copy.deepcopy(template)
        uq_result["test"]["pareto"][f"des-{idx}"]["mu-LCOX(s-1000)[$/kg]"] = -pareto_obj[idx, 0].detach().cpu().numpy()
        uq_result["test"]["pareto"][f"des-{idx}"]["mu-ctg(s-1000)[ton/month]"] = -pareto_obj[idx, 1].detach().cpu().numpy()
        uq_result["test"]["pareto"][f"des-{idx}"]["pfss"] = pareto_con[idx].detach().cpu().numpy() + args.prob_fail

    for idx, des in enumerate(dom_des):
        uq_result["train"]["dominant"][f"des-{idx}"] = copy.deepcopy(template)
        uq_result["train"]["dominant"][f"des-{idx}"]["mu-LCOX(s-1000)[$/kg]"] = -dom_obj[idx, 0].detach().cpu().numpy()
        uq_result["train"]["dominant"][f"des-{idx}"]["mu-ctg(s-1000)[ton/month]"] = -dom_obj[idx, 1].detach().cpu().numpy()
        uq_result["train"]["dominant"][f"des-{idx}"]["pfss"] = dom_con[idx].detach().cpu().numpy() + args.prob_fail

        uq_result["test"]["dominant"][f"des-{idx}"] = copy.deepcopy(template)
        uq_result["test"]["dominant"][f"des-{idx}"]["mu-LCOX(s-1000)[$/kg]"] = -dom_obj[idx, 0].detach().cpu().numpy()
        uq_result["test"]["dominant"][f"des-{idx}"]["mu-ctg(s-1000)[ton/month]"] = -dom_obj[idx, 1].detach().cpu().numpy()
        uq_result["test"]["dominant"][f"des-{idx}"]["pfss"] = dom_con[idx].detach().cpu().numpy() + args.prob_fail

    save_path = os.path.join(log_dir, 'pareto_validation_{type}_dataset_gan_epoch_{epoch}.json'.format(country=args.target_country, region=args.region,
                                                                                                        type = args.validation_type,
                                                                                                        epoch = save_epoch[args.target_country + '/' + args.region]))
    if args.validation_type == "train":
        target_dataset = train_dataset
    else:
        target_dataset = test_dataset

    # %% pareto-point vaidation start: train-data first, and test-data
    # train-data
    print("-*"*100)
    print(f"UQ will be excuted on {args.validation_type}")
    print(f"Length of pareto-des {len(pareto_des)}")
    for idx, des in enumerate(pareto_des):
        start = time.time()
        LCOX, CO2, mu_LCOX, mu_CO2, pfss, _ = problem.planning(des, netG, target_dataset, train_dataset)
        end = time.time()
        uq_result["train"]["pareto"][f"des-{idx}"]["des"] = des.detach().cpu().numpy()
        uq_result["train"]["pareto"][f"des-{idx}"]["LCOX"] = LCOX
        uq_result["train"]["pareto"][f"des-{idx}"]["CO2"] = CO2
        uq_result["train"]["pareto"][f"des-{idx}"]["mu-LCOX(s-10000)[$/kg]"] = mu_LCOX
        uq_result["train"]["pareto"][f"des-{idx}"]["mu-ctg(s-10000)[ton/month]"] = mu_CO2
        uq_result["train"]["pareto"][f"des-{idx}"]["pfss (s-10000)"] = pfss
        print(f"Pareto step {idx} at des {np.round(des.cpu().detach().numpy(), 2)}")
        print(f"Train Simulation time: {end-start}")
        print(f"E[LCOX]-1000: {-pareto_obj[idx, 0].item():.2f} & E[LCOX]-10000: {mu_LCOX.item():.2f}")
        print(f"E[CTG]-1000: {-pareto_obj[idx, 1].item()*100:.2f} & E[CTG]-10000: {mu_CO2.item():.2f}")
        print(f"PFSS-1000: {pareto_con[idx].item() + args.prob_fail:.2f} & PFSS-10000: {pfss.item():.2f}")
    
        with open(save_path, "w") as f:
            json.dump(uq_result, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))
    time.sleep(60)

    for idx, des in enumerate(pareto_des):
        start = time.time()
        LCOX, CO2, mu_LCOX, mu_CO2, pfss, _ = problem.planning(des, None, target_dataset, train_dataset)
        end = time.time()
        uq_result["test"]["pareto"][f"des-{idx}"]["des"] = des.detach().cpu().numpy()
        uq_result["test"]["pareto"][f"des-{idx}"]["LCOX"] = LCOX
        uq_result["test"]["pareto"][f"des-{idx}"]["CO2"] = CO2
        uq_result["test"]["pareto"][f"des-{idx}"]["mu-LCOX(s-10000)[$/kg]"] = mu_LCOX
        uq_result["test"]["pareto"][f"des-{idx}"]["mu-ctg(s-10000)[ton/month]"] = mu_CO2
        uq_result["test"]["pareto"][f"des-{idx}"]["pfss (s-10000)"] = pfss
        print(f"Pareto step {idx} at des {np.round(des.cpu().detach().numpy(), 2)}")
        print(f"Test Simulation time: {end-start}")
        print(f"E[LCOX]-1000: {-pareto_obj[idx, 0].item():.2f} & E[LCOX]-10000: {mu_LCOX.item():.2f}")
        print(f"E[CTG]-1000: {-pareto_obj[idx, 1].item()*100:.2f} & E[CTG]-10000: {mu_CO2.item():.2f}")
        print(f"PFSS-1000: {pareto_con[idx].item() + args.prob_fail:.2f} & PFSS-10000: {pfss.item():.2f}")

    with open(save_path, "w") as f:
        json.dump(uq_result, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))
    time.sleep(60)
    # # %% dominant-point validation start: train-data first, and test-data
    for idx, des in enumerate(dom_des):
        start = time.time()
        LCOX, CO2, mu_LCOX, mu_CO2, pfss, _ = problem.planning(des, netG, target_dataset, train_dataset)
        end = time.time()
        uq_result["train"]["dominant"][f"des-{idx}"]["des"] = des.detach().cpu().numpy()
        uq_result["train"]["dominant"][f"des-{idx}"]["LCOX"] = LCOX
        uq_result["train"]["dominant"][f"des-{idx}"]["CO2"] = CO2
        uq_result["train"]["dominant"][f"des-{idx}"]["mu-LCOX(s-10000)[$/kg]"] = mu_LCOX
        uq_result["train"]["dominant"][f"des-{idx}"]["mu-ctg(s-10000)[ton/month]"] = mu_CO2
        uq_result["train"]["dominant"][f"des-{idx}"]["pfss (s-10000)"] = pfss
        print(f"Dominant step {idx} at des {np.round(des.cpu().detach().numpy(), 2)}")
        print(f"Train Simulation time: {end-start}")
        print(f"E[LCOX]-1000: {-dom_obj[idx, 0].item():.2f} & E[LCOX]-10000: {mu_LCOX.item():.2f}")
        print(f"E[CTG]-1000: {-dom_obj[idx, 1].item()*100:.2f} & E[CTG]-10000: {mu_CO2.item():.2f}")
        print(f"PFSS-1000: {dom_con[idx].item() + args.prob_fail:.2f} & PFSS-10000: {pfss.item():.2f}")
    
    with open(save_path, "w") as f:
        json.dump(uq_result, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))
    time.sleep(60)

    for idx, des in enumerate(dom_des):
        start = time.time()
        LCOX, CO2, mu_LCOX, mu_CO2, pfss, _ = problem.planning(des, None, target_dataset, train_dataset)
        end = time.time()
        uq_result["test"]["dominant"][f"des-{idx}"]["des"] = des.detach().cpu().numpy()
        uq_result["test"]["dominant"][f"des-{idx}"]["LCOX"] = LCOX
        uq_result["test"]["dominant"][f"des-{idx}"]["CO2"] = CO2
        uq_result["test"]["dominant"][f"des-{idx}"]["mu-LCOX(s-10000)[$/kg]"] = mu_LCOX
        uq_result["test"]["dominant"][f"des-{idx}"]["mu-ctg(s-10000)[ton/month]"] = mu_CO2
        uq_result["test"]["dominant"][f"des-{idx}"]["pfss (s-10000)"] = pfss
        print(f"Dominant step {idx} at des {np.round(des.cpu().detach().numpy(), 2)}")
        print(f"Test Simulation time: {end-start}")
        print(f"E[LCOX]-1000: {-dom_obj[idx, 0].item():.2f} & E[LCOX]-10000: {mu_LCOX.item():.2f}")
        print(f"E[CTG]-1000: {-dom_obj[idx, 1].item()*100:.2f} & E[CTG]-10000: {mu_CO2.item():.2f}")
        print(f"PFSS-1000: {dom_con[idx].item() + args.prob_fail:.2f} & PFSS-10000: {pfss.item():.2f}")
    with open(save_path, "w") as f:
        json.dump(uq_result, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))