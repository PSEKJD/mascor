# -*- coding: utf-8 -*-
"""
MOBO utlization function
"""
#Import BO libs
import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.exceptions import BadInitialCandidatesWarning

import warnings
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def generate_initial_data(problem, n, netG, dataset, save_path):
    # generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1, seed = 2024).squeeze(1).to(problem.args.device)
    train_y = problem.objective_function(train_x, netG, dataset, loop_idx = 0, save_path = save_path)
    train_obj = train_y[:,:2].reshape(-1,2)
    train_con = train_y[:,-1].reshape(-1,1)
    return train_x, train_obj, train_con

def initialize_model(problem, train_x, train_obj, train_con, device):
    # define models for objective and constraint
    train_x = normalize(train_x, problem.bounds)
    train_y = torch.cat([train_obj, train_con], dim=-1).to(dtype=torch.float64)
    train_x = train_x.to(device = device)
    train_y = train_y.to(device = device)

    print(f"train-x dtype: {train_x.dtype} & train-y dtype: {train_y.dtype}") #check
    
    models = []
    for i in range(train_y.shape[-1]):
        models.append(
            SingleTaskGP(
                train_x, train_y[..., i : i + 1], outcome_transform=Standardize(m=1)
            )
        )
    model = ModelListGP(*models).to(device = device)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def optimize_qnehvi_and_get_observation(problem, model, train_x, sampler,
                                        BATCH_SIZE, netG, dataset, loop_idx, device, save_path):
    """Optimizes the qNEHVI acquisition function, and returns a new candidate and observation."""
    tkwargs = {
    "dtype": torch.double,
    "device": torch.device(device),}
    train_x = normalize(train_x, problem.bounds)
    
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point.tolist(),
        X_baseline=train_x,
        sampler=sampler,
        prune_baseline=True,
        # define an objective that specifies which outcomes are the objectives
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
        # specify that the constraint is on the last outcome
        constraints=[lambda Z: Z[..., -1]],)
    
    # optimize
    standard_bounds = torch.zeros(2, train_x.shape[1], **tkwargs)
    standard_bounds[1] = 1
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=5,
        raw_samples=512,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 100},
        sequential=True,
        inequality_constraints= [(
            torch.tensor([0, 2], dtype=torch.int64, device=tkwargs["device"]),# indices: x0, x2
            torch.tensor([-1.0, 50.0], dtype=torch.float64, device=tkwargs["device"]),# coefficients: -x0 + 100·x2 ≥ 0
            0.0 # rhs
            )
            ])
    # observe new values
    new_x = unnormalize(candidates, bounds=problem.bounds)
    print('Calculation in new x candidates')
    new_y = problem.objective_function(new_x, netG, dataset, loop_idx, save_path = save_path)
    new_obj = new_y[:,:2].reshape(-1,2)
    new_con = new_y[:,-1].reshape(-1,1)
    return new_x, new_obj, new_con