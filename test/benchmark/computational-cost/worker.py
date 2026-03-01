import argparse
import sys
import json
import time
import os
import concurrent.futures
from pathlib import Path
_GLOBAL_RENEWABLE = None
_GLOBAL_GRID = None

def _limit_threads_1_runtime():
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        import torch; torch.set_num_threads(1)
    except:
        pass
    try:
        import mkl; mkl.set_num_threads(1)   
    except:
        pass
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except:
        pass
try:
    import resource
    HAS_RESOURCE = True
except:
    HAS_RESOURCE = False

def now_usage():
    if HAS_RESOURCE:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "cpu_user_s": ru.ru_utime,
            "cpu_sys_s": ru.ru_stime,
        }
    return {"cpu_user_s":0.0,"cpu_sys_s":0.0}
   
def _milp_worker_init(renewable, grid):
    _limit_threads_1_runtime()    
    print(f"[MILP][worker] PID={os.getpid()} Threads=1", file=sys.stderr, flush=True)

    global _GLOBAL_RENEWABLE, _GLOBAL_GRID
    _GLOBAL_RENEWABLE = renewable
    _GLOBAL_GRID = grid

# ------------------- MILP Solve Logic -------------------
def _milp_solve_range(start_i: int, count: int, env_config, des_local=None, track_status: bool=False):
    # Solving [start_i: start_i+count] episode using global renewalbe, grid
    from mascor.solvers import global_solver
    global _GLOBAL_RENEWABLE, _GLOBAL_GRID
    renewable = _GLOBAL_RENEWABLE
    grid = _GLOBAL_GRID
    ok_cnt = 0
    for k in range(count):
        idx = (start_i + k) % len(renewable)
        r_i = renewable[idx]
        g_i = grid[idx]
        s = global_solver(env_config)
        s.solver_instance(renewable=r_i, SMP=g_i, option=True)
        res = s.solve_planning()
        if track_status:
            ok = True
            if isinstance(res, dict) and "status" in res:
                ok = (str(res["status"]).lower() in {"ok", "optimal", "success"})
            ok_cnt += int(ok)
    return ok_cnt

# ------------------- MILP wrapper -------------------
def milp_function(num_instances: int, parallel: bool, episode, args, track_status: bool=False):
    _, renewable, price = episode
    t0 = time.perf_counter()
    t1 = time.perf_counter()
    solved=0; failed=0

    if not parallel:
        _milp_worker_init(renewable, price)
        solved_cnt = _milp_solve_range(0, num_instances, args.env_config, track_status=track_status)
        if track_status:
            solved = solved_cnt
            failed = num_instances - solved
    else:
        max_workers = args.workers if args.workers else 16
        print(f"[MILP] max_workers={max_workers} (OS scheduler free)", file=sys.stderr, flush=True)
        env_cfg = args.env_config
        base = num_instances // max_workers
        rem  = num_instances % max_workers

        chunks, start = [], 0
        for w in range(max_workers):
            cnt = base + (1 if w < rem else 0)
            if cnt > 0:
                chunks.append((start, cnt))
                start += cnt

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_milp_worker_init,
            initargs=(renewable, price),  
        ) as ex:
            futs = [ex.submit(_milp_solve_range, st, cnt, env_cfg, track_status) for (st, cnt) in chunks]
            if track_status:
                for f in concurrent.futures.as_completed(futs):
                    solved += int(f.result())
                failed = num_instances - solved
            else:
                for f in concurrent.futures.as_completed(futs):
                    f.result()

    t2 = time.perf_counter()

    return ({
        "init_wall_s": t1 - t0,
        "compute_wall_s": t2 - t1,
        "solved": solved,
        "failed": failed
    } if track_status else {
        "init_wall_s": t1 - t0,
        "compute_wall_s": t2 - t1
    })


def transformer_function(device: str, n: int, episode, dataset, args):
    import torch, numpy as np, pandas as pd, time
    from mascor.optimization import uq_problem
    from mascor.solvers import pt_policy
    from mascor.utils.env import env_stack  
     
    if device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("Requested device='gpu' but CUDA is not available.")
    
    noise, renewable, price = episode
    noise = noise.to(device = args.device)
    renewable = torch.as_tensor(renewable).to(device = args.device)
    price = torch.as_tensor(price).to(device = args.device)
    total = len(renewable)
    n = min(n, total)

    # ---------- Instantiate ----------
    t0 = time.perf_counter()
    policy = pt_policy(args)
    problem = uq_problem(args, env_stack, policy, dataset)
    env = problem.env_class(args.env_config, device=args.device)
    netG, dataset = None, None

    using_cuda = (device == "cuda:0" and torch.cuda.is_available())
    if using_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()  
    t1 = time.perf_counter()

    # ---------- Calculation time (episodic-batch)----------
    micro = 10000
    gpu_mem_max = 0
    s = 0
    
    # des-setting
    des = np.array([args.env_config['LH2-cap'], args.env_config['ESS-cap'], 
                   args.env_config['PEM-ratio'], args.env_config['X-flow']],dtype=np.float32)
    des = torch.from_numpy(des).to(device = args.device) 
    num_batches = 0
    while s * micro < n:
        start_idx = s * micro
        end_idx = min((s + 1) * micro, n)
        batch_episode = (noise[start_idx:end_idx],
                         renewable[start_idx:end_idx],
                         price[start_idx:end_idx],)
        problem.args.scenario_size = end_idx-start_idx
        try:
            _ = problem.planning(des, netG=None, dataset=None, backupdataset=None,
                                 episode= batch_episode, env = env)
            if using_cuda:
                torch.cuda.synchronize()
                gpu_mem_max = max(gpu_mem_max, torch.cuda.max_memory_allocated())
            num_batches += 1
            s += 1
        except RuntimeError as e:
            if using_cuda and ("out of memory" in str(e).lower()) and micro > 1:
                micro = max(1, micro // 2)
                torch.cuda.empty_cache()
                continue
            else:
                raise
    t2 = time.perf_counter()
    return {"init_wall_s": t1 - t0,
            "compute_wall_s": t2 - t1,
            "gpu_mem_max_bytes": int(gpu_mem_max),
            "micro_used": int(micro),
            "num_batches": int(num_batches),
            "n_effective": int(n),
            "device_used": "gpu" if using_cuda else "cpu"}

def wind_power_function(Wind_speed):
    ws = Wind_speed.copy()
    ws = ws * (80/50)**(1/7)
    cutin, rated, cutoff = 1.5,12,25
    idx_zero = ws<=cutin
    idx_mid = (cutin<ws)&(ws<=rated)
    idx_cutoff = (ws>rated)&(ws<=cutoff)
    ws[idx_zero]=0
    ws[idx_mid]=(ws[idx_mid]**3 - cutin**3)/(rated**3 - cutin**3)
    ws[idx_cutoff]=1
    ws[ws>cutoff]=0
    return ws

def main():
    import numpy as np, pandas as pd, torch, pickle
    from mascor.utils.gan_data_loader import Dataset
    from mascor.models import generator
    from mascor.utils.helper import select_pareto_and_dominated_min

    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=["milp", "transformer"], required=True)
    ap.add_argument("--device", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--num", type=int, required=True)
    ap.add_argument("--parallel", default="0")  # parallel = 1
    ap.add_argument("--des_path", type=str, default="")       
    ap.add_argument("--track_status", action="store_true")
    ap.add_argument("--limit_threads_1", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    if args.limit_threads_1:
        _limit_threads_1_runtime()
    
    dataset = Dataset('France', 'Dunkirk', uni_seq = 24, max_seq = 24*24, data_type = 'wind-ele', flag='train')

    # optimum-des loading
    log_dir = './optimization/result'
    save_path = os.path.join(log_dir, 'France/Dunkirk/iter_100_history_pfss_0.5_sample_size_1000.pkl')
    (_, _, pareto_des, _, _, _, _, _, _, _) = select_pareto_and_dominated_min(save_path, country = "France", region = "Dunkirk",
                                                                                        min_diff=0.05, dominated_k=30,front_gap=0.5)
    del save_path
    des = pareto_des[0].cpu().detach().numpy().astype(np.float32)
    env_config = {}
    env_config['scale'] = 50000 #50MW
    env_config['op-period'] = 576
    env_config['max-c-tax'] = 132.12
    env_config['min-c-tax'] = 0.10
    env_config['fw'] = 1
    env_config['c-tax'] = 47.96
    env_config['max-SMP'] = dataset.price_scale.data_max_[0]
    env_config['min-SMP'] = dataset.price_scale.data_min_[0]
    env_config['X-flow'] = des[3]
    env_config['PEM-ratio'] = des[2]
    env_config['ESS-cap'] = des[1]
    env_config['LH2-cap'] = des[0]
    env_config['SOC-init'] = env_config['ESS-cap']*0.1
    env_config['L-H2-init'] = 0 
    
    args.env_config = env_config
    args.target_country = 'France'
    args.region = 'Dunkirk'
    args.sample_size = 50000
    args.op_period = 576
    args.design_option = 'c_fax_fix'
    args.data_type = "wind"
    args.prob_fail = 0.5
    args.solver = "PT"
    args.infer_action = "mu"
    args.des_token = True
    args.z_token = True
    args.z_type = 'mv'
    args.critic = True

    # --- episode prepare ---
    args.device = "cuda:0" if (args.device=="gpu" and torch.cuda.is_available()) else "cpu"

    # episode generation
    save='./dataset/France/Dunkirk/checkpoint_gan/wind_20.0'
    ckpt=os.path.join(save,'model_mmd_True_epoch_15000')
    netG = generator_1dcnn_24_v2(ch_dim=1,nz=205).to(args.device)
    netG.load_state_dict(torch.load(ckpt,map_location=args.device)['netG'])
    netG.eval()

    noise = torch.randn(args.num,205,device=args.device)
    weather = netG(noise).detach().cpu().numpy().reshape(-1,576)
    weather[weather<0]=0
    weather = dataset.weather_scale.inverse_transform(weather)
    wind = wind_power_function(weather)
    renewable = wind * env_config['scale']

    df = pd.DataFrame(wind)
    rm = df.T.rolling(window=24).mean().T.values[:,23:]
    noise = torch.tensor(rm[:,::24],dtype=torch.float32)

    si = np.random.randint(0,len(dataset),size=args.num)
    price = np.array([dataset.price_scaled[i:i+dataset.max_seq] for i in si])[:,:,0]
    price = dataset.price_scale.inverse_transform(price)

    episode = (noise, renewable, price)
    del netG, weather, wind, si, df, rm
    
    # --- E2E time (whole-process)--
    t0 = time.perf_counter()
    if args.framework == "transformer":
        res_payload = transformer_function(
            device=args.device, n=args.num, episode=episode, args=args, dataset = dataset)
    else:
        res_payload = milp_function(
            num_instances=args.num,
            parallel=(args.parallel == "1"),
            episode=episode,
            args=args,
            track_status=args.track_status,)
    t1 = time.perf_counter()

    out = {
        "framework": args.framework,
        "device": args.device,
        "num_problems": args.num,
        "parallel": (args.parallel == "1"),
        "e2e_wall_s": t1 - t0,  # E2E
    }
    out |= now_usage()      # cpu_user_s, cpu_sys_s
    out |= res_payload      # init_wall_s, compute_wall_s, gpu_mem_max_bytes 등

    denom = out["e2e_wall_s"] + 1e-12
    count = out.get("n_effective", args.num)
    out["throughput"] = float(count) / denom

    if "compute_wall_s" in out and out["e2e_wall_s"] > 0:
        out["compute_share"] = out["compute_wall_s"] / out["e2e_wall_s"]

    return out

if __name__=="__main__":
    import contextlib, traceback
    try:
        with contextlib.redirect_stdout(sys.stderr):
            out=main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    print(json.dumps(out), flush=True)