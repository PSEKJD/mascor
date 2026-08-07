import pickle
import argparse
import sys
import json
import time
import os
import math
import random
import concurrent.futures
import threading
import torch
import numpy as np

# 수정! revision은 Python module/package가 아니라 일반 directory이므로 file 경로를 직접 등록
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 수정! 같은 directory의 mpec.py를 직접 import
from mpec import PlantConfig, build_model, model_size
from typing import Dict, List, Optional, Sequence, Tuple
import pyomo.environ as pyo

_GLOBAL_RENEWABLE = None
_GLOBAL_GRID = None

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
   
def wind_power_function(wind_speed: torch.Tensor) -> torch.Tensor:
    factor = (80.0 / 50.0) ** (1.0 / 7.0)
    w = wind_speed * factor
    cutin, rated, cutoff = 1.5, 12.0, 25.0
    denom = (rated ** 3) - (cutin ** 3)
    p = (w ** 3 - cutin ** 3) / denom
    p = p.clamp_(0.0, 1.0)
    p = torch.where(w > cutoff, torch.zeros((), dtype=p.dtype, device=p.device), p)
    return p
    
def scenario_sampling(args, netG, dataset):
    weather_min = torch.tensor(dataset.weather_scale.data_min_, dtype=torch.float32, device=args.device)
    weather_max = torch.tensor(dataset.weather_scale.data_max_, dtype=torch.float32, device=args.device)
        
    with torch.inference_mode():
        noise = torch.randn(args.num, 205, device=args.device)
        weather_scenario = netG(noise).reshape(-1, 24 * 24)
        weather_scenario[weather_scenario < 0] = 0
        weather_scenario = weather_scenario * (weather_max - weather_min) + weather_min
        wind_power_scenario = wind_power_function(weather_scenario)
        
        # price data
        si = np.random.randint(0, dataset.__len__(), size=args.num)
        price_scenario = np.array([dataset.price_scaled[idx:idx + dataset.max_seq] for idx in si])[:, :,0]  # N x 576 x 1
        price_scenario = dataset.price_scale.inverse_transform(price_scenario)
        price_scenario = torch.tensor(price_scenario, dtype=wind_power_scenario.dtype, device=args.device)
    return wind_power_scenario, price_scenario

# ------------------- MILP Solve Logic -------------------
def _milp_solve_range(start_i: int, count: int, env_config, des_local=None, track_status: bool=False):
    # Solving [start_i: start_i+count] episode using global renewalbe, grid
    from solvers.GLOBAL_solver import solver as GlobalSolver
    global _GLOBAL_RENEWABLE, _GLOBAL_GRID
    renewable = _GLOBAL_RENEWABLE
    grid = _GLOBAL_GRID
    ok_cnt = 0
    for k in range(count):
        idx = (start_i + k) % len(renewable)
        r_i = renewable[idx]
        g_i = grid[idx]
        s = GlobalSolver(env_config)
        s.solver_instance(renewable=r_i, SMP=g_i, option=True)
        res = s.solve_planning()
        if track_status:
            ok = True
            if isinstance(res, dict) and "status" in res:
                ok = (str(res["status"]).lower() in {"ok", "optimal", "success"})
            ok_cnt += int(ok)
    return ok_cnt

# ------------------- MILP wrapper -------------------
def update_input_data(m: pyo.ConcreteModel,
                      renew: np.ndarray,
                      smp: np.ndarray,) -> None:

    renew = np.asarray(renew, dtype=float)
    smp = np.asarray(smp, dtype=float)

    m.renew.store_values({(s, t): float(renew[s, t])
        for s in range(m._n_scenarios)
        for t in range(m._n_hours)
    })

    m.smp.store_values({(s, t): float(smp[s, t])
        for s in range(m._n_scenarios)
        for t in range(m._n_hours)
    })
        
def solve_model(
    m: pyo.ConcreteModel,
    solver_name: str = "gurobi_direct",
    tee: bool = True,
    threads: Optional[int] = None,
    numeric_focus: int = 1,
    warmstart: bool = False,
    log_file: Optional[str] = None,  # 수정! solve별 Gurobi 로그 저장 경로
):
    solver = pyo.SolverFactory(solver_name)
    if not solver.available(exception_flag=False):
        raise RuntimeError(f"Gurobi solver interface '{solver_name}' unavailable.")
    options: Dict[str, float | int | str] = {  # 수정! LogFile 문자열 허용
       "NonConvex": 2,
        "NumericFocus": 1,
        "Presolve": 2,
        "MIPFocus": 1,
        "OBBT": 1,
        "Heuristics": 0.2,
        "MIPGap": 0.001,}

    if threads is not None:
        options["Threads"] = int(threads)
    if log_file is not None:  # 수정! terminal 출력과 별도로 Gurobi .log 저장
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        options["LogFile"] = os.path.abspath(log_file)
    print(
        f"[Gurobi] requested Threads={options.get('Threads', 'auto')}",
        file=sys.stderr,
        flush=True,)
    return solver.solve(
        m,
        tee=tee,
        options=options,
        warmstart=warmstart,
    )

def solve_with_heartbeat(
    m: pyo.ConcreteModel,
    label: str,
    heartbeat_s: float = 60.0,
    **solve_kwargs,
):
    line = "=" * 110
    start_t = time.perf_counter()
    stop_event = threading.Event()
    result = None

    print(
        f"\n{line}\n"
        f"[START] {label} | heartbeat={heartbeat_s:g}s\n"
        f"{line}",
        file=sys.stderr,
        flush=True,
    )

    def heartbeat():
        while not stop_event.wait(heartbeat_s):
            elapsed = time.perf_counter() - start_t
            print(
                f"\n{line}\n"
                f"[STILL RUNNING] {label} | elapsed={elapsed:.1f}s\n"
                f"{line}",
                file=sys.stderr,
                flush=True,
            )

    heartbeat_thread = None
    if heartbeat_s > 0:
        heartbeat_thread = threading.Thread(
            target=heartbeat,
            daemon=True,
        )
        heartbeat_thread.start()

    try:
        result = solve_model(
            m=m,
            **solve_kwargs,
        )
        return result
    finally:
        stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)

        elapsed = time.perf_counter() - start_t
        termination = (
            str(result.solver.termination_condition)
            if result is not None
            else "exception"
        )
        print(
            f"\n{line}\n"
            f"[FINISHED] {label} | "
            f"termination={termination}, elapsed={elapsed:.1f}s\n"
            f"{line}",
            file=sys.stderr,
            flush=True,
        )


def set_lcox_objective(m: pyo.ConcreteModel) -> None:
    m.co2_epsilon.set_value(0.0)
    #m.co2_epsilon_constraint.activate()
    m.objective.set_value(m.annualized_total_cost / 1.0e7)

def set_co2_objective(m: pyo.ConcreteModel) -> None:  # 수정! epsilon anchor용 CO2 objective
    m.co2_epsilon_constraint.deactivate()
    m.objective.set_value(m.expected_CO2 / 1.0e4)

def set_blended_objective(m: pyo.ConcreteModel, alpha: float,) -> None:
    m.co2_epsilon.set_value(0.0)
    normalized_cost = m.annualized_total_cost / 1.0e7
    normalized_co2 = m.expected_CO2 / 1.0e4
    m.objective.set_value(
        (1.0 - alpha) * normalized_cost
        + alpha * normalized_co2
    )

def set_epsilon_objective(m: pyo.ConcreteModel, co2_epsilon: float) -> None:
    m.co2_epsilon.set_value(co2_epsilon)
    m.co2_epsilon_constraint.activate()
    m.objective.set_value(m.annualized_total_cost)

def acceptable(result) -> bool:
    return result.solver.termination_condition in {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
        pyo.TerminationCondition.maxTimeLimit,
    }

def summary(m: pyo.ConcreteModel) -> Dict[str, float]:
    return {
        "X-flow [kg/hr]": pyo.value(m.X_flow),
        "PEM_P-capacity [kW]": pyo.value(m.PEM_P_cap),
        "LH2-cap [kg]": pyo.value(m.LH2_cap),
        "BESS-cap [kWh]": pyo.value(m.ESS_cap),
        "annualized_total_cost": pyo.value(m.annualized_total_cost),
        "expected_CO2_tonne_per_month": pyo.value(m.expected_CO2),
        "scenario_month_CO2": {int(s): pyo.value(m.scenario_month_CO2[s]) for s in m.S},
        "scenario_annualized_total_cost": {int(s): pyo.value(m.scenario_annualized_total_cost[s]) for s in m.S},
        }

def mpec_function(device: str, num: int, netG, dataset, args):
    # ---------------- Time initialization ----------------
    total_t0 = time.perf_counter()
    model_build_time = 0.0
    model_build_count = 0
    solve_time = 0.0
    solve_count = 0

    # ---------------- Scenario generation ----------------
    env_scale = 50000  # 50 MW
    c_tax_list = {
        "France": 47.96,
        "Denmark": 28.10,
        "Germany": 48.39,
        "Norway": 107.78,
    }
    config = PlantConfig(
        c_tax=c_tax_list[args.target_country]
    )
    renew, smp = scenario_sampling(
        args=args,
        netG=netG,
        dataset=dataset,
    )
    renew = renew.detach().cpu().numpy()
    smp = smp.detach().cpu().numpy()
    renew = renew * env_scale

    # ---------------- Model build ----------------
    build_t0 = time.perf_counter()
    model = build_model(
        renew=renew,
        smp=smp,
        config=config,
    )
    model_build_time += time.perf_counter() - build_t0
    model_build_count += 1

    # 수정! args 조합으로 내부 결과 directory 생성
    result_dir = os.path.join(
        "revision",
        "mpec_solver_simple",
        f"mode_{args.mode}_num_{args.num}_worker_{args.workers}_pareto_{args.pareto_num}",
    )
    os.makedirs(result_dir, exist_ok=True)

    lcox_anchor_path = os.path.join(result_dir, "lcox_anchor.pkl")
    co2_anchor_path = os.path.join(result_dir, "co2_anchor.pkl")
    lcox_anchor_log_path = os.path.join(result_dir, "lcox_anchor_gurobi.log")
    co2_anchor_log_path = os.path.join(result_dir, "co2_anchor_gurobi.log")

    # ---------------- LCOX anchor solve ----------------
    set_lcox_objective(model)
    solve_t0 = time.perf_counter()
    r1 = solve_with_heartbeat(
        m=model,
        label="ANCHOR: LCOX MINIMIZATION",
        heartbeat_s=args.heartbeat_s,
        solver_name="gurobi_direct",
        threads=args.workers,
        warmstart=False,
        log_file=lcox_anchor_log_path,  # 수정! anchor Gurobi log 저장
    )
    lcox_anchor_solve_s = time.perf_counter() - solve_t0
    solve_time += lcox_anchor_solve_s
    solve_count += 1

    if not acceptable(r1):
        raise RuntimeError("LCOX anchor solve failed.")

    lcox_anchor = summary(model)
    lcox_anchor.update({
        "termination": str(r1.solver.termination_condition),
        "solve_wall_s": lcox_anchor_solve_s,
        "gurobi_log_path": lcox_anchor_log_path,  # 수정!
    })

    with open(lcox_anchor_path, "wb") as f:
        pickle.dump(lcox_anchor, f)

    print(
        f"[LCOX anchor] saved={lcox_anchor_path}, "
        f"solve_wall_s={lcox_anchor_solve_s:.3f}",
        file=sys.stderr,
        flush=True,
    )

    # 수정! epsilon mode에서는 epsilon grid 생성을 위해 CO2 anchor를 먼저 계산
    co2_anchor = None
    co2_anchor_solve_s = None

    if args.mode == "epsilon":
        set_co2_objective(model)
        solve_t0 = time.perf_counter()
        r2 = solve_with_heartbeat(
            m=model,
            label="ANCHOR: CO2 MINIMIZATION",
            heartbeat_s=args.heartbeat_s,
            solver_name="gurobi_direct",
            threads=args.workers,
            warmstart=False,
            log_file=co2_anchor_log_path,  # 수정! anchor Gurobi log 저장
        )
        co2_anchor_solve_s = time.perf_counter() - solve_t0
        solve_time += co2_anchor_solve_s
        solve_count += 1

        if not acceptable(r2):
            raise RuntimeError("CO2 anchor solve failed.")

        co2_anchor = summary(model)
        co2_anchor.update({
            "termination": str(r2.solver.termination_condition),
            "solve_wall_s": co2_anchor_solve_s,
            "gurobi_log_path": co2_anchor_log_path,  # 수정!
        })

        with open(co2_anchor_path, "wb") as f:
            pickle.dump(co2_anchor, f)

        print(
            f"[CO2 anchor] saved={co2_anchor_path}, "
            f"solve_wall_s={co2_anchor_solve_s:.3f}",
            file=sys.stderr,
            flush=True,
        )

    # 수정! --mode에 따라 epsilon constraint 또는 weighted objective grid 선택
    if args.mode == "epsilon":
        pareto_grid = np.linspace(
            co2_anchor["expected_CO2_tonne_per_year"],
            lcox_anchor["expected_CO2_tonne_per_year"],
            args.pareto_num,
        )
        grid_key = "co2_epsilon"
    else:
        pareto_grid = np.linspace(
            0.0,
            1.0,
            args.pareto_num,
        )
        grid_key = "alpha"

    # 수정! --pareto-idx는 1부터 시작하며, 해당 point 하나만 실행
    if args.pareto_idx is not None:
        if not 1 <= args.pareto_idx <= args.pareto_num:
            raise ValueError(
                f"--pareto-idx must be between 1 and {args.pareto_num}."
            )
        solve_targets = [(
            args.pareto_idx - 1,
            pareto_grid[args.pareto_idx - 1],
        )]
    else:
        solve_targets = list(enumerate(pareto_grid))

    records = []
    pareto_designs = []
    pareto_lcox = []
    pareto_co2 = []
    pareto_solve_times = []

    # ---------------- Pareto solves ----------------
    for point_idx, grid_value in solve_targets:
        point_no = point_idx + 1

        if args.mode == "epsilon":
            set_epsilon_objective(model, float(grid_value))
            objective_label = (
                f"EPSILON CONSTRAINT (epsilon={float(grid_value):.6f})"
            )
        else:
            set_blended_objective(model, float(grid_value))
            objective_label = (
                f"BLENDED OBJECTIVE (alpha={float(grid_value):.6f})"
            )

        # 수정! mode별 point 파일을 분리하여 서로 덮어쓰지 않도록 저장
        pareto_path = os.path.join(
            result_dir,
            f"pareto_{point_no}_{args.mode}.pkl",
        )
        pareto_log_path = os.path.join(
            result_dir,
            f"pareto_{point_no}_{args.mode}_gurobi.log",
        )

        # 수정! --pareto-idx 단독 실행은 반드시 warm-start=False
        point_warmstart = args.pareto_idx is None

        solve_t0 = time.perf_counter()
        result = solve_with_heartbeat(
            m=model,
            label=(
                f"PARETO {point_no}/{args.pareto_num}: "
                f"{objective_label}"
            ),
            heartbeat_s=args.heartbeat_s,
            solver_name="gurobi_direct",
            threads=args.workers,
            warmstart=point_warmstart,
            log_file=pareto_log_path,  # 수정! point별 Gurobi log 저장
        )
        point_solve_s = time.perf_counter() - solve_t0

        solve_time += point_solve_s
        solve_count += 1

        row = {
            "objective_mode": args.mode,
            "pareto_idx": point_no,
            grid_key: float(grid_value),
            "warmstart": point_warmstart,
            "termination": str(result.solver.termination_condition),
            "solve_wall_s": point_solve_s,
            "gurobi_log_path": pareto_log_path,
        }

        if acceptable(result):
            sol = summary(model)
            row.update(sol)

            pareto_designs.append({
                "X-flow [kg/hr]": sol["X-flow [kg/hr]"],
                "PEM_P-capacity [kW]": sol["PEM_P-capacity [kW]"],
                "LH2-cap [kg]": sol["LH2-cap [kg]"],
                "BESS-cap [kWh]": sol["BESS-cap [kWh]"],
            })
            pareto_lcox.append(sol["annualized_total_cost"])
            pareto_co2.append(sol["expected_CO2_tonne_per_year"])
        else:
            pareto_designs.append(None)
            pareto_lcox.append(None)
            pareto_co2.append(None)

        records.append(row)
        pareto_solve_times.append(point_solve_s)

        # 수정! 전체 결과 하나가 아니라 Pareto point별 pkl을 즉시 저장
        pareto_save = {
            "objective_mode": args.mode,
            "scenario_num": args.num,
            "solver_threads": int(args.workers),
            "pareto_num": args.pareto_num,
            "pareto_idx": point_no,
            "pareto_grid": pareto_grid.tolist(),
            "lcox_anchor_path": lcox_anchor_path,
            "co2_anchor_path": co2_anchor_path,
            "record": row,
        }

        with open(pareto_path, "wb") as f:
            pickle.dump(pareto_save, f)

        print(
            f"[{args.mode}] Saved point {point_no}/{args.pareto_num}: "
            f"{pareto_path}, solve_wall_s={point_solve_s:.3f}",
            file=sys.stderr,
            flush=True,
        )

    # 수정! blend mode도 별도 CO2 anchor와 Gurobi log를 저장하되,
    # 기존 blended Pareto warm-start 흐름을 바꾸지 않도록 Pareto solve 후 계산
    if args.mode == "blend":
        set_co2_objective(model)
        solve_t0 = time.perf_counter()
        r2 = solve_with_heartbeat(
            m=model,
            label="ANCHOR: CO2 MINIMIZATION",
            heartbeat_s=args.heartbeat_s,
            solver_name="gurobi_direct",
            threads=args.workers,
            warmstart=False,
            log_file=co2_anchor_log_path,
        )
        co2_anchor_solve_s = time.perf_counter() - solve_t0
        solve_time += co2_anchor_solve_s
        solve_count += 1

        if not acceptable(r2):
            raise RuntimeError("CO2 anchor solve failed.")

        co2_anchor = summary(model)
        co2_anchor.update({
            "termination": str(r2.solver.termination_condition),
            "solve_wall_s": co2_anchor_solve_s,
            "gurobi_log_path": co2_anchor_log_path,
        })

        with open(co2_anchor_path, "wb") as f:
            pickle.dump(co2_anchor, f)

        print(
            f"[CO2 anchor] saved={co2_anchor_path}, "
            f"solve_wall_s={co2_anchor_solve_s:.3f}",
            file=sys.stderr,
            flush=True,
        )

    # ---------------- Final timing ----------------
    total_t1 = time.perf_counter()

    return {
        "solver_threads": int(args.workers),
        "objective_mode": args.mode,
        "requested_pareto_idx": args.pareto_idx,
        "result_dir": result_dir,
        "total_wall_s": total_t1 - total_t0,

        "avg_model_build_s": (
            model_build_time / model_build_count
            if model_build_count > 0
            else 0.0
        ),
        "total_model_build_s": model_build_time,
        "model_build_s": model_build_time,  # 수정! runner brief_result와 key 이름 연결
        "model_build_count": model_build_count,

        "avg_solve_s": (
            solve_time / solve_count
            if solve_count > 0
            else 0.0
        ),
        "total_solve_s": solve_time,
        "solve_count": solve_count,

        "lcox_anchor": lcox_anchor,
        "lcox_anchor_solve_wall_s": lcox_anchor_solve_s,
        "co2_anchor": co2_anchor,
        "co2_anchor_solve_wall_s": co2_anchor_solve_s,

        "pareto_grid": pareto_grid.tolist(),
        "pareto_designs": pareto_designs,
        "pareto_lcox": pareto_lcox,
        "pareto_co2": pareto_co2,
        "pareto_solve_times_s": pareto_solve_times,
        "pareto_records": records,
    }

def main():
    import numpy as np, pandas as pd, torch, pickle
    from utils.gan_data_loader import Dataset
    from gan.WGAN_GP_renewable_model import generator_1dcnn_24_v2
    from utils.helper import select_pareto_and_dominated_min

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--framework",
        choices=["mpec", "mascor"],
        default="mpec",
    )  # 수정! runner 이전 버전과도 호환되도록 optional 유지
    ap.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
    )  # 수정! runner에서 생략해도 기존 계산은 cpu로 실행
    ap.add_argument("--num", type=int, default=10, required=True)
    ap.add_argument(
        "--workers", "--worker",
        dest="workers",
        type=int,
        default=16,
        required=True,
    )  # 수정! --workers와 --worker 모두 허용
    ap.add_argument(
        "--pareto_num", "--pareto-num",
        dest="pareto_num",
        type=int,
        default=16,
        required=True,
    )  # 수정! underscore/hyphen 모두 허용
    ap.add_argument(
        "--mode",
        choices=["epsilon", "blend"],
        required=True,
    )
    ap.add_argument(
        "--pareto-idx",
        type=int,
        default=None,
    )
    ap.add_argument("--heartbeat-s", type=float, default=60.0)
    ap.add_argument("--track_status", action="store_true")
    ap.add_argument(
        "--target-country", "--country",
        dest="target_country",
        type=str,
        default="France",
    )  # 수정! runner의 --country와 기존 --target-country 모두 허용
    ap.add_argument("--region", type=str, default="Dunkirk")
    args = ap.parse_args()
    
    # Loading dataset & GAN
    dataset = Dataset(args.target_country, args.region, uni_seq = 24, max_seq = 24*24, data_type = 'wind-ele', flag='train')
    save_path = os.path.join('./dataset', f'{args.target_country}/{args.region}/checkpoint_gan/wind_20.0')
    checkpoint_path = os.path.join(save_path, 'model_mmd_True_epoch_15000')
    state_dict = torch.load(checkpoint_path, map_location=args.device)
    netG = generator_1dcnn_24_v2(ch_dim = 1, nz = 205).to(args.device)
    netG.load_state_dict(state_dict['netG'])
    netG.eval()
    del save_path, checkpoint_path, state_dict       
    
    # --- E2E time (whole-process)--
    t0 = time.perf_counter()
    res_payload = mpec_function(device=args.device, num=args.num, 
                                    netG = netG, args=args, dataset=dataset)

    t1 = time.perf_counter()

    out = {
        "framework": args.framework,
        "device": args.device,
        "scenario-num": args.num,
        "e2e_wall_s": t1 - t0,  # E2E
    }
    out |= now_usage()      # cpu_user_s, cpu_sys_s
    out |= res_payload      # init_wall_s, compute_wall_s, gpu_mem_max_bytes 등

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