import os
import sys
import time
import json
import pickle
import subprocess
from datetime import datetime


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _parse_worker_json(stdout):
    
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def run_cmd(env, pyfile="worker.py", args=None):
    
    worker_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        pyfile,
    )
    cmd = [sys.executable, "-u", worker_path] + (args or [])

    t0 = time.perf_counter()
    p = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        check=False,  
    )
    t1 = time.perf_counter()
    stdout = (p.stdout or "").strip()

    if p.returncode != 0:
        return {
            "error": "subprocess_failed",
            "returncode": p.returncode,
            "stdout": stdout,
            "runner_wall_s": t1 - t0,
            "args_used": args or [],
            "cmd_used": cmd,
        }

    payload = _parse_worker_json(stdout)
    if payload is None:
        return {
            "error": "worker_json_missing",
            "returncode": p.returncode,
            "stdout": stdout,
            "runner_wall_s": t1 - t0,
            "args_used": args or [],
            "cmd_used": cmd,
        }

    payload["runner_wall_s"] = t1 - t0
    payload["args_used"] = args or []
    payload["cmd_used"] = cmd
    return payload


def base_env():
    return os.environ.copy()


def brief_result(payload):
    if not isinstance(payload, dict):
        return "ERR: invalid payload"

    if "error" in payload:
        kind = payload.get("error")
        rc = payload.get("returncode", "")
        msg = payload.get("exc_msg") or payload.get("stdout", "")[-300:]
        return f"ERR({kind}, rc={rc}) {msg}"

    threads = payload.get("solver_threads")
    e2e = payload.get("e2e_wall_s", 0.0)
    total = payload.get("total_wall_s", 0.0)
    build = payload.get("model_build_s", 0.0)
    avg_solve = payload.get("avg_solve_s", 0.0)
    solve_count = payload.get("solve_count", 0)

    return (
        f"OK threads={threads} "
        f"e2e={e2e:.2f}s "
        f"mpec_wall={total:.2f}s "
        f"build={build:.2f}s "
        f"avg_solve={avg_solve:.2f}s "
        f"solves={solve_count}"
    )


def main(
    workers=None,
    pareto_num=100,
    country="France",
    region="Dunkirk",
    num=10,
    mode="blend",
    pareto_idx=None,
    heartbeat_s=60.0,
):
    results = {}

    if workers is None:
        Ns = [1, 4, 8, 12, 16, 32]
    else:
        Ns = [workers]

    results["mpec"] = {}

    for n in Ns:
        env = base_env()
        args = [
            "--num", str(num),
            "--workers", str(n),
            "--pareto_num", str(pareto_num),
            "--mode", mode,
            "--heartbeat-s", str(heartbeat_s),
            "--country", country,
            "--region", region,
        ]

        if pareto_idx is not None:
            args += ["--pareto-idx", str(pareto_idx)]

        point_label = (
            f", pareto_idx={pareto_idx}"
            if pareto_idx is not None
            else ""
        )
        log(
            f"MPEC ▶ mode={mode}, scenarios={num}, "
            f"threads={n}, pareto_num={pareto_num}{point_label} ..."
        )
        payload = run_cmd(
            env=env,
            args=args,
        )
        log(
            f"MPEC ◀ mode={mode}, scenarios={num}, "
            f"threads={n}{point_label} → {brief_result(payload)}"
        )
        results["mpec"][str(n)] = payload

    thread_label = "sweep" if workers is None else str(workers)
    idx_label = (
        f"point_{pareto_idx}"
        if pareto_idx is not None
        else "all_points"
    )
    out_file = (
        f"revision/mpec_solver_simple/compcost_results_mpec_"
        f"mode_{mode}_"
        f"threads_{thread_label}_"
        f"scenario_{num}_"
        f"pareto_{pareto_num}_"
        f"{idx_label}.pkl"
    )
    os.makedirs(os.path.dirname(out_file), exist_ok=True)   
    with open(out_file, "wb") as f:
        pickle.dump(results, f)
    log(f"✅ Saved runner summary: {out_file}")

    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--num",
        type=int,
        default=10,
        help="Scenario number",
    )
    ap.add_argument(
        "--workers", "--worker",
        dest="workers",
        type=int,
        default=None,
        help="Gurobi thread number",
    )  
    ap.add_argument(
        "--pareto_num", "--pareto-num",
        dest="pareto_num",
        type=int,
        default=100,
    )
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
    ap.add_argument(
        "--heartbeat-s",
        type=float,
        default=60.0,
    )
    ap.add_argument(
        "--country", "--target-country",
        dest="country",
        type=str,
        default="France",
    )  
    ap.add_argument(
        "--region",
        type=str,
        default="Dunkirk",
    )
    opt = ap.parse_args()
    main(
        workers=opt.workers,
        pareto_num=opt.pareto_num,
        country=opt.country,
        region=opt.region,
        num=opt.num,
        mode=opt.mode,
        pareto_idx=opt.pareto_idx,
        heartbeat_s=opt.heartbeat_s,
    )