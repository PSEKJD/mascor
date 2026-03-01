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

def run_cmd(env,  pyfile="worker.py", args=None):
    cmd = [sys.executable, pyfile] + (args or [])
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        t1 = time.perf_counter()
        stdout = (p.stdout or "").strip()
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            lines = [ln for ln in stdout.splitlines() if ln.strip()]
            payload = json.loads(lines[-1]) if lines else {}
        payload["runner_wall_s"] = t1 - t0
        payload["args_used"] = args or []
        return payload
    except subprocess.CalledProcessError as e:
        t1 = time.perf_counter()
        return {
            "error": "subprocess_failed",
            "returncode": e.returncode,
            "stderr": (e.stderr or "").strip(),
            "stdout": (e.stdout or "").strip(),
            "runner_wall_s": t1 - t0,
            "args_used": args or [],
        }
    
def base_env():
    return os.environ.copy()

def env_threads_1(env):
    env = env.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    return env

def brief_result(payload):
    if not isinstance(payload, dict):
        return "ERR: invalid payload"
    if "error" in payload:
        kind = payload.get("error")
        rc = payload.get("returncode", "")
        msg = payload.get("exc_msg") or payload.get("stderr", "")[:120]
        return f"ERR({kind}, rc={rc}) {msg}"
    e2e = payload.get("e2e_wall_s")
    thr = payload.get("throughput")
    dev = payload.get("device") or payload.get("device_used")
    return f"OK dev={dev} e2e={e2e:.2f}s thr={thr:.3f}/s"

def main(workers=None):
    results = {}

    # === S1: Single MILP (sanity) ===
    results["S1_cpu1"] = {}
    for fw in ["milp"]:
        env = env_threads_1(base_env())
        args = ["--framework", fw, "--device", "cpu", "--num", "1", "--episode_len", "1"]
        log(f"S1 ▶ {fw}/cpu N=1 ...")
        payload = run_cmd(env=env, cpu_list="0", args=args)
        log(f"S1 ◀ {fw}/cpu N=1 → {brief_result(payload)}")
        results["S1_cpu1"][fw] = payload

    # === S2: Transformer benchmarks (CPU & GPU) ===
    Ns = [10, 20, 50,
          100, 200, 500,
          1000, 2000, 5000,
          10000]
    results["S2_transformer"] = {"cpu": {}, "gpu": {}}

    for dev in ["gpu"]:
        for n in Ns:
            env = base_env()
            if dev == "gpu":
                env["CUDA_VISIBLE_DEVICES"] = "0"
            args = [
                "--framework", "transformer",
                "--device", dev,
                "--num", str(n),
            ]
            log(f"S2 ▶ transformer/{dev} N={n} ...")
            payload = run_cmd(env=env, args=args)
            log(f"S2 ◀ transformer/{dev} N={n} → {brief_result(payload)}")
            results["S2_transformer"][dev][str(n)] = payload

    # === S3: MILP Parallel (CPU multi-process) ===
    results["S3_milp_parallel"] = {}
    for n in Ns:
        env = env_threads_1(base_env())
        args = [
            "--framework", "milp",
            "--device", "cpu",
            "--num", str(n),
            "--parallel", "1",
            "--limit_threads_1",
        ]
        if workers:
            args += ["--workers", str(workers)]

        log(f"S3 ▶ milp/cpu-parallel N={n} (workers={workers}) ...")
        payload = run_cmd(env=env, args=args)
        log(f"S3 ◀ milp/cpu-parallel N={n} → {brief_result(payload)}")
        results["S3_milp_parallel"][str(n)] = payload

    out_file = f"compcost_results_workers_{workers or 'auto'}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(results, f)
    log(f"Saved: {out_file}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    opt = ap.parse_args()
    main(workers=opt.workers)