"""Run a list of configs, saving each result as JSON; skip configs that are already done."""
from __future__ import annotations
import json, os, sys, time, hashlib, math
import torch
from .train import train

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def cfg_name(cfg):
    keys = ["task", "width", "depth", "param", "optimizer", "lr", "steps", "seed", "act", "N", "out_init"]
    s = "_".join(f"{k}={cfg.get(k)}" for k in keys)
    return s


def cfg_path(cfg, group):
    d = os.path.join(RESULTS, group)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, cfg_name(cfg) + ".json")


def run_cfg(cfg, device, **extra):
    """Train one config with the project defaults filled in; returns the JSON-serializable result."""
    kw = dict(cfg); kw.update(extra)
    kw.setdefault("depth", 3); kw.setdefault("act", "tanh"); kw.setdefault("N", 512); kw.setdefault("out_init", "default")
    kw.setdefault("steps", 1000); kw.setdefault("seed", 0)
    kw["task_name"] = kw.pop("task")
    r = train(device=device, **kw)
    r["stats"] = {str(k): v for k, v in r["stats"].items()}
    return r


def run_one(cfg, group, device, force=False):
    p = cfg_path(cfg, group)
    if os.path.exists(p) and not force:
        return None
    r = run_cfg(cfg, device)
    with open(p, "w") as f:
        json.dump(r, f)
    return r


def run_many(cfgs, group, device, log=None):
    t0 = time.time()
    for i, cfg in enumerate(cfgs):
        ts = time.time()
        r = run_one(cfg, group, device)
        if r is not None:
            msg = f"[{i+1}/{len(cfgs)}] {cfg_name(cfg)} loss={r['final_loss']:.3e} eval={r['final_eval']:.3e} relL2={r['rel_l2']:.3f} div={r['diverged']} {time.time()-ts:.1f}s"
            print(msg, flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


def load_group(group):
    d = os.path.join(RESULTS, group)
    out = []
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            with open(os.path.join(d, fn)) as f:
                out.append(json.load(f))
    return out


if __name__ == "__main__":
    # usage: python -m mupinn.runner <configs.json> <group> <device>
    cfgs = json.load(open(sys.argv[1]))
    group, device = sys.argv[2], sys.argv[3]
    torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
    run_many(cfgs, group, device)
