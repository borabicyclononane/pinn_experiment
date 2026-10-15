"""Remote LR-transfer sweep (run via agave on a CUDA GPU). Writes one JSON per config into $OUTPUT_DIR/sweep/.

Priority order: phase A (regress1d, ode2), B (ode1, heat2d), C (ode4); widths 1024, 2048, 4096 (ode4: 1024, 2048).
Stops cleanly before BUDGET_S so that Kaggle saves the outputs (a timed-out kernel loses /kaggle/working).
"""
import os, sys, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from mupinn.train import train

BUDGET_S = float(os.environ.get("AGAVE_BUDGET_S", 5.0 * 3600))
OUT = os.path.join(os.environ.get("OUTPUT_DIR", "outputs/local"), "sweep")
os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)

# width-major priority: 4096 first (not run locally), then 2048, then 1024
PRIORITY = [(4096, "regress1d"), (4096, "ode2"), (4096, "ode1"), (4096, "heat2d"),
            (2048, "ode4"), (2048, "ode2"), (2048, "heat2d"), (2048, "regress1d"), (2048, "ode1"),
            (1024, "ode4"), (1024, "ode2"), (1024, "heat2d"), (1024, "regress1d"), (1024, "ode1")]
LRS = {"sgd": [2.0 ** k for k in range(-11, -1)], "adam": [2.0 ** k for k in range(-12, -3)]}
cfgs = []
for w, task in PRIORITY:
    if True:
        for opt in ["sgd", "adam"]:
            for param in ["sp", "mup"]:
                for lr in LRS[opt]:
                    cfgs.append({"task": task, "width": w, "param": param, "optimizer": opt, "lr": lr, "steps": 1000, "seed": 0})


def name(c):
    keys = ["task", "width", "depth", "param", "optimizer", "lr", "steps", "seed", "act", "N", "out_init"]
    return "_".join(f"{k}={c.get(k)}" for k in keys)


t0 = time.time()
done = 0
for i, c in enumerate(cfgs):
    p = os.path.join(OUT, name(c) + ".json")
    if os.path.exists(p):
        continue
    if time.time() - t0 > BUDGET_S:
        print(f"budget reached after {done} runs; stopping at config {i}/{len(cfgs)}", flush=True)
        break
    ts = time.time()
    kw = dict(c); kw["task_name"] = kw.pop("task")
    r = train(device=device, want_grad=False, **kw)
    r["stats"] = {str(k): v for k, v in r["stats"].items()}
    with open(p, "w") as f:
        json.dump(r, f)
    done += 1
    print(f"[{i+1}/{len(cfgs)}] {name(c)} loss={r['final_loss']:.3e} relL2={r['rel_l2']:.3f} div={r['diverged']} {time.time()-ts:.1f}s elapsed={time.time()-t0:.0f}s", flush=True)
print(f"finished {done} runs in {time.time()-t0:.0f}s", flush=True)
