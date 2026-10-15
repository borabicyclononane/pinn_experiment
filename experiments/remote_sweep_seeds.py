"""Seed replicates (seeds 1, 2) of the LR-transfer sweep for regress1d and ode2, widths 64..2048, on a remote GPU."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from mupinn.runner import run_cfg, cfg_name

BUDGET_S = float(os.environ.get("AGAVE_BUDGET_S", 1.9 * 3600))
OUT = os.path.join(os.environ.get("OUTPUT_DIR", "outputs/local"), "sweep"); os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)
LRS = {"sgd": [2.0 ** k for k in range(-11, -1)], "adam": [2.0 ** k for k in range(-12, -3)]}
cfgs = []
for w in [64, 128, 256, 512, 1024, 2048]:
    for task in ["ode2", "regress1d"]:
        for seed in [1, 2]:
            for opt in ["sgd", "adam"]:
                for param in ["sp", "mup"]:
                    for lr in LRS[opt]:
                        cfgs.append({"task": task, "width": w, "param": param, "optimizer": opt, "lr": lr, "steps": 1000, "seed": seed})
t0 = time.time(); done = 0
for i, c in enumerate(cfgs):
    if time.time() - t0 > BUDGET_S:
        print(f"budget reached after {done} runs; stopping at {i}/{len(cfgs)}", flush=True); break
    p = os.path.join(OUT, cfg_name(c) + ".json"); ts = time.time()
    r = run_cfg(c, device, want_grad=False)
    json.dump(r, open(p, "w")); done += 1
    print(f"[{i+1}/{len(cfgs)}] {cfg_name(c)} loss={r['final_loss']:.3e} div={r['diverged']} {time.time()-ts:.1f}s elapsed={time.time()-t0:.0f}s", flush=True)
print(f"finished {done} runs in {time.time()-t0:.0f}s", flush=True)
