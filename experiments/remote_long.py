"""Long-run LR transfer on ode2: 5000 steps, fixed vs resampled collocation points, widths 64/256/1024."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from mupinn.runner import run_cfg, cfg_name

BUDGET_S = float(os.environ.get("AGAVE_BUDGET_S", 2.4 * 3600))
OUT = os.path.join(os.environ.get("OUTPUT_DIR", "outputs/local"), "sweep_long"); os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)
LRS = {"sgd": [2.0 ** k for k in range(-8, -3)], "adam": [2.0 ** k for k in range(-10, -5)]}
cfgs = []
for w in [64, 256, 1024]:
    for resample in [False, True]:
        for opt in ["sgd", "adam"]:
            for param in ["sp", "mup"]:
                for lr in LRS[opt]:
                    cfgs.append({"task": "ode2", "width": w, "param": param, "optimizer": opt, "lr": lr, "steps": 5000, "seed": 0, "resample": resample})
t0 = time.time(); done = 0
for i, c in enumerate(cfgs):
    if time.time() - t0 > BUDGET_S:
        print(f"budget reached after {done} runs; stopping at {i}/{len(cfgs)}", flush=True); break
    p = os.path.join(OUT, cfg_name(c) + ".json"); ts = time.time()
    r = run_cfg(c, device, want_grad=False)
    json.dump(r, open(p, "w")); done += 1
    print(f"[{i+1}/{len(cfgs)}] {cfg_name(c)} loss={r['final_loss']:.3e} relL2={r['rel_l2']:.4f} div={r['diverged']} {time.time()-ts:.1f}s elapsed={time.time()-t0:.0f}s", flush=True)
print(f"finished {done} runs in {time.time()-t0:.0f}s", flush=True)
