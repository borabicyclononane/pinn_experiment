"""Smoke test for the remote GPU: time a few short runs and write them to OUTPUT_DIR."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from mupinn.train import train
OUT = os.environ.get("OUTPUT_DIR", "outputs/local"); os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device, torch.cuda.get_device_name(0) if device == "cuda" else "", torch.__version__, flush=True)
res = {}
for task, w in [("ode2", 1024), ("ode2", 2048), ("ode2", 4096), ("ode4", 2048), ("heat2d", 4096), ("regress1d", 4096)]:
    t0 = time.time()
    r = train(task, w, 3, "mup", 2 ** -5, 50, optimizer="sgd", device=device, want_grad=False, measure_steps=[0, 50])
    dt = time.time() - t0
    res[f"{task}_{w}"] = {"s_per_step": dt / 50, "final_loss": r["final_loss"]}
    print(f"{task} w={w}: {dt/50*1000:.1f} ms/step (50 steps incl. 2 measures), loss {r['hist']['loss'][0]:.3f}->{r['final_loss']:.3f}", flush=True)
json.dump(res, open(os.path.join(OUT, "smoke.json"), "w"), indent=1)
