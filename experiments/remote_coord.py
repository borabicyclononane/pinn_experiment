"""Remaining coordinate-check configs (full instrumentation) on a remote GPU; results into $OUTPUT_DIR/<group>/."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from mupinn.train import train
from mupinn.runner import cfg_name

BUDGET_S = float(os.environ.get("AGAVE_BUDGET_S", 1.75 * 3600))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/local")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device, torch.cuda.get_device_name(0) if device == "cuda" else "", torch.__version__, flush=True)
cfgs = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote_coord_configs.json")))
t0 = time.time(); done = 0
for i, c in enumerate(cfgs):
    if time.time() - t0 > BUDGET_S:
        print(f"budget reached after {done} runs; stopping at {i}/{len(cfgs)}", flush=True); break
    c = dict(c); group = c.pop("group")
    out = os.path.join(OUTPUT_DIR, group); os.makedirs(out, exist_ok=True)
    p = os.path.join(out, cfg_name(c) + ".json")
    ts = time.time()
    kw = dict(c); kw["task_name"] = kw.pop("task")
    r = train(device=device, **kw)
    r["stats"] = {str(k): v for k, v in r["stats"].items()}
    json.dump(r, open(p, "w")); done += 1
    print(f"[{i+1}/{len(cfgs)}] {group} {cfg_name(c)} loss={r['final_loss']:.3e} div={r['diverged']} {time.time()-ts:.1f}s elapsed={time.time()-t0:.0f}s", flush=True)
print(f"finished {done} runs in {time.time()-t0:.0f}s", flush=True)
