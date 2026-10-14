"""Per-neuron (first-layer) gradient magnitude vs |w_i| at init, for each task. Width 1024, muP, 3 hidden layers.
Also records the kurtosis / tail index of per-neuron gradient norms."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from mupinn.model import MLP
from mupinn.tasks import TASKS
torch.set_num_threads(2)
out = {}
for task_name in ["regress1d", "ode1", "ode2", "ode4"]:
    task = TASKS[task_name]()
    pts = task.points(512, torch.Generator().manual_seed(1234), "cpu")
    rows = []
    for seed in range(3):
        m = MLP(1, 1024, 3, param="mup", seed=seed)
        total, terms, _ = task.loss(m, pts)
        g = torch.autograd.grad(terms["res"], [m.W[0], m.b[0]])
        gn = (g[0][:, 0] ** 2 + g[1] ** 2).sqrt()          # per-neuron gradient norm (w_i, b_i)
        w = m.W[0][:, 0].detach()
        rows.append(np.stack([w.numpy(), m.b[0].detach().numpy(), gn.numpy()], 1))
    arr = np.concatenate(rows)
    gn = arr[:, 2]; absw = np.abs(arr[:, 0])
    # binned medians
    bins = np.quantile(absw, np.linspace(0, 1, 9))
    med = [float(np.median(gn[(absw >= bins[i]) & (absw < bins[i + 1])])) for i in range(8)]
    # local power-law exponent of gn vs |w| (fit on log-log over upper half)
    sel = absw > np.median(absw)
    slope = float(np.polyfit(np.log(absw[sel]), np.log(gn[sel] + 1e-30), 1)[0])
    share_top5 = float(np.sort(gn)[-len(gn) // 20:].sum() / gn.sum())
    out[task_name] = {"bin_edges": [float(b) for b in bins], "median_gradnorm": med, "slope_vs_absw": slope,
                      "share_top5pct": share_top5, "gn_mean": float(gn.mean()), "gn_max_over_mean": float(gn.max() / gn.mean()),
                      "w": arr[:, 0].tolist(), "gn": gn.tolist()}
    print(f"{task_name:10s} slope(|grad| vs |w|, upper half)={slope:+.2f}  top-5%-neurons share of grad mass={share_top5:.2f}  max/mean={out[task_name]['gn_max_over_mean']:.1f}")
    print("   median |grad| per |w| octile:", ["%.2e" % v for v in med])
json.dump(out, open("results/neuron_level.json", "w"))
