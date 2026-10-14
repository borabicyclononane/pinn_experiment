"""LR-transfer plots: final training loss vs log2(lr), one line per width, SP vs muP panels.
Usage: python scripts/plot_sweep.py <group> [metric]   metric in {final_loss, final_eval, best_eval, rel_l2}"""
import sys, os, math, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mupinn.runner import load_group

group = sys.argv[1]
metric = sys.argv[2] if len(sys.argv) > 2 else "final_loss"
rs = load_group(group)
tab = collections.defaultdict(lambda: collections.defaultdict(dict))  # (task,opt,param) -> width -> lr -> value
for r in rs:
    v = r[metric]
    if r["diverged"] or v is None or not math.isfinite(v):
        v = float("nan")
    tab[(r["task"], r["optimizer"], r["param"])][r["width"]][r["lr"]] = v
tasks = sorted({k[0] for k in tab}); opts = sorted({k[1] for k in tab})
cmap = plt.get_cmap("viridis")
os.makedirs("figs", exist_ok=True)
summary = {}
for task in tasks:
    for opt in opts:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        for ax, param in zip(axes, ["sp", "mup"]):
            d = tab.get((task, opt, param), {})
            ws = sorted(d)
            for i, w in enumerate(ws):
                lrs = sorted(d[w]); vals = [d[w][l] for l in lrs]
                ax.plot([math.log2(l) for l in lrs], vals, marker="o", ms=3, color=cmap(i / max(1, len(ws) - 1)), label=f"n={w}")
                finite = [(v, l) for v, l in zip(vals, lrs) if np.isfinite(v)]
                if finite:
                    best = min(finite)
                    summary[(task, opt, param, w)] = (math.log2(best[1]), best[0])
            ax.set_yscale("log"); ax.set_xlabel("log2(base learning rate)"); ax.set_title(f"{task} / {opt} / {param.upper()}")
            ax.grid(alpha=.3); ax.legend(fontsize=7)
        axes[0].set_ylabel(metric)
        fig.tight_layout(); fig.savefig(f"figs/sweep_{group}_{task}_{opt}_{metric}.png", dpi=130); plt.close(fig)
print(f"best log2(lr) and {metric} per (task, opt, param, width):")
for k in sorted(summary):
    print(f"  {k[0]:10s} {k[1]:4s} {k[2]:3s} n={k[3]:5d}  best lr=2^{summary[k][0]:+.0f}  {metric}={summary[k][1]:.3e}")
