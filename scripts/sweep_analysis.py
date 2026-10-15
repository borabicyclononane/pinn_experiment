"""LR-transfer analysis over one or more result groups, seed-averaged.
Usage: python scripts/sweep_analysis.py <group>[,<group>...] [metric]
Outputs:
  figs/sweep_<task>_<opt>.png      final loss (geometric mean over seeds) vs log2 lr, line per width, SP | muP
  figs/transfer_summary.png        best log2 lr vs width per task/opt, SP vs muP
  results/sweep_summary.json       per (task,opt,param,width): best lr, max stable lr, loss at best, feature movement
"""
import sys, os, math, json, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mupinn.runner import load_group

groups = (sys.argv[1] if len(sys.argv) > 1 else "sweep").split(",")
metric = sys.argv[2] if len(sys.argv) > 2 else "final_loss"
rs = [r for g in groups for r in load_group(g)]
# (task,opt,param,width,lr) -> list over seeds of (metric, diverged, feature movement)
tab = collections.defaultdict(list)
for r in rs:
    v = r[metric]
    div = r["diverged"] or v is None or not math.isfinite(v) or v > 10
    last = str(max(int(k) for k in r["stats"]))
    st = r["stats"][last]
    fm = {k: st.get(f"{k}/delta", float("nan")) / max(st.get(f"{k}/value", float("nan")), 1e-30) for k in ["h1", "h2", "h3"]}
    tab[(r["task"], r["optimizer"], r["param"], r["width"], r["lr"])].append((float("nan") if div else v, div, fm, r["seed"]))


def gmean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.exp(np.mean(np.log(np.maximum(xs, 1e-30))))) if xs else float("nan")


curves = collections.defaultdict(dict)  # (task,opt,param,width) -> lr -> (gmean loss, n_seeds, any_div)
for (task, opt, param, w, lr), vals in tab.items():
    curves[(task, opt, param, w)][lr] = (gmean([v[0] for v in vals]), len(vals), any(v[1] for v in vals), np.nanmean([v[2]["h3"] for v in vals]))
summary = {}
for key, d in curves.items():
    lrs = sorted(d)
    finite = [(d[l][0], l) for l in lrs if np.isfinite(d[l][0])]
    if not finite:
        continue
    best_loss, best_lr = min(finite)
    stable = [l for l in lrs if not d[l][2] and np.isfinite(d[l][0])]
    summary["|".join(map(str, key))] = {"best_log2lr": math.log2(best_lr), "best_loss": best_loss,
                                        "max_stable_log2lr": math.log2(max(stable)) if stable else float("nan"),
                                        "feature_move_h3_at_best": d[best_lr][3], "n_seeds": d[best_lr][1]}
json.dump(summary, open("results/sweep_summary.json", "w"), indent=1)

os.makedirs("figs", exist_ok=True)
tasks = sorted({k[0] for k in curves}); opts = sorted({k[1] for k in curves})
cmap = plt.get_cmap("viridis")
for task in tasks:
    for opt in opts:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        for ax, param in zip(axes, ["sp", "mup"]):
            ws = sorted({k[3] for k in curves if k[:3] == (task, opt, param)})
            for i, w in enumerate(ws):
                d = curves[(task, opt, param, w)]; lrs = sorted(d)
                ax.plot([math.log2(l) for l in lrs], [d[l][0] for l in lrs], marker="o", ms=3, color=cmap(i / max(1, len(ws) - 1)), label=f"n={w}")
            ax.set_yscale("log"); ax.set_xlabel("log2(base learning rate)"); ax.set_title(f"{task} / {opt} / {'SP' if param == 'sp' else 'μP'}")
            ax.grid(alpha=.3); ax.legend(fontsize=7)
        axes[0].set_ylabel(f"{metric} after 1000 steps (geo-mean over seeds)")
        fig.tight_layout(); fig.savefig(f"figs/sweep_{task}_{opt}.png", dpi=130); plt.close(fig)

# transfer summary: best log2 lr vs width
fig, axes = plt.subplots(1, len(opts), figsize=(5.2 * len(opts), 3.8), squeeze=False)
markers = {"regress1d": "o", "ode1": "s", "ode2": "^", "ode4": "D", "heat2d": "v"}
for ax, opt in zip(axes[0], opts):
    for task in tasks:
        for param, col in [("sp", "#eb6834"), ("mup", "#2a78d6")]:
            ws = sorted({k[3] for k in curves if k[:3] == (task, opt, param)})
            ys = [summary.get(f"{task}|{opt}|{param}|{w}", {}).get("best_log2lr", float("nan")) for w in ws]
            ax.plot(ws, ys, marker=markers.get(task, "o"), ms=5, color=col, alpha=.85, label=f"{task} {'SP' if param == 'sp' else 'μP'}")
    ax.set_xscale("log", base=2); ax.set_xlabel("width n"); ax.set_ylabel("best log2(base lr)"); ax.set_title(f"{opt}: best learning rate vs width")
    ax.grid(alpha=.3); ax.legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig("figs/transfer_summary.png", dpi=130); plt.close(fig)

print(f"{'task':10s} {'opt':4s} {'param':4s} " + " ".join(f"{w:>6d}" for w in [64, 128, 256, 512, 1024, 2048, 4096]) + "   (best log2 lr / max stable log2 lr)")
for task in tasks:
    for opt in opts:
        for param in ["sp", "mup"]:
            cells = []
            for w in [64, 128, 256, 512, 1024, 2048, 4096]:
                s = summary.get(f"{task}|{opt}|{param}|{w}")
                cells.append(f"{s['best_log2lr']:+.0f}/{s['max_stable_log2lr']:+.0f}" if s else "   .  ")
            print(f"{task:10s} {opt:4s} {param:4s} " + " ".join(f"{c:>6s}" for c in cells))
print("\nfeature movement |delta h3|/|h3| at the best lr after 1000 steps:")
for task in tasks:
    for opt in opts:
        for param in ["sp", "mup"]:
            cells = []
            for w in [64, 256, 1024, 4096]:
                s = summary.get(f"{task}|{opt}|{param}|{w}")
                cells.append(f"{s['feature_move_h3_at_best']:.2f}" if s else ".")
            print(f"  {task:10s} {opt:4s} {param:4s} n=64,256,1024,4096: " + ", ".join(cells))
