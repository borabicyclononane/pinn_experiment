"""Coordinate-check figures: RMS vs width per node, lines per step; SP vs muP side by side.
Usage: python scripts/plot_coord.py <task> <opt> [group]
"""
import sys, math, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mupinn.analysis import aggregate_coord, fit_exponent

task, opt = sys.argv[1], sys.argv[2]
group = sys.argv[3] if len(sys.argv) > 3 else "coordcheck"
agg = aggregate_coord(group)
STEPS = [0, 1, 2, 4, 8]
cmap = plt.get_cmap("viridis")


def panel(ax, d, key, title, steps=STEPS):
    if key not in d:
        ax.set_visible(False); return
    ws = sorted(d[key])
    for i, t in enumerate(steps):
        vals = [d[key][w].get(t, (float("nan"),))[0] for w in ws]
        errs = [d[key][w].get(t, (0, 0))[1] for w in ws]
        if all(not np.isfinite(v) for v in vals):
            continue
        e = fit_exponent(ws, vals)
        ax.errorbar(ws, vals, yerr=errs, marker="o", ms=3, color=cmap(i / max(1, len(steps) - 1)), label=f"t={t} (n^{e:+.2f})")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_title(title, fontsize=9); ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=6, loc="best")


def make(quantity, nodes, fname, ncols=None):
    ncols = ncols or len(nodes)
    fig, axes = plt.subplots(2, ncols, figsize=(2.6 * ncols, 5.4), squeeze=False)
    for r, param in enumerate(["sp", "mup"]):
        d = agg.get((task, param, opt), {})
        for c, node in enumerate(nodes):
            panel(axes[r, c], d, f"{node}/{quantity}", f"{param.upper()} {node} {quantity}")
    fig.suptitle(f"{task} / {opt}: RMS of {quantity} vs width (lines = training step, legend = fitted exponent over n>=256)")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


os.makedirs("figs", exist_ok=True)
d_any = agg.get((task, "mup", opt), {})
nodes_h = [n for n in ["h1", "h2", "h3", "f"] if f"{n}/value" in d_any]
nodes_d = [n for n in ["dx_h1", "dx_h2", "dx_h3", "f_x", "dxx_h2", "dxx_h3", "f_xx", "f_xxx", "f_xxxx", "f_t"] if f"{n}/value" in d_any]
nodes_w = [n for n in ["W1", "W2", "W3", "Wout"] if f"{n}/value" in d_any]
for q in ["value", "delta", "grad"]:
    make(q, nodes_h, f"figs/coord_{task}_{opt}_act_{q}.png")
    if nodes_d:
        make(q, nodes_d, f"figs/coord_{task}_{opt}_deriv_{q}.png")
make("update", nodes_w, f"figs/coord_{task}_{opt}_w_update.png")
make("grad", nodes_w, f"figs/coord_{task}_{opt}_w_grad.png")
print("ok", task, opt, nodes_h, nodes_d)
