"""Finite-width effects under muP by derivative order:
 (a) relative deviation of RMS(delta) at width n from its value at the largest width, per node, step 4;
 (b) seed-to-seed relative std of the node values vs width (expected ~ n^-1/2)."""
import sys, os, math, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mupinn.analysis import aggregate_coord, fit_exponent
agg = aggregate_coord("coordcheck")
nodes = ["h3", "f", "f_x", "f_xx", "f_xxx", "f_xxxx", "f_t"]
print("(a) |RMS_delta(n)/RMS_delta(n_max) - 1| at t=4, muP, SGD  (how far the finite width is from the widest run)")
print("| task | n_max | " + " | ".join(nodes) + " |  at n=256 / n=1024")
for task in ["regress1d", "ode1", "ode2", "ode4", "heat2d"]:
    d = agg[(task, "mup", "sgd")]
    row = []
    for node in nodes:
        k = f"{node}/delta"
        if k not in d: row.append(""); continue
        ws = sorted(d[k]); nmax = ws[-1]
        ref = d[k][nmax][4][0]
        dev = lambda n: abs(d[k][n][4][0] / ref - 1) if n in d[k] else float("nan")
        row.append(f"{dev(256):.2f} / {dev(1024):.2f}")
    print(f"| {task} | {ws[-1]} | " + " | ".join(row) + " |")
print("\n(b) seed-to-seed relative std of RMS(value) at t=4, muP SGD: fitted exponent vs width (expect about -0.5) and value at n=1024")
print("| task | " + " | ".join(nodes) + " |")
for task in ["regress1d", "ode1", "ode2", "ode4", "heat2d"]:
    d = agg[(task, "mup", "sgd")]
    row = []
    for node in nodes:
        k = f"{node}/value"
        if k not in d: row.append(""); continue
        ws = sorted(d[k])
        rel = [d[k][w][4][1] * math.sqrt(d[k][w][4][2]) / d[k][w][4][0] for w in ws]   # sem*sqrt(n_seeds) = std
        e = fit_exponent(ws, rel, min_width=128)
        row.append(f"n^{e:+.2f}, {rel[ws.index(1024)]:.3f}")
    print(f"| {task} | " + " | ".join(row) + " |")
