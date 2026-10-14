"""Fitted width exponents (n >= 256) of RMS(delta) and RMS(value) per node at chosen steps, per task/param/opt.
Usage: python scripts/exponent_table.py [group] [steps...]   -> prints markdown tables and writes results/exponents_<group>.json"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mupinn.analysis import aggregate_coord, fit_exponent

group = sys.argv[1] if len(sys.argv) > 1 else "coordcheck"
steps = [int(s) for s in sys.argv[2:]] or [1, 4, 8]
agg = aggregate_coord(group)
NODES = ["h1", "h2", "h3", "f", "dx_h2", "dx_h3", "f_x", "dxx_h3", "f_xx", "f_xxx", "f_xxxx", "f_t", "W1", "W2", "W3", "Wout"]
out = {}
for key in sorted(agg):
    task, param, opt = key
    d = agg[key]
    for q in ["value", "delta", "update", "grad"]:
        for t in steps:
            row = {}
            for node in NODES:
                k = f"{node}/{q}"
                if k not in d:
                    continue
                ws = sorted(d[k]); vals = [d[k][w].get(t, (None,))[0] for w in ws]
                e = fit_exponent(ws, vals)
                # also the max width reached and the ratio of largest/smallest width values
                row[node] = e
            if row:
                out[f"{task}|{param}|{opt}|{q}|t={t}"] = row
json.dump(out, open(f"results/exponents_{group}.json", "w"), indent=1)
# compact print: delta at t=1 and t=8 per (task,opt): SP vs muP
tasks = sorted({k[0] for k in agg}); opts = sorted({k[2] for k in agg})
for q in ["delta", "value"]:
    for t in steps:
        print(f"\n### {q} exponents at t={t}")
        print("| task | opt | param | " + " | ".join(NODES) + " |")
        print("|" + "---|" * (len(NODES) + 3))
        for task in tasks:
            for opt in opts:
                for param in ["sp", "mup"]:
                    row = out.get(f"{task}|{param}|{opt}|{q}|t={t}")
                    if not row:
                        continue
                    cells = [("%+.2f" % row[n]) if (n in row and math.isfinite(row[n])) else "" for n in NODES]
                    print(f"| {task} | {opt} | {param} | " + " | ".join(cells) + " |")
