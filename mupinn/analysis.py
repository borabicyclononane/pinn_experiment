"""Aggregate coordinate-check results and fit width exponents."""
from __future__ import annotations
import collections, math
import numpy as np
from .runner import load_group


def aggregate_coord(group="coordcheck"):
    """Returns dict[(task,param,opt)] -> {node_quantity: {width: {step: (mean, sem, n)}}}"""
    rs = load_group(group)
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list))))
    for r in rs:
        key = (r["task"], r["param"], r["optimizer"])
        for t, st in r["stats"].items():
            for k, v in st.items():
                acc[key][k][r["width"]][int(t)].append(v)
    out = {}
    for key, d in acc.items():
        out[key] = {}
        for k, dw in d.items():
            out[key][k] = {}
            for w, dt in dw.items():
                out[key][k][w] = {}
                for t, vals in dt.items():
                    vals = [v for v in vals if v is not None and math.isfinite(v)]
                    if vals:
                        a = np.array(vals)
                        out[key][k][w][t] = (float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0, len(a))
    return out


def fit_exponent(widths, vals, min_width=256):
    pts = [(w, v) for w, v in zip(widths, vals) if w >= min_width and v is not None and v > 0 and math.isfinite(v)]
    if len(pts) < 2:
        return float("nan")
    x = np.log2([p[0] for p in pts]); y = np.log2([p[1] for p in pts])
    return float(np.polyfit(x, y, 1)[0])


def exponent_table(agg, key, step, quantities=("value", "delta", "grad", "update"), min_width=256):
    """dict node -> {quantity: exponent}"""
    d = agg[key]
    nodes = sorted({k.split("/")[0] for k in d})
    tab = {}
    for node in nodes:
        tab[node] = {}
        for q in quantities:
            k = f"{node}/{q}"
            if k not in d:
                continue
            ws = sorted(d[k])
            vals = [d[k][w].get(step, (None,))[0] for w in ws]
            tab[node][q] = fit_exponent(ws, vals, min_width)
    return tab
