"""LR-transfer sweep configs. Usage: python scripts/gen_sweep.py <phase> <outprefix>
phase A: regress1d, ode2 ; phase B: ode1, heat2d ; phase C: ode4 (narrower)
Writes <outprefix>_small.json (width < 1024, for cpu) and <outprefix>_big.json (width >= 1024, for mps)."""
import json, sys
phase, out = sys.argv[1], sys.argv[2]
TASKS = {"A": ["regress1d", "ode2"], "B": ["ode1", "heat2d"], "C": ["ode4"]}[phase]
WIDTHS = [64, 128, 256, 512, 1024, 2048] if phase != "C" else [64, 128, 256, 512, 1024]
LRS = {"sgd": [2.0 ** k for k in range(-11, -1)], "adam": [2.0 ** k for k in range(-12, -3)]}
STEPS = 1000
small, big = [], []
for task in TASKS:
    for opt in ["sgd", "adam"]:
        for param in ["sp", "mup"]:
            for w in WIDTHS:
                for lr in LRS[opt]:
                    c = {"task": task, "width": w, "param": param, "optimizer": opt, "lr": lr, "steps": STEPS, "seed": 0,
                         "want_grad": False}
                    (big if w >= 1024 else small).append(c)
# run big widths last-LR-first so divergent ones (cheap) come first? keep order simple: sort by width then lr
big.sort(key=lambda c: (c["width"], c["task"], c["optimizer"], c["param"], c["lr"]))
json.dump(small, open(out + "_small.json", "w")); json.dump(big, open(out + "_big.json", "w"))
print(len(small), len(big))
