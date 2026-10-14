"""Coordinate-check configs: few steps, many widths, 3 seeds, measured at every step."""
import json, sys
small, big = [], []
LR = {"sgd": 2.0 ** -5, "adam": 2.0 ** -8}
for task in ["regress1d", "ode1", "ode2", "ode4", "heat2d"]:
    widths = [64, 128, 256, 512, 1024, 2048, 4096] + ([8192] if task not in ("ode4", "heat2d") else [])
    for param in ["sp", "mup"]:
        for opt in ["sgd", "adam"]:
            for w in widths:
                for seed in [0, 1, 2]:
                    c = {"task": task, "width": w, "param": param, "optimizer": opt, "lr": LR[opt], "steps": 8,
                         "seed": seed, "measure_steps": list(range(9))}
                    (big if w >= 1024 else small).append(c)
json.dump(small, open(sys.argv[1] + "_small.json", "w")); json.dump(big, open(sys.argv[1] + "_big.json", "w"))
print(len(small), len(big))
