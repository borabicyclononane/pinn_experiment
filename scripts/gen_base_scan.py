"""LR scan at base width 64 to pick coordinate-check learning rates."""
import json, itertools, sys
cfgs = []
for task in ["regress1d", "ode1", "ode2", "ode4", "heat2d"]:
    for opt in ["sgd", "adam"]:
        lrs = [2.0 ** k for k in range(-14, 1)] if opt == "sgd" else [2.0 ** k for k in range(-14, -3)]
        for lr in lrs:
            cfgs.append({"task": task, "width": 64, "param": "mup", "optimizer": opt, "lr": lr, "steps": 300, "seed": 0})
json.dump(cfgs, open(sys.argv[1], "w"))
print(len(cfgs))
