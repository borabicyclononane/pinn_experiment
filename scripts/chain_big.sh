#!/bin/zsh
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
cd /Users/aryan/Documents/Aryan/CS/pinn_experiment
until grep -q "^done in" results/coord_big.log; do sleep 5; done
for ph in A B C; do NTHREADS=2 $PY -m mupinn.runner results/sweep${ph}_big.json sweep mps > results/sweep${ph}_big.log 2>&1; done
