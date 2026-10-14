#!/bin/zsh
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
cd /Users/aryan/Documents/Aryan/CS/pinn_experiment
until grep -q "^done in" results/coord_big.log; do sleep 5; done
NTHREADS=2 $PY -m mupinn.runner results/coordlow_big.json coordcheck_lowlr mps > results/coordlow_big.log 2>&1
for ph in A B C; do NTHREADS=2 $PY -m mupinn.runner results/sweep${ph}_big.json sweep mps > results/sweep${ph}_big.log 2>&1; done
