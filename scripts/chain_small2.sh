#!/bin/zsh
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
cd /Users/aryan/Documents/Aryan/CS/pinn_experiment
until grep -q "^done in" results/coord_small.log; do sleep 5; done
NTHREADS=6 $PY -m mupinn.runner results/coordlow_small.json coordcheck_lowlr cpu > results/coordlow_small.log 2>&1
for ph in A B C; do NTHREADS=6 $PY -m mupinn.runner results/sweep${ph}_small.json sweep cpu > results/sweep${ph}_small.log 2>&1; done
