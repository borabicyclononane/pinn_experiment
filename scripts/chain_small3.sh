#!/bin/zsh
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
cd /Users/aryan/Documents/Aryan/CS/pinn_experiment
until grep -q "^done in" results/sweepC_small.log 2>/dev/null; do sleep 10; done
for ph in A B C; do NTHREADS=6 $PY -m mupinn.runner results/sweep${ph}_1024.json sweep cpu > results/sweep${ph}_1024.log 2>&1; done
