#!/bin/bash
if [[ $# -eq 0 ]]; then
    echo "$0 <dataset>";
    exit;
fi

DATASET="--dataset $1"

# CUDA_VISIBLE_DEVICES=0 python test.py --mw pre $DATASET && echo "test pre." &
# T1=$!
# CUDA_VISIBLE_DEVICES=0 python test.py --mw pref $DATASET && echo "test pref." &
# T2=$!
CUDA_VISIBLE_DEVICES=0 python test.py --mw train $DATASET && echo "test train." &
T3=$!
CUDA_VISIBLE_DEVICES=0 python test.py --mw final $DATASET && echo "test final." &
T4=$!

wait $T1 $T2 $T3 $T4
echo "finished."

