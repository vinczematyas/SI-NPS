#!/bin/bash

python base/main.py \
    --tqdm=True \
    --log=True \
    --dataset=$'hotel' \
    --lr=1.0 \
    --contxt_k=1 \
    --num_rules=4 \
    --step_size=45 \
    --seed=1

python base/main.py \
    --tqdm=True \
    --log=True \
    --dataset=$'eth' \
    --lr=1.0 \
    --contxt_k=1 \
    --num_rules=4 \
    --step_size=10 \
    --seed=0

python base/main.py \
    --tqdm=True \
    --log=True \
    --dataset=$'univ' \
    --lr=1.0 \
    --contxt_k=1 \
    --num_rules=4 \
    --step_size=45 \
    --seed=5

python base/main.py \
    --tqdm=True \
    --log=True \
    --dataset=$'zara1' \
    --lr=1.0 \
    --contxt_k=1 \
    --num_rules=4 \
    --step_size=45 \
    --seed=7

python base/main.py \
    --tqdm=True \
    --log=True \
    --dataset=$'zara2' \
    --lr=1.0 \
    --contxt_k=1 \
    --num_rules=4 \
    --step_size=45 \
    --seed=0

python base/main.py \
    --tqdm=True \
    --log=True \
    --dataset=$'sdd' \
    --lr=1.0 \
    --contxt_k=1 \
    --num_rules=4 \
    --step_size=45 \
    --seed=9
