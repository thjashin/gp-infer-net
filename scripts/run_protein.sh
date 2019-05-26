#!/bin/bash

trap "exit" INT
for i in $(seq 0 19); do
    echo $i
    CUDA_VISIBLE_DEVICES=$1 python3 regression.py \
    --split=$i \
    --dataset=protein \
    --method=gpnet \
    --batch_size=500 \
    --m=500 \
    --measure=noise \
    --learning_rate=0.003 \
    --n_hidden=1000 \
    --n_layer=1 \
    --n_inducing=100 \
    --n_iters=10000 \
    --pretrain=1000 \
    --net=rf \
    --residual=False \
    --fix_rf_ls=False \
    --hyper_rate=0 \
    --hyper_anneal=False \
    --lr_anneal=False \
    --beta0=0.1 \
    --gamma=0.1 \
    --test_freq=50
done
