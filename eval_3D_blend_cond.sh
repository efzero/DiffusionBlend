#!/bin/bash

Nview=8
T_sampling=50
eta=0.85

python main.py \
    --type '3dblendcond' \
    --config AAPM_256_lsun.yaml \
    --dataset_path "/nfs/turbo/coe-liyues/bowenbw/DDS/indist_samples/CT/L067" \
    --ckpt_load_name "/nfs/turbo/coe-liyues/bowenbw/DDS/checkpoints/AAPM256_1M.pth" \
    --Nview $Nview \
    --eta $eta \
    --deg "SV-CT" \
    --sigma_y 0.00 \
    --T_sampling 100 \
    --T_sampling $T_sampling \
    -i ./results
