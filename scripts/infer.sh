#!/usr/bin/env bash

input_root="/workspace/AgiBotWM/WorldModel/test/info_dataset/"
save_root="/workspace/AgiBotWM/teamace_sub4/"
ckp_path="/workspace/AgiBotWM/log/agibotworld_v3/checkpoints/epoch=1-step=7000.ckpt"
config_path="/workspace/AgiBotWM/configs/agibotworld/train_config_challenge_wm.yaml"
n_pred=3

python evac/main/infer_all.py \
  -i "$input_root" \
  -s "$save_root" \
  --ckp_path "$ckp_path" \
  --config_path "$config_path" \
  --n_pred "$n_pred"
