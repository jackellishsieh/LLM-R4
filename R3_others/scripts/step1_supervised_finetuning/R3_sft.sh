#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# SFT script
# zero_stage can be 2 or 3
# This is unnecessary for 1 device
ZERO_STAGE=2

# for math datasets, train_epochs=2
# for other datasets, train_epochs=5
num_train_epochs=10

# they use a model with 7B parameters and hidden size 4096 + learning rate 2e-5
# we use a model with 0.5B parameters and hidden size 896 + learning rate 1e-4
learning_rate=1e-6

model_name_or_path="Qwen/Qwen2-0.5B-Instruct"
data_path="../../data/gsm8k_cot/gsm8k_nl_train_example.json"
output_base="/home/ubuntu/LLM-R4/output_models/sft_gsm8k_cot/"
output_dir=${output_base}lr${learning_rate}_ep${num_train_epochs}/
data_output_path=${output_base}

eval_input_path="${HOME}/LLM-R4/R3_others/data/gsm8k_cot/gsm8k_test.json"
eval_output_path="${HOME}/LLM-R4/eval_results/sft_model_eval.json"
src_name="gsm8k"

# wandb information
wandb_log="True"
wandb_entity="cs224r_project_team"
wandb_project="R3_qwen"
wandb_run_name="sft_gsm8k_cot"

mkdir -p $output_dir

# for GSM8K-P-CoT, max_seq_len=1024
# for other math datasets, max_seq_len=512 
# for MNLI, SNLI, max_seq_len=512
# for raceHigh, raceMiddle, max_seq_len=1024
# for boardgame, max_seq_len=512
# and we keep total_batch_size=256
deepspeed \
    --master_port 39000 \
    --num_gpus 1 \
main.py \
    --model_name_or_path $model_name_or_path\
    --data_path $data_path \
    --data_split "10,0,0" \
    --data_output_path $data_output_path \
    --learning_rate $learning_rate \
    --zero_stage $ZERO_STAGE \
    --gradient_accumulation_steps 8  \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --per_device_eval_batch_size 4 \
    --max_seq_len 512 \
    --print_loss  \
    --num_train_epochs ${num_train_epochs} \
    --deepspeed \
    --output_dir ${output_dir} \
    --eval_input_path ${eval_input_path} \
    --eval_output_path ${eval_output_path} \
    --src_name $src_name \
    --max_gen_length 700 \
    --wandb_log "${wandb_log}" \
    --wandb_project "${wandb_project}" \
    --wandb_entity "${wandb_entity}" \
    --wandb_run_name "${wandb_run_name}" \
    --verbose \
    > /home/ubuntu/LLM-R4/log_dir/sft_gsm8k_cot.log 2>&1
