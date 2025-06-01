#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# SFT script
# zero_stage can be 2 or 3
# This is unnecessary for 1 device

model_name_or_path="Qwen/Qwen2-0.5B-Instruct"
output_dir="${HOME}/LLM-R4/outputs"
eval_file="${HOME}/LLM-R4/R3_others/data/gsm8k_cot/gsm8k_test.json"

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
standalone_eval.py \
    --model_name_or_path $model_name_or_path\
    --eval_file $eval_file \
    --output_dir $output_dir \
    --per_device_eval_batch_size 4 \
    --max_input_length 700 \
    --max_gen_length 700 \
    --engine "nl" \
    --deepspeed \
    # > "${HOME}/LLM-R4/log_dir/sft_gsm8k_cot_eval.log" 2>&1
