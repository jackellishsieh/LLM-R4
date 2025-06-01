#!/bin/bash

model_name_or_path="Qwen/Qwen2-0.5B-Instruct"
output_dir="${HOME}/LLM-R4/outputs"
src_name="gsm8k"
eval_file="${HOME}/LLM-R4/R3_others/data/gsm8k_cot/gsm8k_test.json"

mkdir -p $output_dir

# for GSM8K-P-CoT, max_seq_len=1024
# for other math datasets, max_seq_len=512 
# for MNLI, SNLI, max_seq_len=512
# for raceHigh, raceMiddle, max_seq_len=1024
# for boardgame, max_seq_len=512
# and we keep total_batch_size=256
python standalone_eval.py \
    --model_name_or_path $model_name_or_path\
    --eval_file $eval_file \
    --src_name $src_name \
    --output_dir $output_dir \
    --max_gen_length 700 \
    --add_eot_token \
    --verbose