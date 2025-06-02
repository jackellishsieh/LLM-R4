#!/bin/bash

model_name_or_path="Qwen/Qwen2-0.5B-Instruct"
src_name="gsm8k"
eval_path="${HOME}/LLM-R4/R3_others/data/gsm8k_cot/gsm8k_test.json"
output_path="${HOME}/LLM-R4/eval_results/base_model_eval.json"


mkdir -p $output_dir

# for GSM8K-P-CoT, max_seq_len=1024
# for other math datasets, max_seq_len=512 
# for MNLI, SNLI, max_seq_len=512
# for raceHigh, raceMiddle, max_seq_len=1024
# for boardgame, max_seq_len=512
# and we keep total_batch_size=256
python standalone_eval.py \
    --model_name_or_path $model_name_or_path\
    --src_name $src_name \
    --eval_path $eval_path \
    --output_path $output_path \
    --max_gen_length 700 \
    --add_eot_token \
    --verbose