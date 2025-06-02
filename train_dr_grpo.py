"""
This file contains the training loop function for the Dr. GRPO algorithm using the Qwen-0.5B model.
"""
import os
from dr_grpo_trainer import DrGRPOTrainer
from trl import GRPOConfig
import rl_util
import json
from datasets import Dataset, concatenate_datasets
import wandb
import torch

TRAINING_FILES = [
    "R3_math/data/gsm8k_original_train.json",
]

def train_dr_grpo(training_files=TRAINING_FILES):
    """Train the Dr. GRPO algorithm on the Qwen-0.5B model."""

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    wandb.init(
        project="dr-grpo-qwen",
        name="dr-grpo-qwen-gsm8k",
        config={
            "model": "dillonkn/qwen2.5-0.5b-reasoning-sft",
            "dataset": "gsm8k",
            "dataset_size": 2000,
            "algorithm": "Dr. GRPO",
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "num_generations": 16,
            "temperature": 0.8,
            "top_p": 0.9,
            "reward_components": ["format", "correctness"],
            "format_reward": 0.2,
            "correctness_reward": 1.0,
        }
    )

    # Set W&B environment variables
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # Log model checkpoints

    rl_util.cot_info = rl_util.prepare_cot_info("gsm8k")
    
    # define reward function
    def reward_function(completions, **kwargs):
        """Multi-component reward function: format (0.2) + correctness (1.0)"""
        total_rewards = []
        # for logging
        format_count = 0
        correctness_count = 0

        answers = kwargs['answer_value']
        
        for completion, correct_answer in zip(completions, answers):
            reward = 0.0

            # check formatting
            if rl_util.check_proper_format(completion):
                reward += 0.2
                format_count += 1
            
            # check correctness
            try:
                predicted_answer = rl_util.cot_info["cot_to_answer"](completion) # extract answer (float)
                predicted_value = rl_util.cot_info["answer_to_value"](predicted_answer)

                correct_value = rl_util.cot_info["answer_to_value"](correct_answer)
                
                if rl_util.cot_info["compare_values"](predicted_value, correct_value):
                    reward += 1.0
                    correctness_count += 1
            except:
                pass
            
            total_rewards.append(reward)
        
        # log rewards
        wandb.log({
            "reward/mean_reward": sum(total_rewards) / len(total_rewards),
            "reward/format_compliance_rate": format_count / len(total_rewards),
            "reward/correctness_rate": correctness_count / len(total_rewards),
            "reward/max_reward": max(total_rewards),
            "reward/min_reward": min(total_rewards),
        })

        return total_rewards
    
    # load training datasets
    datasets = Dataset.from_list([])  # initialize empty huggingface dataset
    for file in training_files:
        # load the dataset, assuming it has a 'question', and 'answer_value' field
        with open(file, 'r') as f:
            raw_data = json.load(f)

        for item in raw_data:
            # replace "question" with "prompt" in each item
            item["prompt"] = rl_util.r1_zero_question_to_prompt(item.pop("question"))
            item.pop("answer_cot", None)  # remove answer_cot if it exists
        
        # convert to huggingface dataset
        datasets = concatenate_datasets([datasets, Dataset.from_list(raw_data)])
    datasets = datasets.select(range(0, 2000))  # limit to first 2000 samples for faster training
    
    # configure training
    training_args = GRPOConfig(
        output_dir="dr_grpo_qwen_0.5b",
        report_to="wandb",
        run_name="dr_grpo_qwen_gsm8k",
        logging_steps=10,
        use_vllm=True,
        vllm_mode="colocate",  # maybe "server", depending on setup
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2, 
        num_generations=16,  # n responses per prompt
        bf16=True,
        gradient_checkpointing=True,
        scale_rewards=False,  # disable std scaling for dr. grpo
        temperature=0.8, # high temp for GRPO
        top_p=0.9,
        max_completion_length=256,  # max length of generated response
        save_steps=1000,
        save_total_limit=1,  # keep only the last checkpoint
        eval_steps=100,  
        # vllm_gpu_memory_utilization=0.4,
    )
    
    # init trainer
    trainer = DrGRPOTrainer(
        model="dillonkn/qwen2.5-0.5b-reasoning-sft",
        args=training_args,
        reward_funcs=reward_function,
        train_dataset=datasets,
    )
    
    trainer.train()

    wandb.finish()

if __name__ == "__main__":
    train_dr_grpo()
