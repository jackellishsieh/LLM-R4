"""
This file contains the training loop function for the Dr. GRPO algorithm using the Qwen-0.5B model.

To run without overrides, simply execute the script: python train_dr_grpo.py --config configs/default_grpo.yaml

To run the script in commmand line with overrides, you can do something like this:
python train_dr_grpo.py \
    --config configs/default_grpo.yaml \
    --experiment_name "grpo-experiment-v2" \
    --dataset_size 5000 \
    --batch_size 16 \
    --temperature 0.7

To create experiment-specific configurations, you can make a copy under a new name using: 
    cp configs/default_grpo.yaml configs/{new_exp_name}.yaml
Then you can edit the new config file directly before running
python train_dr_grpo.py --config configs/{new_exp_name}.yaml
"""

import os
import wandb
import torch
import json
import yaml
from datasets import Dataset, concatenate_datasets

from trl import GRPOConfig
from dr_grpo_trainer import DrGRPOTrainer
import rl_util
from config_manager import load_config, parse_override_args, validate_config

TRAINING_FILES = [
    "R3_math/data/gsm8k_original_train.json",
]

def setup_environment(config):
    """Setup distributed training environment"""
    # currently not using distributed training, but if compute/GPU resources are available, distributed training can be enabled here
    dist_config = config["distributed"]
    os.environ["RANK"] = str(dist_config["rank"])
    os.environ["WORLD_SIZE"] = str(dist_config["world_size"])
    os.environ["LOCAL_RANK"] = str(dist_config["local_rank"])
    os.environ["MASTER_ADDR"] = dist_config["master_addr"]
    os.environ["MASTER_PORT"] = str(dist_config["master_port"])


def create_reward_function(config):
    """Create reward function based on config"""
    reward_config = config["reward"]
    format_reward = reward_config["format_reward"]
    correctness_reward = reward_config["correctness_reward"]
    
    def reward_function(completions, **kwargs):
        """Multi-component reward function"""
        total_rewards = []
        format_count = 0
        correctness_count = 0
        answers = kwargs['answer_value']
        
        for completion, correct_answer in zip(completions, answers):
            reward = 0.0

            # formatted reward
            if rl_util.check_proper_format(completion):
                reward += format_reward
                format_count += 1
            
            # correctness reward
            try:
                predicted_answer = rl_util.cot_info["cot_to_answer"](completion)
                predicted_value = rl_util.cot_info["answer_to_value"](predicted_answer)
                correct_value = rl_util.cot_info["answer_to_value"](correct_answer)
                
                if rl_util.cot_info["compare_values"](predicted_value, correct_value):
                    reward += correctness_reward
                    correctness_count += 1
            except:
                pass
            
            total_rewards.append(reward)
        
        # wandb logging
        if config["wandb"]["enabled"]:
            wandb.log({
                "reward/mean_reward": sum(total_rewards) / len(total_rewards),
                "reward/format_compliance_rate": format_count / len(total_rewards),
                "reward/correctness_rate": correctness_count / len(total_rewards),
                "reward/max_reward": max(total_rewards),
                "reward/min_reward": min(total_rewards),
            })

        return total_rewards
    
    return reward_function


def load_datasets(config):
    """Load and prepare datasets based on config"""
    data_config = config["data"]
    datasets = Dataset.from_list([])
    
    for file in data_config["training_files"]:
        with open(file, 'r') as f:
            raw_data = json.load(f)

        for item in raw_data:
            item["prompt"] = rl_util.r1_zero_question_to_prompt(item.pop("question"))
            item.pop("answer_cot", None)
        
        datasets = concatenate_datasets([datasets, Dataset.from_list(raw_data)])
    
    # apply dataset size limit if specified
    if data_config["dataset_size"] is not None:
        datasets = datasets.select(range(0, data_config["dataset_size"]))
    
    return datasets

def create_training_args(config):
    """Create GRPOConfig from configuration"""
    exp_config = config["experiment"]
    train_config = config["training"]
    
    return GRPOConfig(
        output_dir=exp_config["output_dir"],
        report_to="wandb" if config["wandb"]["enabled"] else None,
        run_name=exp_config["name"],
        logging_steps=train_config["logging_steps"],
        use_vllm=train_config["use_vllm"],
        vllm_mode=train_config["vllm_mode"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        num_generations=train_config["num_generations"],
        bf16=train_config["bf16"],
        gradient_checkpointing=train_config["gradient_checkpointing"],
        scale_rewards=train_config["scale_rewards"],
        temperature=train_config["temperature"],
        top_p=train_config["top_p"],
        max_completion_length=train_config["max_completion_length"],
        save_steps=train_config["save_steps"],
        save_total_limit=train_config["save_total_limit"],
        eval_steps=train_config["eval_steps"],
        # vllm_gpu_memory_utilization=train_config.get("vllm_gpu_memory_utilization"),
    )


def train_dr_grpo(config):
    """Train the Dr. GRPO algorithm on the Qwen-0.5B model."""

    # load configuration
    setup_environment(config)

    # init wandb
    if config["wandb"]["enabled"]:
        os.environ["WANDB_LOG_MODEL"] = config["wandb"]["log_model"]
        
        wandb.init(
            project=config["experiment"]["project"],
            name=config["experiment"]["name"],
            config=config  # log configs to wandb
        )

    # wandb.init(
    #     project="dr-grpo-qwen",
    #     name="dr-grpo-qwen-gsm8k",
    #     config={
    #         "model": "dillonkn/qwen2.5-0.5b-reasoning-sft",
    #         "dataset": "gsm8k",
    #         "dataset_size": 2000,
    #         "algorithm": "Dr. GRPO",
    #         "batch_size": 8,
    #         "gradient_accumulation_steps": 2,
    #         "num_generations": 16,
    #         "temperature": 0.8,
    #         "top_p": 0.9,
    #         "reward_components": ["format", "correctness"],
    #         "format_reward": 0.2,
    #         "correctness_reward": 1.0,
    #     }
    # )

    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log model checkpoints

    # setup cot info
    rl_util.cot_info = rl_util.prepare_cot_info(config["data"]["dataset_name"])

    # setup components from configs
    reward_function = create_reward_function(config)
    datasets = load_datasets(config)
    training_args = create_training_args(config)
    
    # define reward function
    # def reward_function(completions, **kwargs):
    #     """Multi-component reward function: format (0.2) + correctness (1.0)"""
    #     total_rewards = []
    #     # for logging
    #     format_count = 0
    #     correctness_count = 0

    #     answers = kwargs['answer_value']
        
    #     for completion, correct_answer in zip(completions, answers):
    #         reward = 0.0

    #         # check formatting
    #         if rl_util.check_proper_format(completion):
    #             reward += 0.2
    #             format_count += 1
            
    #         # check correctness
    #         try:
    #             predicted_answer = rl_util.cot_info["cot_to_answer"](completion) # extract answer (float)
    #             predicted_value = rl_util.cot_info["answer_to_value"](predicted_answer)

    #             correct_value = rl_util.cot_info["answer_to_value"](correct_answer)
                
    #             if rl_util.cot_info["compare_values"](predicted_value, correct_value):
    #                 reward += 1.0
    #                 correctness_count += 1
    #         except:
    #             pass
            
    #         total_rewards.append(reward)
        
    #     # log rewards
    #     wandb.log({
    #         "reward/mean_reward": sum(total_rewards) / len(total_rewards),
    #         "reward/format_compliance_rate": format_count / len(total_rewards),
    #         "reward/correctness_rate": correctness_count / len(total_rewards),
    #         "reward/max_reward": max(total_rewards),
    #         "reward/min_reward": min(total_rewards),
    #     })

    #     return total_rewards
    
    # # load training datasets
    # datasets = Dataset.from_list([])  # initialize empty huggingface dataset
    # for file in training_files:
    #     # load the dataset, assuming it has a 'question', and 'answer_value' field
    #     with open(file, 'r') as f:
    #         raw_data = json.load(f)

    #     for item in raw_data:
    #         # replace "question" with "prompt" in each item
    #         item["prompt"] = rl_util.r1_zero_question_to_prompt(item.pop("question"))
    #         item.pop("answer_cot", None)  # remove answer_cot if it exists
        
    #     # convert to huggingface dataset
    #     datasets = concatenate_datasets([datasets, Dataset.from_list(raw_data)])
    # datasets = datasets.select(range(0, 2000))  # limit to first 2000 samples for faster training
    
    # # configure training
    # training_args = GRPOConfig(
    #     output_dir="dr_grpo_qwen_0.5b",
    #     report_to="wandb",
    #     run_name="dr_grpo_qwen_gsm8k",
    #     logging_steps=10,
    #     use_vllm=True,
    #     vllm_mode="colocate",  # "server" if local storage issues, but "server" slower
    #     per_device_train_batch_size=8,
    #     gradient_accumulation_steps=2, 
    #     num_generations=16,  # n responses per prompt
    #     bf16=True,
    #     gradient_checkpointing=True,
    #     scale_rewards=False,  # disable std scaling for dr. grpo
    #     temperature=0.8, # high temp for GRPO
    #     top_p=0.9,
    #     max_completion_length=256,  # max length of generated response
    #     save_steps=1000,
    #     save_total_limit=1,  # keep only the last checkpoint
    #     eval_steps=100,  
    #     # vllm_gpu_memory_utilization=0.4,
    # )
    
    # init trainer
    # trainer = DrGRPOTrainer(
    #     model="dillonkn/qwen2.5-0.5b-reasoning-sft",
    #     args=training_args,
    #     reward_funcs=reward_function,
    #     train_dataset=datasets,
    # )
    trainer = DrGRPOTrainer(
        model=config["model"]["name_or_path"],
        args=training_args,
        reward_funcs=reward_function,
        train_dataset=datasets,
    )
    
    trainer.train()

    if config["wandb"]["enabled"]:
        wandb.finish()

def main():
    # parse args and load config
    config_path, overrides = parse_override_args()
    config = load_config(config_path, overrides)

    # validate config
    validate_config(config)

    # print config
    print("=" * 50)
    print("GRPO Training Configuration:")
    print("=" * 50)
    print(yaml.dump(config, default_flow_style=False))
    print("=" * 50)

    # train the model
    train_dr_grpo(config)

if __name__ == "__main__":
    # train_dr_grpo()
    main()
