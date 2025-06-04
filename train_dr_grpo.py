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
from curriculum import StagedDataset
import wandb
import json
import yaml
from datasets import Dataset, concatenate_datasets

from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
import rl_util
from config_manager import load_config, parse_override_args, validate_config
from typing import Literal
from curriculum import StagedDataset

# torch specific optimizations
# Add this after your imports in train_dr_grpo.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
    """Return reward function based on config
    """
    reward_config = config["reward"]
    format_reward = reward_config["format_reward"]
    correctness_reward = reward_config["correctness_reward"]
    
    def reward_function(completions: list[str], **kwargs) -> list[float]:
        """Multi-component reward function"""
        total_rewards = []
        format_count = 0
        correctness_count = 0
        answers = kwargs['answer_value']
        
        for completion, correct_answer in zip(completions, answers):
            reward = 0.0

            # formatted reward
            if rl_util.check_proper_format(completion):
                reward = format_reward
                format_count += 1
            
                # correctness reward
                try:
                    predicted_answer = rl_util.cot_info["cot_to_answer"](completion)
                    predicted_value = rl_util.cot_info["answer_to_value"](predicted_answer)
                    correct_value = rl_util.cot_info["answer_to_value"](correct_answer)
                    
                    if rl_util.cot_info["compare_values"](predicted_value, correct_value):
                        reward = correctness_reward
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


def load_datasets(config, split: Literal["train", "eval"]):
    """Load and prepare train/eval datasets based on config"""
    data_config = config["data"][split]
    datasets = Dataset.from_list([])
    
    for file in data_config["files"]:
        with open(file, 'r') as f:
            raw_data = json.load(f)

        for item in raw_data:
            item["prompt"] = rl_util.r1_zero_question_to_prompt(item.pop("question"))
            item.pop("answer_cot", None)
        
        datasets = concatenate_datasets([datasets, Dataset.from_list(raw_data)])
    
    print(f"Raw dataset size before processing: {len(datasets)}")

    # set shuffle
    if data_config["shuffle"]:
        datasets = datasets.shuffle(seed=config["experiment"]["seed"])

    # apply dataset size limit if specified
    if data_config["dataset_size"] is not None:
        print(data_config["dataset_size"])
        datasets = datasets.select(range(0, data_config["dataset_size"]))
        print(f"Dataset size after processing: {len(datasets)}")
        # print("Dataset Items:", datasets)


    return datasets


class DatasetCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        # At the start of each epoch, update the train_dataset
        if state.epoch > 0:
            train_dataloader = kwargs['train_dataloader']
            if hasattr(train_dataloader.dataset, 'next_stage'):
                train_dataloader.dataset.next_stage()
        # kwargs["train_dataset"].next_stage()
        return

def create_training_args(config):
    """Create GRPOConfig from configuration"""
    exp_config = config["experiment"]
    train_config = config["training"]
    
    return GRPOConfig(
        output_dir=exp_config["output_dir"],
        report_to="wandb" if config["wandb"]["enabled"] else None,
        run_name=exp_config["name"],
        logging_steps=train_config["logging_steps"],
        num_train_epochs=train_config["num_train_epochs"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],

        # vllm parameters
        use_vllm=train_config["use_vllm"],
        vllm_mode=train_config["vllm_mode"],
        vllm_gpu_memory_utilization=train_config["vllm_gpu_memory_utilization"],

        # gen-evaluation parameters
        # generation_batch_size=train_config["generation_batch_size"],
        steps_per_generation=train_config["steps_per_generation"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        num_generations=train_config["num_generations"],

        # training configs
        bf16=train_config["bf16"],
        gradient_checkpointing=train_config["gradient_checkpointing"],
        loss_type=train_config["loss_type"],
        scale_rewards=train_config["scale_rewards"],
        disable_dropout=train_config.get("disable_dropout", True),

        # sampling parameters
        temperature=train_config["temperature"],
        top_p=train_config["top_p"],
        max_completion_length=train_config["max_completion_length"],

        # eval
        eval_strategy=train_config.get("eval_strategy", "epoch"),
        eval_steps=train_config.get("eval_steps", 500),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 16),
        # eval_num_generations=train_config.get("eval_num_generations", 1),
        # eval_temperature=train_config.get("eval_temperature", 0.0),

        # save
        save_steps=train_config["save_steps"],
        save_total_limit=train_config["save_total_limit"],
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

    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log model checkpoints

    # setup cot info
    rl_util.cot_info = rl_util.prepare_cot_info(config["data"]["dataset_name"])

    # setup components from configs
    reward_function = create_reward_function(config)
    train_dataset = StagedDataset(config["data"]["train"]["files"][0], verbose=True)
    train_dataset.next_stage()
    # train_dataset = load_datasets(config, split="train")
    # print(f"sample train dataset items: {train_dataset[0]}")
    # print(f"dataset features: {train_dataset.features}")
    eval_dataset = load_datasets(config, split="eval")
    training_args = create_training_args(config)


    torch.cuda.empty_cache()
    
    trainer = GRPOTrainer(
        model=config["model"]["name_or_path"],
        args=training_args,
        reward_funcs=reward_function,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[DatasetCallback()],
    )
    
    print("Training")
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
    main()
