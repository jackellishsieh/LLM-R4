"""
This files contains the training loop function for the Dr. GRPO algorithm using the Qwen-0.5B model.
"""

from dr_grpo_trainer import DrGRPOTrainer
from trl import GRPOConfig
import rl_util
import json
from datasets import Dataset, concatenate_datasets

TRAINING_FILES = [
    "R3_math/data/gsm8k_original_train.json",
]

def train_dr_grpo(training_files=TRAINING_FILES):
    """Train the Dr. GRPO algorithm on the Qwen-0.5B model."""

    rl_util.cot_info = rl_util.prepare_cot_info("gsm8k")
    
    # define reward function
    def reward_function(completions, questions, answers, **kwargs):
        """Multi-component reward function: format (0.2) + correctness (1.0)"""
        total_rewards = []
        
        for completion, correct_answer in zip(completions, answers):
            reward = 0.0

            # check formatting
            if rl_util.check_proper_format(completion):
                reward += 0.2
            
            # check correctness
            try:
                predicted_answer = rl_util.cot_info["cot_to_answer"](completion) # extract answer (float)
                predicted_value = rl_util.cot_info["answer_to_value"](predicted_answer)

                correct_value = rl_util.cot_info["answer_to_value"](correct_answer)
                
                if rl_util.cot_info["compare_values"](predicted_value, correct_value):
                    reward += 1.0
            except:
                pass
            
            total_rewards.append(reward)
    
        return total_rewards
    
    # load training datasets
    datasets = Dataset.from_list([])  # initialize empty huggingface dataset
    for file in training_files:
        # load the dataset, assuming it has a 'question', and 'answer_value' field
        with open(file, 'r') as f:
            raw_data = json.load(f)

        for item in raw_data:
            # replace "question" with "prompt" in each item
            item["prompt"] = rl_util.deepseek_question_to_prompt(item.pop("question"))
            item.pop("answer_cot", None)  # remove answer_cot if it exists
        
        # convert to huggingface dataset
        datasets = concatenate_datasets([datasets, Dataset.from_list(raw_data)])
        
    
    # configure training
    training_args = GRPOConfig(
        output_dir="dr_grpo_qwen_0.5b",
        use_vllm=True,
        vllm_mode="colocate",  # maybe "server", depending on setup
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16, 
        num_generations=64,  # n responses per prompt
        bf16=True,
        gradient_checkpointing=True,
        scale_rewards=False,  # disable std scaling for dr. grpo
        temperature=0.8, # high temp for GRPO
        top_p=0.9,
        max_completion_length=512,  # max length of generated response
    )
    
    # init trainer
    trainer = DrGRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        args=training_args,
        reward_funcs=reward_function,
        train_dataset=datasets,
    )
    
    trainer.train()

if __name__ == "__main__":
    train_dr_grpo()
