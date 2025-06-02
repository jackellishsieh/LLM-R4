"""
This file contains the class for the trainer for the DRGRPO trainer, which inherits from TRL's GRPOTrainer.
It includes the methods from the Dr. GRPO paper (https://arxiv.org/pdf/2503.20783) to remove response-level 
length bias and question-level difficulty bias.
"""

from trl import GRPOTrainer
import torch
import numpy as np
# from generation import evaluate_vllm, init_vllm, init_sampling_params

class DrGRPOTrainer(GRPOTrainer):
    def compute_advantages(self, rewards):
        """Dr. GRPO advantage calculation - removes std normalization"""
        # group rewards by prompt
        advantages = []
        
        for group_rewards in rewards:  # rewards per prompt group
            group_mean = np.mean(group_rewards)
            # notably no std normalization here, as per dr. grpo
            group_advantages = [r - group_mean for r in group_rewards]
            advantages.extend(group_advantages)
        
        return torch.tensor(advantages, dtype=torch.float32)