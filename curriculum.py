"""
This file contains curriculum dataset construction utilities for RL training.

python curriculum.py \
    --method "vanilla" \
    --num_stages 6 \
    --stage_size 1024 \
    --raw_dataset_path "R3_math/data/gsm8k_original_train.json"    
"""

import json
import random
from datasets import Dataset
import argparse

import rl_util


class StagedDataset(Dataset):
    """
    This class expects a list of list of dictionaries, where each sublist represents a stage
    next_stage() can be called to progress to the next stage of the dataset.
    """

    def __init__(self, staged_dataset_path: str, verbose: bool = True):
        # Load entire JSON once (e.g. a list of list of dicts)
        with open(staged_dataset_path, "r") as file:
            self.all_stages: list[list[dict]] = json.load(file)

        # Assert that the data is a list of lists
        assert isinstance(self.all_stages, list) and all(
            isinstance(stage, list) for stage in self.all_stages
        ), f"{staged_dataset_path} must represent a list of lists"

        # Initialize the active stage
        self.active_stage = 0

        self.verbose = verbose
        return

    def __len__(self):
        # Return the length of the current active stage
        return len(self.all_stages[self.active_stage]) if self.active_stage < len(self.all_stages) else 0

    def __getitem__(self, idx: int):
        # Ensure the active stage is valid
        assert self.active_stage < len(self.all_stages), f"Exhausted all {len(self.all_stages)} stages."
        assert (
            0 <= idx < len(self.all_stages[self.active_stage])
        ), f"Index {idx} out of bounds for the current active stage (length {len(self.all_stages[self.active_stage])})."

        # Return the item from the current active stage
        return self.all_stages[self.active_stage][idx]

    def next_stage(self):
        if self.verbose:
            print(f"Advancing from stage {self.active_stage} to stage {self.active_stage + 1}.")
        self.active_stage += 1
        return


def construct_vanilla_dataset(
    raw_dataset: list[dict], num_stages: int, stage_size: int, seed: int = 42
) -> list[list[dict]]:
    """
    Constructs a vanilla dataset from a JSON file containing a list of lists of dictionaries.
    Each sublist represents a stage in the curriculum.
    """

    # Shuffle the raw dataset
    random.seed(seed)
    random.shuffle(raw_dataset)

    # Format the prompts and answer values
    formatted_dataset = [
        {
            "prompt": rl_util.r1_zero_question_to_prompt(item["question"]),
            "answer_value": item["answer_value"],
            "item_id": item["item_id"],
        }
        for item in raw_dataset
    ]

    # Take and split the dataset into `num_stages` stages, each containing `stage_size` items
    # Only contains the first num_stages * stage_size items
    staged_dataset = [formatted_dataset[stage_size * i : stage_size * (i + 1)] for i in range(0, num_stages)]

    return staged_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Construct a staged dataset from raw data.")
    parser.add_argument("--raw_dataset_path", type=str, required=True, help="Path to the raw dataset JSON file.")
    parser.add_argument("--method", type=str, choices=["vanilla"], help="Method to construct the dataset.")
    parser.add_argument("--num_stages", type=int, default=3, help="Number of stages in the curriculum.")
    parser.add_argument("--stage_size", type=int, default=1, help="Number of items in each stage.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the constructed staged dataset.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.raw_dataset_path, "r") as file:
        raw_dataset = json.load(file)
    print("Raw dataset loaded from", args.raw_dataset_path)

    if args.method == "vanilla":
        staged_dataset = construct_vanilla_dataset(
            raw_dataset, num_stages=args.num_stages, stage_size=args.stage_size, seed=args.seed
        )

    print(
        f"Constructed staged dataset using {args.method} method, with {args.num_stages} stages, each containing {args.stage_size} items."
    )

    # Saved the staged dataset
    if args.output_path is None:
        args.output_path = f"curricula/{args.method}_staged_dataset.json"
    with open(args.output_path, "w") as file:
        json.dump(staged_dataset, file, indent=4)
    print(f"Staged dataset saved to {args.output_path}")
