import os
import json
import argparse
from datasets import load_dataset

"""
    Shuffles the dataset in all cases. Replaces previous JSONs. Adds difficulty if requested.
"""
def prepare_dataset(dataset_name, output_folder, shuffle=False):
    """
    The dataset name from Hugging Face here should have a default train and test split.
    """
    train_json_path = os.path.join(output_folder, f"train_{dataset_name}.json")
    test_json_path = os.path.join(output_folder, f"test_{dataset_name}.json")

    if not os.path.exists(train_json_path):
        dataset = load_dataset(dataset_name)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
    else:
        with open(train_json_path, "r") as f:
            train_dataset = json.load(f)
        with open(test_json_path, "r") as f:
            test_dataset = json.load(f)

    if shuffle:
        if hasattr(train_dataset, "shuffle"):
            train_dataset = train_dataset.shuffle()
            test_dataset = test_dataset.shuffle()
        else:
            raise RuntimeError("Dataset has no shuffle method")

    return train_dataset, test_dataset

def add_difficulty_scores(data):
    """
    Iterate over a list of examples, compute a “difficulty_score” for each example 
    by splitting its 'answer' field line by line, and return the modified list.

    Args:
        data (datadict): from load_dataset 

    Returns:
        list of dict: The same list, but each dict now also has a 
                      'difficulty_score' key (an integer).
    """
    for example in data:
        if "answer" not in example:
            raise RuntimeError("Dataset has no 'answer' field — cannot compute difficulty score")
        cot = example["answer"]
        steps = [step for step in cot.split('\n') if step.strip()] # only keep steps with informative chars
        example["difficulty"] = len(steps)
    return data

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def main(dataset_name, output_folder, difficulty=False, shuffle=False):
    os.makedirs(output_folder, exist_ok=True)
    train_data, test_data = prepare_dataset(dataset_name, output_folder, shuffle=shuffle)

    # save testing data
    test_json_path = os.path.join(output_folder, f"test_{dataset_name}.json")
    save_json(test_data, test_json_path)

    print(f"Saved original train set: {train_json_path}")
    print(f"Saved original test set: {test_json_path}")

    # save training data
    if difficulty:
        train_data_with_difficulty = add_difficulty_scores(train_data)
        train_json_path = os.path.join(output_folder, f"train_{dataset_name}_staged.json")
        save_json(train_data_with_difficulty, train_json_path)
        print(f"Saved difficulty-tiered train set: {train_json_path}")
    else:
        train_json_path = os.path.join(output_folder, f"train_{dataset_name}.json")
        save_json(train_data, train_json_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_name", help="Base name for the GSM8K files, e.g., 'gsm8k'")
    parser.add_argument("outer_folder", help="Folder path to save the GSM8K JSON files")
    parser.add_argument("--difficulty", action="store_true", help="Add difficulty tiers to train set")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle train and test sets")
    args = parser.parse_args()
    main(args.root_name, args.outer_folder, difficulty=args.difficulty, shuffle=args.shuffle)