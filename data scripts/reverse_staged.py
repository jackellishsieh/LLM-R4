import os
import json 
import argparse
import random 
from datasets import load_dataset

"""
    Shuffles the dataset in all cases. Replaces previous JSONs. Adds difficulty if requested.
"""
def prepare_dataset(dataset_name):
    """
    The dataset name from Hugging Face here should have a default train and test split.
    shuffle: to shuffle/randomize the dataset or not
    p: portion of dataset to train and test on
    """
    dataset = load_dataset(dataset_name, "main") # can add support for different config later
    train_dataset = dataset["train"] 
    test_dataset = dataset["test"] 
    return train_dataset, test_dataset
    
def staged_data(data, M):
    """
    Iterate over a list of examples, compute a “difficulty_score” for each example 
    by splitting its 'answer' field line by line, and return the modified list.

    Args:
        data (datadict): Dataset with 'answer' field.

    Returns:
    list[list[dict]]
    """
    data_by_stage = [[] for i in range(M)]
    for example in data:
        if "answer" not in example:
            raise RuntimeError("Dataset has no 'answer' field.")
        cot = example["answer"]
        staged_partial, staged_remaining = [], [] 
        steps = [step for step in cot.split('\n') if step.strip()]
        no_steps = len(steps)
        if no_steps >= M / 2: # if we have enough newlines, sample from newlines 
            for i in range(M):
                step_idx = int((i + 1) * no_steps / (M + 2))
                staged_partial.append("\n".join(steps[:step_idx]))
                staged_remaining.append("\n".join(steps[step_idx:]))
        else:
            tokens = cot.split() # otherwise, uniformly sample by token 
            no_tokens = len(tokens)
            for i in range(M):
                token_idx = int((i + 1) * no_tokens / (M + 2)) # can repeat if M is close to no_tokens
                staged_partial.append(" ".join(tokens[:token_idx]))
                staged_remaining.append(" ".join(tokens[token_idx:])) 
        
        answer_val = cot.split("####")[-1].strip()
        for stage in range(M): 
            staged_example = example.copy()
            staged_example["partial_rationale"] = staged_partial[stage]
            staged_example["answer"] = answer_val
            staged_example["stage"] = M - stage 
            data_by_stage[M - stage - 1].append(staged_example)
        
    return data_by_stage

def prepare_staged_datasets(data_by_stage, batch):
    sampled_by_stage = []
    
    def to_deepseek_style(example):
        example = example.copy()
        question = example.get("question") 
        partial_rationale = example.get("partial_rationale") 
        prompt = f"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. 
        User: {question}
        Assistant: <think> {partial_rationale}""" 
        example["prompt"] = prompt
        example.pop("partial_rationale") # no longer needed
        return example 
    
    for i, stage_list in enumerate(data_by_stage):  # stage_list is List[Dict]
        if len(stage_list) < batch:
            raise RuntimeError(f"Stage {i+1} has only {len(stage_list)} examples, but batch={batch} requested.")
        
        indices = random.sample(range(len(stage_list)), batch)
        sampled = [to_deepseek_style(stage_list[j]) for j in indices]
        sampled_by_stage.append(sampled)  

    return sampled_by_stage 

def save_json(data, path):
    if not isinstance(data, list):
        data = data.to_list()
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
def main(dataset_name, output_folder, M=0, batch=0):
    os.makedirs(output_folder, exist_ok=True)
    train_data, test_data = prepare_dataset(dataset_name)

    # save testing data
    test_json_path = os.path.join(output_folder, f"test_{dataset_name}.json")
    save_json(test_data, test_json_path)
    print(f"Saved original test set: {test_json_path}")

    # save training data
    if M:
        data_by_stage = staged_data(train_data, M)
        if batch:
            train_data = prepare_staged_datasets(data_by_stage, batch)
        train_json_path = os.path.join(output_folder, f"train_{dataset_name}_staged_M{M}.json")
        print(f"Saving staged train set with M={M} stages: {train_json_path}")
    else:
        train_json_path = os.path.join(output_folder, f"train_{dataset_name}.json")
        print(f"Saving staged train set.")
    save_json(train_data, train_json_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("dataset_name", help="Base name for the GSM8K (or dataset with similar enough structure) files, e.g., 'gsm8k'")
    parser.add_argument("output_folder", help="Folder path to save the GSM8K JSON files")
    parser.add_argument("--M", type=int, default=0, help="Number of stages for staged data (default: None)")
    parser.add_argument("--batch", type=int, default=0, help="Number of examples to sample for each batch. (default: All)")
    
    args = parser.parse_args()
    main(args.dataset_name, args.output_folder, M=args.M, batch=args.batch)