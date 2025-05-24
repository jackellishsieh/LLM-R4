import os
import json
import argparse
from datasets import load_dataset

def add_item_ids(data, root_name, start_index=0):
    for i, example in enumerate(data, start=start_index):
        example["item_id"] = f"{root_name}_{i}"
    return data

def add_difficulty_scores(data):
    for example in data:
        answer_cot = example.get("answer", "")
        steps = [step.strip() for step in answer_cot.split('\n') if step.strip()]
        example["difficulty_score"] = len(steps)
    return data

def extract_answer_value(answer_cot):
    last_line = answer_cot.strip().split('\n')[-1].strip()
    return last_line.replace("####", "").strip()

def reorder_fields(data, file_type):
    reordered = []
    for ex in data:
        if file_type == "difficulty_train":
            reordered.append({
                "item_id": ex["item_id"],
                "question": ex["question"],
                "answer_cot": ex["answer"],
                "answer_value": extract_answer_value(ex["answer"]),
                "difficulty": ex["difficulty_score"]
            })
        elif file_type == "original_train":
            reordered.append({
                "item_id": ex["item_id"],
                "question": ex["question"],
                "answer_cot": ex["answer"],
                "answer_value": extract_answer_value(ex["answer"])
            })
        elif file_type == "original_test":
            reordered.append({
                "item_id": ex["item_id"],
                "question": ex["question"],
                "answer_value": extract_answer_value(ex["answer"])
            })
    return reordered

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_or_download_gsm8k(root_name, outer_folder):
    train_original_path = os.path.join(outer_folder, f"{root_name}_original_train.json")
    test_original_path = os.path.join(outer_folder, f"{root_name}_original_test.json")
    train_difficulty_path = os.path.join(outer_folder, f"{root_name}_difficulty_train.json")

    if os.path.exists(train_original_path) and os.path.exists(test_original_path):
        with open(train_original_path) as f:
            train_data = json.load(f)
        with open(test_original_path) as f:
            test_data = json.load(f)
    else:
        print(f"{root_name} files not found, downloading from Hugging Face...")
        dataset = load_dataset(root_name, "main")
        train_data = dataset["train"].to_list()
        test_data = dataset["test"].to_list()

    # Add item_ids
    train_data = add_item_ids(train_data, root_name, start_index=0)
    test_data = add_item_ids(test_data, root_name, start_index=len(train_data))

    # Reorder and save
    reordered_train = reorder_fields(train_data, "original_train")
    reordered_test = reorder_fields(test_data, "original_test")
    os.makedirs(outer_folder, exist_ok=True)
    save_json(reordered_train, train_original_path)
    save_json(reordered_test, test_original_path)

    return train_data, test_data, train_difficulty_path

def main(root_name, outer_folder):
    train_data, test_data, train_difficulty_path = load_or_download_gsm8k(root_name, outer_folder)

    # Add difficulty scores
    train_data_with_difficulty = add_difficulty_scores(train_data)
    reordered_difficulty = reorder_fields(train_data_with_difficulty, "difficulty_train")

    # Save difficulty-annotated version
    save_json(reordered_difficulty, train_difficulty_path)

    print(f"Original train saved as: {os.path.join(outer_folder, root_name + '_original_train.json')}")
    print(f"Difficulty-annotated train saved as: {train_difficulty_path}")
    print(f"Original test saved as: {os.path.join(outer_folder, root_name + '_original_test.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_name", help="Base name for the GSM8K files, e.g., 'gsm8k'")
    parser.add_argument("outer_folder", help="Folder path to save the GSM8K JSON files")
    args = parser.parse_args()
    main(args.root_name, args.outer_folder)