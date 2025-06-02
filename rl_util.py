import re
import torch
from typing import Callable, TypedDict
import constants

class CotInfo(TypedDict):
    """A TypedDict to hold information related to Chain of Thought question processing and evaluation."""

    question_to_prompt: Callable[[str], str]    # function to convert a question to a prompt
    cot_to_answer: Callable[[str], str]         # function to convert a generated Chain of Thought (CoT) to a string answer. Just extraction, not post-processing
    answer_to_value: Callable[[str], float]     # function to convert a string answer to a float value
    compare_values: Callable[[float, float], bool]   # function to compare the extracted answer with the target answer

    # The original CotInfo format is:
    # instruction: str        # the prompt instruction, to be followed by the question
    # cot_trigger: str        # the reponse trigger, to be followed by the generated CoT
    # answer_trigger: str     # the answer trigger, to be followed by the final answer
    # post_process_final_answer_fn_mapper: dict[str, Callable[[str], float]]      # post-process final answers (by removing spaces and commas)
    # post_process_completed_question_answer_fn_mapper: dict[tuple[str, str], Callable[[str], float]] # post-process completed question answers (by extracting the answer from the generated CoT)
    # compare_answer_fn_mapper: dict[str, Callable[[float, float], bool]] # compare the extracted answer with the target answer


def deepseek_question_to_prompt(question: str) -> str:
    """
    Convert a question to a prompt using the DeepSeek r1-zero format.
    """
    return constants.r1_zero_instruction.replace("{question}", question)


def deepseek_cot_to_answer(cot: str) -> float:
    """
    Convert a generated Chain of Thought (CoT) to a final answer using the DeepSeek r1-zero format.
    """
    # Identify the last occurence of <answer>...</answer> in the CoT
    answer_matches = re.findall(
        r"<answer>(.*?)</answer>", cot
    )  # really, there should only be one match
    if not answer_matches:
        return None

    last_answer_match = answer_matches[-1]  # take the last match if it exists
    last_extracted_answer = last_answer_match.strip()
    return last_extracted_answer

def gsm8k_answer_to_value(answer: str) -> float:
    return float(answer.replace(",", "").strip())

def gsm8k_compare_answer(
    predicted_value: float, target_value: float, tolerance=1e-2
) -> bool:
    """
    Compare the extracted answer with the target answer for the GSM8K dataset.
    Returns True if the absolute difference is within a small tolerance, otherwise False.
    """
    return abs(predicted_value - target_value) <= tolerance

def deepseek_answer_cot_to_string(answer_cot: str, target_answer: str) -> str:
    """
    Convert a golden CoT and target answer to a string, formatted using the DeepSeek r1-zero format.
    """
    return f"{answer_cot}</think> <answer> {target_answer} </answer>"


def prepare_cot_info(src_name):
    """
    Given a source name, prepare the instruction, COT trigger, answer trigger, and post-processing functions.
    This is a function only of the dataset
    """
    # assert src_name in [
    #     "gsm8k",
    #     "svamp",
    # ], f"Source name ({src_name}) must be either 'gsm8k' or 'svamp'."
    assert src_name == "gsm8k", f"Source name ({src_name}) must be 'gsm8k' for now."

    return {
        "question_to_prompt": deepseek_question_to_prompt,
        "cot_to_answer": deepseek_cot_to_answer,
        "answer_to_value": gsm8k_answer_to_value,
        "compare_values": gsm8k_compare_answer,
    }

def check_proper_format(completion):
    """Check if completion has proper <think></think> and <answer></answer> formatting"""
    # Check for both think tags and answer tags
    has_think_tags = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL))
    has_answer_tags = bool(re.search(r'<answer>.*?</answer>', completion, re.DOTALL))
    
    return has_think_tags and has_answer_tags

def collate_fn(batch, tokenizer):
    """
    Collate function for DataLoader.
    """
    max_input_length = max([len(item["input_ids"]) for item in batch])
    max_target_length = max([len(item["labels"]) for item in batch])
    max_prefix_length = max([len(item["prefix"]) for item in batch])

    input_ids, input_ids_left_padded = [], []
    attention_mask, attention_mask_left_padded = [], []
    labels, labels_left_padded = [], []
    prefix, prefix_left_padded = [], []
    prefix_attention_mask, prefix_attention_mask_left_padded = [], []

    for item in batch:
        labels_left_padded.append(
            [-100] * (max_target_length - len(item["labels"])) + item["labels"]
        )
        prefix_left_padded.append(
            [tokenizer.pad_token_id] * (max_prefix_length - len(item["prefix"]))
            + item["prefix"]
        )
        prefix_attention_mask_left_padded.append(
            [0] * (max_prefix_length - len(item["prefix_attention_mask"]))
            + item["prefix_attention_mask"]
        )

    ppo_forward_kwargs = {
        "query": [item["prefix_text"] for item in batch],
        "query_tensors": torch.LongTensor(prefix_left_padded),
        "query_tensors_attention_mask": torch.BoolTensor(
            prefix_attention_mask_left_padded
        ),
        "answer_values": [item["answer_value"].replace(",", "") for item in batch],
        "item_ids": torch.LongTensor(
            [int(item["item_id"].split("_")[1]) for item in batch]
        ),
    }
    generate_prefix_kwargs = {
        "input_ids": torch.LongTensor(prefix_left_padded),
        "attention_mask": torch.BoolTensor(prefix_attention_mask_left_padded),
        "labels": torch.LongTensor(labels_left_padded),
    }

    return {
        "ppo_forward_kwargs": ppo_forward_kwargs,
        "generate_prefix_kwargs": generate_prefix_kwargs,
    }
