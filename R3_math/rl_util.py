import torch
from typing import Callable, TypedDict

class CotInfo(TypedDict):
    """A TypedDict to hold information related to Chain of Thought question processing and evaluation."""
    instruction: str        # the prompt instruction, to be followed by the question
    cot_trigger: str        # the reponse trigger, to be followed by the generated CoT
    answer_trigger: str     # the answer trigger, to be followed by the final answer
    post_process_final_answer_fn_mapper: dict[str, Callable[[str], float]]      # post-process final answers (by removing spaces and commas)
    post_process_completed_question_answer_fn_mapper: dict[tuple[str, str], Callable[[str], float]] # post-process completed question answers (by extracting the answer from the generated CoT)
    compare_answer_fn_mapper: dict[str, Callable[[float, float], bool]] # compare the extracted answer with the target answer

def prepare_cot_info(src_name):
    """
    Given a source name, prepare the instruction, COT trigger, answer trigger, and post-processing functions.
    This is a function only of the dataset
    """
    assert src_name in ["gsm8k", "svamp"]

    # default for common datasets
    instruction = "Question:\n"
    cot_trigger = "\nAnswer reasoning:\n"
    answer_trigger = "\nTherefore, the answer is: "

    if src_name in ["gsm8k", "svamp"]:
        # over-write for the simple datasets
        instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"
        cot_trigger = "\n\n### Response:"
        answer_trigger = "\n####"

    # post process final answers by removing commas and spaces
    post_process_final_answer_fn_mapper = {
        "gsm8k": lambda x: float(x.replace(",", "").strip()),
        "svamp": lambda x: float(x.replace(",", "").strip()),
    }

    # run code to extract answer preceding ####
    post_process_completed_question_answer_fn_mapper = {
        # ('python', 'gsm8k'): lambda completed_question_answer: float(run_python_code(code_gen=completed_question_answer.split(cot_trigger)[-1].strip())),
        # ('python', 'svamp'): lambda completed_question_answer: float(run_python_code(code_gen=completed_question_answer.split(cot_trigger)[-1].strip())),
        ("nl", "gsm8k"): lambda completed_question_answer: float(
            completed_question_answer.split(cot_trigger)[-1]
            .split(answer_trigger)[-1]
            .strip()
        ),
        ("nl", "svamp"): lambda completed_question_answer: float(
            completed_question_answer.split(cot_trigger)[-1]
            .split(answer_trigger)[-1]
            .strip()
        ),
    }

    # compare for equality, essentially
    compare_answer_fn_mapper = {
        "gsm8k": lambda extracted_ans, target_answer: abs(extracted_ans - target_answer)
        <= 1e-2,
        "svamp": lambda extracted_ans, target_answer: abs(extracted_ans - target_answer)
        <= 1e-2,
    }

    return {
        "instruction": instruction,
        "cot_trigger": cot_trigger,
        "answer_trigger": answer_trigger,
        "post_process_final_answer_fn_mapper": post_process_final_answer_fn_mapper,
        "post_process_completed_question_answer_fn_mapper": post_process_completed_question_answer_fn_mapper,
        "compare_answer_fn_mapper": compare_answer_fn_mapper,
    }


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
