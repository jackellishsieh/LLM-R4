import argparse
from collections import defaultdict
from functools import partial
import json
from typing import List
from tqdm import tqdm
import torch
from R3_math.src.utils import timeout
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from accelerate.utils import pad_across_processes
import deepspeed

from transformers import (
    AutoModelForCausalLM,
)
from R3_others.dschat.utils.ds_utils import get_train_ds_config
from R3_others.dschat.utils.model.model_utils import create_hf_model
from R3_others.dschat.utils.utils import (
    load_hf_tokenizer,
)

import util

TIMEOUT = 2


def prepare_eval_dataset(
    eval_file: str,
    tokenizer,
    engine: str = "nl",
    max_input_length: int = 700,
    eval_batch_size: int = 8,
    num_workers: int = 0,
):
    # Load raw dataset
    eval_dataset = (
        Dataset.from_list(json.load(open(eval_file, "r")))
        if not eval_file.rstrip("/").endswith("_cache")
        else load_from_disk(eval_file)
    )

    src_name = eval_dataset["item_id"][0].split("_")[0]  # e.g., "nl", "python"
    cot_info = util.prepare_cot_info(src_name)
    instruction = cot_info["instruction"]
    cot_trigger = cot_info["cot_trigger"]
    answer_trigger = cot_info["answer_trigger"]

    def tokenize_fn(batch):
        assert tokenizer.eos_token_id is not None, (
            tokenizer.eos_token_id,
            tokenizer.eos_token,
        )

        new_batch = defaultdict(list)
        all_keys = list(batch.keys())
        for item_values in zip(*(batch[k] for k in all_keys)):
            item = {k: item_values[i] for i, k in enumerate(all_keys)}
            item_id, question, answer_value, answer_cot = (
                item["item_id"],
                item["question"],
                item["answer_value"],
                item.get("answer_cot", None),
            )
            # question = question.strip()
            if answer_value is not None:
                answer_value = answer_value.strip()

            if answer_cot:
                # answer_cot = answer_cot.strip()
                if engine == "nl" and src_name in ["gsm8k"]:
                    answer_cot += f"{answer_trigger} {answer_value}"

                input = f"{instruction}{question}"
                output = f"{answer_cot}"
                prefix_text = f"{instruction}{question}"

                if answer_cot.startswith("def"):
                    question_1 = question.replace("\n\n### Response:", "")
                    if src_name in ["gsm8k", "svamp"] and engine == "python":
                        prefix_text += f'def solution():\n    """{question_1}"""\n'

            else:
                input = f"{instruction}{question}{cot_trigger}"
                output = f"{answer_cot}"
                prefix_text = f"{instruction}{question}{cot_trigger}"

                if src_name in ["gsm8k", "svamp"] and engine == "python":
                    prefix_text += f'def solution():\n    """{question}"""\n'

            input_encode = tokenizer(input, add_special_tokens=False)
            output_encode = tokenizer(output, add_special_tokens=False)
            prefix_encode = tokenizer(prefix_text, add_special_tokens=False)

            input_ids = (
                input_encode["input_ids"]
                + output_encode["input_ids"]
                + [tokenizer.eos_token_id]
            )
            labels = (
                [-100] * len(input_encode["input_ids"])
                + output_encode["input_ids"]
                + [tokenizer.eos_token_id]
            )
            attention_mask = [1] * len(input_ids)
            prefix = prefix_encode["input_ids"]
            prefix_attention_mask = prefix_encode["attention_mask"]

            # Truncation
            input_ids = input_ids[:max_input_length]
            labels = labels[:max_input_length]
            attention_mask = attention_mask[:max_input_length]
            prefix = prefix[:max_input_length]
            prefix_attention_mask = prefix_attention_mask[:max_input_length]

            ##
            new_batch["input_ids"].append(input_ids)
            new_batch["labels"].append(labels)
            new_batch["attention_mask"].append(attention_mask)
            new_batch["prefix"].append(prefix)
            new_batch["prefix_attention_mask"].append(prefix_attention_mask)
            ##
            new_batch["item_id"].append(item_id)
            new_batch["question"].append(question)
            new_batch["prefix_text"].append(prefix_text)
            new_batch["answer_cot"].append(answer_cot)
            new_batch["answer_value"].append(answer_value)

        return new_batch

    tokenized_eval_dataset = eval_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=eval_dataset.column_names,
        num_proc=None,
        load_from_cache_file=True,
        keep_in_memory=False,
    )

    eval_dataloader = DataLoader(
        tokenized_eval_dataset,
        shuffle=False,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(util.collate_fn, tokenizer=tokenizer),
    )

    return (tokenized_eval_dataset, eval_dataloader), cot_info


def evaluate_generation(
    model,
    dataset,
    dataloader,
    tokenizer,
    cot_info,
    output_dir: str,
    max_gen_length: int,
    engine: str,
):
    """
    Evaluate the accuracy of the model on the dataset using the dataloader.
    """
    model.eval()
    predictions: List[str] = []
    targets: List[str] = []

    # Iterate through the evaluation dataloader and collect the generated predictions as strings
    for idx, batch in tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Evaluation Gen Loop"
    ):
        output_ = model.generate(
            **batch["generate_prefix_kwargs"],
            max_length=max_gen_length,
            output_scores=True,
            return_dict_in_generate=True,
            num_beams=1,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ids = output_.sequences
        generated_ids = pad_processes(
            generated_ids, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True
        )

        labels = batch["generate_prefix_kwargs"]["labels"]
        labels = pad_across_processes(
            labels, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True
        )
        labels[labels == -100] = tokenizer.pad_token_id

        # generated_ids, labels = accelerator.gather(generated_ids), accelerator.gather(labels)

        preds = [
            tokenizer.decode(
                g.cpu().numpy().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            for g in generated_ids
        ]
        predictions.extend(preds)
        target = [
            tokenizer.decode(
                t.cpu().numpy().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            for t in labels
        ]
        targets.extend(target)

    predictions = predictions[: len(dataset)]
    targets = targets[: len(dataset)]

    post_process_final_answer_fn_mapper = cot_info[
        "post_process_final_answer_fn_mapper"
    ]
    compare_answer_fn_mapper = cot_info["compare_answer_fn_mapper"]
    post_process_completed_question_answer_fn_mapper = cot_info[
        "post_process_completed_question_answer_fn_mapper"
    ]

    results = [
        {
            "prediction": prediction,
            "target": target,
            "item_id": item.get("item_id", None),
            "answer_value": item.get("answer_value", None),
            "answer_type": item.get("answer_type", None),
        }
        for prediction, target, item in zip(predictions, targets, dataset)
    ]

    corr_value = 0
    for cur_res in results:
        prediction, target, item_id = (
            cur_res["pred"],
            cur_res["tar"],
            cur_res["item_id"],
        )
        src_name = item_id.split("_")[0]
        answer_value = cur_res["answer_value"]

        ## Processing target
        target_cot = target.strip()
        target_value = post_process_final_answer_fn_mapper[src_name](answer_value)
        cur_res["target_cot"] = target_cot
        cur_res["target_value"] = target_value

        ## Processing prediction
        try:
            with timeout(seconds=TIMEOUT):
                prediction_cot = prediction.strip()
                prediction_value = post_process_completed_question_answer_fn_mapper[
                    (engine, src_name)
                ](prediction_cot)
        except:
            prediction_cot = None
            prediction_value = None
        cur_res["prediction_cot"] = prediction_cot
        cur_res["prediction_value"] = prediction_value

        # Compute correctness
        is_correct = (
            compare_answer_fn_mapper[src_name](prediction_value, target_value)
            if prediction_value is not None
            else False
        )
        corr_value += is_correct
        cur_res["is_correct"] = is_correct

    with open(output_dir, "w") as f:
        json.dump(results, f, indent=2)
    value_accuracy = corr_value / len(predictions) * 100
    print(f"[Eval Info] value_accuracy: {value_accuracy:.5g}%")
    value_accuracy = torch.FloatTensor([value_accuracy])

    # Metric summary:
    model.train()
    return value_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on a json dataset.")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the model.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )

    parser.add_argument(
        "--eval_file", type=str, help="The input evaluation data file (a json file)."
    )

    parser.add_argument(
        "--max_input_length",
        type=int,
        default=700,
        help="The maximum total input sequence length after tokenization.",
    )

    parser.add_argument(
        "--max_gen_length",
        type=int,
        default=700,
        help="The maximum total sequence length for generation.",
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="nl",
        help="The engine to use for generation. Options: nl, python",
    )

    parser.add_argument(
        "--add_eot_token",
        action="store_true",
        help="Add <|endoftext|> as additional special token to tokenizer",
    )

    parser.add_argument(
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16"],
        help="Training data type",
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    print("DDP: ", torch.distributed.is_initialized())

    if not torch.distributed.is_initialized():
        device = torch.device(deepspeed.get_accelerator().device_name())
    else:
        deepspeed.get_accelerator().set_device(args.local_rank)
        device = torch.device(
            deepspeed.get_accelerator().device_name(), args.local_rank
        )
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    print("Device = ", device)

    # Initialize the tokenizer
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = (
        args.end_of_conversation_token if args.add_eot_token else None
    )
    tokenizer = load_hf_tokenizer(
        args.model_name_or_path,
        fast_tokenizer=True,
        add_special_tokens=additional_special_tokens,
    )

    # Initialize the model
    ds_config = get_train_ds_config(
        offload=args.offload,
        # gradient_clipping="auto",
        dtype=args.dtype,
        tb_name="step1_model",
    )
    model = create_hf_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        tokenizer,
        ds_config,
        dropout=args.dropout,
    )

    # Evaluate the model
    (tokenized_eval_dataset, eval_dataloader), cot_info = prepare_eval_dataset(
        args.eval_file, tokenizer=tokenizer, max_input_length=args.max_input_length
    )

    value_accuracy = evaluate_generation(
        model,
        tokenized_eval_dataset,
        eval_dataloader,
        tokenizer,
        cot_info,
        output_dir=args.output_dir,
        max_gen_length=args.max_gen_length,
        engine=args.engine,
    )
    print(f"Final value_accuracy: {value_accuracy:.5g}%")


if __name__ == "__main__":
    main()
