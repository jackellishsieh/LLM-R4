import argparse
import json
import torch
from R3_math.src.utils import timeout
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
)
from R3_others.dschat.utils.ds_utils import get_train_ds_config
from R3_others.dschat.utils.model.model_utils import create_hf_model
from R3_others.dschat.utils.utils import (
    load_hf_tokenizer,
)
from vllm import LLM, SamplingParams

import rl_util
import generation

TIMEOUT = 2

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on a json dataset.")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    # parser.add_argument(
    #     "--per_device_eval_batch_size",
    #     type=int,
    #     default=16,
    #     help="Batch size (per device) for the evaluation dataloader.",
    # )
    # parser.add_argument(
    #     "--dropout",
    #     type=float,
    #     default=None,
    #     help="If dropout configured, use it. "
    #     "Otherwise, keep the default dropout configuration of the model.",
    # )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--src_name",
        type=str,
        help="Source name for the evaluation dataset (e.g., 'gsm8k', 'svamp').",
    )

    parser.add_argument(
        "--eval_file", type=str, help="The input evaluation data file (a json file)."
    )

    parser.add_argument(
        "--max_gen_length",
        type=int,
        default=700,
        help="The maximum total sequence length for generation.",
    )

    parser.add_argument(
        "--add_eot_token",
        action="store_true",
        help="Add <|endoftext|> as additional special token to tokenizer",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
        help="Training data type",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, will print additional information during evaluation.",
    )

    # parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main(args):
    """
    Main function to evaluate a model on a json dataset.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.verbose:
        print(f"\nUsing device {device}")

    # # Initialize the tokenizer
    # args.end_of_conversation_token = "<|endoftext|>"
    # additional_special_tokens = (
    #     args.end_of_conversation_token if args.add_eot_token else None
    # )
    # tokenizer = load_hf_tokenizer(
    #     args.model_name_or_path,
    #     fast_tokenizer=True,
    #     add_special_tokens=additional_special_tokens,
    # )
    # if args.verbose:
    #     print(f"Initialized tokenizer from {args.model_name_or_path}")

    # Read the eval file as a json list of dictionaries
    eval_examples: list[generation.EvalExample] = json.load(open(args.eval_file, "r"))
    if args.verbose:
        print(f"Loaded {len(eval_examples)} evaluation examples from {args.eval_file}")

    # Initialize the vLLM model
    vllm_model = LLM(model=args.model_name_or_path,
                     dtype=args.dtype,
                     enable_prefix_caching=True,
                     gpu_memory_utilization=0.5
                )
    if args.verbose:
        print(f"Initialized vLLM model from {args.model_name_or_path}")

    # Initialize the sampling parameters
    eval_sampling_params = generation.init_sampling_params(
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        max_tokens=args.max_gen_length,
    )
    if args.verbose:
        print("Initialized sampling parameters for evaluation")

    # Run evaluation on the model, saving the results to the output directory
    cot_info = rl_util.prepare_cot_info(args.src_name)  # Assuming "nl" is the source name for the evaluation
    if args.verbose:
        print(f"Prepared COT info for source name {args.src_name}")

    results = generation.evaluate_vllm(
        vllm_model,
        eval_examples,
        eval_sampling_params,
        cot_info,
        output_path=args.output_dir + "/base_model_eval.json",
        verbose=args.verbose,
    )
    if args.verbose:
        print(f"Evaluation completed. Results saved to {args.output_dir}/eval_results.json")

    # # Implement accuracy calculation
    return results

if __name__ == "__main__":
    args = parse_args()
    # main(args)

    input_path = args.output_dir + "/base_model_eval.json"
    # Read as a json list of dictionaries
    with open(input_path, "r") as f:
        eval_outputs = json.load(f)
    if args.verbose:
        print(f"Loaded evaluation results from {input_path}")

    # Compute metrics
    eval_metrics = generation.compute_eval_metrics(eval_outputs)
    if args.verbose:
        print(f"Computed metrics: {json.dumps(eval_metrics, indent=4)}")