import torch
from vllm import LLM, SamplingParams, RequestOutput
from typing import Callable, TypedDict, Optional

import rl_util
import constants

class EvalExample(TypedDict):  # what is required at a minimum for each evaluation example
    item_id: str
    question: str
    answer_value: str
    # additional keys can be added, e.g., difficulty, answer_cot, etc.

class EvalOutput(TypedDict):  # what is returned for each evaluation example
    item_id: str
    question: str
    answer_value: str                   # same values as EvalExample
    # additional keys can be added, e.g., difficulty, answer_cot, etc.

    target_value: float                 # the target answer value, post-processed to a float
    prompt: str                         # the prompt used for generation (i.e., instructions + question)

    prediction_cot: str                 # the generated COT as a string
    prediction_value: Optional[float]   # the predicted answer value, post-processed to a float. None if an answer could not be extracted from the generated CoT
    is_correct: bool                    # whether the predicted answer is correct or not, by comparing it with the target answer


def init_vllm(
    model_name_or_path: str,
    gpu_memory_utilization: float = 0.5,
):
    """Initializes a vLLM model with the given model name or path and GPU memory utilization (on the same device).

    Args:
        model_name_or_path (str): The base model name or path to use for the vLLM model.
        gpu_memory_utilization (float, optional): The fraction of GPU memory to utilize for the model. Defaults to 0.5.
    """
    vllm_model = LLM(
        model=model_name_or_path,
        dtype=torch.bfloat16,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    return vllm_model


def init_sampling_params(
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    max_tokens: int = 700,
) -> SamplingParams:
    """Initializes the sampling parameters for the vLLM model.

    Args:
        temperature (float, optional): The temperature to use for sampling. Defaults to 0.0.
        top_p (float, optional): The top-p value to use for sampling. Defaults to 1.0.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 700.

    Returns:
        SamplingParams: The sampling parameters for the vLLM model.
    """
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop=[constants.r1_zero_instruction_stop],  # the end of the answer
        include_stop_str_in_output=True,  # include the stop string in the output
    )


def evaluate_vllm(
    vllm_model: LLM,
    eval_examples: list[EvalExample],
    eval_sampling_params: SamplingParams,
    cot_info: rl_util.CotInfo,
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> list[EvalOutput]:
    """Given a vllm model and a list of evaluation examples, this function generates rationales from the vllm and grades them according to cot_info.
    Returns a list of dictionaries, containing the original evaluation example, the generated rationale, the generated answers, and the feedback.

    Args:
        vllm_model (LLM): The vllm model to use for generation.
        eval_examples (list[EvalExample]): A list of evaluation examples, each containing at minimum an item_id, question, and answer_value.
        eval_sampling_params (SamplingParams): The sampling parameters to use for generation.
        cot_info (rl_util.CotInfo): CoT info necessary for producing prompts and grading answers against ground truth

        output_path (str | Optional[str]): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        list[EvalOutput]: A list of dictionaries, each containing the original evaluation example, the generated rationale, the generated answers, and the feedback.
            See EvalOutput for the structure of each dictionary.
    """
    # Construct the prompt for each example
    prompts = [cot_info["question_to_prompt"](example["question"]) for example in eval_examples]

    # Sample completions, in the same order as the prompts
    outputs: list[RequestOutput] = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    for example, prompt, output in zip(eval_examples, prompts, outputs):
        # Unused for now, since no code and CoT info has been reworked for our purposes
        # src_name = example["item_id"].split("_")[0]           # extract the source name from the item_id.
        # engine = "nl"  # default engine is natural language

        # copy the example and the prompt
        result = example.copy()
        result["prompt"] = prompt

        # post-process the target answer to a float
        result["target_value"] = cot_info["answer_to_value"](example["answer_value"])

        # set the predicted CoT
        result["prediction_cot"] = output.outputs[0].text.strip()

        # post-process the predicted question answer to a float
        try:
            # with timeout(seconds=TIMEOUT):    # timeout is irrelevant, as we are not running any code
            result["prediction_value"] = cot_info["answer_to_value"](
                cot_info["cot_to_answer"](result["prediction_cot"])
            )
        except:
            result["prediction_value"] = None

        # compare the predicted answer with the target answer
        result["is_correct"] = (
            cot_info["compare_values"](result["prediction_value"], result["target_value"])
            if result["prediction_value"] is not None
            else False
        )

        results.append(result)

    # Save the results to a json file if output_path is provided
    if output_path is not None:
        import json

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

    # Return the results
    return results
