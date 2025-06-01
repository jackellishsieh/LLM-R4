import torch
from vllm import LLM, SamplingParams, RequestOutput
from typing import Callable, TypedDict

import util

class EvalExample(TypedDict):       # what is required for each evaluation example
    item_id: str
    question: str
    answer_value: str

def init_vllm(
    model_name_or_path: str,
    gpu_memory_utilization: float = 0.5,
):
    """Initializes a vLLM model with the given model name or path and GPU memory utilization (on the same device).

    Args:
        model_name_or_path (str): The base model name or path to use for the vLLM model.
        gpu_memory_utilization (float, optional): The fraction of GPU memory to utilize for the model. Defaults to 0.5.
    """
    vllm_model = LLM(model=model_name_or_path,
                     dtype=torch.bfloat16,
                    enable_prefix_caching=True,
                    gpu_memory_utilization=gpu_memory_utilization,)
    return vllm_model
    
def init_sampling_params(
    tokenizer,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 700,
) -> SamplingParams:
    """Initializes the sampling parameters for the vLLM model.

    Args:
        tokenizer: The tokenizer to use for the model. Used to set the stop and pad tokens.
        temperature (float, optional): The temperature to use for sampling. Defaults to 0.0.
        top_p (float, optional): The top-p value to use for sampling. Defaults to 1.0.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 700.

    Returns:
        SamplingParams: The sampling parameters for the vLLM model.
    """
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_tokens=[tokenizer.eos_token_id],  # use the end of sentence token as the stop token. Should be excluded, as it is a special token
    )


def evaluate_vllm(
    vllm_model: LLM,
    eval_examples: list[EvalExample],
    eval_sampling_params: SamplingParams,
    cot_info: util.CotInfo,
    append_cot_trigger: bool,
    output_path: str | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Given a vllm model and a list of evaluation examples, this function generates rationales from the vllm and grades them according to cot_info.
    Returns a list of dictionaries, containing the original evaluation example, the generated rationale, the generated answers, and the feedback.

    Args:
        vllm_model (LLM): The vllm model to use for generation.
        eval_examples (list[EvalExample]): A list of evaluation examples, each containing at minimum an item_id, question, and answer_value.
        eval_sampling_params (SamplingParams): The sampling parameters to use for generation.
        cot_info (util.CotInfo): CoT info necessary for producing prompts and grading answers against ground truth
        append_cot_trigger (bool): Whether to append the CoT trigger to the question-filled prompt.

        output_path (str | None, optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        list[dict]: A list of dictionaries, each containing the original evaluation example, the generated rationale, the generated answers, and the feedback.
    """
    # Construct the prompt for each example
    prompts = [
        f"{cot_info["instruction"]}{example["question"]}" + (cot_info["cot_trigger"] if append_cot_trigger else "")
        for example in eval_examples
    ]   # append the question and appending the CoT trigger if required

    # Sample completions, in the same order as the prompts
    outputs: list[RequestOutput] = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    for example, output in zip(eval_examples, outputs):
        src_name = example["item_id"].split("_")[0] # extract the source name from the item_id
        engine = "nl"       # default engine is natural language

        result = example.copy()

        # post-process the target answer to a float
        result["target_value"] = cot_info["post_process_final_value_fn_mapper"][src_name](example["answer_value"])
        
        # set the predicted CoT
        result["prediction_cot"] = output.outputs[0].text.strip()

        # post-process the predicted question answer to a float
        try:
            # with timeout(seconds=TIMEOUT):    # timeout is irrelevant, as we are not running any code
            result["prediction_value"] = cot_info["post_process_completed_question_value_fn_mapper"][(engine, src_name)](result["prediction_cot"])
        except:
            result["prediction_value"] = None

        # compare the predicted answer with the target answer
        result["is_correct"] = cot_info["compare_value_fn_mapper"][src_name](result["prediction_value"], result["target_value"]) if result["prediction_value"] is not None else False

        results.append(result)

    # Save the results to a json file if output_path is provided
    if output_path is not None:
        import json
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

    # Return the results
    return results