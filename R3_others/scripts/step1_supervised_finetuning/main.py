#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import json
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..") # for generation
sys.path.append(
    "/opt/tiger/xzh-agent/DeepSpeedExamples-master/applications/DeepSpeed-Chat/dschat"
)


from tqdm import tqdm

import argparse
import math

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from vllm import LLM, SamplingParams
import wandb

from dschat.utils.data.data_utils import create_prompt_dataset, customized_data_collator
from dschat.utils.utils import (
    print_rank_0,
    to_device,
    save_hf_format,
    set_random_seed,
    get_all_reduce_mean,
    get_optimizer_grouped_parameters,
    save_zero_three_model,
    load_hf_tokenizer,
)
from dschat.utils.ds_utils import get_train_ds_config
from dschat.utils.module.lora import (
    convert_linear_layer_to_lora,
    convert_lora_to_linear_layer,
    only_optimize_lora_parameters,
    make_model_gradient_checkpointing_compatible,
)
from dschat.utils.model.model_utils import create_hf_model, causal_lm_model_to_fp32_loss
from dschat.utils.perf import print_throughput

from dschat.utils.data.data_utils import PromptDataset
torch.serialization.add_safe_globals([PromptDataset])

import generation
import rl_util


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["Dahoas/rm-static"],
        help="Path to the training dataset. Accepted format:"
        "1) a single data path, 2) multiple datasets in the"
        "form: dataset1-path dataset2-path ...",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="2,4,4",
        help="Comma-separated list of proportions for training"
        "phase 1, 2, and 3 data. For example the split `6,2,2`"
        "will use 60%% of data for phase 1, 20%% for phase 2"
        "and 20%% for phase 3.",
    )
    parser.add_argument(
        "--sft_only_data_path",
        nargs="*",
        default=[],
        help="Path to the dataset for only using in SFT phase.",
    )
    parser.add_argument(
        "--data_output_path",
        type=str,
        default="/tmp/data_files",
        help="Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for model.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the model.",
    )
    # deepspeed features
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
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )
    ## LoRA for efficient training setting
    parser.add_argument(
        "--lora_dim",
        type=int,
        default=0,
        help="If > 0, use LoRA for efficient training.",
    )
    parser.add_argument(
        "--lora_module_name",
        type=str,
        default="decoder.layers.",
        help="The scope of LoRA.",
    )
    parser.add_argument(
        "--only_optimize_lora",
        action="store_true",
        help="Only optimize the LoRA parameters.",
    )
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help="Initial LoRA learning rate (after the potential warmup period) to use.",
    )
    ## low precision
    parser.add_argument(
        "--compute_fp32_loss",
        action="store_true",
        help="Relevant for low precision dtypes (fp16, bf16, etc.). "
        "If specified, loss is calculated in fp32.",
    )
    ## Tensorboard logging
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    parser.add_argument("--tensorboard_path", type=str, default="step1_tensorboard")
    ## Tokenizer
    parser.add_argument(
        "--add_eot_token",
        action="store_true",
        help="Add <|endoftext|> as additional special token to tokenizer",
    )
    ## Print loss
    parser.add_argument(
        "--print_loss", action="store_true", help="Prints loss at each step."
    )


    # Add the evaluation parameters
    parser.add_argument(
        "--eval_input_path",
        type=str,
        default="R3_others/data/r1_zero_instruction_eval.json",
        help="Path to the evaluation dataset. This should be a json file with the evaluation examples.",
    )
    parser.add_argument(
        "--eval_output_path",
        type=str,
        help="Filepath to save the evaluation results."
    )
    parser.add_argument(
        "--src_name",
        type=str,
        help="Source name for the evaluation dataset (e.g., 'gsm8k', 'svamp').",
    )
    parser.add_argument(
        "--max_gen_length",
        type=int,
        default=700,
        help="The maximum total sequence length for generation.",
    )


    # Add wandb arguments wandb_log, wand_entity, wandb_project, wandb_name
    parser.add_argument(
        "--wandb_log",
        type=bool,
        default=False,
        help="Enable logging to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity to use for logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="Weights & Biases project to use for logging.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="Weights & Biases run name to use for logging.",
    )

    # Optional verbosity
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose information during training.",
    )

    # Add deepspeed config arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def initialize_ds_config(args):
    ds_config = get_train_ds_config(
        offload=args.offload,
        # gradient_clipping="auto",
        dtype=args.dtype,
        stage=args.zero_stage,
        enable_tensorboard=args.enable_tensorboard,
        tb_path=args.tensorboard_path,
        tb_name="step1_model",
    )
    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    ds_config["train_batch_size"] = (
        args.per_device_train_batch_size
        * (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
        * args.gradient_accumulation_steps
    )  # overall batch size, across gradient accumulation steps and devices

    return ds_config

def initialize_train_dataloader(args, tokenizer):
    train_phase = 1
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase,
        args.seed,
        tokenizer,
        args.max_seq_len,
        sft_only_data_path=args.sft_only_data_path,
        reload=True,
    )   # eval dataset not used for SFT

    print("train length:{}".format(len(train_dataset)))

    # Create the dataloaders
    train_sampler = DistributedSampler(train_dataset) if torch.distributed.is_initialized() else RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=customized_data_collator,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
    )

    return train_dataloader

def initialize_model(args, tokenizer, ds_config, num_train_examples: int):
    # Import the model
    model = create_hf_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        tokenizer,
        ds_config,
        dropout=args.dropout,
    )

    # Set the model to the correct dtype
    if args.compute_fp32_loss:
        print_rank_0(
            f"Using model {model.__class__.__name__} with loss in fp32",
            args.global_rank,
        )
        causal_lm_model_to_fp32_loss(model)

    # Set lora model
    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(
            model, args.lora_module_name, args.lora_dim
        )
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)
            model = make_model_gradient_checkpointing_compatible(model)

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate
    )

    # Initialize the model, optimizer, and learning rate scheduler
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(
        optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95)
    )

    num_update_steps_per_epoch = math.ceil(
        num_train_examples / args.gradient_accumulation_steps
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model, optimizer, lr_scheduler

def initialize_vllm(args):
    # # Initialize the vLLM model
    # dtype = {
    #     "fp16": "float16",
    #     "bf16": "bfloat16",
    #     "fp32": "float32",
    # }[args.dtype]   # convert dtype string to vLLM dtype

    # vllm_model = LLM(model=args.model_name_or_path,
    #                  dtype=dtype,
    #                  enable_prefix_caching=True,
    #                  gpu_memory_utilization=0.5
    #             )
    # if args.verbose:
    #     print(f"Initialized vLLM model from {args.model_name_or_path}")
    vllm_model = None

    # Initialize the sampling parameters
    eval_sampling_params = generation.init_sampling_params(
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        max_tokens=args.max_gen_length,
        seed=args.seed,
    )
    if args.verbose:
        print("Initialized sampling parameters for evaluation")

    # Run evaluation on the model, saving the results to the output directory
    cot_info = rl_util.prepare_cot_info(args.src_name)  # Assuming "nl" is the source name for the evaluation
    
    return vllm_model, eval_sampling_params, cot_info

def main():
    args = parse_args()

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    '''
    Initialize the device and backend
    '''
    print("DDP: ", torch.distributed.is_initialized())

    if not torch.distributed.is_initialized():
        device = torch.device(deepspeed.get_accelerator().device_name())
    else:
        deepspeed.get_accelerator().set_device(args.local_rank)
        device = torch.device(deepspeed.get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()
    args.global_rank = (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0)
    if args.verbose:
        print("Device = ", device)


    '''
    Initialize the wandb logging
    '''
    if args.wandb_log:
        print("Project = ", args.wandb_project)
        print("Entity = ", args.wandb_entity)
        print("Run name = ", args.wandb_run_name)

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
        )
        wandb.config.update(args)
        if args.verbose:
            print("Initialized wandb logging")


    '''
    Initialize the DeepSpeed configuration
    '''
    ds_config = initialize_ds_config(args)
    if args.verbose:
        print("Initialized DeepSpeed config: ", ds_config)
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


    '''
    Load the tokenizer
    '''
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = (
        args.end_of_conversation_token if args.add_eot_token else None
    )
    tokenizer = load_hf_tokenizer(
        args.model_name_or_path,
        fast_tokenizer=True,
        add_special_tokens=additional_special_tokens,
    )


    '''
    Initialize the training dataloader
    '''
    train_dataloader = initialize_train_dataloader(args, tokenizer)
    if args.verbose:
        print(
            f"Initialized training dataloader with {len(train_dataloader)} batches, "
            f"batch size {args.per_device_train_batch_size}, "
            f"total batch size {ds_config['train_batch_size']}"
        )
    

    '''
    Initialize the model, optimizer, and learning rate scheduler
    '''
    model, optimizer, lr_scheduler = initialize_model(
        args,
        tokenizer,
        ds_config,
        num_train_examples=len(train_dataloader),  # will be set later
    )


    '''
    Load eval examples
    '''
    eval_examples: list[generation.EvalExample] = json.load(open(args.eval_input_path, "r"))
    if args.verbose:
        print(f"Loaded {len(eval_examples)} evaluation examples from {args.eval_input_path}")


    '''
    Initialize the vLLM model and sampling parameters
    '''
    vllm_model, eval_sampling_params, cot_info = initialize_vllm(args)
    if args.verbose:
        print(f"Prepared COT info for source name {args.src_name}")

    '''
    This function will be called at the start of each epoch to evaluate the model (+ a final at the end of training)    
    '''
    def eval_subroutine(model):
        # Save the model to the output directory
        if args.output_dir is not None and args.global_rank == 0:
            model = convert_lora_to_linear_layer(model)
            save_hf_format(model, tokenizer, args)
            if args.verbose:
                print(f"Saved the model to the output directory {args.output_dir}")

        # Load a vllm model for evaluation with those saved weights
        dtype = {
            "fp16": "float16",
            "bf16": "bfloat16",
            "fp32": "float32",
        }[args.dtype]   # convert dtype string to vLLM dtype

        vllm_model = LLM(model=args.output_dir,
                        dtype=dtype,
                        enable_prefix_caching=True,
                        gpu_memory_utilization=0.5
                    )
        if args.verbose:
            print(f"Initialized vLLM model from {args.output_dir}")
    
        # Run evaluation on the vLLM model
        eval_outputs = generation.evaluate_vllm(
            vllm_model,
            eval_examples,
            eval_sampling_params,
            cot_info,
            output_path=args.eval_output_path,
            verbose=args.verbose,
        )
        eval_metrics = generation.compute_eval_metrics(eval_outputs)
        if args.verbose:
            print(f"Evaluation completed. Results: {json.dumps(eval_metrics, indent=4)}")
        
        if args.wandb_log:
            wandb.log(
                {
                    "eval/answer_accuracy": eval_metrics["answer_accuracy"],
                    "eval/format_accuracy": eval_metrics["format_accuracy"],
                    "eval/epoch": epoch,
                }
            )
        return

    '''
    SFT Train loop
    '''
    for epoch in range(args.num_train_epochs):
        '''
        Evaluate the vllm model on the eval examples
        '''
        eval_subroutine(model)

        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank,
        )
        model.train()
        import time

        epoch_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for step, batch in epoch_bar:
            start = time.time()

            # Take a gradient step
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            if args.print_loss:
                # print(
                #     f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
                # )
                epoch_bar.set_description(
                    f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
                )
            model.backward(loss)
            model.step()  # incorporates optimizer/learning rate

            end = time.time()
            if torch.distributed.get_rank() == 0:
                print_throughput(model.model, args, end - start, args.global_rank)

            if args.wandb_log:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/epoch": epoch + (step + 1) / len(train_dataloader),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=step + epoch * len(train_dataloader),
                )

        model.tput_timer.update_epoch_count()

    # Final evaluation after training
    print_rank_0("Final evaluation after training ...", args.global_rank)
    eval_subroutine(model)

    # Save the final model
    if args.output_dir is not None:
        print_rank_0("saving the final model ...", args.global_rank)
        model = convert_lora_to_linear_layer(model)

        if args.global_rank == 0:
            print_rank_0("saving the final model and tokenizer ...", args.global_rank)
            save_hf_format(model, tokenizer, args)

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(
                model, args.global_rank, args.output_dir, zero_stage=args.zero_stage
            )

if __name__ == "__main__":
    main()
