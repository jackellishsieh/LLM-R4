from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import pad_across_processes, broadcast
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
from datetime import timedelta
from functools import partial
import json
import os
import random
from src.python_engine import run_python_code, process_code
# from src.wolfram_engine import run_wolfram_code
from src.python_stdout_engine import run_python_stdout_code, compare_both_string_and_number_format, number_it
from src.utils import set_seed, is_numeric, timeout, discount_cumsum, do_gather
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from trl import AutoModelForCausalLMWithValueHead
from trl.core import masked_mean, masked_var, masked_whiten
from src.modeling_rl import AutoModelForCausalLMWithValueModel
import numpy as np
import wandb
import shutil
from prettytable import PrettyTable

def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor, gather: bool = True) -> torch.Tensor:
    """
    Taken from https://github.com/huggingface/trl/blob/9410874787db47ce0864d8dc16d91a415c6f7406/trl/core.py
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)

    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy

tqdm = partial(tqdm, ncols=0, leave=False)
TIMEOUT = 2

def prepare_deepspeed_ref_model(model):
    # Adopted from: https://github.com/huggingface/trl/blob/02f5c1d8cee73045c837d01d7f1577a57779b035/trl/trainer/ppo_trainer.py#L1399
    import deepspeed

    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepspeed_plugin.deepspeed_config
    if model is not None:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                    }
                )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


def prepare_cot_info(src_name):
    assert src_name in ['gsm8k', 'svamp']

    # default for common datasets
    instruction = 'Question:\n'
    cot_trigger = '\nAnswer reasoning:\n'
    answer_trigger = '\nTherefore, the answer is: '


    if src_name in ['gsm8k', 'svamp']:
        # over-write for the simple datasets
        instruction = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n'
        cot_trigger = '\n\n### Response:'
        answer_trigger = '\n####'


    post_process_final_answer_fn_mapper = {
        'gsm8k': lambda x: float(x.replace(',', '').strip()),
        'svamp': lambda x: float(x.replace(',', '').strip()),
    }
    post_process_completed_question_answer_fn_mapper = {
        ('python', 'gsm8k'): lambda completed_question_answer: float(run_python_code(code_gen=completed_question_answer.split(cot_trigger)[-1].strip())),
        ('python', 'svamp'): lambda completed_question_answer: float(run_python_code(code_gen=completed_question_answer.split(cot_trigger)[-1].strip())),
        
        ('nl', 'gsm8k'): lambda completed_question_answer: float(completed_question_answer.split(cot_trigger)[-1].split(answer_trigger)[-1].strip()),
        ('nl', 'svamp'): lambda completed_question_answer: float(completed_question_answer.split(cot_trigger)[-1].split(answer_trigger)[-1].strip()),
    }
    compare_answer_fn_mapper = {
        'gsm8k': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
        'svamp': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    }

    return {
        'instruction': instruction,
        'cot_trigger': cot_trigger,
        'answer_trigger': answer_trigger,
        'post_process_final_answer_fn_mapper': post_process_final_answer_fn_mapper,
        'post_process_completed_question_answer_fn_mapper': post_process_completed_question_answer_fn_mapper,
        'compare_answer_fn_mapper': compare_answer_fn_mapper,
    }


def prepare_datasets_and_data_loaders(args, tokenizer, current_difficulty_score_range=None):
    with accelerator.main_process_first():

        # Load full train dataset only once and cache it as static attribute
        if not hasattr(prepare_datasets_and_data_loaders, '_full_raw_train_dataset'):
            accelerator.print(f"Loading full training dataset from {args['train_file']}...")
            if args['train_file'].rstrip('/').endswith('_cache'):
                prepare_datasets_and_data_loaders._full_raw_train_dataset = load_from_disk(args['train_file'])
            else:
                prepare_datasets_and_data_loaders._full_raw_train_dataset = Dataset.from_list(
                    json.load(open(args['train_file'], 'r')))
            accelerator.print('Full raw training data loaded.')

        current_raw_train_dataset = prepare_datasets_and_data_loaders._full_raw_train_dataset

        # Filter by difficulty_score if specified
        if current_difficulty_score_range:
            min_score, max_score = current_difficulty_score_range
            accelerator.print(f"Filtering training data for difficulty_score between {min_score} and {max_score}")
            current_raw_train_dataset = current_raw_train_dataset.filter(
                lambda x: min_score <= x.get('difficulty_score', -1) <= max_score,
                num_proc=None,
                load_from_cache_file=False
            )
            accelerator.print(f"Filtered training data size: {len(current_raw_train_dataset)}")

        # Prepare test dataset (load or from disk)
        if args['test_file'].rstrip('/').endswith('_cache'):
            test_dataset = load_from_disk(args['test_file'])
        else:
            test_dataset = Dataset.from_list(json.load(open(args['test_file'], 'r')))

        raw_dataset = DatasetDict({
            'train': current_raw_train_dataset,
            'test': test_dataset
        })
        accelerator.print('Raw data for current stage:', raw_dataset)

        # Get cot info from the first train example's item_id prefix
        src_name = raw_dataset['train']['item_id'][0].split('_')[0]
        cot_info = prepare_cot_info(src_name)
        instruction = cot_info['instruction']
        cot_trigger = cot_info['cot_trigger']
        answer_trigger = cot_info['answer_trigger']

        # Define the tokenize function here so it has access to args, tokenizer, cot_info, src_name
        def tokenize_fn(batch):
            assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
            new_batch = defaultdict(list)
            all_keys = list(batch.keys())
            for item_values in zip(*(batch[k] for k in all_keys)):
                item = {k: item_values[i] for i, k in enumerate(all_keys)}
                item_id = item['item_id']
                question = item['question']
                answer_value = item['answer_value']
                answer_cot = item.get('answer_cot', None)

                if answer_value is not None:
                    answer_value = answer_value.strip()

                if answer_cot:
                    if args['engine'] == 'nl' and src_name in ['gsm8k']:
                        answer_cot += f'{answer_trigger} {answer_value}'

                    input_text = f'{instruction}{question}'
                    output_text = f'{answer_cot}'
                    prefix_text = f'{instruction}{question}'

                    if answer_cot.startswith('def'):
                        question_1 = question.replace("\n\n### Response:", "")
                        if src_name in ['gsm8k', 'svamp'] and args['engine'] == 'python':
                            prefix_text += f'def solution():\n    """{question_1}"""\n'

                else:
                    input_text = f'{instruction}{question}{cot_trigger}'
                    output_text = f'{answer_cot}'  # Possibly None or empty string here?
                    prefix_text = f'{instruction}{question}{cot_trigger}'

                    if src_name in ['gsm8k', 'svamp'] and args['engine'] == 'python':
                        prefix_text += f'def solution():\n    """{question}"""\n'

                input_encode = tokenizer(input_text, add_special_tokens=False)
                output_encode = tokenizer(output_text if output_text else '', add_special_tokens=False)
                prefix_encode = tokenizer(prefix_text, add_special_tokens=False)

                input_ids = input_encode['input_ids'] + output_encode['input_ids'] + [tokenizer.eos_token_id]
                labels = [-100] * len(input_encode['input_ids']) + output_encode['input_ids'] + [tokenizer.eos_token_id]
                attention_mask = [1] * len(input_ids)
                prefix = prefix_encode['input_ids']
                prefix_attention_mask = prefix_encode['attention_mask']

                # Truncation
                input_ids = input_ids[:args['max_input_length']]
                labels = labels[:args['max_input_length']]
                attention_mask = attention_mask[:args['max_input_length']]
                prefix = prefix[:args['max_input_length']]
                prefix_attention_mask = prefix_attention_mask[:args['max_input_length']]

                # Append to batch output
                new_batch['input_ids'].append(input_ids)
                new_batch['labels'].append(labels)
                new_batch['attention_mask'].append(attention_mask)
                new_batch['prefix'].append(prefix)
                new_batch['prefix_attention_mask'].append(prefix_attention_mask)

                new_batch['item_id'].append(item_id)
                new_batch['question'].append(question)
                new_batch['prefix_text'].append(prefix_text)
                new_batch['answer_cot'].append(answer_cot)
                new_batch['answer_value'].append(answer_value)

            return new_batch

        # Tokenize datasets (train filtered by difficulty, test constant)
        tokenized_dataset = DatasetDict({
            mode: dataset.map(
                tokenize_fn, batched=True,
                remove_columns=dataset.column_names,
                num_proc=None, load_from_cache_file=True, keep_in_memory=False,
            ) for mode, dataset in raw_dataset.items()
        })
        accelerator.print('Processed data for current stage:', tokenized_dataset)

        if accelerator.is_main_process and args['wandb_log']:
            wandb_config_update_dict = {
                "src_name": src_name,
                "instruction": instruction,
                "cot_trigger": cot_trigger,
                "answer_trigger": answer_trigger,
            }
            if current_difficulty_score_range:
                wandb_config_update_dict['current_difficulty_score_range'] = current_difficulty_score_range
            wandb.config.update(wandb_config_update_dict)

    train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=args['batch_size'],
                                  num_workers=args['num_workers'], pin_memory=True,
                                  collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))

    if not hasattr(prepare_datasets_and_data_loaders, '_cached_test_dataloader'):
        prepare_datasets_and_data_loaders._cached_test_dataloader = DataLoader(
            tokenized_dataset['test'], shuffle=False, batch_size=args['eval_batch_size'],
            num_workers=args['num_workers'], pin_memory=True,
            collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))

    return (tokenized_dataset['train'], train_dataloader), \
           (tokenized_dataset['test'], prepare_datasets_and_data_loaders._cached_test_dataloader), cot_info


def do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths=None):
    os.makedirs(save_path, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_path, is_main_process=accelerator.is_main_process,
                                    save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
    tokenizer.save_pretrained(save_path)
    if accelerator.is_main_process and most_recent_ckpts_paths is not None:
        most_recent_ckpts_paths.append(save_path)
        if args['keep_num_ckpt'] is not None and len(most_recent_ckpts_paths) > args['keep_num_ckpt']:
            ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
            # os.remove(ckpt_to_be_removed)
            shutil.rmtree(ckpt_to_be_removed)


def allgather(tensor, group=None):
    """smantic sugar for torch.distributed.all_gather.

    Args:
        tensor: (bs, ...)
        group:

    Returns:
        All gathered tensor (world_size, bs, ...)
    """
    if group is None:
        group = torch.distributed.group.WORLD
    allgather_tensor = [torch.zeros_like(tensor) for _ in range(group.size())]
    torch.distributed.all_gather(allgather_tensor, tensor, group=group)
    allgather_tensor = torch.stack(allgather_tensor, dim=0)
    return allgather_tensor


def allgather_masked_whiten(values, mask, shift_mean=False):
    """Whiten values with all-gathered masked values.

    Args:
        values: (bs, ...)
        mask: (bs, ...)
        shift_mean: bool

    Returns:
        whitened values, (bs, ...)
    """
    allgather_values = allgather(values)  # (n_proc, bs, ...)
    # accelerator.print(f'allgather_values {allgather_values.shape}, {allgather_values[0, 0:3]}')

    allgather_mask = allgather(mask)  # (n_proc, bs, ...)
    # accelerator.print(f'allgather_mask {allgather_mask.shape}, {allgather_mask[0, 0:3]}')

    global_mean = masked_mean(allgather_values, allgather_mask)
    global_var = masked_var(allgather_values, allgather_mask)
    whitened = (values - global_mean) * torch.rsqrt(global_var + 1e-8)
    if shift_mean:
        whitened += global_mean
    return whitened


def logging_values(ids, vpreds, rets, advs, old_vpreds, rews, score_rews, masks, tokenizer):
    

    get_str_digits = lambda x: str(round(x.item(), 3))

    for tmp_i in range(ids.size(0)):
        mk_ids = torch.masked_select(ids[tmp_i], masks[tmp_i].bool())  # (valid_resp_len,)
        mk_vpreds = torch.masked_select(vpreds[tmp_i], masks[tmp_i].bool())  # (valid_resp_len,)
        mk_rets = torch.masked_select(rets[tmp_i], masks[tmp_i].bool())  # (valid_resp_len,)
        mk_advs = torch.masked_select(advs[tmp_i], masks[tmp_i].bool())  # (valid_resp_len,)
        mk_old_vpreds = torch.masked_select(old_vpreds[tmp_i], masks[tmp_i].bool())  # (valid_resp_len,)
        mk_rews = torch.masked_select(rews[tmp_i], masks[tmp_i].bool())  # (valid_resp_len,)
        mk_score_rews = torch.masked_select(score_rews[tmp_i], masks[tmp_i].bool())  # (valid_resp_len,)

        accelerator.print(tokenizer.decode(mk_ids))

        table = PrettyTable()
        table.field_names = ["a_t", "V(s_t)", "Ret(s_t)", "A(s_t,a_t)", "old_V(s_t)", "r(s_t,a_t)", "score_r(s_t,a_t)"]
        for tmp_j in range(mk_ids.nelement()):
            a_t = tokenizer.decode(mk_ids[tmp_j])
            table.add_row([
                a_t if a_t != '\n' else '\\n', 
                get_str_digits(mk_vpreds[tmp_j]), 
                get_str_digits(mk_rets[tmp_j]), 
                get_str_digits(mk_advs[tmp_j]), 
                get_str_digits(mk_old_vpreds[tmp_j]),
                get_str_digits(mk_rews[tmp_j]),
                get_str_digits(mk_score_rews[tmp_j])
            ])
        table_str = table.get_string()
        accelerator.print(table_str)
        accelerator.print('\n')


def rollout(args, model, ref_model, tokenizer, query_tensors, query_tensors_attention_mask, answer_values, src_name, cot_info):
    model.eval()
    with torch.no_grad():
        gen_output = accelerator.unwrap_model(model).generate(
            input_ids=query_tensors,
            attention_mask=query_tensors_attention_mask,
            top_k=0.0, top_p=1.0,
            do_sample=True,
            # output_scores=True,
            # return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            # bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args['max_gen_length'],
        )
        # completed_tensors, logits_per_steps = gen_output[0], gen_output[1]
        completed_tensors = gen_output
        completed_tensors = pad_across_processes(completed_tensors, dim=1, pad_index=tokenizer.pad_token_id, pad_first=False)

    # Evaluate score
    post_process_final_answer_fn_mapper = cot_info['post_process_final_answer_fn_mapper']
    compare_answer_fn_mapper = cot_info['compare_answer_fn_mapper']
    post_process_completed_question_answer_fn_mapper = cot_info['post_process_completed_question_answer_fn_mapper']
    correctness = []
    for completed_ids, a_value in zip(completed_tensors, answer_values):
        completed_text = tokenizer.decode(completed_ids.cpu().numpy().tolist(), skip_special_tokens=True)
        try:
            with timeout(TIMEOUT):
                extracted_ans = post_process_completed_question_answer_fn_mapper[(args['engine'], src_name)](completed_text.strip())
        except:
            extracted_ans = None

        target_value = post_process_final_answer_fn_mapper[src_name](a_value)
        if extracted_ans is not None:
            if compare_answer_fn_mapper[src_name](extracted_ans, target_value):
                is_correct = 1
            else:
                is_correct = 0.2
        else:
            is_correct = 0
        correctness.append(is_correct)

    model_input_ids = completed_tensors
    model_attention_mask = (completed_tensors != tokenizer.pad_token_id)
    with torch.no_grad():
        # Get old logprob and val
        lm_logits, _dummy2, val = model(input_ids=model_input_ids, attention_mask=model_attention_mask)
        old_logprob = logprobs_from_logits(lm_logits[:, :-1, :], labels=model_input_ids[:, 1:])  # (bs, seqlen-1)

        # Get the ref model logprob
        ref_logprob = None
        if ref_model is not None:
            ref_lm_logits, _dummy2, _dummy3 = ref_model(input_ids=model_input_ids, attention_mask=model_attention_mask)
            ref_logprob = logprobs_from_logits(ref_lm_logits[:, :-1, :], labels=model_input_ids[:, 1:])  # (bs, seqlen-1)

    # Masking the last prompt token up untils the token before eos_token_id
    prompt_len = query_tensors.size(1)
    mask = torch.zeros_like(model_input_ids, dtype=torch.bool)  # (bs, seqlen)
    mask[:, query_tensors.size(1) - 1: -1] = 1
    score_rew = np.zeros(mask.shape)  # (bs, seqlen)
    score_rew[:, -2] = np.array(correctness)
    nonzero = (model_input_ids == tokenizer.eos_token_id).nonzero()
    for (bidx, tidx) in nonzero:
        mask[bidx][tidx:] = 0
        score_rew[bidx][tidx:] = 0
        score_rew[bidx][tidx - 1] = correctness[bidx]

    # Make the kl reward and the full reward
    kl_rew = None
    rew = score_rew
    if ref_logprob is not None:
        kl = old_logprob - ref_logprob  # (bs, seqlen-1)
        kl = (kl.float() * mask[:, :-1]).cpu().numpy()
        kl_rew = np.zeros(mask.shape)  # (bs, seqlen)
        kl_rew[:, :-1] = -kl # NOTE the minus sign
 
        kl_coef = args["kl_coef"]
        rew = score_rew + kl_coef * kl_rew

    # Process val ret adv logprob
    val = (val.float() * mask).cpu().numpy()
    gamma = args["gamma"]
    lam = args["lam"]
    # ret = np.zeros_like(rew)
    adv = np.zeros_like(rew)
    for i in range(len(rew)):
        cur_rew, cur_val = rew[i], val[i]
        cur_delta = -cur_val[:-1] + cur_rew[:-1] + gamma * cur_val[1:]
        cur_adv = discount_cumsum(cur_delta, discount=gamma * lam)
        cur_adv[:prompt_len - 1] = 0
        adv[i][:-1] = cur_adv

    # lambda_return = GAE + values
    ret = adv + val  # (bs, seqlen)

    rew = torch.tensor(rew, device=mask.device, dtype=old_logprob.dtype) * mask
    score_rew = torch.tensor(score_rew, device=mask.device, dtype=old_logprob.dtype) * mask
    if kl_rew is not None:
        kl_rew = torch.tensor(kl_rew, device=mask.device, dtype=old_logprob.dtype) * mask
    ret = torch.tensor(ret, device=mask.device, dtype=old_logprob.dtype) * mask
    val = torch.tensor(val, device=mask.device, dtype=old_logprob.dtype) * mask
    adv = torch.tensor(adv, device=mask.device, dtype=old_logprob.dtype) * mask
    old_logprob = old_logprob * mask[:, :-1]

    model.train()
    return model_input_ids, model_attention_mask, mask, rew, score_rew, kl_rew, ret, correctness, val, old_logprob, ref_logprob, adv


def train_one_epoch(args, model, ref_model, train_dataset, train_dataloader, optimizer, scheduler, tokenizer,
                    global_step, global_iter_num, test_dataset, test_dataloader, cot_info,
                    prefix, epoch, best_eval_log_dict, most_recent_ckpts_paths):
    max_epoch = args['n_epochs']
    model_dir = args['model_dir']
    clip_grad_norm = args.get('clip_grad_norm', None)
    vf_coef = args['vf_coef']
    evaluating_step_freq = args.get('evaluating_step_freq', None)
    logging_step_freq = args.get('logging_step_freq', None)
    saving_step_freq = args.get('saving_step_freq', None)

    model.train()
    epoch_result_dict = defaultdict(list)
    for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                           disable=not accelerator.is_main_process, desc='Train Loop'):
        result_dict = defaultdict(list)

        # Do rollout first
        model.eval()
        model_input_ids, model_attention_mask, mask, rew, score_rew, kl_rew, ret, correctness, val, old_logprob, ref_logprob, adv = rollout(
            args, model, ref_model, tokenizer,
            query_tensors=batch['ppo_forward_kwargs']['query_tensors'],
            query_tensors_attention_mask=batch['ppo_forward_kwargs']['query_tensors_attention_mask'],
            answer_values=batch['ppo_forward_kwargs']['answer_values'],
            src_name=train_dataset[0]['item_id'].split('_')[0],
            cot_info=cot_info,
        )
        model.train()

        # printing sample sequence
        if args['logging_seq_str_step_freq'] and global_step % args['logging_seq_str_step_freq'] == 0:
            if accelerator.is_main_process:
                accelerator.print(f'\n---\nglobal_step: {global_step}')
                for tmp_i in range(model_input_ids.size(0)):
                    sequence_str = tokenizer.decode(model_input_ids[tmp_i])
                    accelerator.print('---')
                    accelerator.print(sequence_str)
                    accelerator.print(f'correctness: {correctness[tmp_i]}')
                accelerator.print('---')    

        # preprocess
        raw_adv = adv
        if args['adv_whitening'] == 'global':
            adv = allgather_masked_whiten(adv, mask) # (mini_bs, seqlen)
        elif args['adv_whitening'] == 'local':
            adv = masked_whiten(adv, mask)

        batch_size_per_gpu = len(batch['ppo_forward_kwargs']['query'])
        mini_batch_size_per_gpu = args["mini_batch_size"]
        ppo_epochs = args["ppo_epochs"]
        train_stats = {}
        for _ in range(ppo_epochs):
            perms = torch.randperm(batch_size_per_gpu)
            for mini_idx in range(0, len(perms), mini_batch_size_per_gpu):
                b_inds = perms[mini_idx: mini_idx + mini_batch_size_per_gpu]
                # Subset to batch
                cur_val = val[b_inds].contiguous()  # mini_bs x seqlen
                cur_old_logprob = old_logprob[b_inds].contiguous()  # mini_bs x seqlen
                cur_mask = mask[b_inds].contiguous()  # mini_bs x seqlen
                cur_rew = rew[b_inds].contiguous()  # mini_bs x seqlen
                cur_score_rew = score_rew[b_inds].contiguous() # mini_bs x seqlen
                cur_kl_rew = None if kl_rew is None else kl_rew[b_inds].contiguous()  # mini_bs x seqlen
                cur_ret = ret[b_inds].contiguous()  # mini_bs x seqlen
                cur_adv = adv[b_inds].contiguous()  # mini_bs x seqlen
                cur_raw_adv = raw_adv[b_inds].contiguous()  # mini_bs x seqlen
                cur_model_input_ids = model_input_ids[b_inds].contiguous()  # mini_bs x seqlen
                cur_model_attention_mask = model_attention_mask[b_inds].contiguous()  # mini_bs x seqlen
                
                resp_len_per_sample = torch.clamp(torch.sum(cur_mask, dim=1), min=1.0)  # (mini_bs,)
                cur_query_mask = torch.logical_xor(cur_mask, cur_model_attention_mask)  # (mini_bs, seqlen)
                query_len_per_sample = torch.clamp(torch.sum(cur_query_mask, dim=1), min=1.0)  # (mini_bs,)

                # Preprocess advantage and get metrics  
                cur_mask = cur_mask.type(cur_adv.dtype).contiguous()
                mean_adv, var_adv = masked_mean(cur_adv, cur_mask), masked_var(cur_adv, cur_mask)

                # Forward current model
                model.eval()
                lm_logits, _, vpreds = model(input_ids=cur_model_input_ids, attention_mask=cur_model_attention_mask)
                logprob = logprobs_from_logits(lm_logits[:, :-1, :], cur_model_input_ids[:, 1:])  # (mini_bs, seqlen-1)

                # logging values
                if args['logging_values_step_freq'] is not None and global_step % args['logging_values_step_freq'] == 0:
                    # it's fine that the value logging occurs multiple times in the inner most loop.
                    if accelerator.is_main_process:
                        accelerator.print(f'\n---\nglobal_step: {global_step}')
                        logging_values(cur_model_input_ids, vpreds, cur_ret, cur_raw_adv, cur_val, cur_rew, cur_score_rew, cur_mask, tokenizer)

                # Compute losses
                loss = 0

                # policy gradient loss
                ratio = torch.exp(logprob - cur_old_logprob)
                pg_losses = -cur_adv[:, :-1] * ratio
                pg_losses2 = -cur_adv[:, :-1] * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                pg_loss = ((torch.max(pg_losses, pg_losses2) * cur_mask[:, :-1]).sum(dim=-1) / resp_len_per_sample).mean()

                # value loss
                vpredclipped = torch.max(torch.min(vpreds, cur_val + 0.2), cur_val - 0.2)
                vf_losses1 = (vpreds - cur_ret) ** 2
                vf_losses2 = (vpredclipped - cur_ret) ** 2
                vf_loss = 0.5 * ((torch.max(vf_losses1, vf_losses2) * cur_mask).sum(dim=-1) / resp_len_per_sample).mean()
                # vf_loss = 0.5 * ((torch.max(vf_losses1, vf_losses2) * cur_mask).sum() / cur_mask.sum())

                # total loss
                loss += pg_loss + vf_coef * vf_loss

                # token related metrics
                mean_query_len = torch.mean(allgather(torch.mean(query_len_per_sample)))
                std_query_len = torch.mean(allgather(torch.std(query_len_per_sample)))
                mean_resp_len = torch.mean(allgather(torch.mean(resp_len_per_sample)))
                std_resp_len = torch.mean(allgather(torch.std(resp_len_per_sample)))

                # value related metrics
                # vf_expl_var_num = torch.var(torch.masked_select(cur_ret - vpreds, cur_mask.bool())) 
                # vf_expl_var_dem = torch.var(torch.masked_select(cur_ret, cur_mask.bool()))
                vf_expl_var_num = masked_var(cur_ret - vpreds, cur_mask)
                vf_expl_var_dem = masked_var(cur_ret, cur_mask)
                vf_expl_var = 1.0 - vf_expl_var_num / (vf_expl_var_dem + 1e-8)
                vf_expl_var = max(-1.0, vf_expl_var.item())  # the truncated value suffices
                mean_vpred = masked_mean(vpreds, cur_mask)
                mean_return = masked_mean(cur_ret, cur_mask)
                mean_reward = masked_mean(cur_rew, cur_mask)
                mean_score_reward = masked_mean(cur_score_rew, cur_mask)
                mean_kl_reward = 0.0 if cur_kl_rew is None else masked_mean(cur_kl_rew, cur_mask)
                mean_kcxkl_reward = args["kl_coef"] * mean_kl_reward

                # policy related metrics
                mean_ratio = masked_mean(ratio, cur_mask[:, :-1])
                #mean_adv = masked_mean(cur_adv[:, :-1], cur_mask[:, :-1])
                mean_logprob = masked_mean(logprob, cur_mask[:, :-1])
                # sequence-level kl
                mean_seq_kl = -1.0
                if cur_kl_rew is not None:
                    cur_kl = -cur_kl_rew
                    seq_kl = torch.sum(cur_kl * cur_mask, dim=1)  # (mini_bs,)
                    mean_seq_kl = torch.mean(seq_kl)

                # Update
                epoch_result_dict['loss'].append(loss.item())

                # accelerator.backward(loss)
                # accelerator.deepspeed_engine_wrapped.backward(loss)
                # runs backpropagation and handles mixed precision
                if accelerator.distributed_type == "DEEPSPEED":
                    accelerator.deepspeed_engine_wrapped.engine.backward(loss)
                    total_grad_norm = 0.0
                    for n, p in model.named_parameters():
                        cur_grad = deepspeed.utils.safe_get_full_grad(p).view(-1)
                        cur_grad_norm_sqrt = torch.norm(cur_grad, 2)
                        if cur_grad_norm_sqrt < 1e-8:
                            accelerator.print(f'{n} grad_norm_sqrt: {cur_grad_norm_sqrt}')
                        total_grad_norm += cur_grad_norm_sqrt ** 2
                    total_grad_norm = total_grad_norm ** 0.5
                    # Deepspeed's `engine.step` performs the following operations:
                    # - gradient accumulation check
                    # - gradient clipping
                    # - optimizer step
                    # - zero grad
                    # - checking overflow
                    # - lr_scheduler step (only if engine.lr_scheduler is not None)
                    accelerator.deepspeed_engine_wrapped.engine.step()
                else:
                    accelerator.backward(loss)
                    #accelerator.backward(loss)
                    total_grad_norm = -1.0
                    if clip_grad_norm is not None:
                        total_grad_norm = accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()

                # Update running stats
                n_correct, total = do_gather([sum(correctness), len(correctness)])
                train_stats["acc"] = n_correct / total
                train_stats["ncor"] = n_correct
                train_stats["total"] = total
                train_stats['pg_loss'] = pg_loss.item()
                train_stats['vf_loss'] = vf_loss.item()
                train_stats['vf_expl_var'] = vf_expl_var

                for k, v in train_stats.items():
                    result_dict[k].append(v)

                total_param_norm = 0.0
                if accelerator.distributed_type == "DEEPSPEED":
                    for n, p in model.named_parameters():
                        cur_param = deepspeed.utils.safe_get_full_fp32_param(p).view(-1)
                        total_param_norm += torch.norm(cur_param, 2) ** 2
                    total_param_norm = total_param_norm ** 0.5
                else:
                    total_param_norm = torch.norm(
                        torch.cat([p.view(-1) for p in model.parameters()]),
                        p=2  # L2 norm
                    )
                # logging
                if accelerator.is_main_process and args['wandb_log']:
                    wandb.log({
                        "nn/total_grad_norm": total_grad_norm,
                        "nn/total_param_norm": total_param_norm,
                        "nn/lr": scheduler.get_last_lr()[0],
                    }, step=global_iter_num)
                    wandb.log({
                        "acc/acc": train_stats["acc"],
                        "acc/ncor": train_stats["ncor"],
                        "acc/total": train_stats["total"],
                    }, step=global_iter_num)
                    wandb.log({
                        "loss/loss:": loss,
                        "loss/pg_loss": pg_loss,
                        "loss/vf_loss": vf_loss,
                    }, step=global_iter_num)
                    wandb.log({
                        "tokens/mean_query_len": mean_query_len,
                        "tokens/std_query_len": std_query_len,
                        "tokens/mean_resp_len": mean_resp_len,
                        "tokens/std_resp_len": std_resp_len,
                    }, step=global_iter_num)
                    wandb.log({
                        "policy/mean_ratio": mean_ratio,
                        "policy/mean_adv": mean_adv,
                        "policy/var_adv": var_adv,
                        "policy/mean_logprob": mean_logprob,
                        "policy/mean_seq_kl": mean_seq_kl,
                    }, step=global_iter_num)
                    wandb.log({
                        "value/vf_expl_var": vf_expl_var,
                        "value/mean_vpred": mean_vpred,
                        "value/mean_return": mean_return,
                        "value/mean_reward": mean_reward,
                        "value/mean_score_reward": mean_score_reward,
                        "value/mean_kl_reward": mean_kl_reward,
                        "value/mean_kcxkl_reward": mean_kcxkl_reward,
                    }, step=global_iter_num)
                # Update iter num
                # torch.distributed.barrier()
                global_iter_num += 1

        scheduler.step()
        global_step += 1
        # accelerator.empty_cache()
        # Step update metric
        epoch_result_dict['loss'].append(loss.item())
        for k, v in train_stats.items():
            epoch_result_dict[k].append(v)

        # Step evaluating
        eval_log_dict = {}
        is_best = False
        if evaluating_step_freq is not None and global_step % evaluating_step_freq == 0:
            evaluate_result_dict = {f'Eval.Gen.{k}': v for k, v in
                                    evaluate_generation(args, model, test_dataset, test_dataloader, tokenizer, cot_info).items()}
            eval_log_dict.update(evaluate_result_dict)
            if eval_log_dict['Eval.Gen.value_accuracy'] > best_eval_log_dict.get('Eval.Gen.value_accuracy_best', 0):
                is_best = True
                best_eval_log_dict['Eval.Gen.value_accuracy_best'] = eval_log_dict['Eval.Gen.value_accuracy']

        # Step logging
        train_log_dict = {}
        if logging_step_freq is not None and global_step % logging_step_freq == 0:
            train_log_dict = {f'Train.{k}': sum(v) / len(v) if isinstance(v, list) else v for k, v in epoch_result_dict.items()}

        if eval_log_dict or train_log_dict:
            log_dict = {'lr': scheduler.get_last_lr()[0], **train_log_dict, **eval_log_dict, **best_eval_log_dict}
            if accelerator.is_main_process and args['wandb_log']:
                wandb.log(log_dict, step=global_iter_num)
                log_dict = {'wandb': args['wandb_project'] + '|' + args['wandb_run_name'] + '|' + wandb.run.id, **log_dict}

            log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k, v in log_dict.items()}
            accelerator.print(f"{prefix}[Epoch={epoch}/{max_epoch}, Step={global_step}] {log_dict}")

        # Step saving
        if saving_step_freq is not None and global_step % saving_step_freq == 0:
            if is_best:
                save_path = os.path.join(model_dir, f'best')
                do_checkpoint(args, model, tokenizer, save_path)
            #
            if args['keep_num_ckpt'] > 0:
                # save only if keep num ckpt > 0
                save_path = os.path.join(model_dir, f'global_step_{str(global_step)}')
                do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)

        # Keep only max_record items
        for k, v in epoch_result_dict.items():
            if len(v) > 1:
                epoch_result_dict[k] = v[-1:]

    # Metric summary:
    epoch_result_dict = {k: (sum(v) / len(v) if isinstance(v, list) else v) for k, v in epoch_result_dict.items()}
    return epoch_result_dict, global_step, global_iter_num


def main(args):
    set_seed(args['seed'] + accelerator.process_index)

    # if torch.distributed.get_rank() == 0:
    #     import ipdb; ipdb.set_trace()

    if accelerator.is_main_process and args['wandb_log']:
        wandb.init(project=args['wandb_project'], entity=args['wandb_entity'], name=args['wandb_run_name'])
        wandb.config.update(args)
        
    if args.get('use_small_vocab',False) and args['engine'] == 'game24':
        from src.game24_tokenizer import Calc24Vocab32Tokenizer
        tokenizer = Calc24Vocab32Tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args['tokenizer_name_or_path'], use_fast=True)
        # For Galactica model
        tokenizer.pad_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 3

    (train_dataset, train_dataloader), (test_dataset, test_dataloader), cot_info = \
        prepare_datasets_and_data_loaders(args, tokenizer)

    # Stages 
    difficulty_tiers = [
        (1, 2),    # Tier 1: Very Easy problems (e.g., 1-2 steps)
        (3, 4),    # Tier 2: Easy problems (e.g., 3-4 steps)
        (5, 6),    # Tier 3: Medium problems (e.g., 5-6 steps)
        (7, 8),    # Tier 4: Hard problems (e.g., 7-8 steps)
        (9, 100)   # Tier 5: Very Hard problems (e.g., 9+ steps, up to a high max)
    ]
    
    dummy_args_for_test_load = args.copy()
    dummy_args_for_test_load['train_file'] = args['test_file'] # Hack to get the test split loaded initially
    _, (test_dataset, test_dataloader), cot_info = prepare_datasets_and_data_loaders(dummy_args_for_test_load, tokenizer, current_difficulty_score_range=None)
    
    MODEL_CLASS = AutoModelForCausalLMWithValueHead if not args['separate_vf'] else AutoModelForCausalLMWithValueModel
    model = MODEL_CLASS.from_pretrained(args['model_name_or_path'])
    # accelerator.print(f'[Vocab size]: {len(tokenizer)}')
    # model.resize_token_embeddings(len(tokenizer))


    # initialize ref model (if any)
    ref_model = None
    if args['ref_model_name_or_path']:
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args['ref_model_name_or_path'])
        # from copy import deepcopy
        # ref_model = deepcopy(model)

    # optimizer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": args['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=1e-8)
    # Adjust scheduler for total number of steps across all tiers
    # num_training_steps_total = sum([len(prepare_datasets_and_data_loaders._full_raw_train_dataset) // args['batch_size'] for _ in difficulty_tiers]) * args['epochs_per_tier']
    # warmup_step = args['warmup_step'] if args['warmup_step'] is not None and args['warmup_step'] >= 0 else int(0.1 * num_training_steps_total)
    # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step)
    # Keeping the original scheduler for now, assuming it's adjusted elsewhere or works fine.
    # For a constant schedule, total steps might not be as critical.
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args['warmup_step']) # Using original setup
    
    # Accelerator prepare should be done *before* the tier loop, on the model and optimizer
    # DataLoaders will be prepared within the loop for each tier
    model, optimizer = accelerator.prepare(model, optimizer)
    if ref_model is not None:
        if accelerator.distributed_type == "DEEPSPEED":
            ref_model = prepare_deepspeed_ref_model(ref_model)
        else:
            ref_model = accelerator.prepare(ref_model)


    global_step = 0
    global_iter_num = 0
    # Note: n_epochs is now effectively n_epochs_per_tier * len(difficulty_tiers)
    # So, adjust your logging/saving frequencies if they were tied to total epochs.
    evaluating_epoch_freq = args['evaluating_epoch_freq']
    logging_epoch_freq = args['logging_epoch_freq']
    saving_epoch_freq = args['saving_epoch_freq']
    model_dir = args['model_dir']
    best_eval_log_dict = {}
    os.makedirs(model_dir, exist_ok=True)

    most_recent_ckpts_paths = []

    # --- NEW: Outer loop for difficulty tiers ---
    for tier_idx, difficulty_range in enumerate(difficulty_tiers):
        accelerator.print(f"\n=======================================================")
        accelerator.print(f"Starting Training Tier {tier_idx+1}/{len(difficulty_tiers)}: Difficulty Score Range {difficulty_range}")
        accelerator.print(f"=======================================================\n")

        # Load and prepare data for the current tier
        # The `train_dataloader` will now be specific to this difficulty tier
        (tokenized_train_dataset, train_dataloader), _, _ = \
            prepare_datasets_and_data_loaders(args, tokenizer, current_difficulty_score_range=difficulty_range)
        
        # Prepare the dataloader with accelerator after it's been instantiated for the current tier
        # Note: Test dataloader is already prepared and cached.
        train_dataloader = accelerator.prepare(train_dataloader)

        accelerator.print(
            f"***** Running training for Tier {tier_idx+1} *****\n"
            f"  Num examples (current tier) = {len(tokenized_train_dataset)}\n"
            f"  Num Epochs per Tier = {args['epochs_per_tier']}\n"
            f"  Instantaneous batch size per device = {args['batch_size']}\n"
            f"  Total train batch size (w. parallel, distributed & accumulation) = {args['batch_size'] * accelerator.num_processes}\n"
            f"  Current Learning rate: {optimizer.param_groups[0]['lr']}\n" # Log current LR
        )


        for epoch_in_tier in range(1, args['epochs_per_tier'] + 1):
            current_overall_epoch = (tier_idx * args['epochs_per_tier']) + epoch_in_tier # Track overall epoch for logging
            accelerator.print(f"--- Tier {tier_idx+1} Epoch {epoch_in_tier}/{args['epochs_per_tier']} (Overall Epoch {current_overall_epoch}) ---")
            
            kwargs = {
                'args': args,
                'model': model,
                'ref_model': ref_model,
                'train_dataset': tokenized_train_dataset, # This is the tier-specific dataset
                'train_dataloader': train_dataloader,     # This is the tier-specific dataloader
                'test_dataset': test_dataset,             # Keep the full test dataset constant
                'test_dataloader': test_dataloader,       # Keep the full test dataloader constant
                'cot_info': cot_info,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'global_step': global_step,
                'global_iter_num': global_iter_num,
                'tokenizer': tokenizer,
                'prefix': f'[Tier {tier_idx+1}-Epoch {epoch_in_tier}]', # More descriptive prefix
                'epoch': current_overall_epoch, # Pass overall epoch
                'best_eval_log_dict': best_eval_log_dict,
                'most_recent_ckpts_paths': most_recent_ckpts_paths,
            }
            train_epoch_result_dict, global_step, global_iter_num = train_one_epoch(**kwargs)

            eval_log_dict = {}
            is_best = False
            # Evaluation and Logging should still happen on the full test set
            if evaluating_epoch_freq is not None and current_overall_epoch % evaluating_epoch_freq == 0:
                evaluate_result_dict = {f'Eval.Gen.{k}': v for k, v in
                                        evaluate_generation(args, model, test_dataset, test_dataloader, tokenizer, cot_info).items()}
                eval_log_dict.update(evaluate_result_dict)
                if eval_log_dict['Eval.Gen.value_accuracy'] > best_eval_log_dict.get('Eval.Gen.value_accuracy_best', 0):
                    is_best = True
                    best_eval_log_dict['Eval.Gen.value_accuracy_best'] = eval_log_dict['Eval.Gen.value_accuracy']

            train_log_dict = {}
            if logging_epoch_freq is not None and current_overall_epoch % logging_epoch_freq == 0:
                train_log_dict = {f'Train.{k}': sum(v) / len(v) if isinstance(v, list) else v for k, v in
                                  train_epoch_result_dict.items()}

            if eval_log_dict or train_log_dict:
                log_dict = {'lr': scheduler.get_last_lr()[0], **train_log_dict, **eval_log_dict, **best_eval_log_dict}
                if accelerator.is_main_process and args['wandb_log']:
                    wandb.log(log_dict, step=global_iter_num)
                    log_dict = {'wandb': args['wandb_project'] + '|' + args['wandb_run_name'] + '|' + wandb.run.id, **log_dict}

                log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k, v in log_dict.items()}
                accelerator.print(
                    f"[Overall Epoch={current_overall_epoch}/{len(difficulty_tiers)*args['epochs_per_tier']}, "
                    f"Tier={tier_idx+1}, Step={global_step}] {log_dict}")

            if saving_epoch_freq is not None and current_overall_epoch % saving_epoch_freq == 0:
                if is_best:
                    save_path = os.path.join(model_dir, f'best_tier_{tier_idx+1}') # Save best per tier or overall?
                    do_checkpoint(args, model, tokenizer, save_path)
                
                # save the checkpoint if keep num ckpt > 0
                if args['keep_num_ckpt'] > 0:
                    save_path = os.path.join(args['model_dir'], f'global_step_{str(global_step)}_tier_{tier_idx+1}_epoch_{epoch_in_tier}')
                    do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)

    # Final saving after all tiers are complete
    if accelerator.is_main_process:
        accelerator.print("\nTraining complete across all difficulty tiers.")
        final_save_path = os.path.join(model_dir, 'final_model')
        do_checkpoint(args, model, tokenizer, final_save_path)

def evaluate_generation(args, model, dataset, dataloader, tokenizer, cot_info):
    model.eval()
    predictions = []
    targets = []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process,
                           desc='Evaluation Gen Loop'):
        output_ = accelerator.unwrap_model(model).generate(
            **batch['generate_prefix_kwargs'],
            max_length=args['max_gen_length'],
            output_scores=True,
            return_dict_in_generate=True,
            num_beams=1,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ids = output_.sequences
        generated_ids = pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)

        labels = batch['generate_prefix_kwargs']['labels']
        labels = pad_across_processes(labels, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)
        labels[labels == -100] = tokenizer.pad_token_id

        generated_ids, labels = accelerator.gather(generated_ids), accelerator.gather(labels)

        preds = [tokenizer.decode(g.cpu().numpy().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in
                 generated_ids]
        predictions.extend(preds)
        target = [tokenizer.decode(t.cpu().numpy().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for t in
                  labels]
        targets.extend(target)

    predictions = predictions[:len(dataset)]
    targets = targets[:len(dataset)]

    post_process_final_answer_fn_mapper = cot_info['post_process_final_answer_fn_mapper']
    compare_answer_fn_mapper = cot_info['compare_answer_fn_mapper']
    post_process_completed_question_answer_fn_mapper = cot_info['post_process_completed_question_answer_fn_mapper']
    if accelerator.is_main_process and accelerator.is_local_main_process:
        results = [{
            'pred': pred,
            'tar': tar,
            'item_id': item.get('item_id', None),
            'answer_value': item.get('answer_value', None),
            'answer_type': item.get('answer_type', None),
        } for pred, tar, item in zip(predictions, targets, dataset)]

        corr_value = 0
        for cur_res in results:
            prediction, target, item_id = cur_res['pred'], cur_res['tar'], cur_res['item_id']
            src_name = item_id.split('_')[0]
            answer_value = cur_res['answer_value']

            ## Processing target
            target_cot = target.strip()
            target_value = post_process_final_answer_fn_mapper[src_name](answer_value)
            cur_res['target_cot'] = target_cot
            cur_res['target_value'] = target_value

            ## Processing prediction
            try:
                with timeout(seconds=TIMEOUT):
                    prediction_cot = prediction.strip()
                    prediction_value = post_process_completed_question_answer_fn_mapper[(args['engine'], src_name)](prediction_cot)
            except:
                prediction_cot = None
                prediction_value = None
            cur_res['prediction_cot'] = prediction_cot
            cur_res['prediction_value'] = prediction_value

            # Compute correctness
            is_correct = compare_answer_fn_mapper[src_name](prediction_value, target_value) if prediction_value is not None else False
            corr_value += is_correct
            cur_res['is_correct'] = is_correct
        res_path = args['model_dir'].rstrip('/')+ '/' + '_res.json'
        with open(res_path, 'w') as f:
            json.dump(results, f, indent=2)
        value_accuracy = corr_value / len(predictions) * 100
        accelerator.print(f"[Eval Info] value_accuracy: {value_accuracy:.5g}%")
        value_accuracy = torch.FloatTensor([value_accuracy]).to(accelerator.device)
    else:
        value_accuracy = torch.FloatTensor([-1.0]).to(accelerator.device)
    value_accuracy = broadcast(value_accuracy).cpu().numpy().tolist()[0]

    # Metric summary:
    model.train()
    return {'value_accuracy': value_accuracy}


def collate_fn(batch, args, tokenizer):
    max_input_length = max([len(item['input_ids']) for item in batch])
    max_target_length = max([len(item['labels']) for item in batch])
    max_prefix_length = max([len(item['prefix']) for item in batch])

    input_ids, input_ids_left_padded = [], []
    attention_mask, attention_mask_left_padded = [], []
    labels, labels_left_padded = [], []
    prefix, prefix_left_padded = [], []
    prefix_attention_mask, prefix_attention_mask_left_padded = [], []

    for item in batch:
        labels_left_padded.append([-100] * (max_target_length - len(item['labels'])) + item['labels'])
        prefix_left_padded.append([tokenizer.pad_token_id] * (max_prefix_length - len(item['prefix'])) + item['prefix'])
        prefix_attention_mask_left_padded.append(
            [0] * (max_prefix_length - len(item['prefix_attention_mask'])) + item['prefix_attention_mask'])

    ppo_forward_kwargs = {
        'query': [item['prefix_text'] for item in batch],
        'query_tensors': torch.LongTensor(prefix_left_padded),
        'query_tensors_attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
        'answer_values': [item['answer_value'].replace(',', '') for item in batch],
        'item_ids': torch.LongTensor([int(item['item_id'].split('_')[1]) for item in batch]),
    }
    generate_prefix_kwargs = {
        'input_ids': torch.LongTensor(prefix_left_padded),
        'attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
        'labels': torch.LongTensor(labels_left_padded)
    }

    return {
        'ppo_forward_kwargs': ppo_forward_kwargs,
        'generate_prefix_kwargs': generate_prefix_kwargs,
    }


if __name__ == '__main__':
    from transformers import HfArgumentParser

    NONE_INT = -100
    NONE_STR = 'None'


    @dataclass
    class Arguments:
        model_name_or_path: str
        tokenizer_name_or_path: str
        model_dir: str
        train_file: str
        test_file: str
        batch_size: int = field(default=8)
        mini_batch_size: int = field(default=8)
        eval_batch_size: int = field(default=8)
        ppo_epochs: int = field(default=1)
        n_epochs: int = field(default=40)
        num_workers: int = field(default=8)
        learning_rate: float = field(default=2e-5)
        weight_decay: float = field(default=1e-6)
        warmup_step: int = field(default=0)
        clip_grad_norm: float = field(default=1)
        vf_coef: float = field(default=1.0)
        kl_coef: float = field(default=0.1)
        gamma: float = field(default=0.98)
        lam: float = field(default=0.95)
        ref_model_name_or_path: str = field(default="")
        evaluating_epoch_freq: int = field(default=1)
        logging_epoch_freq: int = field(default=1)
        saving_epoch_freq: int = field(default=1000)
        evaluating_step_freq: int = field(default=NONE_INT)
        logging_step_freq: int = field(default=NONE_INT)
        logging_seq_str_step_freq: int = field(default=NONE_INT)
        logging_values_step_freq: int = field(default=NONE_INT)
        saving_step_freq: int = field(default=NONE_INT)
        seed: int = field(default=42)
        max_input_length: int = field(default=700)
        max_gen_length: int = field(default=700)
        keep_num_ckpt: int = field(default=5)
        # value model and head
        separate_vf: int = field(default=0)  # separate value function
        init_value_model_with_rm: int = field(default=0)
        init_value_head_with_rm: int = field(default=0)
        rm_model_name_or_path: str = field(default="/your_model_path_here")
        # wandb stuff
        wandb_log: bool = field(default=False)
        wandb_project: str = field(default='tmp')
        wandb_entity: str = field(default='default_entity_name')
        wandb_run_name: str = field(default='default_run_name')
        ###
        engine: str = field(default='nl')
        use_small_vocab: int = field(default=0)
        adv_whitening: str = field(default='global')

    parser = HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()
    args = asdict(args)
    for k, v in args.items():
        if v in [NONE_INT, NONE_STR]:
            args[k] = None
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))])
    accelerator.print(args)
    accelerator.print(json.dumps(args, indent=2, ensure_ascii=False))
    main(args)
