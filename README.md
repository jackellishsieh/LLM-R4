# Custom Set-up Instructions

1. Run `huggingface-cli dillonkn/qwen2.5-0.5b-reasoning-sft`, which downloads our SFT'd Qwen model to 
`/home/ubuntu/.cache/huggingface/hub/models--dillonkn--qwen2.5-0.5b-reasoning-sft/snapshots/4b10ef8d143d5e4ae10bbb481a746dd8fa72beb2`

2. Setup the right conda environment by running 
```
conda create -n R3 python=3.9 -y
conda activate R3
pip install -r requirements.txt
```

3. Log into wandb by running `wandb login` and pasting your wandb account API key.

4. Run `mkdir output_models` and `mkdir log_dir`

5. To train our SFT'd Qwen model on THEIR GMS8K dataset, run
```
cd R3_math
bash R3_cot_gsm8k.sh
```

<h1 align="left"><strong>R</strong><sup>3</sup>: Training Large Language Models for <strong>R</strong>easoning through <strong>R</strong>everse Curriculum <strong>R</strong>einforcement Learning</h1>

<space for arxiv badge>
Implementation of the "Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning" presented by Zhiheng Xi, Wenxiang Chen, Boyang Hong, et al.

Paper Link: https://arxiv.org/abs/2402.05808
## 💡 Introduction

![](src/figures/main.png)

## 🛠️ Set up

It is suggested to use a **python 3.9** environment to run the experiment. Run the following commands to set up your environment:

```
git clone https://github.com/xxxxx.git

conda create -n R3_math python=3.9 -y
cd R3_math/
pip install -r requirements.txt

conda create -n R3_others python=3.9 -y
cd R3_others/
pip install -r requirements.txt
```

## ⚡️Usage

### Step1: SFT Training

To train a sft model, first set the model path and output path in the  `R3_others/scripts/step1_supervised_finetuning/R3_sft.sh`script. Then, run the following command:

```
cd R3_others/scripts/step1_supervised_finetuning/
bash R3_sft.sh
```

### Step2: R<sup>3</sup> Training

To train a reinforced model using **R**$^3$ on GSM8K (or other math datasets), first set the actor model path (it should be a sft model checkpoint from **Step1**) and output path in `R3_math/scripts/R3_cot_gsm8k.sh`, and run the following command:

```
cd R3_math/scripts/
bash R3_cot_gsm8k.sh
```

**Note**: If you want to try **R**$^3$ on other datasets like MNLI or race@High, set the SFT model path in `R3_others/scripts/step3_rlhf_finetuning/R3_mix.sh`. Then, run the folloing command:

```
cd R3_others/scripts/step3_rlhf_finetuning/
bash R3_mix.sh
```

### Evaluation

> It is not required for math datasets. Results will be saved in *wandb*.

To evaluate the model performance, first run the evaluation script `R3_others/scripts/eval/eval_single.sh`. Then, get your results in `output_{dataset_name}.py`. Here's an example for MNLI dataset:

```
cd R3_others/scripts/eval
bash eval_single.sh
# after evaluation
# you will get a result file like: eval_mnli/R3_test.txt

python output_mnli.py
# then you will get acc result
```

### Data

For the purpose of security review, we provide some examples of the data, formatted as follows:

```
Dataset: MNLI
	---- mnli_train_example.json # for SFT
	---- mnli_mix_example.json # fot R^3
	---- mnli_test.json
```

## ✏️ Citation

If you find **R**$^3$ useful for your your research and applications, please cite using this BibTeX:

```
@misc{xi2024training,
      title={Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning}, 
      author={Zhiheng Xi and Wenxiang Chen and Boyang Hong and Senjie Jin and Rui Zheng and Wei He and Yiwen Ding and Shichun Liu and Xin Guo and Junzhe Wang and Honglin Guo and Wei Shen and Xiaoran Fan and Yuhao Zhou and Shihan Dou and Xiao Wang and Xinbo Zhang and Peng Sun and Tao Gui and Qi Zhang and Xuanjing Huang},
      year={2024},
      eprint={2402.05808},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```


## Contact
- zhxi22@m.fudan.edu.cn
