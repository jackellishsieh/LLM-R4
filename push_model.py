"""
This script currently uploads GRPO checkpint 6000 to the Hugging Face Hub.
Please update files and model name as needed.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your trained model
model = AutoModelForCausalLM.from_pretrained("./dr_grpo_qwen_0.5b/checkpoint-6000/")
tokenizer = AutoTokenizer.from_pretrained("./dr_grpo_qwen_0.5b/checkpoint-6000/")

# Push to Hub with descriptive name
model_name = "dillonkn/qwen2.5-0.5b-grpo-gsm8k"

model.push_to_hub(model_name)
tokenizer.push_to_hub(model_name)

print(f"Model uploaded to: https://huggingface.co/{model_name}")