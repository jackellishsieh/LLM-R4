
# gsm8k_instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"   # the original
gsm8k_instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"

gsm8k_instruction = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer.
"""

" "
gsm8k_instruction += " "
gsm8k_instruction += "The final answer must be single number prefaced with\n#### "
gsm8k_instruction += "For example, if the final answer to the question is 5, the Assistant should end the solution with \n#### 5 \n"
