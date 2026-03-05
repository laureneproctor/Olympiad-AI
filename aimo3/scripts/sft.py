# Input: prepared train/val splits from 00
# - Format each row:
#   - Prompt = system_prompt + problem + solution
#   - Completion = gnerated_solution + final_answer: expected_answer
# - Loads tokenizer + base model (4bit? Quant)
# - Applied LoRA
# - Trains with SFTTrainer
# - Saves model + tokenizer
# Output
# - out_deepseekmath7b_omr_lora_aimo3/ (Our SFT checkpoint)