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


# ---------------------------------------------------------------------------------------------------------
### Imports for the script
# ---------------------------------------------------------------------------------------------------------
import random
import torch
from datasets import load_from_disk
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

# ---------------------------------------------------------------------------------------------------------
### Set seed, configs, and paths
# ---------------------------------------------------------------------------------------------------------
SEED = 42
CURRENT_MODEL = "deepseek-ai/deepseek-math-7b-instruct"
PREPARED_DATA_PATH = "/content/drive/MyDrive/Math Olympiad Competition/processed_splits/omr_aimo3/hf"
OUT_DIRECTORY = "/content/drive/MyDrive/Math Olympiad Competition/models/out_deepseekmath7b_omr_lora_aimo3"

SYSTEM_PROMPT = (
    "You are a mathematical problem solver.\n"
    "Solve step-by-step.\n"
    "At the end, output a line exactly of the form:\n"
    "FINAL_ANSWER: <integer>\n"
    "Where <integer> is an integer between 0 and 99999."
)

# ---------------------------------------------------------------------------------------------------------
### Set seed function
# ---------------------------------------------------------------------------------------------------------
def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------------------------------------
### Prompt formatting
# ---------------------------------------------------------------------------------------------------------
def convert_to_promp_format(example):
    problem = example["problem"].strip()
    reasoning = example["generated_solution"].strip()
    answer = str(example["expected_answer"]).strip()

    prompt = (
        SYSTEM_PROMPT + "\n\nProblem:\n" + problem + "\n\nSolution:\n")
    completion = reasoning.rstrip() + "\nFINAL_ANSWER: " + answer
    return {"prompt": prompt, "completion": completion, "expected_answer": answer}

# ---------------------------------------------------------------------------------------------------------
### Load Splits from drive
# ---------------------------------------------------------------------------------------------------------

def load_prepared_splits():
    print("Loading processed Train/Val/Test splits from drive")
    prepared = load_from_disk(PREPARED_DATA_PATH)

    ds_train_raw = prepared["train"]
    ds_val_raw = prepared["val"]
    ds_test_raw = prepared["test"]

    print("Train rows:", len(ds_train_raw))
    print("Val rows:", len(ds_val_raw))
    print("Test rows:", len(ds_test_raw))


    return ds_train_raw, ds_val_raw, ds_test_raw

# ---------------------------------------------------------------------------------------------------------
### Format them to SFT format (prompt + completion)
# ---------------------------------------------------------------------------------------------------------

def format_sft_splits(ds_train_raw, ds_val_raw):
    print("Formatting train/val splits for SFT...")

    remove_cols = ds_train_raw.column_names

    fmtd_train = ds_train_raw.map(convert_to_promp_format, remove_columns=remove_cols)
    fmtd_val = ds_val_raw.map(convert_to_promp_format, remove_columns=remove_cols)


    print("Formatted train columns:", fmtd_train.column_names)
    print("Formatted val columns:", fmtd_val.column_names)

    return fmtd_train , fmtd_val