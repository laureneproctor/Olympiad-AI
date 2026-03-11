# Input: prepared train/val/test splits from step 00
# - Formats each row for supervised fine-tuning (SFT)
# - Prompt = SYSTEM_PROMPT + problem + "\n\nSolution:\n"
# - Completion = generated_solution + "\nFINAL_ANSWER: expected_answer"
# - Loads tokenizer and 4-bit quantized base model
# - Applies LoRA for parameter-efficient fine-tuning
# - Trains with SFTTrainer
# - Saves model adapter weights and tokenizer
# Output:
# - out_deepseekmath7b_omr_lora_aimo3/ (SFT checkpoint)

# ---------------------------------------------------------------------------------------------------------
### Imports for the script
# ---------------------------------------------------------------------------------------------------------
import random
import torch
from datasets import load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------------------------------------
### Prompt formatting
# ---------------------------------------------------------------------------------------------------------
def convert_to_prompt_format(example):
    problem = example["problem"].strip()
    reasoning = example["generated_solution"].strip()
    answer = str(example["expected_answer"]).strip()

    prompt = (
        SYSTEM_PROMPT
        + "\n\nProblem:\n"
        + problem
        + "\n\nSolution:\n"
    )
    completion = reasoning.rstrip() + "\nFINAL_ANSWER: " + answer

    return {
        "prompt": prompt,
        "completion": completion,
        "expected_answer": answer,
    }

# ---------------------------------------------------------------------------------------------------------
### Load splits from drive
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
### Format train/val splits to SFT prompt-completion format
# ---------------------------------------------------------------------------------------------------------
def format_sft_splits(ds_train_raw, ds_val_raw):
    print("Formatting train/val splits for SFT")

    fmtd_train = ds_train_raw.map(
        convert_to_prompt_format,
        remove_columns=ds_train_raw.column_names,
    )
    fmtd_val = ds_val_raw.map(
        convert_to_prompt_format,
        remove_columns=ds_val_raw.column_names,
    )

    print("Formatted train columns:", fmtd_train.column_names)
    print("Formatted val columns:", fmtd_val.column_names)

    return fmtd_train, fmtd_val

# ---------------------------------------------------------------------------------------------------------
### Load tokenizer
# ---------------------------------------------------------------------------------------------------------
def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(
        CURRENT_MODEL,
        use_fast=True,
        trust_remote_code=True,
    )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "right"
    return tok

# ---------------------------------------------------------------------------------------------------------
### Load 4-bit quantized base model
# ---------------------------------------------------------------------------------------------------------
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        CURRENT_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    return model

# ---------------------------------------------------------------------------------------------------------
### Build LoRA config
# ---------------------------------------------------------------------------------------------------------
def build_peft_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

# ---------------------------------------------------------------------------------------------------------
### Build SFT trainer config
# ---------------------------------------------------------------------------------------------------------
def build_sft_config():
    return SFTConfig(
        output_dir=OUT_DIRECTORY,
        max_length=2048,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-5,
        warmup_steps=50,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=1,
        bf16=True,
        report_to=[],
    )

# ---------------------------------------------------------------------------------------------------------
### Main training function
# ---------------------------------------------------------------------------------------------------------
def train_model():
    set_seeds(SEED)

    ds_train_raw, ds_val_raw, ds_test_raw = load_prepared_splits()
    ds_train, ds_val = format_sft_splits(ds_train_raw, ds_val_raw)

    tok = load_tokenizer()
    model = load_model()
    peft_config = build_peft_config()
    sft_config = build_sft_config()

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tok,
        peft_config=peft_config,
    )

    print("Starting training")
    trainer.train()

    print("Saving model + tokenizer")
    trainer.save_model(sft_config.output_dir)
    tok.save_pretrained(sft_config.output_dir)

    print("Done. Saved to:", sft_config.output_dir)