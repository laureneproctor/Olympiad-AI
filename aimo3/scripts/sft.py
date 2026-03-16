# Input: prepared train/val/test splits from step 00
# - Formats each row for supervised fine-tuning (SFT)
# - Prompt = SYSTEM_PROMPT + problem + "\n\nSolution:\n"
# - Completion = generated_solution + "\nFINAL_ANSWER: expected_answer"
# - Loads tokenizer and 4-bit quantized base model
# - Applies LoRA for parameter-efficient fine-tuning
# - Trains with SFTTrainer
# - Saves model adapter weights and tokenizer
# Output:
# - output_dir/ (SFT checkpoint)

# Input: prepared train/val/test splits from step 00
# - Formats each row for supervised fine-tuning (SFT)
# - Prompt = SYSTEM_PROMPT + problem + "\n\nSolution:\n"
# - Completion = generated_solution + "\nFINAL_ANSWER: expected_answer"
# - Loads tokenizer and 4-bit quantized base model
# - Applies LoRA for parameter-efficient fine-tuning
# - Trains with SFTTrainer
# - Saves model adapter weights and tokenizer
# Output:
# - output_dir/ (SFT checkpoint)

# ---------------------------------------------------------------------------------------------------------
### Imports for the script
# ---------------------------------------------------------------------------------------------------------
import os
import random

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------------------------------------------------------------------------------------------------
### Config paths
# ---------------------------------------------------------------------------------------------------------
SFT_CONFIG_PATH = "/content/drive/MyDrive/Math Olympiad Competition/configs/sft.yaml"

def load_yaml(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------------------------------------
### Load configs and resolve selected run
# ---------------------------------------------------------------------------------------------------------
def load_configs(sft_config_path=SFT_CONFIG_PATH):
    sft_config = load_yaml(sft_config_path)

    data_config_path = sft_config["paths"]["data_config_path"]
    data_config = load_yaml(data_config_path)

    exp_name = sft_config["run"]["experiment_name"]
    model_key = sft_config["run"]["model_key"]

    if exp_name not in data_config["experiments"]:
        raise KeyError(f"Experiment '{exp_name}' not found in data.yaml")

    if model_key not in data_config["models"]:
        raise KeyError(f"Model '{model_key}' not found in data.yaml")

    exp_config = data_config["experiments"][exp_name]
    model_config = data_config["models"][model_key]

    allowed_models = exp_config.get("models", [])
    if allowed_models and model_key not in allowed_models:
        raise ValueError(
            f"Model '{model_key}' is not listed for experiment '{exp_name}'. "
            f"Allowed models: {allowed_models}"
        )

    n = exp_config["N"]
    prepared_data_path = os.path.join(
        sft_config["paths"]["prepared_splits_root"],
        "splits",
        str(n),
    )

    output_directory = os.path.join(
        sft_config["paths"]["output_root"],
        f"sft_{model_key}_{exp_name}",
    )

    return sft_config, data_config, exp_config, model_config, prepared_data_path, output_directory

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
def convert_to_prompt_format(example, system_prompt):
    problem = example["problem"].strip()
    reasoning = example["generated_solution"].strip()
    answer = str(example["expected_answer"]).strip()

    prompt = (
        system_prompt
        + "\n\nProblem:\n"
        + problem
        + "\n\nSolution:"
    )

    completion = (
        "\n"
        + reasoning.rstrip()
        + "\nFINAL_ANSWER: "
        + answer
    )

    return {
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion,
        "expected_answer": answer,
    }

# ---------------------------------------------------------------------------------------------------------
### Load splits from drive
# ---------------------------------------------------------------------------------------------------------
def load_prepared_splits(prepared_data_path):
    print("Loading processed Train/Val/Test splits from drive")
    print("Prepared path:", prepared_data_path)

    prepared = load_from_disk(prepared_data_path)

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
def format_sft_splits(ds_train_raw, ds_val_raw, system_prompt):
    print("Formatting train/val splits for SFT")

    fmtd_train = ds_train_raw.map(
        convert_to_prompt_format,
        fn_kwargs={"system_prompt": system_prompt},
        remove_columns=ds_train_raw.column_names,
    )
    fmtd_val = ds_val_raw.map(
        convert_to_prompt_format,
        fn_kwargs={"system_prompt": system_prompt},
        remove_columns=ds_val_raw.column_names,
    )

    print("Formatted train columns:", fmtd_train.column_names)
    print("Formatted val columns:", fmtd_val.column_names)

    return fmtd_train, fmtd_val

# ---------------------------------------------------------------------------------------------------------
### Load tokenizer
# ---------------------------------------------------------------------------------------------------------
def load_tokenizer(current_model):
    tok = AutoTokenizer.from_pretrained(
        current_model,
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
def load_model(current_model, quantization_config_yaml):
    compute_dtype_name = quantization_config_yaml["bnb_4bit_compute_dtype"]

    if compute_dtype_name == "bfloat16":
        compute_dtype = torch.bfloat16
    elif compute_dtype_name == "float16":
        compute_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported compute dtype: {compute_dtype_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization_config_yaml["load_in_4bit"],
        bnb_4bit_quant_type=quantization_config_yaml["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quantization_config_yaml["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        current_model,
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
def build_peft_config(lora_yaml):
    return LoraConfig(
        r=lora_yaml["r"],
        lora_alpha=lora_yaml["lora_alpha"],
        lora_dropout=lora_yaml["lora_dropout"],
        bias=lora_yaml["bias"],
        task_type=lora_yaml["task_type"],
        target_modules=lora_yaml["target_modules"],
    )

# ---------------------------------------------------------------------------------------------------------
### Build SFT trainer config
# ---------------------------------------------------------------------------------------------------------
def build_sft_config(training_yaml, output_directory):
    return SFTConfig(
        output_dir=output_directory,
        max_length=training_yaml["max_length"],
        per_device_train_batch_size=training_yaml["per_device_train_batch_size"],
        per_device_eval_batch_size=training_yaml["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_yaml["gradient_accumulation_steps"],
        num_train_epochs=training_yaml["num_train_epochs"],
        learning_rate=training_yaml["learning_rate"],
        warmup_steps=training_yaml["warmup_steps"],
        logging_steps=training_yaml["logging_steps"],
        eval_strategy=training_yaml["eval_strategy"],
        eval_steps=training_yaml["eval_steps"],
        save_steps=training_yaml["save_steps"],
        save_total_limit=training_yaml["save_total_limit"],
        bf16=training_yaml["bf16"],
        report_to=training_yaml["report_to"],
    )

# ---------------------------------------------------------------------------------------------------------
### Save run configs
# ---------------------------------------------------------------------------------------------------------
def save_run_metadata(output_directory, sft_config, data_config, exp_config, model_config, prepared_data_path):
    os.makedirs(output_directory, exist_ok=True)

    with open(os.path.join(output_directory, "sft_config_used.yaml"), "w") as f:
        yaml.safe_dump(sft_config, f, sort_keys=False)

    with open(os.path.join(output_directory, "data_config_used.yaml"), "w") as f:
        yaml.safe_dump(data_config, f, sort_keys=False)

    run_summary = {
        "experiment_name": sft_config["run"]["experiment_name"],
        "model_key": sft_config["run"]["model_key"],
        "model_name": model_config["name"],
        "N": exp_config["N"],
        "prepared_data_path": prepared_data_path,
        "output_directory": output_directory,
    }

    with open(os.path.join(output_directory, "run_summary.yaml"), "w") as f:
        yaml.safe_dump(run_summary, f, sort_keys=False)

# ---------------------------------------------------------------------------------------------------------
### Print trainable parameter count
# ---------------------------------------------------------------------------------------------------------
def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    pct = 100 * trainable_params / all_params if all_params > 0 else 0.0

    print(f"Trainable params: {trainable_params:,}")
    print(f"All params: {all_params:,}")
    print(f"Trainable %: {pct:.4f}")

# ---------------------------------------------------------------------------------------------------------
### Main training function
# ---------------------------------------------------------------------------------------------------------
def train_model():
    sft_config, data_config, exp_config, model_config, prepared_data_path, output_directory = load_configs()

    seed = sft_config["run"]["seed"]
    current_model = model_config["name"]
    system_prompt = sft_config["prompting"]["system_prompt"]

    print("Experiment:", sft_config["run"]["experiment_name"])
    print("Model key:", sft_config["run"]["model_key"])
    print("Model name:", current_model)
    print("Prepared data path:", prepared_data_path)
    print("Output directory:", output_directory)

    set_seeds(seed)

    ds_train_raw, ds_val_raw, ds_test_raw = load_prepared_splits(prepared_data_path)
    ds_train, ds_val = format_sft_splits(ds_train_raw, ds_val_raw, system_prompt)

    tok = load_tokenizer(current_model)
    model = load_model(current_model, sft_config["quantization"])
    peft_config = build_peft_config(sft_config["lora"])
    sft_trainer_config = build_sft_config(sft_config["training"], output_directory)

    save_run_metadata(
        output_directory,
        sft_config,
        data_config,
        exp_config,
        model_config,
        prepared_data_path,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_trainer_config,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tok,
        peft_config=peft_config,
        dataset_text_field="text",
    )

    print_trainable_parameters(trainer.model)

    print("Starting training")
    trainer.train()

    print("Saving model + tokenizer")
    trainer.save_model(sft_trainer_config.output_dir)
    tok.save_pretrained(sft_trainer_config.output_dir)

    print("Done. Saved to:", sft_trainer_config.output_dir)
    print("Test split was loaded but not used for training:", len(ds_test_raw))