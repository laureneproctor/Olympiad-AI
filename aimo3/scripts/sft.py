import os
import random
from pathlib import Path
import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# ------------------------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------------------------
def get_repo_root():
    return Path(__file__).resolve().parents[1]
def get_sft_config_path():
    return get_repo_root() / "configs" / "sft.yaml"
def get_data_config_path():
    return get_repo_root() / "configs" / "data.yaml"

# ------------------------------------------------------------------------------
# YAML loading
# ------------------------------------------------------------------------------
def load_yaml(config_path):
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ------------------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------------------
def load_configs(sft_config_path=None):
    if sft_config_path is None:
        sft_config_path = get_sft_config_path()

    sft_config = load_yaml(sft_config_path)

    data_config_path = sft_config.get("paths", {}).get("data_config_path")
    if not data_config_path:
        data_config_path = get_data_config_path()
    data_config = load_yaml(data_config_path)
    exp_name = sft_config["run"]["experiment_name"]
    model_key = sft_config["run"]["model_key"]
    exp_config = data_config["experiments"][exp_name]
    model_config = data_config["models"][model_key]
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
    return (
        sft_config,
        data_config,
        exp_config,
        model_config,
        prepared_data_path,
        output_directory,
    )

# ------------------------------------------------------------------------------
# Seed
# ------------------------------------------------------------------------------
def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------------------------------------------------------------------
# Prompt formatting
# ------------------------------------------------------------------------------
def convert_to_prompt_format(example, system_prompt):
    problem = str(example["problem"]).strip()
    reasoning = str(example["generated_solution"]).strip()
    answer = str(example["expected_answer"]).strip()

    text = (
        system_prompt
        + "\n\nProblem:\n"
        + problem
        + "\n\nSolution:\n"
        + reasoning
        + "\nFINAL_ANSWER: "
        + answer
    )

    return {
        "text": text,
        "expected_answer": answer,
    }

# ------------------------------------------------------------------------------
# Load prepared splits
# ------------------------------------------------------------------------------
def load_prepared_splits(prepared_data_path):
    print("Loading processed Train/Val/Test splits")
    print("Prepared path:", prepared_data_path)
    prepared = load_from_disk(prepared_data_path)
    ds_train_raw = prepared["train"]
    ds_val_raw = prepared["val"]
    ds_test_raw = prepared["test"]
    print("Train rows:", len(ds_train_raw))
    print("Val rows:", len(ds_val_raw))
    print("Test rows:", len(ds_test_raw))
    return ds_train_raw, ds_val_raw, ds_test_raw


# ------------------------------------------------------------------------------
# Format splits
# ------------------------------------------------------------------------------
def format_sft_splits(ds_train_raw, ds_val_raw, system_prompt):
    print("Formatting train/val splits for SFT")

    fmtd_train = ds_train_raw.map(
        convert_to_prompt_format,
        fn_kwargs={"system_prompt": system_prompt},
        remove_columns=ds_train_raw.column_names,
        desc="Formatting train split",
    )

    fmtd_val = ds_val_raw.map(
        convert_to_prompt_format,
        fn_kwargs={"system_prompt": system_prompt},
        remove_columns=ds_val_raw.column_names,
        desc="Formatting val split",
    )
    print("Formatted train columns:", fmtd_train.column_names)
    print("Formatted val columns:", fmtd_val.column_names)
    return fmtd_train, fmtd_val


# ------------------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------------------
def load_tokenizer(model_name_or_path):
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

# ------------------------------------------------------------------------------
# Quantization helpers
# ------------------------------------------------------------------------------
def maybe_get_quantization_config(quantization_yaml):
    enabled = quantization_yaml.get("enabled", True)
    if not enabled:
        return None
    compute_dtype_name = quantization_yaml["bnb_4bit_compute_dtype"]
    if compute_dtype_name == "bfloat16":
        compute_dtype = torch.bfloat16
    elif compute_dtype_name == "float16":
        compute_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported compute dtype")
    return BitsAndBytesConfig(
        load_in_4bit=quantization_yaml["load_in_4bit"],
        bnb_4bit_quant_type=quantization_yaml["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quantization_yaml["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

def load_model(model_config, quantization_yaml):
    model_name = model_config["name"]
    quantization_config = maybe_get_quantization_config(quantization_yaml)
    model_kwargs = {"trust_remote_code": True,
                    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    return model

# ------------------------------------------------------------------------------
# LoRA
# ------------------------------------------------------------------------------
def build_peft_config(lora_yaml):
    return LoraConfig(
        r=lora_yaml["r"],
        lora_alpha=lora_yaml["lora_alpha"],
        lora_dropout=lora_yaml["lora_dropout"],
        bias=lora_yaml["bias"],
        task_type=lora_yaml["task_type"],
        target_modules=lora_yaml["target_modules"],
    )

# ------------------------------------------------------------------------------
# Training config
# ------------------------------------------------------------------------------
def build_sft_config(training_yaml, output_directory):
    max_length = training_yaml.get("max_length", training_yaml.get("max_seq_length"))
    eval_strategy = training_yaml.get("eval_strategy", training_yaml.get("evaluation_strategy"))

    config_kwargs = {
        "output_dir": output_directory,
        "max_length": max_length,
        "per_device_train_batch_size": training_yaml["per_device_train_batch_size"],
        "per_device_eval_batch_size": training_yaml["per_device_eval_batch_size"],
        "gradient_accumulation_steps": training_yaml["gradient_accumulation_steps"],
        "num_train_epochs": training_yaml["num_train_epochs"],
        "learning_rate": training_yaml["learning_rate"],
        "warmup_steps": training_yaml["warmup_steps"],
        "logging_steps": training_yaml["logging_steps"],
        "eval_strategy": eval_strategy,
        "eval_steps": training_yaml["eval_steps"],
        "save_steps": training_yaml["save_steps"],
        "save_total_limit": training_yaml["save_total_limit"],
        "bf16": training_yaml.get("bf16", False),
        "report_to": training_yaml.get("report_to", []),
    }
    if "gradient_checkpointing" in training_yaml:
        config_kwargs["gradient_checkpointing"] = training_yaml["gradient_checkpointing"]
    if "packing" in training_yaml:
        config_kwargs["packing"] = training_yaml["packing"]

    return SFTConfig(**config_kwargs)

# ------------------------------------------------------------------------------
# Metadata saving
# ------------------------------------------------------------------------------
def save_run_metadata(
    output_directory,
    sft_config,
    data_config,
    exp_config,
    model_config,
    prepared_data_path,
):
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

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def train_model(sft_config_path=None):
    (
        sft_config,
        data_config,
        exp_config,
        model_config,
        prepared_data_path,
        output_directory,
    ) = load_configs(sft_config_path=sft_config_path)

    seed = sft_config["run"]["seed"]
    system_prompt = sft_config["prompting"]["system_prompt"]
    model_name = model_config["name"]

    print("Experiment:", sft_config["run"]["experiment_name"])
    print("Model key:", sft_config["run"]["model_key"])
    print("Model name:", model_name)
    print("Prepared data path:", prepared_data_path)
    print("Output directory:", output_directory)
    set_seeds(seed)
    ds_train_raw, ds_val_raw, ds_test_raw = load_prepared_splits(prepared_data_path)
    ds_train, ds_val = format_sft_splits(ds_train_raw, ds_val_raw, system_prompt)

    tok = load_tokenizer(model_name)
    model = load_model(model_config, sft_config["quantization"])
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
    )
    print("Starting training")
    trainer.train()

    print("Saving model + tokenizer")
    trainer.save_model(sft_trainer_config.output_dir)
    tok.save_pretrained(sft_trainer_config.output_dir)

    print("Done. Saved to:", sft_trainer_config.output_dir)
    print("Test split was loaded but not used for training:", len(ds_test_raw))