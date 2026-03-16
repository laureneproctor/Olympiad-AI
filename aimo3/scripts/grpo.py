# Input: data/rl_train/ + starting model checkpoint (SFT model)
#   - For each prompt:
#       - sample a group of outputs (group_size)
#       - extract predicted final int from each
#       - compute reward for each output (correct/wrong/invalid/no-answer)
#       - update model to increase the probability of higher-reward outputs
#       - apply KL penalty to stay close to the SFT model
# Output:
# - checkpoints/grpo_model/ (your RL-tuned checkpoint)

import os
import re
import random
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

# ------------------------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------------------------
def get_repo_root():
    return Path(__file__).resolve().parents[1]

def get_grpo_config_path():
    return get_repo_root() / "configs" / "grpo.yaml"

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
def load_configs(grpo_config_path=None):
    if grpo_config_path is None:
        grpo_config_path = get_grpo_config_path()
    grpo_config = load_yaml(grpo_config_path)
    data_config_path = grpo_config.get("paths", {}).get("data_config_path")
    if not data_config_path:
        data_config_path = get_data_config_path()
    data_config = load_yaml(data_config_path)
    exp_name = grpo_config["run"]["experiment_name"]
    model_key = grpo_config["run"]["model_key"]
    exp_config = data_config["experiments"][exp_name]
    model_config = data_config["models"][model_key]
    starting_checkpoint = grpo_config["paths"]["starting_checkpoint"]
    rl_train_path = grpo_config["paths"]["rl_train_path"]
    output_directory = os.path.join(
        grpo_config["paths"]["output_root"],
        f"grpo_{model_key}_{exp_name}",
    )
    return (
        grpo_config,
        data_config,
        exp_config,
        model_config,
        starting_checkpoint,
        rl_train_path,
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
# Dataset loading
# ------------------------------------------------------------------------------
def load_rl_dataset(rl_train_path):
    """
    Supports:
      - dataset saved with datasets.save_to_disk(...)
      - a DatasetDict with split 'train'
    """
    print("Loading RL dataset from:", rl_train_path)

    if not os.path.exists(rl_train_path):
        raise FileNotFoundError(f"RL dataset path not found: {rl_train_path}")
    ds = load_from_disk(rl_train_path)
    if hasattr(ds, "keys") and "train" in ds:
        ds = ds["train"]
    print("RL train rows:", len(ds))
    print("Columns:", ds.column_names)
    required = {"prompt", "expected_answer"}
    missing = required - set(ds.column_names)
    if missing:
        raise KeyError(
            f"RL dataset missing required columns: {missing}. "
            f"Expected at least {required}"
        )
    return ds


# ------------------------------------------------------------------------------
# Prompt formatting
# ------------------------------------------------------------------------------
def normalize_prompt(example, system_prompt):
    prompt = str(example["prompt"]).strip()
    answer = str(example["expected_answer"]).strip()
    full_prompt = f"{system_prompt}\n\n{prompt}"
    return {
        "prompt": full_prompt,
        "expected_answer": answer,
    }


def format_rl_dataset(ds, system_prompt):
    print("Formatting RL dataset")
    ds = ds.map(
        normalize_prompt,
        fn_kwargs={"system_prompt": system_prompt},
        desc="Formatting RL dataset",
    )

    print("Formatted columns:", ds.column_names)
    return ds


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
    tok.padding_side = "left"
    return tok

# ------------------------------------------------------------------------------
# Quantization
# ------------------------------------------------------------------------------
def maybe_get_quantization_config(quantization_yaml):
    enabled = quantization_yaml.get("enabled", True)
    if not enabled:
        return None

    dtype_name = quantization_yaml["bnb_4bit_compute_dtype"]
    if dtype_name == "bfloat16":
        compute_dtype = torch.bfloat16
    elif dtype_name == "float16":
        compute_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported compute dtype: {dtype_name}")

    return BitsAndBytesConfig(
        load_in_4bit=quantization_yaml["load_in_4bit"],
        bnb_4bit_quant_type=quantization_yaml["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quantization_yaml["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )


# ------------------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------------------
def load_model_for_grpo(starting_checkpoint, quantization_yaml):
    quantization_config = maybe_get_quantization_config(quantization_yaml)

    model_kwargs = {
        "trust_remote_code": True,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        starting_checkpoint,
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
# Answer extraction
# ------------------------------------------------------------------------------
def extract_final_answer(text):
    """
    Looks for:
      FINAL_ANSWER: 123
    and falls back to last standalone integer if needed.
    Returns:
      str or None
    """
    if text is None:
        return None
    text = str(text)
    match = re.search(r"FINAL_ANSWER\s*:\s*(-?\d+)", text)
    if match:
        return match.group(1)
    matches = re.findall(r"(?<!\d)-?\d+(?!\d)", text)
    if matches:
        return matches[-1]

    return None


def normalize_integer_string(value):
    """
    Converts to canonical integer string and filters to [0, 99999].
    Returns None if invalid.
    """
    try:
        ivalue = int(str(value).strip())
    except Exception:
        return None

    if ivalue < 0 or ivalue > 99999:
        return None

    return str(ivalue)


# ------------------------------------------------------------------------------
# Reward functions
# ------------------------------------------------------------------------------
def reward_correctness(completions, expected_answer, **kwargs):
    """
    Main task reward:
      correct answer -> +1.0
      wrong valid integer -> 0.0
      invalid / no answer -> -1.0
    """
    rewards = []

    for completion, gold in zip(completions, expected_answer):
        pred = extract_final_answer(completion)
        pred = normalize_integer_string(pred)
        gold = normalize_integer_string(gold)
        if pred is None:
            rewards.append(-1.0)
        elif gold is not None and pred == gold:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def reward_format(completions, **kwargs):
    """
    Small positive reward for obeying the expected format.
    """
    rewards = []
    for completion in completions:
        text = str(completion)
        ok = re.search(r"FINAL_ANSWER\s*:\s*-?\d+", text) is not None
        rewards.append(0.2 if ok else -0.2)
    return rewards

def reward_range(completions, **kwargs):
    """
    Small reward if extracted answer is within allowed competition range.
    """
    rewards = []
    for completion in completions:
        pred = extract_final_answer(completion)
        pred = normalize_integer_string(pred)
        rewards.append(0.1 if pred is not None else -0.1)
    return rewards

def build_reward_functions():
    return [
        reward_correctness,
        reward_format,
        reward_range,
    ]

# ------------------------------------------------------------------------------
# GRPO config
# ------------------------------------------------------------------------------
def build_grpo_training_config(training_yaml, output_directory):
    config_kwargs = {
        "output_dir": output_directory,
        "learning_rate": training_yaml["learning_rate"],
        "per_device_train_batch_size": training_yaml["per_device_train_batch_size"],
        "gradient_accumulation_steps": training_yaml["gradient_accumulation_steps"],
        "num_train_epochs": training_yaml.get("num_train_epochs", 1),
        "max_prompt_length": training_yaml["max_prompt_length"],
        "max_completion_length": training_yaml["max_completion_length"],
        "num_generations": training_yaml["group_size"],   # important
        "beta": training_yaml.get("beta", 0.04),          # KL penalty strength
        "logging_steps": training_yaml["logging_steps"],
        "save_steps": training_yaml["save_steps"],
        "save_total_limit": training_yaml["save_total_limit"],
        "bf16": training_yaml.get("bf16", False),
        "report_to": training_yaml.get("report_to", []),
    }

    optional_keys = [
        "max_steps",
        "warmup_steps",
        "lr_scheduler_type",
        "gradient_checkpointing",
        "weight_decay",
        "max_grad_norm",
        "seed",
        "scale_rewards",
        "loss_type",
    ]

    for key in optional_keys:
        if key in training_yaml:
            config_kwargs[key] = training_yaml[key]

    return GRPOConfig(**config_kwargs)

# ------------------------------------------------------------------------------
# Metadata saving
# ------------------------------------------------------------------------------
def save_run_metadata(
    output_directory,
    grpo_config,
    data_config,
    exp_config,
    model_config,
    starting_checkpoint,
    rl_train_path,
):
    os.makedirs(output_directory, exist_ok=True)

    with open(os.path.join(output_directory, "grpo_config_used.yaml"), "w") as f:
        yaml.safe_dump(grpo_config, f, sort_keys=False)

    with open(os.path.join(output_directory, "data_config_used.yaml"), "w") as f:
        yaml.safe_dump(data_config, f, sort_keys=False)

    summary = {
        "experiment_name": grpo_config["run"]["experiment_name"],
        "model_key": grpo_config["run"]["model_key"],
        "model_name": model_config["name"],
        "starting_checkpoint": starting_checkpoint,
        "rl_train_path": rl_train_path,
        "output_directory": output_directory,
    }

    with open(os.path.join(output_directory, "run_summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

# ------------------------------------------------------------------------------
# Trainer construction
# ------------------------------------------------------------------------------
def build_trainer(model, tokenizer, peft_config, train_dataset, trainer_config):
    reward_funcs = build_reward_functions()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=trainer_config,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    return trainer

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def train_grpo(grpo_config_path=None):
    (
        grpo_config,
        data_config,
        exp_config,
        model_config,
        starting_checkpoint,
        rl_train_path,
        output_directory,
    ) = load_configs(grpo_config_path=grpo_config_path)

    seed = grpo_config["run"]["seed"]
    system_prompt = grpo_config["prompting"]["system_prompt"]

    print("Experiment:", grpo_config["run"]["experiment_name"])
    print("Model key:", grpo_config["run"]["model_key"])
    print("Base model:", model_config["name"])
    print("Starting checkpoint:", starting_checkpoint)
    print("RL train path:", rl_train_path)
    print("Output directory:", output_directory)

    set_seeds(seed)

    ds_train = load_rl_dataset(rl_train_path)
    ds_train = format_rl_dataset(ds_train, system_prompt)

    tokenizer = load_tokenizer(starting_checkpoint)
    model = load_model_for_grpo(starting_checkpoint, grpo_config["quantization"])
    peft_config = build_peft_config(grpo_config["lora"])
    trainer_config = build_grpo_training_config(
        grpo_config["training"],
        output_directory,
    )

    save_run_metadata(
        output_directory,
        grpo_config,
        data_config,
        exp_config,
        model_config,
        starting_checkpoint,
        rl_train_path,
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=ds_train,
        trainer_config=trainer_config,
    )

    print("Starting GRPO training")
    trainer.train()

    print("Saving GRPO model + tokenizer")
    trainer.save_model(trainer_config.output_dir)
    tokenizer.save_pretrained(trainer_config.output_dir)

    print("Done. Saved to:", trainer_config.output_dir)


if __name__ == "__main__":
    train_grpo()