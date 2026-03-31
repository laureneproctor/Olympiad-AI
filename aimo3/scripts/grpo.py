# Input: data/rl_train/ + starting model checkpoint (SFT model)
#   - For each prompt:
#       - sample a group of outputs (group_size)
#       - extract predicted final int from each
#       - compute reward for each output (correct/wrong/invalid/no-answer)
#       - update model to increase the probability of higher-reward outputs
#       - apply KL penalty to stay close to the SFT model
# Output:
# - checkpoints/grpo_model/ (RL-tuned checkpoint)

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
from . import helpers

# ------------------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------------------
def load_configs(grpo_config_path=None):
    """ 
    This function loads the GRPO config and the data config, and extracts relevant sections for the experiment.
    Input:
    - grpo_config_path: Optional path to the GRPO config YAML file. If None, defaults to "grpo.yaml" in the config directory.
    Output: Tuple containing:
    - grpo_config: The full GRPO config dictionary.
    - data_config: The full data config dictionary.
    - exp_config: The experiment-specific config extracted from the data config using the experiment name in the GRPO config.
    - model_config: The model-specific config extracted from the data config using the model key in the GRPO config.
    - starting_checkpoint: The path to the starting model checkpoint, extracted from the GRPO config.
    - rl_train_path: The path to the RL training dataset, extracted from the GRPO config.
    - output_directory: The directory where outputs will be saved, constructed from the GRPO config paths and identifiers.
    """
    if grpo_config_path is None:
        grpo_config_path = helpers.get_config_path("grpo.yaml")

    grpo_config = helpers.load_yaml(grpo_config_path)
    data_config_path = grpo_config.get("paths", {}).get("data_config_path")

    if not data_config_path:
        data_config_path = helpers.get_config_path("data.yaml")

    data_config = helpers.load_yaml(data_config_path)

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
# Dataset loading
# ------------------------------------------------------------------------------
def load_rl_dataset(rl_train_path):
    """
    This function loads the RL training dataset from the specified path, checks for required columns, and returns the dataset object.
    Input:
        - rl_train_path: The path to the RL training dataset, expected to be in a format
        compatible with Hugging Face Datasets (e.g., a directory containing a saved dataset).
    Output:
        - A Hugging Face Dataset object containing the training data, with at least the columns "prompt
        and "expected_answer". If the dataset is a DatasetDict with a "train" split, it will return that split.
        Otherwise, it returns the loaded dataset as is.
    Raises:
        - FileNotFoundError: If the specified dataset path does not exist.
        - KeyError: If the loaded dataset does not contain the required columns "prompt" and "expected_answer".
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
    """
    This function takes a dataset example and a system prompt, and constructs the full prompt to be fed to the model.
    It ensures that the "prompt" and "expected_answer" fields are strings, and concatenates the system prompt with the
    example prompt. The expected answer is also stripped of leading/trailing whitespace and returned in the output dictionary.
    Input:
        - example: A dictionary containing at least the keys "prompt" and "expected_answer".
        - system_prompt: A string containing the system prompt to be prepended to the example prompt
    Output:
        - A dictionary with the following keys:
        - "prompt": The full prompt string, constructed by concatenating the system prompt and the example prompt.
        - "expected_answer": The expected answer string, stripped of leading/trailing whitespace. 
    """
    # Ensure prompt and expected_answer are strings and strip whitespace
    prompt = str(example["prompt"]).strip()
    answer = str(example["expected_answer"]).strip()
    # Concatenate system prompt and example prompt to form the full prompt
    full_prompt = f"{system_prompt}\n\n{prompt}"
    # Return the full prompt and the expected answer in a new dictionary
    return {
        "prompt": full_prompt,
        "expected_answer": answer,
    }

def format_rl_dataset(ds, system_prompt):
    """
    This function applies the normalize_prompt function to each example in the dataset, using the provided system prompt.
    It returns a new dataset with the formatted prompts and expected answers, and prints the columns of the formatted dataset.
    Input:
    - ds: A Hugging Face Dataset object containing the training data, with at least the columns "prompt" and "expected_answer".
    - system_prompt: A string containing the system prompt to be prepended to each example prompt.
    Output:
    - A new Hugging Face Dataset object where each example has been processed by the normalize_prompt function, resulting in
     a "prompt" that includes the system prompt and the original prompt, and an "expected_answer" that is stripped of whitespace.
    The function also prints the columns of the formatted dataset for verification.
    """
    # Map
    print("Formatting RL dataset")
    ds = ds.map(
        normalize_prompt,
        fn_kwargs={"system_prompt": system_prompt},
        desc="Formatting RL dataset",
    )
    # Print the columns of the formatted dataset for verification
    print("Formatted columns:", ds.column_names)
    return ds


# ------------------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------------------
def load_tokenizer(model_name_or_path):
    """
    This function loads a tokenizer from the specified model name or path, ensuring that it has a padding token defined.
    If token is none it sets the padding token to be the same as the end-of-sequence token. It also sets the padding side 
    to "left" to ensure that prompts are padded on the left side.
    """
    # Load the tokenizer, allowing for fast tokenization and trusting remote code if necessary
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )
    # Ensure the tokenizer has a padding token defined; if not, set it to be the same as the end-of-sequence token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Set the padding side to "left" to ensure that prompts are padded on the left side
    tok.padding_side = "left"
    return tok

# ------------------------------------------------------------------------------
# Quantization
# ------------------------------------------------------------------------------
def maybe_get_quantization_config(quantization_yaml):
    """
    In here we read the quantization config from the YAML, and if quantization is enabled, we construct and 
    return a BitsAndBytesConfig object with the appropriate settings. If quantization is not enabled, we return None.
    Input:
        - quantization_yaml: A dictionary containing the quantization settings, expected to have at least the key "enabled" (boolean) and, if enabled, the keys "bnb_4bit_compute
        _dtype", "load_in_4bit", "bnb_4bit_quant_type", and "bnb_4bit_use_double_quant".
    Output:
        - If quantization is enabled, returns a BitsAndBytesConfig object configured according to the settings in the quantization_yaml. If quantization is not enabled, returns None.
    """
    # Check if quantization is enabled in the YAML config; if not, return None to indicate no quantization configuration
    enabled = quantization_yaml.get("enabled", True)
    if not enabled:
        return None
    # Read the compute dtype setting from the YAML and convert it to the corresponding torch dtype
    dtype_name = quantization_yaml["bnb_4bit_compute_dtype"]
    # Map the string dtype name to the corresponding torch dtype; raise an error if the dtype is unsupported
    if dtype_name == "bfloat16":
        compute_dtype = torch.bfloat16
    elif dtype_name == "float16":
        compute_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported compute dtype: {dtype_name}")
    # Return object with quantization settings based on the YAML config, which will be used when loading the model to enable 4-bit quantization with the specified options
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
    """
    This function loads the model for GRPO training from the specified starting checkpoint, applying 
    quantization settings if enabled in the YAML config.
    Input:
        - starting_checkpoint: The path to the model checkpoint to be loaded, which can be a local path or a model identifier from Hugging Face Model Hub.
        - quantization_yaml: A dictionary containing the quantization settings, expected to have at least the key "enabled" (boolean) and, if enabled, the 
        keys "bnb_4bit_compute_dtype", "load_in_4bit", "bnb_4bit_quant_type", and "bnb_4bit_use_double_quant".
    Output:
        - A model loaded from the specified checkpoint, with quantization applied if enabled. If quantization is enabled, the model will be prepared for 
        k-bit training using the appropriate settings. The model will also have its config updated to disable caching, which is important for RL training 
        where we need to compute rewards based on the generated outputs.
    """
    quantization_config = maybe_get_quantization_config(quantization_yaml)

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

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
    """
    This function reads the LoRA configuration from the provided YAML dictionary and constructs a LoraConfig object with the appropriate settings.
    Input:
        - lora_yaml: A dictionary containing the LoRA settings, expected to have the keys "r", "lora_alpha", "lora_dropout", "bias", "task_type",
        and "target_modules".
    Output:
        - A LoraConfig object configured according to the settings in the lora_yaml, which will be used for applying LoRA fine-tuning to the model during GRPO training.
    """
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
    This function compares the predicted answer extracted from the model's completion with the expected answer. 
    It assigns a reward of +1.0 if the predicted answer matches the expected answer, 0.0 if the predicted answer is 
    a valid integer but does not match the expected answer, and -1.0 if the predicted answer is invalid or if no answer 
    could be extracted. The function returns a list of rewards corresponding to each completion in the input list.
    Input:
        - completions: A list of model-generated completions (strings) from which the predicted answers will be extracted.
        - expected_answer: A list of expected answer strings corresponding to each completion, which will be used to determine the correctness of the predicted answers.
    Output:
        - A list of rewards (floats) where each reward corresponds to a completion in the input list. The reward is +1.0 if the predicted answer matches the expected answer,
        0.0 if the predicted answer is a valid integer but does not match the expected answer, and -1.0 if the predicted answer is invalid or if no answer could be extracted.
    """
    # Rewards list to store the computed reward for each completion
    rewards = []
    # For each completion and its corresponding expected answer, extract the predicted answer and normalize both predicted and expected answers
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
    Here we check if the completion matches pattern "FINAL_ANSWER: <integer>". If it does, 
    we give a small positive reward (e.g., +0.2). If it doesn't, we give a small negative reward (e.g., -0.2).
    Input:
        - completions: A list of model-generated completions (strings) that will be evaluated for format adherence.
    Output:
        - A list of rewards (floats) where each reward corresponds to a completion in the input list.
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
    Input:
    - completions: A list of model-generated completions (strings) from which the predicted answers will be extracted and evaled for being within the allowed competition range (e.g., 0 to 99999).
    Output:
    - A list of rewards (floats) where each reward corresponds to a completion in the input list. The reward is +0.1 if the extracted answer is a valid integer within the
    allowed range, and -0.1 if the extracted answer is invalid or out of range.
    """
    rewards = []
    for completion in completions:
        pred = extract_final_answer(completion)
        pred = normalize_integer_string(pred)
        rewards.append(0.1 if pred is not None else -0.1)
    return rewards

def build_reward_functions():
    """
    This function constructs and returns a list of reward functions that will be used during GRPO training. 
    """
    return [
        reward_correctness,
        reward_format,
        reward_range,
    ]

# ------------------------------------------------------------------------------
# GRPO config
# ------------------------------------------------------------------------------
def build_grpo_training_config(training_yaml, output_directory):
    """
    This function reads the GRPO training configuration from the provided YAML dictionary and constructs a GRPOConfig object with the appropriate settings.
    It also filters the configuration parameters to only include those that are accepted by the installed version of the TRL library, and prints a note if
    any unsupported parameters are dropped. The resulting GRPO configuration object will be used to configure the GRPOTrainer for training the model.
    Input:
    - training_yaml: A dictionary containing the GRPO training settings, expected to have keys such as "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps"....
    - output_directory: The directory where GRPO training outputs (e.g., checkpoints, logs) will be saved, which will be included in the GRPOConfig.
    Output:
    - A GRPOConfig object configured according to the settings in the training_yaml, with unsupported parameters filtered out based on the installed TRL version. 
    This configuration object will be used to set up the GRPOTrainer for training the model with the specified hyperparameters and settings.
    """
    import inspect

    config_kwargs = {
        "output_dir": output_directory,
        "learning_rate": training_yaml["learning_rate"],
        "per_device_train_batch_size": training_yaml["per_device_train_batch_size"],
        "gradient_accumulation_steps": training_yaml["gradient_accumulation_steps"],
        "num_train_epochs": training_yaml.get("num_train_epochs", 1),
        "max_prompt_length": training_yaml["max_prompt_length"],
        "max_completion_length": training_yaml["max_completion_length"],
        "num_generations": training_yaml["group_size"],
        "beta": training_yaml.get("beta", 0.04),
        "logging_steps": training_yaml["logging_steps"],
        "save_steps": training_yaml["save_steps"],
        "save_total_limit": training_yaml["save_total_limit"],
        "bf16": training_yaml.get("bf16", False),
        "report_to": training_yaml.get("report_to", []),
    }

    optional_keys = [
        "optim",
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

    # Filter to only params the installed TRL version accepts
    valid_params = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    filtered_kwargs = {k: v for k, v in config_kwargs.items() if k in valid_params}

    dropped = set(config_kwargs.keys()) - set(filtered_kwargs.keys())
    if dropped:
        print(f"Note: dropped unsupported GRPOConfig params for this TRL version: {dropped}")

    return GRPOConfig(**filtered_kwargs)

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
    """
    This function saves the GRPO run metadata, including the GRPO config used, the data config used, and a summary of the run parameters, to the specified output directory.
    It creates the output directory if it does not exist, and saves the GRPO config and data config as separate YAML files. It also constructs a summary dictionary containing 
    key information about the run and saves it as "run_summary.yaml" in the output directory. This metadata will be useful for tracking the details of the experiment and for reproducibility.
    Input:
    - output_directory: The directory where the metadata files will be saved.
    - grpo_config: The full GRPO config dictionary used for the run, which will be saved as "grpo_config_used.yaml".
    - data_config: The full data config dictionary used for the run, which will be saved as "data_config_used.yaml".
    - exp_config: The experiment-specific config extracted from the data config, which is not saved directly but can be included in the run summary if desired.
    - model_config: The model-specific config extracted from the data config, which is not saved directly but can be included in the run summary if desired.
    - starting_checkpoint: The path to the starting model checkpoint used for the run, which will be included in the run summary.
    - rl_train_path: The path to the RL training dataset used for the run, which will be included in the run summary.
    Output:
    - Saves the GRPO config, data config, and a run summary as YAML files in the specified output directory. The run summary includes key information about the experiment name, model key, model
    name, starting checkpoint, RL training dataset path, and output directory.
    """
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
    """
    This function costructs and returns a GRPOTrainer object, which will be used to train the model using the GRPO algorithm. 
    It first builds the reward functions that will be used during training, and then initializes the GRPOTrainer with the model,
    tokenizer, reward functions, training configuration, training dataset, and PEFT configuration. The resulting trainer object 
    is returned and can be used to start the training process.
    Input:
    - model: The model to be trained using GRPO, which should be a causal language model loaded and prepared according to the specified configurations (e.g., 
    with quantization and/or LoRA applied as needed).
    - tokenizer: The tokenizer corresponding to the model, which will be used for processing the prompts and completions during training.
    - peft_config: The configuration for Parameter-Efficient Fine-Tuning (PEFT), such as LoRA settings, which will be applied to the model during training.
    - train_dataset: The training dataset to be used for GRPO training, which should be a Hugging Face Dataset object containing the formatted prompts and expected answers.
    - trainer_config: The configuration for the GRPO training process, which should be a GRPOConfig object containing the hyperparameters and settings for training 
    (e.g., learning rate, batch size, number of epochs, etc.).
    Output:
    - A GRPOTrainer object initialized with the provided model, tokenizer, reward functions, training configuration, training dataset, and PEFT configuration. 
    This trainer can be used to start the GRPO training.
    """
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
    """
    this function orchestrates the entire GRPO training process. It loads the necessary configurations, sets seeds for reproducibility, 
    loads and formats the RL training dataset, loads the tokenizer and model with the appropriate settings (including quantization and LoRA),
      saves the run metadata, builds the GRPO trainer, and starts the training process.
    
    Input:
    - grpo_config_path: Optional path to the GRPO config YAML file. If None, defaults to "grpo.yaml" in the config directory. This config file should contain all the
necessary settings for the GRPO training, including paths to data and checkpoints, model and training hyperparameters, and other relevant configurations.
    
    Output:
    - The function does not return a value, but it performs the GRPO training and saves the trained model, tokenizer, and metadata to the specified output directory as 
    defined in the GRPO config. The output directory will contain the final trained model checkpoint, the tokenizer files, and YAML files with the GRPO config used, 
    the data config used, and a summary of the run parameters.
    """ 

    (
        grpo_config,
        data_config,
        exp_config,
        model_config,
        starting_checkpoint,
        rl_train_path,
        output_directory,
    ) = load_configs(grpo_config_path=grpo_config_path)

    # get seed
    seed = grpo_config["run"]["seed"]

    # get system prompt
    system_prompt = grpo_config["prompting"]["system_prompt"]

    # print summary of experiment parameters
    print("Experiment:", grpo_config["run"]["experiment_name"])
    print("Model key:", grpo_config["run"]["model_key"])
    print("Base model:", model_config["name"])
    print("Starting checkpoint:", starting_checkpoint)
    print("RL train path:", rl_train_path)
    print("Output directory:", output_directory)

    # set seed
    helpers.set_seeds(seed)

    # load and format dataset
    ds_train = load_rl_dataset(rl_train_path)
    ds_train = format_rl_dataset(ds_train, system_prompt)

    # load tokenizer, the model and configurations for quantization and LoRA
    tokenizer = load_tokenizer(starting_checkpoint)
    model = load_model_for_grpo(starting_checkpoint, grpo_config["quantization"])
    peft_config = build_peft_config(grpo_config["lora"])
    trainer_config = build_grpo_training_config(
        grpo_config["training"],
        output_directory,
    )

    # save run data 
    save_run_metadata(
        output_directory,
        grpo_config,
        data_config,
        exp_config,
        model_config,
        starting_checkpoint,
        rl_train_path,
    )

    # building trainer
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=ds_train,
        trainer_config=trainer_config,
    )

    # begin training
    print("Starting GRPO training")
    trainer.train()

    # save model and tokenizer
    print("Saving GRPO model + tokenizer")
    trainer.save_model(trainer_config.output_dir)
    tokenizer.save_pretrained(trainer_config.output_dir)

    print("Done. Saved to:", trainer_config.output_dir)