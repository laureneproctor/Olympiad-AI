import os

from . import helpers
from .grpo import (
    build_grpo_training_config,
    build_peft_config,
    build_trainer,
    format_rl_dataset,
    load_model_for_grpo,
    load_rl_dataset,
    load_tokenizer,
    save_run_metadata,
)


def load_configs(grpo_config_path=None):
    """
    Load baseline GRPO config and resolve all runtime paths/sections.
    """
    if grpo_config_path is None:
        grpo_config_path = helpers.get_config_path("grpo_baseline.yaml")

    grpo_config = helpers.load_yaml(grpo_config_path)

    data_config_path = grpo_config.get("paths", {}).get("data_config_path")
    if not data_config_path:
        data_config_path = helpers.get_config_path("data.yaml")
    data_config = helpers.load_yaml(data_config_path)

    run_cfg = grpo_config.get("run", {})
    paths = grpo_config.get("paths", {})

    exp_name = run_cfg.get("experiment_name", "baseline")
    model_key = run_cfg.get("model_key")
    if not model_key:
        raise ValueError("run.model_key is required in grpo_baseline config")

    model_config = data_config["models"][model_key]
    exp_config = data_config.get("experiments", {}).get(exp_name, {})

    starting_model_or_checkpoint = paths.get("starting_model_or_checkpoint")
    if not starting_model_or_checkpoint:
        starting_model_or_checkpoint = model_config["name"]

    rl_train_path = paths.get("rl_train_path")
    if not rl_train_path:
        raise ValueError("paths.rl_train_path is required in grpo_baseline config")

    output_root = paths.get("output_root", "runs/models")
    output_directory = paths.get("output_directory")
    if not output_directory:
        output_directory = os.path.join(
            output_root,
            f"grpo_baseline_{model_key}_{exp_name}",
        )

    return (
        grpo_config,
        data_config,
        exp_config,
        model_config,
        starting_model_or_checkpoint,
        rl_train_path,
        output_directory,
    )


def train_grpo_baseline(grpo_config_path=None):
    """
    Train GRPO from a baseline model (foundation model, not SFT checkpoint).
    """
    (
        grpo_config,
        data_config,
        exp_config,
        model_config,
        starting_model_or_checkpoint,
        rl_train_path,
        output_directory,
    ) = load_configs(grpo_config_path=grpo_config_path)

    seed = grpo_config["run"].get("seed", 42)
    system_prompt = grpo_config["prompting"]["system_prompt"]

    print("Experiment:", grpo_config["run"].get("experiment_name", "baseline"))
    print("Model key:", grpo_config["run"]["model_key"])
    print("Base model:", model_config["name"])
    print("Starting model/checkpoint:", starting_model_or_checkpoint)
    print("RL train path:", rl_train_path)
    print("Output directory:", output_directory)

    helpers.set_seeds(seed)

    ds_train = load_rl_dataset(rl_train_path)
    ds_train = format_rl_dataset(ds_train, system_prompt)

    tokenizer = load_tokenizer(starting_model_or_checkpoint)
    model = load_model_for_grpo(starting_model_or_checkpoint, grpo_config["quantization"])
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
        starting_model_or_checkpoint,
        rl_train_path,
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=ds_train,
        trainer_config=trainer_config,
    )

    print("Starting baseline GRPO training")
    trainer.train()

    print("Saving baseline GRPO model + tokenizer")
    trainer.save_model(trainer_config.output_dir)
    tokenizer.save_pretrained(trainer_config.output_dir)

    print("Done. Saved to:", trainer_config.output_dir)
