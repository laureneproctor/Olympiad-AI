import argparse
import json
import os
from pathlib import Path

import optuna
import yaml

from . import helpers
from .evaluate import run_evaluation
from .sft import train_model


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def get_metric(results, metric_name):
    if metric_name == "pass1":
        return float(results["pass1"]["pass@1"])
    if metric_name == "majn":
        key = next((k for k in results.get("majn", {}) if k.startswith("maj@")), None)
        if key is None:
            raise ValueError("Could not find maj@N metric in evaluation results.")
        return float(results["majn"][key])
    raise ValueError(f"Unsupported metric_name: {metric_name}")


def build_trial_sft_config(base_sft, trial, output_root, force_packing_false):
    cfg = dict(base_sft)
    cfg["run"] = dict(base_sft.get("run", {}))
    cfg["paths"] = dict(base_sft.get("paths", {}))
    cfg["training"] = dict(base_sft.get("training", {}))

    cfg["paths"]["output_root"] = output_root
    cfg["training"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    cfg["training"]["gradient_accumulation_steps"] = trial.suggest_categorical(
        "gradient_accumulation_steps", [4, 8, 16]
    )
    cfg["training"]["max_seq_length"] = trial.suggest_categorical("max_seq_length", [1024, 1536, 2048])
    cfg["training"]["warmup_steps"] = trial.suggest_categorical("warmup_steps", [10, 50, 100])

    # Keep external logging disabled during local tuning unless already requested by user.
    cfg["training"].setdefault("report_to", [])

    if force_packing_false:
        cfg["training"]["packing"] = False

    return cfg


def build_trial_eval_config(base_eval, trial_sft_path, checkpoint_path, output_dir, max_items, majn_max_items):
    cfg = dict(base_eval)
    cfg["paths"] = dict(base_eval.get("paths", {}))
    cfg["evaluation"] = dict(base_eval.get("evaluation", {}))
    cfg["reporting"] = dict(base_eval.get("reporting", {}))

    cfg["paths"]["sft_config_path"] = trial_sft_path
    cfg["paths"]["model_checkpoint"] = checkpoint_path
    cfg["paths"]["output_dir"] = output_dir

    if max_items is not None:
        cfg["evaluation"]["max_items"] = max_items
    if majn_max_items is not None:
        cfg["evaluation"]["majn_max_items"] = majn_max_items

    cfg["reporting"]["save_report"] = True
    cfg["reporting"]["report_filename"] = "evaluation_report_trial.json"
    cfg["reporting"]["verbose"] = True

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Tune SFT hyperparameters with Optuna.")
    parser.add_argument(
        "--base-sft-config",
        type=str,
        default=str(helpers.get_config_path("sft.yaml")),
        help="Path to base sft.yaml.",
    )
    parser.add_argument(
        "--base-eval-config",
        type=str,
        default=str(helpers.get_config_path("evaluate.yaml")),
        help="Path to base evaluate.yaml.",
    )
    parser.add_argument("--n-trials", type=int, default=5, help="Number of Optuna trials.")
    parser.add_argument(
        "--workdir",
        type=str,
        default="runs/optuna_sft",
        help="Directory for trial configs, reports, and best params.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="pass1",
        choices=["pass1", "majn"],
        help="Optimization target metric.",
    )
    parser.add_argument(
        "--eval-max-items",
        type=int,
        default=100,
        help="Cap for pass@1 evaluation rows per trial.",
    )
    parser.add_argument(
        "--eval-majn-max-items",
        type=int,
        default=50,
        help="Cap for maj@N evaluation rows per trial.",
    )
    parser.add_argument(
        "--allow-packing",
        action="store_true",
        help="Allow packing from base config. By default, packing is forced off for safer non-flash-attn runs.",
    )
    args = parser.parse_args()

    base_sft = load_yaml(args.base_sft_config)
    base_eval = load_yaml(args.base_eval_config)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    exp_name = base_sft["run"]["experiment_name"]
    model_key = base_sft["run"]["model_key"]

    def objective(trial):
        trial_dir = workdir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        trial_output_root = str(trial_dir / "models")
        trial_sft = build_trial_sft_config(
            base_sft=base_sft,
            trial=trial,
            output_root=trial_output_root,
            force_packing_false=not args.allow_packing,
        )
        trial_sft_path = str(trial_dir / "sft.yaml")
        save_yaml(trial_sft_path, trial_sft)

        train_model(trial_sft_path)

        checkpoint_path = os.path.join(trial_output_root, f"sft_{model_key}_{exp_name}")
        trial_eval = build_trial_eval_config(
            base_eval=base_eval,
            trial_sft_path=trial_sft_path,
            checkpoint_path=checkpoint_path,
            output_dir=str(trial_dir / "eval"),
            max_items=args.eval_max_items,
            majn_max_items=args.eval_majn_max_items,
        )
        trial_eval_path = str(trial_dir / "evaluate.yaml")
        save_yaml(trial_eval_path, trial_eval)

        results = run_evaluation(trial_eval_path)
        score = get_metric(results, args.metric)

        trial.set_user_attr("checkpoint_path", checkpoint_path)
        trial.set_user_attr("results", results)
        trial.set_user_attr("score", score)

        with open(trial_dir / "summary.json", "w") as f:
            json.dump(
                {
                    "trial": trial.number,
                    "params": trial.params,
                    "score": score,
                    "metric": args.metric,
                    "checkpoint_path": checkpoint_path,
                    "results": results,
                },
                f,
                indent=2,
            )

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    best = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "metric": args.metric,
        "best_params": study.best_trial.params,
        "checkpoint_path": study.best_trial.user_attrs.get("checkpoint_path"),
    }

    best_path = workdir / "best_result.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

    print("Best trial summary:")
    print(json.dumps(best, indent=2))
    print(f"Saved best result to: {best_path}")


if __name__ == "__main__":
    main()
