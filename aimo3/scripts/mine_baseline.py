import json
import os
from collections import Counter

from datasets import Dataset
from tqdm import tqdm

from . import helpers
from .mine import (
    build_prompt,
    generate_n_solutions,
    get_mining_reasons,
    load_mining_split,
    load_model_and_tokenizer,
    majority_vote_answer,
    normalize_to_int_str,
)


def load_configs(mine_config_path=None):
    """
    Load baseline mining config and resolve all runtime paths/sections.
    """
    if mine_config_path is None:
        mine_config_path = helpers.get_config_path("mine_baseline.yaml")
    mine_config = helpers.load_yaml(mine_config_path)

    data_config_path = mine_config.get("paths", {}).get("data_config_path")
    if not data_config_path:
        data_config_path = helpers.get_config_path("data.yaml")
    data_config = helpers.load_yaml(data_config_path)

    run_cfg = mine_config.get("run", {})
    paths = mine_config.get("paths", {})
    generation = mine_config.get("generation", {})
    criteria = mine_config.get("failure_criteria", {})
    reporting = mine_config.get("reporting", {})

    model_key = run_cfg.get("model_key")
    if not model_key:
        raise ValueError("run.model_key is required in mine_baseline config")

    model_name_or_path = paths.get("model_name_or_path")
    if not model_name_or_path:
        model_name_or_path = data_config["models"][model_key]["name"]

    dataset_path = paths.get("dataset_path")
    if not dataset_path:
        raise ValueError("paths.dataset_path is required in mine_baseline config")

    dataset_split = paths.get("dataset_split", "test")
    max_items = paths.get("max_items")

    output_path = paths.get("output_path")
    if not output_path:
        exp_name = run_cfg.get("experiment_name", "baseline")
        output_path = os.path.join(
            "runs",
            "rl_train",
            f"mined_{model_key}_{exp_name}_baseline",
        )

    return (
        mine_config,
        data_config,
        model_name_or_path,
        dataset_path,
        dataset_split,
        output_path,
        max_items,
        generation,
        criteria,
        reporting,
    )


def mine_failures_baseline(mine_config_path=None):
    """
    Mine RL examples using a baseline model (foundation model, not SFT checkpoint).
    """
    (
        mine_config,
        _data_config,
        model_name_or_path,
        dataset_path,
        dataset_split,
        output_path,
        max_items,
        generation_cfg,
        criteria_cfg,
        reporting_cfg,
    ) = load_configs(mine_config_path)

    seed = int(mine_config.get("run", {}).get("seed", 42))
    helpers.set_seeds(seed)

    k = int(generation_cfg.get("k_solutions", 8))
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 512))
    temperature = float(generation_cfg.get("temperature", 0.7))
    top_p = float(generation_cfg.get("top_p", 0.95))

    print("Baseline mining model:", model_name_or_path)
    print("Mining dataset:", dataset_path)
    print("Mining split:", dataset_split)

    ds = load_mining_split(dataset_path, dataset_split)
    if max_items is not None:
        capped_total = min(int(max_items), len(ds))
        ds = ds.select(range(capped_total))
        print(f"Applied max_items cap: {capped_total}")

    tokenizer, model = load_model_and_tokenizer(model_name_or_path)

    mined_rows = []
    stats = {
        "total_examples": len(ds),
        "mined_examples": 0,
        "skipped_invalid_gold": 0,
        "reasons_count": Counter(),
    }

    for row in tqdm(ds, desc="Mining examples (baseline)", unit="example"):
        problem = row.get("problem", "")
        gold_raw = row.get("expected_answer", None)
        gold = normalize_to_int_str(gold_raw)

        if gold is None:
            stats["skipped_invalid_gold"] += 1
            continue

        prompt = build_prompt(problem)

        generations = generate_n_solutions(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            n=k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        majority, counts, extracted, agreement = majority_vote_answer(generations)

        has_any_valid = len(extracted) > 0
        format_validity_rate = len(extracted) / float(k) if k > 0 else 0.0

        reasons = get_mining_reasons(
            majority_answer=majority,
            gold_answer=gold,
            agreement=agreement,
            format_validity_rate=format_validity_rate,
            has_any_valid=has_any_valid,
            criteria=criteria_cfg,
        )

        if reasons:
            for reason in reasons:
                stats["reasons_count"][reason] += 1

            mined_rows.append(
                {
                    "prompt": prompt,
                    "expected_answer": gold,
                    "problem": str(problem),
                    "majority_answer": majority,
                    "agreement": float(agreement),
                    "format_validity_rate": float(format_validity_rate),
                    "n_valid_extractions": int(len(extracted)),
                    "vote_counts": dict(counts),
                    "mine_reasons": reasons,
                }
            )

    stats["mined_examples"] = len(mined_rows)

    os.makedirs(output_path, exist_ok=True)

    if mined_rows:
        mined_ds = Dataset.from_list(mined_rows)
    else:
        mined_ds = Dataset.from_dict(
            {
                "prompt": [],
                "expected_answer": [],
                "problem": [],
                "majority_answer": [],
                "agreement": [],
                "format_validity_rate": [],
                "n_valid_extractions": [],
                "vote_counts": [],
                "mine_reasons": [],
            }
        )

    mined_ds.save_to_disk(output_path)
    print(f"Saved mined baseline dataset to: {output_path}")
    print(f"Mined examples: {len(mined_rows)} / {len(ds)}")

    if reporting_cfg.get("save_mining_report", True):
        report_path = reporting_cfg.get("report_path", "runs/mining_report_baseline.json")
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)

        report = {
            "config_path": str(mine_config_path if mine_config_path else helpers.get_config_path("mine_baseline.yaml")),
            "dataset_path": dataset_path,
            "dataset_split": dataset_split,
            "max_items": int(max_items) if max_items is not None else None,
            "model_name_or_path": model_name_or_path,
            "output_path": output_path,
            "generation": {
                "k_solutions": k,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            "failure_criteria": criteria_cfg,
            "stats": {
                "total_examples": stats["total_examples"],
                "mined_examples": stats["mined_examples"],
                "mined_rate": (stats["mined_examples"] / stats["total_examples"]) if stats["total_examples"] else 0.0,
                "skipped_invalid_gold": stats["skipped_invalid_gold"],
                "reasons_count": dict(stats["reasons_count"]),
            },
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved baseline mining report to: {report_path}")
