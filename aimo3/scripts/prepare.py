# Input: OpenMathReasoning ds
# - Filters rows using our rules
# - Shuffle with seed
# - Option cap to N (debug)
# - Split into Train/Val/Test 
# - Save splits so later steps are fast and repeatable
# Output:
# - data/prepared/train
# - data/prepared/test
# - data/prepared/val

from collections import defaultdict
from datasets import load_from_disk, Dataset, DatasetDict
from pathlib import Path
import os
import random
import yaml

def get_data_config_path():
    return Path(__file__).resolve().parents[1] / "configs" / "data.yaml"

def load_yaml(config_path=None):
    if config_path is None:
        config_path = get_data_config_path()

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(dataset_path):
    print("Loading OpenMathReasoning dataset...")
    cot_data = load_from_disk(dataset_path)
    print("Loaded:", len(cot_data), "rows")
    print("Columns:", cot_data.column_names)
    return cot_data

# --------------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------------
def filter_rows(row, filtering_config):
    for key, value in filtering_config.items():
        if key == "non_null_columns":
            continue
        if row.get(key) != value:
            return False
    for col in filtering_config.get("non_null_columns", []):
        val = row.get(col)
        if val is None:
            return False
        if isinstance(val, str) and not val.strip():
            return False

    return True

def apply_filter(cot_data, config):
    filtered = cot_data.filter(lambda row: filter_rows(row, config))
    return filtered

    filtered = cot_data.filter(
        filter_rows,
        fn_kwargs={"filtering_config": filtering_config},
        load_from_cache_file=False,
        desc="Filtering rows"
    )

    print("Filtered size:", len(filtered))
    return filtered

# --------------------------------------------------------------------------------
# Take N rows
# --------------------------------------------------------------------------------
def take_n_rows(filtered, n, SEED):
    print(f"Taking {n} rows...")
    row_selection = filtered.shuffle(seed=SEED).select(range(n))
    print("Selected size:", len(row_selection))
    return row_selection

# --------------------------------------------------------------------------------
# Split Train / Val / Test by unique problem text
# --------------------------------------------------------------------------------
def split_train_val_test(row_selection, split_config, seed):
    print("Splitting into Train/Val/Test...")

    total_split = split_config["train"] + split_config["val"] + split_config["test"]
    if abs(total_split - 1.0) > 1e-8:
        raise ValueError(f"Splits must sum to 1.0, got {total_split}")

    problem_groups = defaultdict(list)

    for row in row_selection:
        problem = row["problem"].strip()
        problem_groups[problem].append(row)

    unique_problems = list(problem_groups.keys())

    rng = random.Random(seed)
    rng.shuffle(unique_problems)

    total_problems = len(unique_problems)

    train_frac = split_config["train"]
    val_frac = split_config["val"]

    train_end = int(train_frac * total_problems)
    val_end = train_end + int(val_frac * total_problems)

    train_problems = unique_problems[:train_end]
    val_problems = unique_problems[train_end:val_end]
    test_problems = unique_problems[val_end:]

    train_rows = []
    val_rows = []
    test_rows = []

    for problem in train_problems:
        train_rows.extend(problem_groups[problem])

    for problem in val_problems:
        val_rows.extend(problem_groups[problem])

    for problem in test_problems:
        test_rows.extend(problem_groups[problem])

    ds_train_raw = Dataset.from_list(train_rows)
    ds_val_raw = Dataset.from_list(val_rows)
    ds_test_raw = Dataset.from_list(test_rows)

    print("Unique problems total:", total_problems)
    print("Train problems:", len(train_problems))
    print("Val problems:", len(val_problems))
    print("Test problems:", len(test_problems))

    print("Train rows:", len(ds_train_raw))
    print("Val rows:", len(ds_val_raw))
    print("Test rows:", len(ds_test_raw))

    return ds_train_raw, ds_val_raw, ds_test_raw

# --------------------------------------------------------------------------------
# Save splits
# --------------------------------------------------------------------------------
def save_splits(ds_train_raw, ds_val_raw, ds_test_raw, save_dir):
    print("Saving Train/Val/Test splits...")

    if len(ds_train_raw) == 0 and len(ds_val_raw) == 0 and len(ds_test_raw) == 0:
        raise ValueError("All splits are empty. Check filtering and sampling before saving.")

    prepared = DatasetDict({
        "train": ds_train_raw,
        "val": ds_val_raw,
        "test": ds_test_raw,
    })

    os.makedirs(save_dir, exist_ok=True)
    prepared.save_to_disk(save_dir)

    print(f"Saved prepared splits to: {save_dir}")

# --------------------------------------------------------------------------------
# Run one experiment from YAML
# --------------------------------------------------------------------------------
def run_exp(full_config, exp_name, save_root):
    dataset_path = full_config["dataset"]["path"]
    filtering_config = full_config["dataset"]["filtering"]

    exp_config = full_config["experiments"][exp_name]
    seed = exp_config["seed"]
    n = exp_config["N"]
    split_config = exp_config["splits"]

    cot_data = load_data(dataset_path)
    filtered = apply_filter(cot_data, filtering_config)
    row_selection = take_n_rows(filtered, n, seed)
    ds_train_raw, ds_val_raw, ds_test_raw = split_train_val_test(row_selection, split_config, seed)

    save_dir = os.path.join(save_root, "splits", str(n))
    save_splits(ds_train_raw, ds_val_raw, ds_test_raw, save_dir)

    return DatasetDict({
        "train": ds_train_raw,
        "val": ds_val_raw,
        "test": ds_test_raw,
    })