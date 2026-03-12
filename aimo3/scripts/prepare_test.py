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

from collections import Counter, defaultdict
from datasets import load_from_disk, Dataset, DatasetDict
import os
import random
import yaml

#print("This file now works for Colab")
#SEED = 42
#DATASET_PATH = "/content/drive/MyDrive/Math Olympiad Competition/datasets/OpenMathReasoning_cot"

def load_yaml(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_data(dataset_path):
    # Output of loading dataset
    print("Loading OpenMathReasoning dataset")
    cot_data = load_from_disk(dataset_path)

    # Output information of dataset
    print("Loaded:", len(cot_data), "rows")
    print("Columns:", cot_data.column_names)
    return cot_data

# ---------------------------------------------------------------------------------------------------------
### Define filtering function & apply to dataset.
# ---------------------------------------------------------------------------------------------------------
def filter_rows(row, config):
    for key, value in config.items():
        if key == "Non_nulls":
            continue
        if row.get(key) != value:
            return False

    for col in config.get("Non_nulls", []):
        if row.get(col) in [None, ""]:
            return False

    return True

def apply_filter(cot_data, config):
    filtered = cot_data.filter(filter_rows)
    filtered = cot_data.filter(lambda row: filter_rows(row, config))
    return filtered

# ---------------------------------------------------------------------------------------------------------
### Take N rows
# ---------------------------------------------------------------------------------------------------------
def take_n_rows(filtered, n, SEED):
    print(f"Taking {n} rows...")
    row_selection = filtered.shuffle(seed=SEED).select(range(N))
    print("Selected size:", len(row_selection))
    return row_selection

# ---------------------------------------------------------------------------------------------------------
### Split into Train/Val/Test
# ---------------------------------------------------------------------------------------------------------
def split_train_val_test(row_selection, split_config, seed):
    print("Splitting into Train/Val/Test...")

    problem_groups = defaultdict(list)

    for row in row_selection:
        problem = row["problem"].strip()
        problem_groups[problem].append(row)

    unique_problems = list(problem_groups.keys())


    random.seed(seed)
    random.shuffle(unique_problems)

    total_problems = len(unique_problems)

    train_frac = split_config["train"]
    val_frac = split_config["val"]
    test_frac = split_config["test"]
    
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

def save_splits(ds_train_raw, ds_val_raw, ds_test_raw):
    print("Saving Train/Val/Test splits")

    prepared = DatasetDict({
        "train": ds_train_raw,
        "val": ds_val_raw,
        "test": ds_test_raw,
    })

    prepared.save_to_disk("/content/drive/MyDrive/Math Olympiad Competition/processed_splits/omr_aimo3/hf")
    print("Saved prepared splits to: /content/drive/MyDrive/Math Olympiad Competition/processed_splits/omr_aimo3/hf")
    
def run_exp(config):
    dataset_path = config["dataset_path"]
    SEED = config["seed"]
    N = config["N"]
    split_config = config["split"]

    cot_data = load_data(dataset_path)
    filtered = apply_filter(cot_data, config)
    row_selection = take_n_rows(filtered, N, SEED)
    ds_train_raw, ds_val_raw, ds_test_raw = split_train_val_test(row_selection, split_config, SEED)

""" def main():
    config = load_yaml("./configs/data.yaml")

    experiments = config["experiments"]
    cot_data = load_data(config["dataset"]["path"])
    filtered_data = apply_filter(cot_data, config["dataset"]["filtering"])

    for exp_name, exp_config in experiments.items():
        run_exp(exp_name, exp_config, filtered_data)


if __name__ == "__main__":
    main() """
