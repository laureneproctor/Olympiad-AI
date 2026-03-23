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
from . import helpers


def load_data(dataset_path):
    """
    This function loads the OpenMathReasoning dataset from disk. It prints out the number of rows and the column names for verification.
    Input: 
    - dataset_path: The file path to the OpenMathReasoning dataset saved on disk.
    Output:
    - cot_data: A Hugging Face Dataset object containing the loaded OpenMathReasoning data.
    """
    print("Loading OpenMathReasoning dataset...")
    cot_data = load_from_disk(dataset_path)
    print("Loaded:", len(cot_data), "rows")
    print("Columns:", cot_data.column_names)
    return cot_data

# --------------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------------
def filter_rows(row, filtering_config):
    """
    This function check if a given row from the dataset meets the criteria specified in the filtering configuration.
    It checks for specific key-value pairs and ensures that certain columns are non-null and non-empty.
    Input:
        - row: A single row from the dataset, represented as a dictionary.
        - filtering_config: A dictionary containing the filtering criteria, which may include specific key-value pairs and 
    a list of columns that must be non-null.
    Output:
        - A boolean value indicating whether the row meets the filtering criteria (True) or not (False).
    """
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

def apply_filter(cot_data, filtering_config):
    """
    This function applies the specified filtering criteria to the entire dataset. 
    It uses the filter_rows function to check each row against the filtering configuration and 
    creates a new filtered dataset containing only the rows that meet the criteria.
    Input:
        - cot_data: A Hugging Face Dataset object containing the OpenMathReasoning data.
        - filtering_config: A dictionary containing the filtering criteria, which may include specific 
        key-value pairs and a list of columns that must be non-null.
    Output:
        - filtered: A new Hugging Face Dataset object containing only the rows from cot_data that meet the filtering criteria specified in filtering_config.
    """
    print("Applying filters")
    print("Filtering config:", filtering_config)
    print("First row passes filter:", filter_rows(cot_data[0], filtering_config))
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
def take_n_rows(filtered, n, seed):
    """
    This function takes the filtered dataset and selects a specified number of rows (N) from it.
    It shuffles the filtered dataset using a provided seed to ensure reproducibility, 
    and then selects the first N rows from the shuffled dataset.
    Input:
        - filtered: A Hugging Face Dataset object that has already been filtered based on certain criteria
        - n: The number of rows to select from the filtered dataset.
        - seed: A random seed used for shuffling the dataset to ensure reproducibility.
    Output:
        - row_selection: A new Hugging Face Dataset object containing the selected N rows from the filtered dataset after shuffling. 
        If N is greater than the number of rows in the filtered dataset, it will return all available rows.
    """
    print(f"Extracting {n} rows.")
    shuffled = filtered.shuffle(seed=seed)
    n = min(n, len(shuffled))
    row_selection = shuffled.select(range(n))
    print("Selected size:", len(row_selection))
    return row_selection

# --------------------------------------------------------------------------------
# Split Train / Val / Test by unique problem text
# --------------------------------------------------------------------------------
def split_train_val_test(row_selection, split_config, seed):
    """
    This function splits the selected rows into training, validation, and test sets based on unique problem texts.
    Input:
        - row_selection: A Hugging Face Dataset object containing the selected rows that need to be split into train, val, and test sets.
        - split_config: A dictionary containing the proportions for the train, validation, and test splits. The values should sum to 1 (e.g., {"train": 0.8, "val": 0.1, "test": 0.1}).
        - seed: A random seed used for shuffling the unique problems to ensure reproducibility of the splits.
    Output:
        - ds_train_raw: A Hugging Face Dataset object containing the training set rows.
        - ds_val_raw: A Hugging Face Dataset object containing the validation set rows.
        - ds_test_raw: A Hugging Face Dataset object containing the test set rows.
    """
    print("Splitting into Train/Val/Test")
    total_split = split_config["train"] + split_config["val"] + split_config["test"]
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

    print("Train rows:", len(ds_train_raw))
    print("Val rows:", len(ds_val_raw))
    print("Test rows:", len(ds_test_raw))

    return ds_train_raw, ds_val_raw, ds_test_raw

# --------------------------------------------------------------------------------
# Save splits
# --------------------------------------------------------------------------------
def save_splits(ds_train_raw, ds_val_raw, ds_test_raw, save_dir):
    """
    This function saves the training, validation, and test splits to disk in a specified directory.
    Input:
    - ds_train_raw: A Hugging Face Dataset object containing the training set rows.
    - ds_val_raw: A Hugging Face Dataset object containing the validation set rows.
    - ds_test_raw: A Hugging Face Dataset object containing the test set rows.
    - save_dir: The directory path where the splits should be saved. The function will create
    the directory if it does not already exist.
    Output:
    - The function does not return any value but saves the train, val, and test splits to disk in 
    the specified directory using the Hugging Face Dataset's save_to_disk method.
    """
    print("Saving Train/Val/Test splits")
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
def run_exp(exp_name):
    """
    This function runs the data preparation process for a specific experiment defined in a YAML configuration file.
    It loads the dataset, applies filtering, selects a specified number of rows, splits the data into training,
    validation, and test sets, and saves the splits to disk.
    Input:
    - exp_name: The name of the experiment as defined in the YAML configuration file. 
    Output:
    - A Hugging Face DatasetDict object containing the training, validation, and test splits for the specified experiment.
    """
    full_config = helpers.get_config_path("data.yaml")
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

    sr = helpers.get_repo_root()
    save_dir = os.path.join(sr, "splits", str(n))
    save_splits(ds_train_raw, ds_val_raw, ds_test_raw, save_dir)
    return DatasetDict({"train": ds_train_raw, "val": ds_val_raw, "test": ds_test_raw})