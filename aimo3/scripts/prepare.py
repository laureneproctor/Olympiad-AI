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
import re
import yaml
from transformers import AutoTokenizer
from . import helpers


FINAL_ANSWER_CONFIG_KEYS = {
    "non_null_columns",
    "require_single_integer_final_answer",
    "final_answer_column",
    "final_answer_min",
    "final_answer_max",
    "debug_print_dropped_samples",
    "debug_dropped_sample_limit",
    "debug_dropped_scan_limit",
}

QUALITY_RATING_COLUMN = "reasoning_quality_rating"
_LAST_INT_RE = re.compile(r"\b(-?\d+)\b")
_ASSIGN_RE = re.compile(r"\b([a-zA-Z])\s*=\s*(-?\d+)\b")
_STEP_CUES = ("therefore", "thus", "hence", "so", "because", "then", "implies")
_CORRECTION_CUES = ("actually", "wait", "i was wrong", "correction", "contradiction")


def parse_single_integer_final_answer(value):
    """
    Parse a value as exactly one integer answer.

    Returns:
        int | None: Parsed integer, or None if parsing fails.
    """
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    # Strict mode: answer text must be exactly one integer token.
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)

    return None


def has_valid_integer_final_answer(row, filtering_config):
    """
    Check whether the row has a parseable single integer final answer within bounds.
    """
    require_integer = filtering_config.get("require_single_integer_final_answer", True)
    if not require_integer:
        return True

    answer_col = filtering_config.get("final_answer_column", "expected_answer")
    min_value = filtering_config.get("final_answer_min", 0)
    max_value = filtering_config.get("final_answer_max", 99999)

    parsed = parse_single_integer_final_answer(row.get(answer_col))
    if parsed is None:
        return False

    return min_value <= parsed <= max_value


def get_drop_reason(row, filtering_config):
    """
    Return a short reason string for why a row would be dropped by filter_rows.
    """
    for key, value in filtering_config.items():
        if key in FINAL_ANSWER_CONFIG_KEYS:
            continue
        if row.get(key) != value:
            return f"mismatch:{key}"

    for col in filtering_config.get("non_null_columns", []):
        val = row.get(col)
        if val is None:
            return f"null:{col}"
        if isinstance(val, str) and not val.strip():
            return f"blank:{col}"

    require_integer = filtering_config.get("require_single_integer_final_answer", True)
    if require_integer:
        answer_col = filtering_config.get("final_answer_column", "expected_answer")
        min_value = filtering_config.get("final_answer_min", 0)
        max_value = filtering_config.get("final_answer_max", 99999)
        parsed = parse_single_integer_final_answer(row.get(answer_col))
        if parsed is None:
            return f"invalid_integer:{answer_col}"
        if not (min_value <= parsed <= max_value):
            return f"out_of_range:{answer_col}"

    return "unknown"


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
        if key in FINAL_ANSWER_CONFIG_KEYS:
            continue
        if row.get(key) != value:
            return False
    for col in filtering_config.get("non_null_columns", []):
        val = row.get(col)
        if val is None:
            return False
        if isinstance(val, str) and not val.strip():
            return False

    if not has_valid_integer_final_answer(row, filtering_config):
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

    if filtering_config.get("debug_print_dropped_samples", False):
        sample_limit = int(filtering_config.get("debug_dropped_sample_limit", 10))
        scan_limit = int(filtering_config.get("debug_dropped_scan_limit", 50000))
        scan_count = min(len(cot_data), max(0, scan_limit))
        answer_col = filtering_config.get("final_answer_column", "expected_answer")

        dropped_samples = []
        for idx in range(scan_count):
            row = cot_data[idx]
            if filter_rows(row, filtering_config):
                continue

            dropped_samples.append({
                "idx": idx,
                "reason": get_drop_reason(row, filtering_config),
                "expected_answer": row.get(answer_col),
            })
            if len(dropped_samples) >= sample_limit:
                break

        print(
            f"Dropped sample preview (first {len(dropped_samples)} from first {scan_count} scanned rows):"
        )
        for sample in dropped_samples:
            print(
                f"  idx={sample['idx']} reason={sample['reason']} expected_answer={repr(sample['expected_answer'])}"
            )

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


def _extract_last_int(text):
    matches = _LAST_INT_RE.findall(str(text))
    if not matches:
        return None
    return matches[-1]


def _repetition_ratio(text, n=3):
    tokens = str(text).lower().split()
    if len(tokens) < n:
        return 0.0
    ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def compute_reasoning_quality_rating(row):
    """
    Compute a heuristic reasoning quality rating in [1, 10], where 1 is best and 10 is worst.
    """
    solution = str(row.get("generated_solution", ""))
    answer = str(row.get("expected_answer", "")).strip()

    pred = _extract_last_int(solution)
    align_ok = 1.0 if pred is not None and pred == answer else 0.0

    assignments = _ASSIGN_RE.findall(solution)
    seen = {}
    contradictions = 0
    for var, val in assignments:
        if var in seen and seen[var] != val:
            contradictions += 1
        seen[var] = val
    contradictions += sum(1 for cue in _CORRECTION_CUES if cue in solution.lower())

    token_count = max(1, len(solution.split()))
    cue_count = sum(solution.lower().count(cue) for cue in _STEP_CUES)
    clarity = 100.0 * cue_count / token_count

    length = len(solution)
    length_ok = 1.0 if 120 <= length <= 2200 else 0.0

    repetition = _repetition_ratio(solution, n=3)

    score_0_100 = (
        45.0 * align_ok
        + 20.0 * max(0.0, 1.0 - contradictions / 3.0)
        + 15.0 * min(1.0, clarity / 2.0)
        + 10.0 * length_ok
        + 10.0 * max(0.0, 1.0 - repetition / 0.35)
    )

    # Map continuous score to a 1-10 integer bucket, inverted so lower is better.
    rating = int(max(1, min(10, 11 - round(score_0_100 / 10.0))))
    return rating


def add_reasoning_quality_rating(ds):
    """
    Add a 1-10 reasoning quality rating column used for downstream filtering.
    """
    print(f"Adding {QUALITY_RATING_COLUMN} column")

    def _add(row):
        return {QUALITY_RATING_COLUMN: compute_reasoning_quality_rating(row)}

    rated = ds.map(_add, desc=f"Computing {QUALITY_RATING_COLUMN}")
    print(f"Added {QUALITY_RATING_COLUMN} to dataset")
    return rated


def filter_by_quality_rating(ds, min_rating=1, max_rating=10):
    """
    Keep rows whose reasoning quality rating is within [min_rating, max_rating].
    """
    min_rating = int(min_rating)
    max_rating = int(max_rating)
    if min_rating > max_rating:
        raise ValueError(f"quality rating range invalid: {min_rating} > {max_rating}")

    print(f"Filtering by {QUALITY_RATING_COLUMN} in [{min_rating}, {max_rating}]")
    filtered = ds.filter(
        lambda row: min_rating <= int(row.get(QUALITY_RATING_COLUMN, 0)) <= max_rating,
        desc="Filtering by reasoning quality",
    )
    print("Quality-filtered size:", len(filtered))
    return filtered

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
    print("Splitting into Train/Val1/Val2/Test")
    total_split = split_config["train"] + split_config["val"] + split_config.get("val2", 0.0) + split_config["test"]
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total_split}")

    problem_groups = defaultdict(list)
    for row in row_selection:
        problem = row["problem"].strip()
        problem_groups[problem].append(row)
    unique_problems = list(problem_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_problems)
    total_problems = len(unique_problems)

    train_frac = split_config["train"]
    val1_frac = split_config["val"]
    val2_frac = split_config.get("val2", 0.0)
    test_frac = split_config["test"]

    train_end = int(train_frac * total_problems)
    val1_end = train_end + int(val1_frac * total_problems)
    val2_end = val1_end + int(val2_frac * total_problems)

    train_problems = unique_problems[:train_end]
    val1_problems = unique_problems[train_end:val1_end]
    val2_problems = unique_problems[val1_end:val2_end]
    test_problems = unique_problems[val2_end:]

    train_rows = []
    val1_rows = []
    val2_rows = []
    test_rows = []

    for problem in train_problems:
        train_rows.extend(problem_groups[problem])

    for problem in val1_problems:
        val1_rows.extend(problem_groups[problem])

    for problem in val2_problems:
        val2_rows.extend(problem_groups[problem])

    for problem in test_problems:
        test_rows.extend(problem_groups[problem])

    ds_train_raw = Dataset.from_list(train_rows)
    ds_val1_raw = Dataset.from_list(val1_rows)
    ds_val2_raw = Dataset.from_list(val2_rows)
    ds_test_raw = Dataset.from_list(test_rows)

    print("Train rows:", len(ds_train_raw))
    print("Val1 rows:", len(ds_val1_raw))
    print("Val2 rows:", len(ds_val2_raw))
    print("Test rows:", len(ds_test_raw))

    return ds_train_raw, ds_val1_raw, ds_val2_raw, ds_test_raw

# --------------------------------------------------------------------------------
# Save splits
# --------------------------------------------------------------------------------
def save_splits(ds_train_raw, ds_val1_raw, ds_val2_raw, ds_test_raw, save_dir):
    """
    This function saves the training, validation1, validation2, and test splits to disk in a specified directory.
    Input:
    - ds_train_raw: A Hugging Face Dataset object containing the training set rows.
    - ds_val1_raw: A Hugging Face Dataset object containing the val1 set rows.
    - ds_val2_raw: A Hugging Face Dataset object containing the val2 set rows.
    - ds_test_raw: A Hugging Face Dataset object containing the test set rows.
    - save_dir: The directory path where the splits should be saved. The function will create
    the directory if it does not already exist.
    Output:
    - The function does not return any value but saves the train, val1, val2, and test splits to disk in 
    the specified directory using the Hugging Face Dataset's save_to_disk method.
    """
    print("Saving Train/Val1/Val2/Test splits")
    prepared = DatasetDict({
        "train": ds_train_raw,
        "val": ds_val1_raw,
        "val2": ds_val2_raw,
        "test": ds_test_raw,
    })
    os.makedirs(save_dir, exist_ok=True)
    prepared.save_to_disk(save_dir)
    print(f"Saved prepared splits to: {save_dir}")


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
    output = {"text": text, "expected_answer": answer}
    if QUALITY_RATING_COLUMN in example:
        output[QUALITY_RATING_COLUMN] = int(example[QUALITY_RATING_COLUMN])
    return output


def format_sft_splits(prepared):
    sft_config = helpers.load_yaml(helpers.get_config_path("sft.yaml"))
    system_prompt = sft_config["prompting"]["system_prompt"]

    fmtd = {}
    for split in ["train", "val", "val2", "test"]:
        ds = prepared[split]
        ds_text = ds.map(
            lambda ex: convert_to_prompt_format(ex, system_prompt),
            remove_columns=ds.column_names,
            batched=False,
            desc=f"Formatting {split} split"
        )
        fmtd[split] = ds_text

    return DatasetDict(fmtd)



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
    full_config_path = helpers.get_config_path("data.yaml")
    full_config = helpers.load_yaml(full_config_path)
    dataset_path = full_config["dataset"]["path"]
    filtering_config = full_config["dataset"]["filtering"]
    exp_config = full_config["experiments"][exp_name]

    seed = exp_config["seed"]
    n = exp_config["N"]
    split_config = exp_config["splits"]
    cot_data = load_data(dataset_path)

    filtered = apply_filter(cot_data, filtering_config)
    filtered = add_reasoning_quality_rating(filtered)

    quality_min = exp_config.get("quality_rating_min", 1)
    quality_max = exp_config.get("quality_rating_max", 10)
    filtered = filter_by_quality_rating(filtered, quality_min, quality_max)

    row_selection = take_n_rows(filtered, n, seed)
    ds_train_raw, ds_val1_raw, ds_val2_raw, ds_test_raw = split_train_val_test(row_selection, split_config, seed)

    sft_config = helpers.load_yaml(helpers.get_config_path("sft.yaml"))
    prepared_root = sft_config["paths"]["prepared_splits_root"]
    save_dir = os.path.join(prepared_root, "splits", str(n))
    save_splits(ds_train_raw, ds_val1_raw, ds_val2_raw, ds_test_raw, save_dir)

    return DatasetDict({"train": ds_train_raw, "val": ds_val1_raw, "val2": ds_val2_raw, "test": ds_test_raw})


def run_all_experiments(exp_names=None):
    """
    Run data preparation for multiple experiments while loading/filtering the dataset only once.

    Input:
    - exp_names: optional list of experiment names. If None, runs all experiments from data.yaml.
    Output:
    - dict mapping experiment name -> DatasetDict(train/val/test)
    """
    # Load configurations/ yaml
    full_config_path = helpers.get_config_path("data.yaml")
    full_config = helpers.load_yaml(full_config_path)
    dataset_path = full_config["dataset"]["path"]
    filtering_config = full_config["dataset"]["filtering"]
    experiments = full_config["experiments"]
    if exp_names is None:
        selected_experiments = list(experiments.keys())
    else:
        selected_experiments = list(exp_names)
        missing = [name for name in selected_experiments if name not in experiments]
        if missing:
            raise ValueError(f"Unknown experiment names: {missing}")
        
    # Loads the dataset & Applies filtering
    cot_data = load_data(dataset_path)
    filtered = apply_filter(cot_data, filtering_config)
    filtered = add_reasoning_quality_rating(filtered)

    sft_config = helpers.load_yaml(helpers.get_config_path("sft.yaml"))
    prepared_root = sft_config["paths"]["prepared_splits_root"]

    shuffled_by_seed = {}
    outputs = {}

    # Process for each experiment:
    # - Shuffle with seed
    # - Define N, which is size of split
    # - Split into Train/Val/Test
    # - Save splits so later steps are fast and repeatable
    for exp_name in selected_experiments:
        print(f"Rebuilding {exp_name} ...")
        exp_config = experiments[exp_name]
        seed = exp_config["seed"]
        n = exp_config["N"]
        split_config = exp_config["splits"]
        quality_min = exp_config.get("quality_rating_min", 1)
        quality_max = exp_config.get("quality_rating_max", 10)

        if seed not in shuffled_by_seed:
            shuffled_by_seed[seed] = filtered.shuffle(seed=seed)

        shuffled = shuffled_by_seed[seed]
        quality_filtered = filter_by_quality_rating(shuffled, quality_min, quality_max)
        n_eff = min(n, len(quality_filtered))
        row_selection = quality_filtered.select(range(n_eff))

        ds_train_raw, ds_val1_raw, ds_val2_raw, ds_test_raw = split_train_val_test(row_selection, split_config, seed)
        save_dir = os.path.join(prepared_root, "splits", str(n))
        save_splits(ds_train_raw, ds_val1_raw, ds_val2_raw, ds_test_raw, save_dir)

        outputs[exp_name] = DatasetDict({"train": ds_train_raw, "val": ds_val1_raw, "val2": ds_val2_raw, "test": ds_test_raw})

    return outputs