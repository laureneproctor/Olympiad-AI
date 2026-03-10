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


# ---------------------------------------------------------------------------------------------------------
### Load the dataset, set seed.
# ---------------------------------------------------------------------------------------------------------
from datasets import load_from_disk
from collections import Counter

print("This file now works for Colab")
SEED = 42
DATASET_PATH = "/content/drive/MyDrive/Math Olympiad Competition/datasets/OpenMathReasoning_cot"

def load_data():
    # Output of loading dataset
    print("Loading OpenMathReasoning dataset")
    cot_data = load_from_disk(DATASET_PATH)

    # Output information of dataset
    print("Loaded:", len(cot_data), "rows")
    print("Columns:", cot_data.column_names)
    return cot_data


# ---------------------------------------------------------------------------------------------------------
### Define filtering function & apply to dataset.
# ---------------------------------------------------------------------------------------------------------
def filter_rows(row):
    return (
        row.get("problem_type") == "has_answer_extracted"
        and row.get("expected_answer") not in [None, ""]
        and row.get("generated_solution") not in [None, ""]
        and row.get("problem") not in [None, ""]
    )

def apply_filter(cot_data):
    # Ouput of filtering dataset
    print("Filtering rows...")
    filtered = cot_data.filter(filter_rows)

    # Output information of filtered dataset
    print("Filtered size:", len(filtered))
    print("Filtered columns:", filtered.column_names)
    return filtered


# ---------------------------------------------------------------------------------------------------------
### Take N rows
# ---------------------------------------------------------------------------------------------------------
def take_n_rows(filtered):
    N = 10000
    print(f"Taking {N} rows...")
    row_selection = filtered.shuffle(seed=SEED).select(range(N))
    print("Selected size:", len(row_selection))
    return row_selection


# ---------------------------------------------------------------------------------------------------------
### Split into Train/Val/Test
# ---------------------------------------------------------------------------------------------------------
def split_train_val_test(row_selection):
    from collections import defaultdict
    from datasets import Dataset

    print("Splitting into Train/Val/Test...")

    problem_groups = defaultdict(list)

    for row in row_selection:
        problem = row["problem"].strip()
        problem_groups[problem].append(row)

    unique_problems = list(problem_groups.keys())

    import random
    random.seed(SEED)
    random.shuffle(unique_problems)

    total_problems = len(unique_problems)
    train_end = int(0.90 * total_problems)
    val_end = train_end + int(0.05 * total_problems)

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