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

from datasets import load_from_disk

print("This file now works for Colab")
SEED = 42
DATASET_PATH = "/content/drive/MyDrive/Math Olympiad Competition/datasets/OpenMathReasoning_cot"


def filter_rows(row):
    return (
        row.get("problem_type") == "has_answer_extracted"
        and row.get("expected_answer") not in [None, ""]
        and row.get("generated_solution") not in [None, ""]
        and row.get("problem") not in [None, ""]
    )
# Output of loading dataset
print("Loading OpenMathReasoning dataset")
cot_data = load_from_disk(DATASET_PATH)

# Output information of dataset
print("Loaded:", len(cot_data), "rows")
print("Columns:", cot_data.column_names)

# Ouput of filtering dataset
print("Filtering rows...")
filtered =  cot_data.filter(filter_rows)

# Output information of filtered dataset
print("Filtered size:", len(filtered))
print("Filtered columns:", filtered.column_names)