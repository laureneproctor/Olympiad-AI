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


# ---------------------------------------------------------------------------------------------------------
### Analyze duplicate structure
# ---------------------------------------------------------------------------------------------------------
def analyze_duplicates(filtered):
    print("Analyzing duplicate structure...")

    problem_counter = Counter()
    problem_reasoning_counter = Counter()

    for row in filtered:
        problem = row["problem"].strip()
        reasoning = row["generated_solution"].strip()

        problem_counter[problem] += 1
        problem_reasoning_counter[(problem, reasoning)] += 1

    total_rows = len(filtered)
    num_unique_problems = len(problem_counter)
    num_unique_problem_reasoning = len(problem_reasoning_counter)

    duplicate_problem_rows = sum(c - 1 for c in problem_counter.values() if c > 1)
    duplicate_problem_reasoning_rows = sum(c - 1 for c in problem_reasoning_counter.values() if c > 1)

    print("Total filtered rows:", total_rows)
    print("Unique problems:", num_unique_problems)
    print("Unique (problem, reasoning) pairs:", num_unique_problem_reasoning)
    print("Rows belonging to repeated problems:", duplicate_problem_rows)
    print("Exact duplicate (problem, reasoning) extra rows:", duplicate_problem_reasoning_rows)
    print("Max repeats for one problem:", max(problem_counter.values()))

    return problem_counter, problem_reasoning_counter


def main():
    cot_data = load_data()
    filtered = apply_filter(cot_data)
    row_selection = take_n_rows(filtered)
    problem_counter, problem_reasoning_counter = analyze_duplicates(filtered)
    return cot_data, filtered, row_selection, problem_counter, problem_reasoning_counter


if __name__ == "__main__":
    main()