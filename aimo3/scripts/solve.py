# Input: final model + target problem set (competition set)
#   - for each problem:
#       - sample N solutions (N depends on difficulty or fixed)
#       - extract answers
#       - majority vote (optionally tie-break)
#       - write predictions file
# Output:
# - runs/preds.jsonl (submission-like: problem_id → integer answer)