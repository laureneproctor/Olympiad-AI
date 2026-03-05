# Input: A dataset of problems to mine ( usually val.test or a “seed hard set”) + current model
# - For each problem 
#   - Generates K sampled solutions
#   - Extract each predicted int (extracted_answer)
#   - Majority vote
#   - Computes:
#       - Majority correctness vs gold 
#       - Agreement rate
#       - Format validity rate
#   - If it meets “failure” criteria -> save it
# Output
# - data/rl_train/ ( a dataset of hard/ failure prompts with gold answers)
# This will be what GRPO trains on