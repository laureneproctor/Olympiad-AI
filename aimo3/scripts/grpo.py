# Input: data/rl_train/ + starting model checkpoint (SFT model)
#   - For each prompt:
#       - sample a group of outputs (group_size)
#       - extract predicted final int from each
#       - compute reward for each output (correct/wrong/invalid/no-answer)
#       - update model to increase the probability of higher-reward outputs
#       - apply KL penalty to stay close to the SFT model
# Output:
# - checkpoints/grpo_model/ (your RL-tuned checkpoint)