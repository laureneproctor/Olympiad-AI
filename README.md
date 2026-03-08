# Olympiad-AI
Building a system that parses LaTeX- formatted math problems and uses a model to solve math olympiad-level questions.
# Time-line

<img width="772" height="262" alt="Screenshot 2026-03-03 at 9 29 52 PM" src="https://github.com/user-attachments/assets/b9e98a81-0739-40a8-9b81-03742ff306c5" />


Current Routine for Running:
prepare.py once
sft.py → get baseline SFT model
eval.py → measure pass@1 and maj@N
mine.py → build RL dataset from failures
grpo.py → train RL model
eval.py again → verify improvement
solve.py → produce final predictions
