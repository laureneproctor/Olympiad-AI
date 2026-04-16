
## Olympiad-AI
Building a system that parses LaTeX- formatted math problems and uses a model to solve math olympiad-level questions.

## Time-line
<img width="785" height="289" alt="Screenshot 2026-04-16 at 4 34 16 PM" src="https://github.com/user-attachments/assets/32c498d9-7fe0-4eb8-8df2-17dae0ee8e61" />

## Project Structure

Olympiad-AI/
├── aimo3/
│   ├── configs/
│   │   ├── old_configs/
│   │   ├── data.yaml
│   │   ├── evaluate_grpo_deepseek*.yaml
│   │   ├── evaluate_grpo_qwen*.yaml
│   │   ├── evaluate_sft_deepseek*.yaml
│   │   ├── evaluate_sft_qwen*.yaml
│   │   ├── grpo_baseline_deepseek*.yaml
│   │   ├── grpo_baseline_qwen*.yaml
│   │   ├── grpo_deepseekmath*.yaml
│   │   ├── grpo_qwen.yaml
│   │   ├── grpo_qwen_exp1.yaml
│   │   ├── grpo_qwen_exp3.yaml
│   │   ├── large_grpo_qwen_v2*.yaml
│   │   ├── large_mine_qwen_v2*.yaml
│   │   ├── mine_baseline_deepseek*.yaml
│   │   ├── mine_baseline_qwen*.yaml
│   │   ├── mine_deepseekmath*.yaml
│   │   ├── mine_qwen.yaml
│   │   ├── mine_qwen_exp1.yaml
│   │   ├── mine_qwen_exp3.yaml
│   │   ├── sft_deepseekmath.yaml
│   │   ├── sft_qwen.yaml
│   │   ├── sft.yaml
│   │   └── solve.yaml
│   │
│   ├── extras/
│   │   ├── Baseline_GRPO_Pipeline*
│   │   └── Baseline_Models_Eval*
│   │
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── grpo.py
│   │   ├── grpo_baseline.py
│   │   ├── helpers.py
│   │   ├── mine.py
│   │   ├── mine_baseline.py
│   │   ├── optuna_tune_sft.py
│   │   ├── prepare.py
│   │   ├── sft.py
│   │   ├── solve.py
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── notebooks/
│   ├── EDA/
│   │   ├── eda_1.ipynb
│   │   └── sft_dataset_preview.ipynb
│   │
│   ├── Parser Work/
│   │   ├── ec-pdf-parser.ipynb
│   │   ├── Final SVSU Parser.ipynb
│   │   ├── Olympiad-PDF-Scrape.ipynb
│   │   ├── SVSU Parsing All.ipynb
│   │   ├── All Evaluation.ipynb
│   │   ├── demo.ipynb
│   │   ├── DSM_V2_for_Presentation.ipynb
│   │   ├── Evaluation_GRPO_Deep*.ipynb
│   │   ├── Evaluation_GRPO_Qwen*.ipynb
│   │   ├── GRPO_Execution_Deep*.ipynb
│   │   ├── GRPO_Execution_Qwen*.ipynb
│   │   ├── Main Script.ipynb
│   │   ├── Mine_Execution_Deep*.ipynb
│   │   ├── Mine_Execution_Qwen*.ipynb
│   │   └── README.md
│
├── data/
├── pdf_data/
│   └── raw_pdfs/
│
├── .gitignore
└── README.md
## Full Pipeline
This repository implements a training pipeline for solving olympiad-style math problems using a combination of Supervised Fine-Tuning (SFT) and Reinforcement Learning with GRPO.
```
Raw Datasets
     │
     ▼
prepare.py
     │
     ▼
sft.py  →  Baseline Model
     │
     ▼
eval.py (baseline metrics)
     │
     ▼
mine.py
     │
     ▼
grpo.py  →  RL-Improved Model
     │
     ▼
eval.py (compare with baseline)
     │
     ▼
solve.py → Competition predictions
```
<img width="919" height="668" alt="image" src="https://github.com/user-attachments/assets/d7628db9-bb8c-4d39-9c7e-b50c70b4fed1" />



## Pipeline in Details

1. Raw Dataset: NVIDIA OpenMathReasoning dataset

2. prepare.py:
This script prepares the raw dataset into a clean and reusable format for training and evaluation. It loads the OpenMathReasoning dataset, removes incomplete or invalid entries, and normalizes the answers into integer strings within the range [0, 99999].
It also formats the prompts so that each training example contains the problem statement followed by a reasoning trace that ends with: FINAL_ANSWER: arrows(integer)
The script then creates dataset splits for supervised training, validation, evaluation, and RL mining, and saves the processed datasets to disk.

4. sft.py:
This script performs Supervised Fine-Tuning (SFT) on the prepared dataset. The base model used is deepseek-ai/deepseek-math-7b-instruct, which is fine-tuned using 4-bit quantization and LoRA adapters for efficient training.
The objective of this stage is to teach the model to understand olympiad-style math problems, generate step-by-step reasoning, and consistently output a correctly formatted final answer.
The resulting model is saved as the baseline SFT checkpoint.

5. eval.py (Baseline Evaluation):
After SFT training, the model is evaluated on a held-out evaluation set.
This stage measures baseline performance using metrics such as:

- pass@1: accuracy using greedy decoding
- maj@N: majority vote accuracy across multiple sampled solutions
- format validity: percentage of outputs containing a valid final answer
- agreement rate: consistency of answers across samples
These metrics establish a baseline before applying reinforcement learning.

5. mine.py:
This script mines useful training examples for reinforcement learning.
The baseline SFT model is run on a held-out set of problems, generating multiple sampled solutions for each.
The script then identifies cases where the model:
- produces incorrect answers
- generates inconsistent outputs
- fails to follow the expected answer format
- occasionally finds the correct answer but inconsistently
These failure cases provide valuable learning signals and are saved as RL training data.

6. grpo.py:
This stage applies GRPO (Group Relative Policy Optimization) to further improve the model.
The model generates multiple candidate solutions for each prompt, and a reward signal is assigned based on factors such as:
- correctness of the final answer
- valid output formatting
- reasoning quality
Unlike SFT, which imitates existing solutions, GRPO allows the model to explore different reasoning paths and reinforce those that lead to correct answers.
The resulting model is saved as the GRPO-trained checkpoint.

7. eval.py (Post-RL Evaluation):
The evaluation script is rerun on the GRPO-trained model to quantify improvements.
Metrics from this stage are compared with the baseline SFT results to determine whether reinforcement learning improves accuracy, consistency, and formatting reliability.

8. solve.py:
This script is used for final inference during competition or evaluation.
It loads the final trained model, processes one problem at a time, generates candidate solutions, extracts the final integer answer, and returns a prediction within the required range [0, 99999].
