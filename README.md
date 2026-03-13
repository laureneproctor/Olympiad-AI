## Olympiad-AI
Building a system that parses LaTeX- formatted math problems and uses a model to solve math olympiad-level questions.

## Time-line

<img width="777" height="285" alt="Screenshot 2026-03-12 at 9 05 05 AM" src="https://github.com/user-attachments/assets/82d3410b-ad45-4f0c-a27a-6424b5032000" />

## Project Structure

```
Olympiad-AI/
├── aimo3/
│   ├── configs/
│   │   ├── data.yaml
│   │   ├── evaluate.yaml
│   │   ├── grpo.yaml
│   │   ├── mine.yaml
│   │   ├── sft.yaml
│   │   └── solve.yaml
│   │
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── grpo.py
│   │   ├── mine.py
│   │   ├── prepare.py
│   │   ├── sft.py
│   │   └── solve.py
│   │
│   └── __init__.py
│
├── data/
├── notebooks/
│   ├── Baseline 1.ipynb
│   ├── DSM_V2_for_Presentation.ipynb
│   ├── Final SVSU Parser.ipynb
│   ├── Olympiad-PDF-Scrape.ipynb
│   ├── README.md
│   ├── SVSU Parsing All.ipynb
│   └── ec-pdf-parser.ipynb
│
├── pdf_data/
│   └── raw_pdfs/
│
├── .gitignore
└── README.md
```

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
## Pipeline in Details

1. Raw Dataset: NVIDIA OpenMathReasoning dataset

2. prepare.py:
This script prepares the raw dataset into a clean and reusable format for training and evaluation. It loads the OpenMathReasoning dataset, removes incomplete or invalid entries, and normalizes the answers into integer strings within the range [0, 99999].
It also formats the prompts so that each training example contains the problem statement followed by a reasoning trace that ends with:
```
FINAL_ANSWER: <integer>
```
The script then creates dataset splits for supervised training, validation, evaluation, and RL mining, and saves the processed datasets to disk.

3. sft.py:
This script performs Supervised Fine-Tuning (SFT) on the prepared dataset. The base model used is deepseek-ai/deepseek-math-7b-instruct, which is fine-tuned using 4-bit quantization and LoRA adapters for efficient training.
The objective of this stage is to teach the model to understand olympiad-style math problems, generate step-by-step reasoning, and consistently output a correctly formatted final answer.
The resulting model is saved as the baseline SFT checkpoint.

4. eval.py (Baseline Evaluation):
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
