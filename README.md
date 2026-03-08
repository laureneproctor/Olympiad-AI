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
