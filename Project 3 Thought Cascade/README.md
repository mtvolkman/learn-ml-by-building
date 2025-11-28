# Project 3: Thought Cascade
**Building and Evaluating Agentic AI Systems with Small Language Models**

## Overview
In this assignment, you will explore the transition from traditional "single-shot" prompting to **Agentic AI**. You will build a system that uses the **ReAct (Reasoning + Acting)** pattern to solve complex debugging problems by iterating, testing, and self-correcting.

## Prerequisites
- Python 3.10+
- Basic familiarity with PyTorch and Hugging Face Transformers
- (Optional) A GPU is recommended for faster inference, but the code is optimized to run on CPU (Apple Silicon MPS or standard CPU).

## Setup

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Model Setup**
    The notebook is configured to use **Gemma-2-2b-it** (Google) or **SmolLM-360M** (HuggingFaceTB).
    - If you have models from Project 2, you can point the notebook to their local paths.
    - Otherwise, the notebook will automatically download the necessary weights from Hugging Face (approx. 2-4GB) on the first run.

## Running the Assignment

1.  Start Jupyter:
    ```bash
    jupyter notebook
    ```
2.  Open `p03_Thought_Cascade.ipynb` and follow the instructions.

## Directory Structure
- `p03_Thought_Cascade.ipynb`: The main assignment file.
- `data/QuixBugs/`: Dataset of algorithmic bugs used for evaluation.
- `images/`: Helper diagrams for the notebook.

## Submission
Complete the questions marked with `TODO` in the notebook and submit your final `.ipynb` or exported PDF as per the course guidelines.
