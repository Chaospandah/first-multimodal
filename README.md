#Multimodal Model

I built a small multimodal PyTorch model that combines an MNIST image with task context to make a yes/no prediction.

The task setup is:

- Image: MNIST digit
- Task ID: question about the digit
- Label: yes/no answer

Tasks:

| Task ID | Meaning |
|---:|---|
| 0 | Is the digit even? |
| 1 | Is the digit odd? |
| 2 | Is the digit greater than 5? |
| 3 | Is the digit less than 3? |

## Models Compared

| Model | Inputs |
|---|---|
| Image-only baseline | Image |
| Task-only baseline | Task ID |
| Fusion model | Image + Task ID |

## Why This Matters

This is a simplified version of an agentic systems problem.

A real agent may need to combine:

- visual observations
- task instructions
- state
- memory
- previous actions

before making a decision.

This project practices the same pattern:

raw observation + task context → representation → fusion → decision

## How to Run

```bash
pip install -r requirements.txt
python multimodal_task_predictor.py
