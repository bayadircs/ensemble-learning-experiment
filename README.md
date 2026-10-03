# Ensemble Learning Experiment

A handwritten experiment using scikit-learn to compare individual classifiers with soft voting and stacking. The script creates a synthetic nonlinear classification problem, measures pairwise prediction disagreement, and compares test accuracy with a theoretical ceiling computed from the data-generating probabilities.

## Run

```bash
pip install -r requirements.txt
python MAIN.py
```

The script uses a fixed seed and a train/test split. It compares a varied group of models with a second group of similar classifiers to explore how model diversity affects voting.

## Scope

The labels and features are synthetic. The theoretical ceiling is specific to this simulated distribution; it is not an accuracy target for real-world data. This is an experiment with library estimators, not a from-scratch implementation of those estimators.
