# Practical 1 — Perceptron using NumPy

### Problem Statement
Write a simple Python program to simulate a perceptron using NumPy.

### What's in this notebook
A perceptron built completely from scratch with NumPy — no scikit-learn, no TensorFlow — trained on logic gates using the actual perceptron learning rule (not gradient descent, that's Practical 2).

**Pipeline:** inputs → weighted sum + bias → step activation → compare with target → update weights → repeat.

### How it works
- `step_function(x)` — the decision rule. Returns 1 if the weighted sum is ≥ 0, else 0. Hard cutoff, nothing smooth about it.
- `Perceptron.predict(x)` — computes `z = w·x + b`, passes it through the step function.
- `Perceptron.train(X, y, epochs)` — for every input, checks whether the prediction matches the target. If it doesn't, `error = target - prediction` tells us which way to nudge, and `weights += lr * error * x` does the nudging. Repeats for every sample, every epoch.

### Results
Trained and tested on all three 2-input logic gates, not just the AND gate the practical asks for:

| Gate | Converges? | Errors per epoch | Why |
|------|:---:|---|---|
| AND | Yes, by epoch 4 | `2, 3, 3, 0, 0, 0, 0, 0, 0, 0` | linearly separable |
| OR | Yes, by epoch 4 | `2, 2, 1, 0, 0, 0, 0, 0, 0, 0` | linearly separable |
| XOR | No, never | `3, 3, 4, 4, 4, 4, ...` (flat at 4, doesn't matter how many epochs) | **not** linearly separable |

### My understanding — why XOR matters here
AND and OR both have one straight line that separates their 0s from their 1s, so the perceptron finds it within a handful of epochs. XOR's four points are arranged diagonally — no single straight line can put both 1s on the same side without also catching a 0. That's not a training problem, more epochs won't fix it, it's a structural limit of a single neuron. This is the actual historical reason neural networks moved to multiple layers — one neuron only ever draws one straight decision boundary, and you need to combine several to bend that boundary around something like XOR.

### Files
- `Lab1_Perceptron_using_NumPy.ipynb` — full notebook, already run top to bottom, includes the AND/OR/XOR comparison plus the convergence and decision-boundary plots.
