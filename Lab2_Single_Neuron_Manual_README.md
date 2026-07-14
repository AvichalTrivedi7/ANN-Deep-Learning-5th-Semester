# Practical 2 - Single Neuron Model (Manual Implementation)

### Problem Statement
Implement a single neuron model manually using Python.

### What's in this notebook
One neuron - sigmoid activation, loss, and the gradient descent update, all derived by hand and coded manually. No TensorFlow, no Keras, no autograd. The whole point of "manually" here is opening up what a framework normally hides.

**Pipeline:** inputs → weighted sum + bias → sigmoid → loss vs. target → gradient (hand-derived via chain rule) → update weights → repeat.

### How this differs from Practical 1
Practical 1's perceptron used a hard step function and a heuristic update rule - no real loss function, no calculus. This neuron swaps the step for a smooth, differentiable sigmoid, defines an actual loss (`(a - target)^2`), and updates weights using the *derivative* of that loss - real gradient descent, derived by hand:

```
dL/dz = 2*(a - target) * a*(1 - a)
dL/dw = dL/dz * x
dL/db = dL/dz
```

### How it works
- `sigmoid(z)` - squashes any real number into (0, 1), so the output is a probability, not a hard 0/1.
- `SingleNeuron.forward(x)` - same weighted sum as Practical 1, followed by sigmoid.
- `SingleNeuron.train(X, y, epochs)` - for every sample, computes the loss, works out `dL/dz` using the chain rule above, and moves the weights a small step opposite the gradient. Runs for 2000 epochs (a lot more than Practical 1's 10, since gradient steps are small - especially once sigmoid starts saturating near 0 or 1).

### Results
Same three logic gates as Practical 1, trained with this new rule:

| Gate | Final loss (epoch 2000) | Confident? |
|------|:---:|---|
| AND | 0.00123 | yes - outputs land at 0.0001–0.04 (class 0) and 0.955 (class 1) |
| OR | 0.00064 | yes - outputs land at 0.038 (class 0) and 0.976–1.0 (class 1) |
| XOR | 0.28422 (basically unchanged from epoch 1's 0.2839) | no - every output sits at ~0.43–0.53, the neuron is guessing |

### My understanding - why XOR still fails
Sigmoid only changes *how* the boundary is drawn - soft and probability-based instead of a hard cutoff - it doesn't change *what shape* the boundary can take. `w1*x1 + w2*x2 + b` is a straight line no matter what activation sits on top of it, so a single neuron hits the exact same wall the perceptron did on XOR, just measured differently this time (loss flatlining instead of the error count flatlining). Fixing it needs more neurons, not a different activation or more epochs on this one - stack a few in a hidden layer and the combined boundary can actually bend.

### Files
- `Lab2_Single_Neuron_Manual.ipynb` - full notebook, already run top to bottom, includes the training comparison plus the loss-curve and decision-boundary plots.
