# Practical 3 - Feedforward Neural Network (TensorFlow or PyTorch)

### Problem Statement
Implement a feedforward neural network using TensorFlow or PyTorch.

### What's in this notebook
An actual hidden layer this time - implemented twice, once in PyTorch (primary) and once in TensorFlow/Keras (since the practical allows either), and tested on the same three logic gates as Practicals 1 and 2. This is the one that finally solves XOR.

**Pipeline:** inputs → hidden layer (weights + ReLU) → output layer (weights + sigmoid) → loss → backprop (autograd, not hand-derived this time) → update every weight → repeat.

### How this differs from Practicals 1 and 2
Both previous practicals hit the same wall: a single neuron can only ever draw one straight decision boundary, and XOR needs a bent one. A hidden layer fixes that - several neurons each learn their own line, and the output neuron combines them into a shape that isn't a single line anymore. The other change is that gradients are no longer derived by hand; `loss.backward()` (PyTorch) and `model.fit()` (Keras) run the chain rule automatically through both layers.

### How it works
- **PyTorch (`FeedforwardNN`)** - `Linear(2, 4) → ReLU → Linear(4, 1) → Sigmoid`. Trained with `BCELoss` and `Adam(lr=0.05)` for 1000 epochs.
- **TensorFlow/Keras** - `Dense(8, relu) → Dense(1, sigmoid)`, same loss and similar optimizer settings, 300 epochs.

### Results

| Gate | PyTorch | Keras (fixed) |
|------|---|---|
| AND | 4/4 correct, loss 0.0012 | 4/4 correct, loss 0.0052 |
| OR | 4/4 correct, loss 0.00002 | 4/4 correct, loss 0.0044 |
| XOR | 4/4 correct, loss 0.0006 | 4/4 correct, loss 0.0022 |

### On the reference demo's broken Keras cell
The demo notebook's Practical 3 Keras attempt (and Practical 2's, for that matter) gets stuck at 75% accuracy. Reproduced it exactly to confirm - the cause is the default Adam learning rate (0.001) plus only 100 epochs on a 4-sample dataset, which is too few gradient steps to move a freshly-initialised network far enough. Fixed by raising the learning rate to 0.05 and training for 300 epochs; same fix would resolve Practical 2's demo too.

### My understanding - why 4 hidden neurons instead of the textbook minimum of 2
Tested `hidden_size=2` vs `hidden_size=4` on XOR across 5 random seeds each, everything else held identical: 2 neurons reached 4/4 correct on 0 of 5 seeds, 4 neurons reached it on 4 of 5. Two neurons is enough in theory, but it leaves no room for a "dead" ReLU unit to wreck the whole thing - four neurons have redundancy built in. This is a small version of a general pattern in deep learning: extra width/depth beyond the mathematical minimum usually buys training stability, not extra representational power.

### Files
- `Lab3_Feedforward_NN.ipynb` - full notebook, already run top to bottom, includes both frameworks, the hidden-size comparison, and a curved decision-boundary plot for all three gates.
