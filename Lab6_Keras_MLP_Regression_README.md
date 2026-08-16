# Practical 6: Keras MLP for Regression

### Problem Statement
Implement an MLP using Keras/TensorFlow for a regression problem and analyse the effect of different activation functions and loss functions on model performance.

### What's in this notebook
Practicals 4 and 5 predicted a category. This one predicts a number: the median house price of a California census district.

The practical says explicitly not to just swap activations and losses and report scores, and that the choices must be justified from the characteristics of the dataset. So the notebook answers two questions by measurement rather than assertion: why the activation changes convergence speed, and why anyone would use MAE or Huber when MSE is the standard.

**Pipeline:** census features, scaling, hidden layers (activation under test), single linear output neuron, regression loss under test, predicted price.

### What changes from Practicals 4 and 5

| | Practical 4 | Practical 5 | Practical 6 |
|---|---|---|---|
| Output neurons | 4 | 1 | 1 |
| Output activation | softmax | sigmoid | **none (linear)** |
| Loss | categorical crossentropy | binary crossentropy | MSE / MAE / Huber |
| Reading the answer | `argmax` | `>= 0.5` | the number itself |
| Metric | accuracy | accuracy | MAE, RMSE, R2 |

The key line is the output layer having **no activation**. Every previous practical squashed the output into a fixed range, and a price is not bounded like that.

### Dataset and source
**California Housing**, 1990 US Census, 20640 districts.
Source: `https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv` (the copy used in Geron's *Hands-On Machine Learning*, from Pace and Barry, 1997).

Chosen over a cleaner dataset deliberately, because it needs real preprocessing:

- **207 missing values** in `total_bedrooms`, imputed with the median since the column is right skewed
- **A categorical column**, `ocean_proximity`, one-hot encoded rather than integer encoded so no false ordering is implied
- **Feature scales spanning four orders of magnitude**, latitude around 35 against `total_rooms` up to 39320
- Three ratio features engineered, since raw room counts mostly reflect district size rather than house size

### The dataset characteristic that drives the loss choice
992 rows (4.81%) are recorded at exactly 500001 dollars, a hard price cap applied at collection. Verified as an artefact rather than genuine top-end pricing: those rows span the entire income range of the dataset (0.5 to 15.0) and include 28 inland districts. Districts with median income 0.5 and 15.0 carrying an identical price is not something real prices do.

This matters because a model that has learned the genuine income-to-price relationship predicts high for a wealthy capped district, is charged a large error for being right, and under MSE that error gets **squared**.

### Results

Baselines first, so the MLP number means something:

| Model | MAE | RMSE | R2 |
|---|:---:|:---:|:---:|
| Predict the mean | 0.9061 | 1.1449 | -0.0002 |
| Linear regression | 0.5089 | 0.7267 | 0.5970 |

Full grid, every activation with every loss:

| Experiment | Activation | Loss | MAE | RMSE | R2 | R2 uncapped |
|---|---|---|:---:|:---:|:---:|:---:|
| Model 1 | relu | mse | 0.3777 | 0.5522 | 0.7673 | 0.7324 |
| Model 2 | relu | mae | 0.3644 | 0.5609 | 0.7599 | 0.7233 |
| **Model 3** | **relu** | **huber** | **0.3700** | **0.5455** | **0.7729** | **0.7421** |
| Model 4 | tanh | mse | 0.3794 | 0.5540 | 0.7658 | 0.7329 |
| Model 5 | tanh | mae | 0.3660 | 0.5547 | 0.7652 | 0.7344 |
| Model 6 | tanh | huber | 0.3774 | 0.5531 | 0.7666 | 0.7359 |
| Model 7 | sigmoid | mse | 0.4074 | 0.5921 | 0.7325 | 0.6940 |
| Model 8 | sigmoid | mae | 0.3964 | 0.5899 | 0.7344 | 0.6979 |
| Model 9 | sigmoid | huber | 0.3988 | 0.5854 | 0.7385 | 0.7036 |

**Final model: ReLU with Huber.** R2 0.7706, MAE 0.3687 (about a 36900 dollar average miss), RMSE 0.5483, R2 0.7382 on uncapped rows.

### My understanding: why the activation changed convergence
Sigmoid needs 18 epochs to reach a validation loss of 0.40 where ReLU needs 7, and never reaches ReLU's final loss. Rather than repeat "vanishing gradients", I measured gradient norms on a freshly initialised network:

```
activation     grad L1   grad L2    L1/L2   mean|output| L1
relu            2.7195    5.8540    0.465          0.2409
tanh            3.0173    3.8987    0.774          0.3752
sigmoid         0.1873    1.5021    0.125          0.4990
```

Sigmoid's first layer receives a gradient **14.5 times weaker** than ReLU's before any weight has moved, because backprop multiplies by the activation's derivative at every layer and sigmoid's `s(1-s)` never exceeds 0.25. Its mean output of 0.4990 is the second handicap: sigmoid only emits positives, so the next layer never sees zero-centred input. Tanh is symmetric and avoids that, which is why it beats sigmoid despite also saturating.

ReLU and tanh finished within noise of each other, which surprised me. On two hidden layers with well-scaled inputs, tanh's saturation is not deep enough to hurt.

### My understanding: the experiment that actually proves the loss choice
On the natural dataset the three losses differ by about 1 point of R2, exactly the size of effect I wrongly inflated into a finding in Practical 4. So I built a controlled version, the same shape as Practical 5's fixed-versus-moving defect test.

Remove the capped rows for genuinely clean data. Train on it, and separately on a copy where 3% of **training** targets are multiplied by 10 (a realistic extra-zero typo). Evaluate both on the same untouched clean test set, three seeds.

```
loss       R2 clean train   R2 contaminated    damage
mse                0.7464            0.0816    0.6649
mae                0.7412            0.7429   -0.0017
huber              0.7460            0.7482   -0.0022
```

All three are equivalent on clean data. Corrupt 471 labels out of 15718 and **MSE loses 66 points of R2**, collapsing to barely better than predicting the average. MAE and Huber do not move. Because the losses match on clean data and the test set is identical in both conditions, the cause is isolated to how each loss weights a large error.

The same effect appears in miniature on the natural data: MSE predicts highest on the truncated rows (4.1559) and pays for it with the worst R2 on the honest 95.5% (0.7283 against Huber's 0.7390).

### One trap worth naming
Ranking losses by MAE makes MAE loss win, and ranking by RMSE makes MSE win. Training on a loss optimises the matching metric by construction, so that comparison is circular. This is why the capped versus uncapped split does the real work.

### Overfitting and underfitting
No meaningful overfitting: 16512 training rows against a small network, and validation tracks training closely. Sigmoid **underfits**, which is easy to read backwards since it has among the smallest gaps. The giveaway is a small gap combined with high loss on *both* curves and the worst test score. Its capacity is nominally identical, so the cause is the gradient starvation above.

### Files
- `Lab6_Keras_MLP_Regression.ipynb`, full notebook, run top to bottom
- `housing.csv`, the dataset, so the notebook runs without a download
