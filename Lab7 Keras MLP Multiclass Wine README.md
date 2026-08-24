# Practical 7: Keras MLP for Multiclass Classification

### Problem Statement
Implement an MLP using Keras/TensorFlow for a multiclass classification problem and analyse the effect of different activation functions, optimizers and model configurations on classification performance.

### What's in this notebook
Predicting the quality grade of a wine from its chemical measurements. Seven grades, so multiclass, and the architecture is essentially the one from Practical 4.

**That overlap is deliberate and stated up front.** Practical 4 was also a multiclass MLP with softmax, so repeating it would teach nothing. Two things are genuinely different:

1. **The data is real and difficult.** Practical 4 used a dataset I generated from rules I chose, so classes were separable by construction and the model reached 96%. This one has severe imbalance, overlapping classes and noisy human labels. The model reaches about 54%, and understanding why is the content of the practical.
2. **The new experimental axis is the optimizer**, which no previous practical compared.

**Pipeline:** chemical measurements, scaling, two hidden layers (activation under test), softmax over 7 classes, optimizer under test, predicted grade.

### Dataset and source
**UCI Wine Quality**, physicochemical measurements of Portuguese Vinho Verde wines (Cortez et al., 2009), red and white combined.

- `https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv`
- `https://raw.githubusercontent.com/zygmuntz/wine-quality/master/winequality/winequality-white.csv`

Chosen over Iris or digits deliberately. Those separate cleanly and a model hits the high 90s without the student learning why. This one exhibits every property the practical asks about:

| property | value |
|---|---|
| Classes | 7 (quality grades 3 to 9) |
| Imbalance | **567:1**, grades 5 and 6 are 76.6% of the data |
| Majority baseline | 43.70% |
| Rare classes | grade 9 has 5 rows total, grade 3 has 30 |
| Categorical feature | wine type (red/white) |
| Missing values | none, checked and reported |
| Label noise | each grade is the median of at least 3 human tasters |

### A preprocessing decision, measured not assumed
18.1% of rows (1177) are exact duplicates. With a random split the same wine lands in both train and test, so the model is tested on rows it memorised.

```
duplicates KEPT (leaky)      test accuracy 0.5554
duplicates DROPPED (honest)  test accuracy 0.5385
```

Leakage is worth **1.7 points** of fake accuracy. Small, free to remove, and it flatters the model, which is exactly why it is worth checking.

### Architecture
`Input(12) -> Dense(64, act) -> Dense(32, act) -> Dense(7, softmax)`, batch size 64, 60 epochs, 20% validation split, learning rate 0.001 for Adam and RMSprop, 0.01 for SGD. Loss is sparse categorical crossentropy. Every experiment reuses the identical split.

### Results

| Baseline | Accuracy | Macro F1 |
|---|:---:|:---:|
| Random guess (7 classes) | 0.1429 | |
| Always predict grade 6 | 0.4370 | 0.0869 |
| Multinomial logistic regression | 0.5320 | 0.2309 |

Comparison table, means over 3 seeds with spread in brackets, all metrics macro-averaged:

| Experiment | Activation | Optimizer | Test Accuracy | Macro F1 | Weighted F1 |
|---|---|---|:---:|:---:|:---:|
| Model 1 | relu | adam | 0.5451 (+-0.008) | 0.2528 (+-0.018) | 0.5229 |
| Model 2 | sigmoid | adam | 0.5370 (+-0.002) | 0.2173 (+-0.002) | 0.5120 |
| Model 3 | tanh | adam | 0.5417 (+-0.002) | 0.2528 (+-0.008) | 0.5253 |
| Model 4 | relu | sgd | 0.5451 (+-0.003) | 0.2223 (+-0.003) | 0.5143 |
| Model 5 | relu | rmsprop | 0.5432 (+-0.006) | 0.2595 (+-0.022) | 0.5143 |

### My understanding: the differences above are not real
```
spread of accuracy ACROSS the five configurations : 0.0081
largest spread WITHIN a single configuration      : 0.0169
```

Changing only the random seed moves accuracy twice as much as changing the activation or optimizer does. **So no configuration can be ranked above another on this data.** Macro F1 is worse still, swinging up to 0.0444 within one configuration, because with 1 test sample in grade 9 a single lucky prediction moves that class from recall 0 to 1 and macro averaging weights it equally with grade 6's 465 samples.

Being unable to declare a winner is the finding. Activation and optimizer barely matter here; the data properties dominate.

### What the optimizer does change
Adam and RMSprop reach 54% validation accuracy at **epoch 2**, plain SGD at **epoch 11**, five times slower, because they adapt the step size per weight instead of applying one fixed rate.

But the ranking inverts on generalisation. SGD ends with a train/validation gap of **0.0133** against Adam's **0.0946**, seven times smaller, and the joint-best test accuracy. Its slower noisier updates act as a mild regulariser. Fastest convergence is not the best final model.

### My understanding: 54% is not what it looks like
```
strict accuracy (exact grade) : 0.5423
accuracy allowing +-1 grade   : 0.9530
```

**89.7% of every error is off by exactly one grade.** Only 3 of 487 errors are off by 3 or more. The most confused pairs are the adjacent ones: grades 5 and 6 (229 wines), grades 6 and 7 (158), together 79% of all errors.

This is a property of the labels, not a flaw in the network. Each grade is the median of at least three tasters, so a wine on the boundary between average and good genuinely splits a panel. The chemistry of a borderline 5 and a borderline 6 is nearly identical, so the ambiguity is inside the label itself.

### Class imbalance, and why reweighting fails
Three classes (grades 3, 8, 9) are **never predicted correctly at all**, recall 0.000 each, with 24, 118 and 4 training rows between them. Macro F1 0.259 against weighted F1 0.525 is the same predictions scored two ways.

Balanced class weighting is the standard remedy and it costs far more than it returns: accuracy drops **11.87 points** while macro F1 rises **0.0078**, a gain five times smaller than the run-to-run noise of 0.0363.

Weighting changes the penalty, not the evidence. Grade 9 has four training wines, and multiplying its loss by 152 makes the model guess it more often without giving it anything to learn from.

### Number of classes
Merging the seven grades into three bands (low 3-5, medium 6, high 7-9) lifts accuracy 0.5423 to 0.5893, but macro F1 goes 0.2589 to **0.5828**, more than doubling. The majority baseline stays at 43.7%, so the task did not become trivial. It became answerable at a resolution the features support.

### Final model
**ReLU with plain SGD**, chosen on generalisation gap (0.0133) and seed stability (spread 0.0056) rather than accuracy, since accuracy could not separate the candidates. Macro F1 was deliberately not the deciding metric because its most favourable configuration also carries the largest swing.

### Files
- `Lab7_Keras_MLP_Multiclass_Wine.ipynb`, full notebook, run top to bottom
- `winequality-red.csv` and `winequality-white.csv`, so the notebook runs without a download
