# Practical 4 — Keras MLP for Multiclass Classification

### Problem Statement
Implement a Keras MLP for multiclass classification.

### What's in this notebook
Practicals 1-3 were all binary — one output neuron answering yes/no. This one moves to multiclass, built around an actual use case rather than a stock dataset: a **smart climate controller** that reads room sensors and decides which appliance to fire.

**Pipeline:** sensor readings → scaling → hidden layers (ReLU) → 4-neuron output layer (softmax) → probability per action → `argmax` → fire that appliance.

### The use case
Three sensors — temperature, humidity, occupancy. Four possible actions:

| Class | Action |
|:---:|---|
| 0 | Do nothing |
| 1 | Fan |
| 2 | AC |
| 3 | Heater |

The rules the data is generated from (the network never sees these — it only sees readings and the resulting action):

```
room EMPTY                          -> Do nothing   (regardless of temperature)
temp < 18                           -> Heater
18 <= temp < 24                     -> Do nothing   (comfortable)
24 <= temp < 32, humidity < 60      -> Fan          (warm but dry)
24 <= temp < 32, humidity >= 60     -> AC           (warm and sticky)
temp >= 32                          -> AC
```

Chosen deliberately so the dataset contains three different kinds of logic: plain thresholds (linear), a two-feature interaction (Fan vs AC), and a gate (occupancy overrides everything — structurally the same problem as XOR from Practicals 1-3). 5% of labels are randomly flipped to stand in for sensor glitches, which puts a hard ~96% ceiling on achievable accuracy.

### What changes from Practical 3
Only the output end. The hidden layers are identical.

| | Practical 3 (binary) | Practical 4 (multiclass) |
|---|---|---|
| Output neurons | 1 | 4 — one per class |
| Output activation | sigmoid | softmax |
| Loss | binary crossentropy | categorical crossentropy |
| Reading the answer | `output >= 0.5` | `argmax(outputs)` |

Softmax matters because it normalises across all 4 outputs so they sum to 1 — making them a real probability distribution where classes compete. Four independent sigmoids would happily claim 90% confidence in two different appliances at once.

### Results

| Model | Test accuracy |
|---|:---:|
| Always predict majority class | 38.8% |
| No hidden layer (linear/softmax) | 91.2% |
| MLP [16, 8] | **96.5%** |

Per-class recall is the more useful reading:

| Class | Baseline | MLP |
|---|:---:|:---:|
| Nothing | 94.8% | 96.6% |
| **Fan** | **57.5%** | **89.0%** |
| AC | 94.9% | 98.3% |
| Heater | 99.2% | 98.3% |

Sweeping the entire input space and comparing against the known ground-truth rules: the MLP reproduces **98.6%** of the true decision map, the linear model only 90.6%. With the room empty, the MLP predicts "do nothing" across **100%** of the temperature/humidity space — it learned the gate without ever being shown the rule.

### My understanding — the hidden layer fixed one class, not all of them
Overall accuracy moving 91.2% → 96.5% is easy to dismiss as a small gain. The per-class split is the actual finding. Heater sat at ~99% under both models and even dropped by one sample — it's a pure temperature threshold, a straight line handles it, there was nothing to add. Fan went 57.5% → 89.0%, and Fan→AC confusions collapsed from 23 to 4.

Fan is the only class whose rule depends on two features interacting (warm **and** dry). Checking the region areas confirms the mechanism: the linear model draws the Fan region at 11.9% of the plane when it should be 15.2%, because it can't cut the horizontal humidity boundary at 60% and approximates it with a slanted line instead. Same lesson as XOR in Practicals 1-3 — thresholds on single features are linear and easy, decisions depending on a combination need a hidden layer — just showing up in a realistic problem instead of a toy truth table.

### On feature scaling — I had this wrong at first
My first run tested scaled vs raw features on one seed. The unscaled model collapsed to 38.8% (predicting a single class for everything) and I'd written it up as "scaling is mandatory or the model breaks."

Re-running across five seeds showed that was overclaiming. Raw features reached 83-94% on four of the five seeds and only collapsed on one:

```
SCALED  mean 0.9603   worst 0.9583   spread 0.0067
RAW     mean 0.8087   worst 0.3883   spread 0.5567
```

The honest finding is that unscaled training is a coin flip — a 55 point spread across seeds versus 0.7 points when scaled. Occupancy is the most important feature but only takes values 0 or 1 next to a temperature feature spanning 10-40, so whether the model learns to use it at all depends on the luck of the initialisation. Same practical advice (always scale), different and more defensible reason.

### Architecture choice
Averaged over 3 seeds each: `[8]` 95.6%, `[16]` 95.9%, `[16,8]` **96.6%**, `[32,16]` 96.2%. Doubling capacity from `[16,8]` to `[32,16]` made it marginally worse — once the network can represent the function, extra capacity stops helping. Same conclusion as the 2-vs-4 hidden neuron test in Practical 3. Went with `[16,8]` as the smallest architecture that reaches the noise ceiling.

### Sanity check on real data
Same code pattern on scikit-learn's handwritten digits — real measured data, 10 classes, only the input width (64) and output width (10) changed. Test accuracy 96.9%.

### Files
- `Lab4_Keras_MLP_Multiclass.ipynb` — full notebook, run top to bottom, includes the baseline comparison, decision-map reconstruction, occupancy gate test, live firing-decision scenarios, both multi-seed experiments, and the digits sanity check.
