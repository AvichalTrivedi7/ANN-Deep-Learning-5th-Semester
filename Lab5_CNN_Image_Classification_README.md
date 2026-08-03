# Practical 5: CNN for Two-Class Image Classification

### Problem Statement
Using TensorFlow/Keras and the Cats vs Dogs dataset (or any two-class image / domain dataset), perform the CNN model.

### What's in this notebook
Practicals 1-4 classified rows of numbers. This one classifies images, and the whole point is a property images have that plain numbers don't: **a pattern means the same thing wherever it appears**.

Instead of Cats vs Dogs I used a domain dataset, **automated fabric defect detection**, deciding whether a woven swatch is clean or defective. It's a real two-class vision problem, it's what an actual textile inspection line needs, and critically it let me control the data precisely enough to *prove* what the convolution buys rather than assert it.

**Pipeline:** image → conv filters slide over it looking for local patterns → pooling shrinks the map → more filters → flatten → dense layer → sigmoid → clean or defective.

### What changes from Practical 4

| | Practical 4 (MLP) | Practical 5 (CNN) |
|---|---|---|
| Input | 3 numbers | 32x32 grid of pixels |
| First layer | `Dense` (every input to every neuron | `Conv2D`) one small filter slid across the image |
| Weights | one per input, per neuron | **one small filter reused at every position** |
| Downsizing | none | `MaxPooling2D` |
| Output | 4 neurons + softmax | 1 neuron + sigmoid |

The output head goes back to Practical 3's binary setup. The genuinely new idea is entirely in the first layers.

### The dataset
Every image is a freshly generated plain weave with its own random lighting, exposure and yarn fuzz. Defective samples get exactly one localised flaw of one of four types: **slub** (thick yarn lump, bright bar), **hole** (dark blob), **thread break** (snapped warp thread, weave missing in a strip), **misweave** (patch with the weave pattern shifted).

Two deliberate design decisions, for the same reason as Practical 4's climate rules. So the experiment can prove something:

1. **The defect can appear anywhere in the frame.** This is what makes position-independence necessary rather than optional.
2. **Defects don't change overall brightness.** A slub brightens a few pixels, a hole darkens a few; averaged over the image they cancel. This blocks the cheap solution and forces the model to detect local structure.

3% of labels are flipped, standing in for an inspector mislabelling a borderline swatch.

### Results

| Model | Parameters | Test accuracy |
|---|:---:|:---:|
| Always guess one class |, | 50.4% |
| Brightness statistics (mean/std/min/max) |, | 68.6% |
| MLP (flattened pixels) | 139,521 | **50.4%** |
| CNN | **5,889** | **89.4%** |

The MLP at chance is not a tuning failure, four configurations up to 1,050,625 parameters across two learning rates all landed on chance.

### The experiment that isolates why
The MLP failing and the CNN working is suggestive, not proof. Two competing explanations: (a) the MLP can't detect these local patterns at all, or (b) it can detect them fine but can't cope with them **moving**. So I rebuilt the dataset with the defect pinned to the exact centre, same defect types, same sizes, same subtlety, only position variability removed.

```
                              MLP       CNN
defect FIXED at centre     0.8450    0.9200
defect at RANDOM spot      0.5040    0.9240
```

Pin the defect and the MLP jumps from chance to **84.5%**. Let it move and it falls 34 points back to chance, while the CNN doesn't move at all. So the MLP could always detect these defects, **position was the entire problem**.

### Generalising to positions never seen
Trained the CNN with defects confined to the **left half** of the frame, tested on the **right half** where it saw no defects at all during training: 93.8% → 92.5%, a 1.3 point drop. The filter transfers to unseen territory essentially for free, because it's the same filter sliding over both halves.

(The MLP row of that table is uninformative. It scored 50.3% on the half it *was* trained on, so it had nothing to transfer.)

### My understanding: the CNN is not uniformly good
Per defect type: slub 98.2%, hole 100%, misweave 100%, **thread break 32.2%**.

The error budget reconciles exactly: 78 missed thread breaks + 2 missed slubs + 13 noise-flipped labels with no actual defect = 93, matching the confusion matrix. So **84% of every mistake is a missed thread break**.

My reading: slub, hole and misweave all *add* something, a bright bar, a dark blob, a shifted patch. A thread break *removes* something; it's a strip where the regular weave texture is simply absent, at roughly the same brightness as its surroundings. Detecting "the expected rhythm stopped here" is a harder ask than "there is a dark spot here", especially for 3x3 filters that barely span one weave period.

On a real inspection line this is the number that decides deployment. 89% overall sounds shippable; "misses two-thirds of thread breaks" isn't. And you'd only find that out by breaking results down by type. Same lesson as Practical 4's Fan class.

### The real-data result, and why the tie is the point
Fashion-MNIST, pullover vs shirt (the two hardest classes to separate):

```
MLP: 89.50%   108,801 parameters
CNN: 89.55%     5,889 parameters
```

A dead tie, on fabric the same two models were 39 points apart. Fashion-MNIST garments are **centred and size-normalised**, so there's essentially no position variability, which is exactly the condition where a convolution's advantage disappears. The two results agree: the CNN's edge is specifically translation invariance. What it still gets here is the same accuracy for 18x fewer parameters.

A CNN isn't automatically better because it's a CNN. It's better when the thing you're looking for can appear anywhere.

### Files
- `Lab5_CNN_Image_Classification.ipynb`, full notebook, run top to bottom: dataset generation, statistics and MLP baselines, the MLP fairness sweep, the CNN, per-defect-type breakdown, the fixed-vs-random isolating experiment, the left/right generalisation test, feature map visualisation, and the Fashion-MNIST control.
