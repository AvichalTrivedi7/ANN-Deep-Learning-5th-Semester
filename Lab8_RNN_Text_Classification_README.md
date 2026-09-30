# Practical 8: RNN-Based Text Classification

### Problem Statement
Implement a Recurrent Neural Network (RNN) using TensorFlow/Keras for binary text classification and evaluate its performance on the IMDB movie-review dataset.

### What's in this notebook
Every earlier practical fed the network a fixed set of features. A review is a sequence of variable length where word order carries meaning, so this practical introduces the RNN: a layer that reads one word at a time and carries a hidden state forward.

**Pipeline:** review text, tokenise to integers, pad to 200, Embedding turns each word id into a vector, SimpleRNN reads the sequence into one summary vector, Dense with sigmoid gives the probability the review is positive.

### Dataset and loading
IMDB Large Movie Review Dataset (Maas et al., 2011): 50,000 reviews, balanced 25,000 / 25,000. Loaded from its raw-text CSV, `https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv`, which the notebook downloads automatically on first run.

Raw text rather than `tf.keras.datasets.imdb`, so that Task 2 is visible instead of hidden inside `load_data()`: vocabulary building, frequency ranking, integer mapping and out-of-vocabulary handling are all done explicitly, with the same index conventions Keras uses (0 pad, 1 start, 2 out-of-vocabulary).

| | value |
|---|---|
| Duplicates removed before splitting | 418 |
| Train / test | 24,791 / 24,791 |
| Distinct words in training corpus | 88,945 |
| Vocabulary kept | 10,000 indices, covering 94.12% of word occurrences |
| Reviews truncated at 200 tokens | 41.3% |

### Architecture
`Embedding(10000, 128) -> SimpleRNN(64) -> Dense(1, sigmoid)`, Adam, binary crossentropy, batch 64, 20% validation split.

1,292,417 parameters, and 99.0% of them are the embedding table. The SimpleRNN that actually processes sequence has 12,352.

### Results

| Baseline | Test accuracy |
|---|:---:|
| Majority class | 0.5037 |
| Bag-of-words logistic regression (no word order) | **0.8750** |

Main model: best validation accuracy at epoch 2 (0.8185). Test accuracy 0.7304 at the specified 5 epochs and 0.7571 after 10, against training accuracy 0.9984.

Results table. Experiment B is one run read at three checkpoints, and A2, B2 and C2 are that same run at epoch 5.

| Experiment | Parameter | Training Acc | Validation Acc | Test Acc |
|---|---|:---:|:---:|:---:|
| A1 | RNN units = 32 | 0.9948 | 0.7143 | 0.7069 |
| A2 | RNN units = 64 | 0.9801 | 0.7274 | 0.7304 |
| A3 | RNN units = 128 | 0.8634 | 0.7750 | 0.7746 |
| B1 | Epochs = 3 | 0.9090 | 0.6675 | 0.6629 |
| B2 | Epochs = 5 | 0.9801 | 0.7274 | 0.7304 |
| B3 | Epochs = 10 | 0.9984 | 0.7588 | 0.7571 |
| C1 | Batch size = 32 | 0.8257 | 0.7391 | 0.7442 |
| C2 | Batch size = 64 | 0.9801 | 0.7274 | 0.7304 |
| C3 | Batch size = 128 | 0.9851 | 0.7637 | 0.7582 |

Beyond the brief:

| Model | Test accuracy | Training time |
|---|:---:|:---:|
| 1-layer SimpleRNN, 5 epochs | 0.7304 | 92s |
| 2-layer SimpleRNN, 5 epochs | 0.7748 | 207s |
| SimpleRNN, 3 epochs | 0.6629 | 57s |
| GRU, 3 epochs | 0.8566 | 124s |
| LSTM, 3 epochs | 0.8438 | 124s |

Predictions on five of my own reviews: 4 of 5 correct.

### My understanding: this model overfits hard and trains unstably
Validation accuracy peaks at epoch 2 (0.8185), then drops 15.1 points in a single epoch and never regains that level, while training accuracy climbs to 0.9984 and validation loss rises to 2.6 times its minimum. With 99% of its parameters in a word table, the model memorises the training reviews.

That instability changes how the results table should be read. Experiment B is one model at three checkpoints, and its test accuracy moves 9.42 points between epochs 3 and 10 with no hyperparameter changing at all. The spread across all nine configurations is 11.17 points. So the table mostly records where each unstable run happened to stop, not what the hyperparameter did. Same lesson as Practical 7 from a different direction: there the noise came from the random seed, here from the stopping epoch.

### My understanding: the word-order test that failed
I built a test to check whether the RNN actually uses word order: shuffle the words inside every review, train the identical model, and compare at epoch 3. The scrambled reviews scored 3.97 points *higher* (0.7026 against 0.6629). Scrambling cannot genuinely help, so this is the instability showing through, since epoch 3 is exactly where the ordered model's validation accuracy had just crashed. As designed, the test cannot answer the question; doing it properly needs several seeds per condition and best-validation comparisons.

What the notebook can say is that order is not what drives accuracy here. Bag of words, with no order at all, beat every recurrent model, GRU and LSTM included.

### My understanding: why GRU and LSTM win
After only 3 epochs, GRU reached 0.8566 and LSTM 0.8438, beating the best SimpleRNN anywhere in the notebook (0.7748) by 8.18 and 6.90 points. Their additive, gated memory lets the training signal reach early words that a SimpleRNN, multiplying by its recurrent weights and a tanh derivative 200 times over, effectively cannot learn from.

### The prediction it got wrong
Review 5, "Not the kind of film I usually enjoy, but I was completely won over by the second half", was predicted Negative at 0.1612. It is the one review I wrote to need more than vocabulary: the positive verdict exists only because "but" overrides the opening denial. That is the case a sequence model should handle better than bag of words, and this one did not.

### Files
- `Lab8_RNN_Text_Classification.ipynb`, full notebook, run top to bottom. The dataset (66 MB) downloads automatically on first run, so it does not need to be committed.
