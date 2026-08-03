# ANN-Deep-Learning-5th-Semester

Practicals for my 5th semester Artificial Neural Networks & Deep Learning course. Everything is built from scratch in NumPy first and only moves to a framework once the maths behind it has been done by hand, so the progression matters more than any single notebook.

Every notebook is committed already run, so the outputs, numbers and plots are from actual executions, not typed in.

| # | Practical | Key idea |
|:-:|---|---|
| 1 | [Perceptron using NumPy](Lab1_Perceptron_using_NumPy_README.md) | one neuron, step activation, heuristic weight updates |
| 2 | [Single Neuron, manual](Lab2_Single_Neuron_Manual_README.md) | sigmoid + loss + gradient derived by hand, real gradient descent |
| 3 | [Feedforward NN](Lab3_Feedforward_NN_README.md) | hidden layer + ReLU, solves XOR, autograd via PyTorch/Keras |
| 4 | [Keras MLP, multiclass](Lab4_Keras_MLP_Multiclass_README.md) | softmax + crossentropy, 4-class smart climate controller |
| 5 | [CNN, two-class images](Lab5_CNN_Image_Classification_README.md) | convolution + parameter sharing, fabric defect detection |

The running thread across the first four: a single neuron can only ever draw one straight decision boundary. Practicals 1 and 2 hit that wall from two different directions (a heuristic rule and true gradient descent, both failing XOR identically). Practical 3 breaks it with a hidden layer. Practical 4 shows the same limitation and the same fix turning up in a realistic multiclass problem instead of a toy truth table.

Practical 5 runs into a different wall. A Dense layer learns a pattern at one specific position and cannot reuse it anywhere else, so it sits at chance on defects that move around the frame. Convolution fixes that by sliding one filter across every position, and the notebook measures the difference rather than asserting it.
