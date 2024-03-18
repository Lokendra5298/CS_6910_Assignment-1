# CS_6910
### Fashion-MNIST Neural Network Implementation

### Introduction

This project involves the implementation of a feedforward neural network along with backpropagation for training. The neural network is trained and tested using the Fashion-MNIST dataset. The objective is to classify images into one of ten classes based on the input image of size 28 x 28 pixels (784 pixels in total).

### Installation

To install the Fashion-MNIST dataset, you can use pip:

```bash
pip install fashion-mnist
```

### Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from keras.datasets import fashion_mnist
```

### Question 1: Visualization

An image from each category has been visualized in the `wandb` media environment.

### Question 2: Neural Network Implementation

We've developed a class called `FeedForward_NeuralNetwork`, which contains functions for both forward propagation and backward propagation. During forward propagation, the module accepts:
- Input parameters such as batch size, , epochs, target data, loss
- Optimizers
- Number of hidden layers
- Number of neurons in each hidden layer
- Weight initialization (random or Xavier)

Activation functions such as Softmax, Tanh, ReLU, and Sigmoid are implemented during the construction of the forward propagation.

### Question 3: Optimizers Implementation

Various optimizers have been implemented, including:

- SGD (Stochastic Gradient Descent)
- Momentum-based Gradient Descent
- Nesterov Accelerated Gradient Descent
- RMSProp
- Adam
- Nadam
```
Dictionary_optimizers = {"SGD":do_stochastic_gradient_descent(), "Momentum_GD":do_momentum_based_gradient_descent(),\
"NAG":do_nesterov_accelerated_gradient_descent(), "RMS_Prop":do_rmsprop(), "Adam":do_adam(), "Nadam":do_nadam()}
```
By default, the learning rate for all optimizers is set to 0.001. Specific parameters such as gamma, beta, and epsilon are set accordingly for each optimizer.

### Question 4: Hyperparameter Tuning

A sweep function is implemented to tune hyperparameters and obtain results for validation loss, validation accuracy, general loss, step, and accuracy. Hyperparameters include:

- Number of epochs
- Number of hidden layers
- Size of hidden layers
- Learning rate
- Optimizer
- Batch size
- Weight initialization
- Activation function
- Loss function
```
 parameters_dict = {
                "num_epochs": {
                    "values": [5,10]
                }, \
                "num_hidden_layers": {
                    "values": [3, 4, 5]
                }, \
                "size_hidden_layer": {
                    "values": [32, 64,128]
                }, \
                "learning_rate": {
                    "values": [1e-3, 1e-4]
                }, \
                "optimizer": {
                    "values": ["SGD", "Momentum_GD", "NAG", "RMS_Prop", "Adam", "Nadam"]
                }, \
                "batch_size": {
                    "values": [16, 32, 64]
                }, \
                "weight_init": {
                    "values": ["random","xavier"]
                } , \
                "activation": {
                    "values": ["Sigmoid", "Tanh", "Relu"]
                }, \
                "loss": {
                    "values": ["Cross_Entropy", "Squared_Error"]
                }
                  }
```                  
The code snippet provided helps generate the desired results.

### Question 7: Confusion Matrix

The 7th question generates the confusion matrix for the model trained. The code that implememts the matrix is as follows.
```python
wandb.sklearn.plot_confusion_matrix(y_test,
                                            y_test_pred,
                                            ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])
```
### Question 10: Parallel Coordinate Plot

Based on the parallel coordinate plot, three specific sets of hyperparameters are selected and tested with the MNIST digit dataset.
The hyperparameters tuned and used are:

```
 parameters_dict = {
                "num_epochs": {
                    "values": [5,10]
                }, \
                "num_hidden_layers": {
                    "values": [3, 4, 5]
                }, \
                "size_hidden_layer": {
                    "values": [32, 64,128]
                }, \
                "learning_rate": {
                    "values": [1e-3, 1e-4]
                }, \
                "optimizer": {
                    "values": ["SGD", "Momentum_GD", "NAG", "RMS_Prop", "Adam", "Nadam"]
                }, \
                "batch_size": {
                    "values": [16, 32, 64]
                }, \
                "weight_init": {
                    "values": ["random","xavier"]
                } , \
                "activation": {
                    "values": ["Sigmoid", "Tanh", "Relu"]
                }, \
                "loss": {
                    "values": ["Cross_Entropy", "Squared_Error"]
                }
                  }
```                  

The following code snippet helps us generate the results above. 

```python
wandb.log({"val_loss_end": loss_val/t_val.shape[1], \
                   "val_acc_end": acc_val/t_val.shape[1], \
                   "test_loss_end": loss_test/t_test.shape[1], \
                   "test_acc_end": acc_test/t_test.shape[1], \
                   "epoch":config.num_epochs})

####################################################################
sweep_id = wandb.sweep(sweep_config, project = "Assignment1")
```

**Question 7**

The 7th question generates the confusion matrix for the model trained. The code that implememts the matrix is as follows.
```python
wandb.sklearn.plot_confusion_matrix(y_test,
                                            y_test_pred,
                                            ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])
```

The insights from the confusion matrix are:
We have achieved an accuracy of 61.75%.

**Question 10**

By infering from the parallel coordinate plot, 3 speccific set of hyper parameters are taken and then test with the MNIST digit dataset.  
