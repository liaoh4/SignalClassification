# Deep Space Signal Classification

## Introduction

The objective of this project was to implement a neural network capable of distinguishing true pulsar candidates from noise using eight integrated statistical features. The primary challenge involved the scarcity of positive pulsar samples relative to the majority noise class.

## Data Preprocessing Strategy

Data quality significantly impact the convergence of a neural network. Two primary strategies were employed:
### (1) Z-score Normalization
To ensure that features with different magnitudes contribute equally to the gradient updates, Z-score normalization was applied. 

### (2) Handling Class Imbalance
The dataset is highly imbalanced. To prevent the model from predicting "0" for all inputs, a Weighted Cross-Entropy approach was implemented. A scale factor, weight_pos, was calculated as the ratio of negative to positive samples. During backpropagation, the gradients for positive samples were scaled by this factor, forcing the model to penalize misclassifications of pulsars more heavily.

### (3) Optimization of Decision Threshold
During the inference phase, the classification threshold was adjusted to optimize the MCC.  While the default threshold of 0.5 yielded a competitive score, empirical testing on the validation set demonstrated that increasing the threshold to 0.8 improved the model's performance.

## Neural Network Architecture
The model consists of a 3-layer Feed-Forward Network (8->128->64-1) designed to capture non-linear relationships between the statistical moments. Hidden layers ReLU activation while output applies Sigmoid activation. He-Normal initialization was used for ReLU layers to prevent gradient vanishing, while Xavier initialization was applied to the Sigmoid output layer to maintain variance stability.

## Model Training and Optimization
Training was conducted over 100 epochs using Mini-batch Gradient Descent. The mini batch size is 64.

### (1) Advanced Optimization
In the NumPy version, Momentum-based SGD was implemented to accelerate convergence. In the Framework version, the AdamW optimizer was utilized for its superior adaptive learning rate and weight decay properties.

### (2) Best Model Saving
To prevent the final model from being affected by late-stage overfitting, a model checkpointing strategy was implemented. The network weights were preserved whenever a new peak in Validation MCC was achieved, ensuring the most generalized version of the model was used for final testing.
