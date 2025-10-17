# Neural Network from Scratch for Iris Classification

A complete implementation of a neural network built from scratch using only NumPy to classify Iris flower species. This project demonstrates the fundamental mechanics of deep learning without using high-level frameworks like TensorFlow or PyTorch.

## Features

- Pure NumPy implementation
- Forward and backward propagation
- ReLU and Softmax activation functions
- Gradient descent optimization
- Cross-entropy loss function
- Training progress monitoring

## Model Architecture

```
Input Layer (4) → Hidden Layer (5, ReLU) → Output Layer (3, Softmax)
```

- **Input**: 4 features (sepal length, sepal width, petal length, petal width)
- **Hidden Layer**: 5 neurons with ReLU activation
- **Output Layer**: 3 neurons with Softmax activation (Setosa, Versicolor, Virginica)
- **Parameters**: 43 total parameters

## Installation

```bash
git clone https://github.com/yourusername/iris-neural-network.git
cd iris-neural-network
```

## Usage

```python
# Train the model
W1, b1, W2, b2, losses = train_simple_ann(X_train, y_train, epochs=1000)

# Make predictions
predictions = predict(X_test, W1, b1, W2, b2)
accuracy = np.mean(predictions == y_test)
```

## Training Results

After 1000 training epochs:
- **Final Loss**: 0.6246
- **Final Accuracy**: 69%
- **Loss Reduction**: 43% improvement from initial loss

## Implementation Details

The project includes:
- Forward propagation with matrix operations
- Backward propagation with manual gradient calculation
- Weight initialization and updates
- Loss computation and accuracy tracking

## Requirements

- Python 
- NumPy
- scikit-learn (for dataset)

## Educational Value

This project serves as an excellent learning resource for understanding:
- Neural network mathematics
- Backpropagation algorithm
- Gradient descent optimization
- Activation functions and their derivatives

## License

MIT License
