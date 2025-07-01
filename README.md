# Build My Own Neural Network

A complete implementation of a fully connected feedforward neural network built from scratch using Python and NumPy, without using any deep learning libraries. The project includes both a traditional **least squares classifier** and a custom **neural network** for solving the MNIST handwritten digit classification problem.

## Project Highlights

- Implements a neural network with linear layers, ReLU activation, and MSE loss
- Custom forward and backward propagation for each component
- Gradient descent optimization with numerical and analytical gradient validation
- Classification performance evaluated on the MNIST dataset (digits 0–9)
- One-hot encoding for labels and accuracy/loss tracking throughout training
- Compares model performance between least-squares and neural network approaches

## Technologies Used

- Python 3
- NumPy
- Jupyter Notebook (for experimentation & visualization)
- Matplotlib (optional, for loss/accuracy visualization)

## Architecture Overview

1. **Least Squares Classifier**
   - Implements a linear model trained using normal equations
   - Performs prediction via matrix multiplication
   - Evaluates performance with classification accuracy

2. **Neural Network Components**
   - **Linear Layer**: Matrix multiplication with stored weights
   - **ReLU Activation**: Element-wise nonlinearity
   - **Loss Layer**: Mean Squared Error with backward pass
   - **Forward/Backward Propagation**: Modular, layer-by-layer
   - **Gradient Descent**: Custom training loop with weight updates

3. **Training Process**
   - Uses batches of training data
   - Propagates through layers (forward)
   - Computes gradients (backward)
   - Updates weights using gradients and a learning rate

4. **Evaluation & Testing**
   - Accuracy measured on held-out test data
   - Failure cases visualized
   - Optional hyperparameter tuning for optimization

## Sample Results

```
Training Accuracy (Least Squares): 82.5%
Training Accuracy (Neural Network): 91.2%
Test Accuracy (Neural Network): 88.7%
```

## Repository Structure

```
Build-My-Own-Neural-Network/
├── least_squares_classifier.py      # Least squares classification
├── neural_network.py                # Neural network model
├── layers/
│   ├── linear_layer.py              # Linear layer class
│   ├── relu_layer.py                # ReLU activation
│   └── loss_layer.py                # MSE loss function
├── train_and_test.py                # Training logic and evaluation
├── data/
│   ├── train_data.txt               # 1000 samples (28x28 -> 784 vector)
│   ├── test_data.txt
│   ├── train_labels.txt
│   └── test_labels.txt
└── README.md
```

## How to Run

Ensure you have Python 3 and NumPy installed:

```bash
pip install numpy
```

Run the training script:

```bash
python train_and_test.py
```

You can tweak architecture settings, learning rate, and epochs in the script.

## Future Improvements

- Add bias terms to the linear layers
- Implement new activation functions (e.g., Leaky ReLU, Sigmoid)
- Extend to convolutional layers for better performance on image data
- Use PyTorch or TensorFlow to compare with industry-standard frameworks
