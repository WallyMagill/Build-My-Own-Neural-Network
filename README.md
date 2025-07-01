# Build My Own Neural Network

This project demonstrates how to implement a feedforward neural network entirely from scratch using **Python** and **NumPy**. A baseline least-squares classifier is provided for comparison. Both methods are evaluated on a small subset of the MNIST handwritten digit dataset contained in this repository.

## Key Features
- Simple least-squares classifier as a baseline
- Custom linear layers with manual forward and backward passes
- **ReLU** activation and **Mean Squared Error** loss
- Gradient descent training loop with optional loss/accuracy plots
- Visualization of sample digits and predictions

## Technologies Used
- Python 3
- NumPy
- Matplotlib (for plotting loss/accuracy and digit images)
- Jupyter Notebook

## Network Architecture and Training
The neural network defined in `Entree_Task_Walter_Magill_f0055t2.ipynb` uses the architecture:
```
Input (28×28 → 784) → Linear(784, 256) → ReLU → Linear(256, 10)
```
Training is performed with mini-batch gradient descent. All operations – forward propagation, backward propagation, and weight updates – are programmed manually without external deep learning libraries.

The notebook also includes a function to plot training loss and testing accuracy over epochs.

## Evaluation Results
From the included notebooks:
- **Least Squares Classifier** (baseline) reaches ~72% training accuracy and ~42% test accuracy.
  ```
  Training accuracy is: 0.721
  Testing accuracy is: 0.42
  ```
- **Neural Network** achieves around 86% test accuracy after 200 epochs.
  ```
  Epoch:  200 / 200  | Train loss:  0.02902708029259433  | Test Accuracy :  0.86
  ```

## How to Run
1. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```
2. Launch Jupyter Notebook and open one of the notebooks:
   - `Appetizer_Task_Walter_Magill_f0055t2.ipynb` – least-squares baseline
   - `Entree_Task_Walter_Magill_f0055t2.ipynb` – neural network implementation
3. Execute the cells to train and evaluate the models. The notebooks load the dataset from the **MNIST_Sub** folder by default.

## Example Outputs
During training the notebooks print loss and accuracy for each epoch and display example digit images with predicted labels. Misclassified digits are highlighted in red while correct predictions are shown in green.

## Directory Structure
```
Build-My-Own-Neural-Network/
├── Appetizer_Task_Walter_Magill_f0055t2.ipynb  # Baseline classifier
├── Entree_Task_Walter_Magill_f0055t2.ipynb     # Neural network notebook
├── MNIST_Sub/                                  # MNIST subset (1000 train / 200 test)
│   ├── train_data.txt
│   ├── train_labels.txt
│   ├── test_data.txt
│   └── test_labels.txt
├── README.md
```
This repository is completely self-contained and offers a clear demonstration of building and training a neural network without relying on external deep learning frameworks.
