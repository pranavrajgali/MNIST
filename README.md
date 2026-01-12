MNIST Handwritten Digit Classifier
Overview
This project implements a neural network to classify handwritten digits from the MNIST dataset. The MNIST dataset consists of 60,000 training and 10,000 test grayscale images of handwritten digits (0-9), each of size 28x28 pixels.

Features
Loads and preprocesses the MNIST dataset by normalizing pixel values to the range [0,1].
Flattens 28x28 images into 784-dimensional vectors for input into the model.
Implements a feedforward neural network with the following architecture:
Dense layer with 128 neurons and ReLU activation
Dense layer with 64 neurons and sigmoid activation
Dense layer with 32 neurons and sigmoid activation
Output Dense layer with 10 neurons and softmax activation for classification
Uses Adam optimizer and sparse categorical cross-entropy loss for training.
Trains the model for 10 epochs achieving approximately 97.8% accuracy on the test set.
Evaluates model performance with a confusion matrix and detailed classification report (precision, recall, F1-score).
Requirements
Python 3.x
TensorFlow (with Keras API)
NumPy
Matplotlib (for visualization)
scikit-learn (for classification report)
seaborn (for confusion matrix visualization)
Usage
Clone the repository or download the notebook.
Install the required packages if not already installed:
pip install tensorflow numpy matplotlib scikit-learn seaborn
Run the notebook or script to train and evaluate the model on MNIST.
Potential Improvements
Replace the fully connected network with a convolutional neural network (CNN) for improved accuracy.
Add validation split and early stopping to prevent overfitting.
Incorporate regularization techniques such as dropout or batch normalization.
Visualize misclassified examples for error analysis.
References
MNIST Dataset: Modified National Institute of Standards and Technology database
TensorFlow Keras Documentation: 
