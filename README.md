# Neural-Net-for-MNIST
A simple Neural Network to classify MNIST data.

Implementations from scratch using Numpy or using the Keras library.

Feed forward fully connected artificial neural network for classification of handwritten digits data from MNIST using relu for hidden layers activations and softmax for output layer activation.

Cost function is minimised using the Adam optimisation algorithm.

The script shows that even with a relatively simple neural network with one hidden layer and few nodes could classify the digits quite accurately (>90 % for test set).

The number of layers of the neural network in this script could be easily modified by redefining the layerdims variable.

The full set of training and testing hand written digits data from the MNIST dataset (uncompressed) must be present for the Python script to run.

Download the MNIST hand written digits dataset (compressed) here:
http://yann.lecun.com/exdb/mnist/
