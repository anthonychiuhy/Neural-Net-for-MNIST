# -*- coding: utf-8 -*-
"""
Keras implementation of fully connected feed forward artificial neural network for hand written digits classification on the MNIST dataset
"""
import numpy as np
import matplotlib.pyplot as plt

from MNISTutil import readMNIST, prepareXY

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras import optimizers

# Define neural network architecture
nX = 28*28
nh1 = 100
nY = 10
layerdims = (nX, nh1, nY)

# Set optimasation hyperparameters
lambd = 0.0001 # Regularisation parameter

alpha = 0.001 # learning rate
beta1 = 0.9 # First moment
beta2 = 0.999 # Second moment
epochs = 10
batchsize = 100

# Define regularizer and optimizer
l2regularizer = kernel_regularizer=regularizers.l2(lambd)
Adamoptimizer = optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)

# Load X and Y
mtrain = 60000
mtest = 10000
Xtrain, Xtest, Ytrain, Ytest = prepareXY(*readMNIST(), mtrain, mtest, nX, nY, 'first')

# Create graph for neural network
inputs = Input(shape=(layerdims[0],))
x = inputs
for l in range(1, len(layerdims)-1):
    x = Dense(layerdims[l], activation='relu', kernel_regularizer=l2regularizer)(x)
y = Dense(layerdims[-1], activation='softmax', kernel_regularizer=l2regularizer)(x)

# Create model
model = Model(inputs=inputs, outputs=y)

# Train model
model.compile(optimizer=Adamoptimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(Xtrain, Ytrain, batch_size=batchsize, epochs=epochs)

# Evaluate model
model.summary()
print('Model accuracies:')
print('Train accuracy:', model.evaluate(x=Xtrain, y=Ytrain)[-1])
print('Test accuracy:', model.evaluate(x=Xtest, y=Ytest)[-1])

# Plot losses
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Cross entropy loss')
