# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 17:09:03 2018

@author: HOME1
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 23:49:23 2017

MNIST Fully Connected Articifical Neural Network for handwritten digits recognition using tanh as activation
funtion

one hidden layer and one output layer

input size = 28*28
hidden layer size = 100
output size = 10


@author: Anthony
"""

import numpy as np, cv2

imgsize = 28*28

numtrain = 60000
numtest = 10000

#np.random.seed(12123)


# Data Management and Initialisation
#############################################################################################

# read from training data
with open('train-images.idx3-ubyte', 'rb') as fi, open('train-labels.idx1-ubyte', 'rb') as fl:
    trainimagesheader = [int.from_bytes(fi.read(4),'big') for i in range(4)]
    trainlabelsheader = [int.from_bytes(fl.read(4),'big') for i in range(2)]

    trainimages = [[int.from_bytes(fi.read(1), 'big') for i in range(imgsize)] for j in range(numtrain)]
    trainlabels = [int.from_bytes(fl.read(1), 'big') for i in range(numtrain)]
    
    print('Train data loaded')

# read from testing data
with open('t10k-images.idx3-ubyte', 'rb') as fi, open('t10k-labels.idx1-ubyte', 'rb') as fl:
    testimagesheader = [int.from_bytes(fi.read(4),'big') for i in range(4)]
    testlabelsheader = [int.from_bytes(fl.read(4),'big') for i in range(2)]
    
    testimages = [[int.from_bytes(fi.read(1), 'big') for i in range(imgsize)] for j in range(numtest)]
    testlabels = [int.from_bytes(fl.read(1), 'big') for i in range(numtest)]
    
    print('Test data loaded')

# train images to neural network inputs and outputs numpy arrays
traininputs = np.concatenate([np.array(trainimages),np.ones([numtrain,1])], 1) # add bias
traininputs[traininputs >= 1] = 1

# test images to neural network inputs and outputs numpy arrays
testinputs = np.concatenate([np.array(testimages),np.ones([numtest,1])], 1) # add bias
testinputs[testinputs >= 1] = 1
#############################################################################################


# Neural Network
#############################################################################################

# neural network parameters
inputsize = 28*28 + 1
hiddensize = 30
outputsize = 10

learnrate = 0.001

correcttrain = 0
correcttest = 0

# generate random weights within interval [-5,5], -1 for bias consideration
def randweights(inputsize,outputsize):
    return np.random.normal(0,0.1, [inputsize,outputsize])

# -1 for bias consideration
weightsin = randweights(inputsize, hiddensize-1)
weightsout = randweights(hiddensize, outputsize)

# tanh as activation
def fwdpropagate(inputs, weights):
    return np.tanh(np.dot(inputs,weights))

# back propagation of del, del_j^(l) = d Ein/d S_j^(l)
def backpropagate(inputs, weights, layer):
    return (1 - layer**2) * np.dot(weights,inputs)

    
epoch = 0
while True:
    
    correcttrain = 0
    
    weightsinori = weightsin.copy()
    weightsoutori = weightsout.copy()
    
    for i in np.random.permutation(numtrain):
        
        inputs = traininputs[i]
        label = trainlabels[i]
        
        hiddens = fwdpropagate(inputs, weightsin)
        hiddens = np.concatenate([hiddens, np.ones(1)]) # add bias
        
        outputs = fwdpropagate(hiddens, weightsout)
        
        targets = -0.8*np.ones(10)
        targets[label] = 0.8
        
        err = np.sum((outputs - targets)**2) # define in-sample error to be this
        
        outputdels = 2 * (outputs - targets) * (1 - outputs**2) # derivative of err wrt output weights
        hiddendels = backpropagate(outputdels,weightsout,hiddens)
    
        gradweightsout = np.outer(hiddens, outputdels) # find gradient of output weights
        gradweightsin = np.outer(inputs, hiddendels[:-1]) # find gradient, -1 excluding bias
    
        weightsin -= learnrate * gradweightsin
        weightsout -= learnrate * gradweightsout
        
        #if i % 10000 == 0:
        #    print('Err: ', err, 'label: ', label, ' predicted: ', np.argmax(outputs))
        
        if np.argmax(outputs) == label:
            correcttrain += 1

    epoch += 1
    wchange = np.sqrt(np.sum((weightsin - weightsinori)**2) + np.sum((weightsoutori - weightsout)**2))
    Ein = correcttrain/numtrain

    print('epoch = ', epoch, ', w change = ', wchange, ', accuracy = ', Ein)
    
    # if weight change < 0.1 then finish SGD
    if wchange < 0.6:
        break


for i in range(numtest):
    
    inputs = testinputs[i]
    label = testlabels[i]
    
    hiddens = fwdpropagate(inputs, weightsin)
    hiddens = np.concatenate([hiddens, np.ones(1)]) # add bias
    
    outputs = fwdpropagate(hiddens, weightsout)
    
    if np.argmax(outputs) == label:
        correcttest += 1

Eout = correcttest/numtest
print('Out of sample accuracy = ', Eout)

''' 
#############################################################################################

def imgtoinputs(filename):
    img = cv2.imread(filename,0)
    img[img > 0] = 1;
    img = img.reshape(28*28)
    img = np.concatenate([img, np.ones(1)])
    return img

def inputstolabel(inputs,weightsin,weightsout):
    hiddens = fwdpropagate(inputs, weightsin)
    hiddens = np.concatenate([hiddens, np.ones(1)]) # add bias
        
    outputs = fwdpropagate(hiddens, weightsout)
    
    return np.argmax(outputs)

def imgtolabel(filename,weightsin,weightsout):
    img = imgtoinputs(filename)
    return inputstolabel(img,weightsin,weightsout)
'''