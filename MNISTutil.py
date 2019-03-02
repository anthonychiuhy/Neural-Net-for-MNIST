# -*- coding: utf-8 -*-
"""
MNIST dataset utilities
"""

import numpy as np

def readMNIST():
    # Read raw MNIST data into numpy arrays
    with open('train-images.idx3-ubyte', 'rb') as fi, open('train-labels.idx1-ubyte', 'rb') as fl:
        # Read training data
        fi.seek(16)
        fl.seek(8)
        Xtrain = np.fromfile(fi, dtype=np.uint8) # pixels as unsigned interger (0 - 255)
        Ytrainlabel = np.fromfile(fl, dtype=np.uint8) # labels as unsigned interger (0 - 255)
    with open('t10k-images.idx3-ubyte', 'rb') as fi, open('t10k-labels.idx1-ubyte', 'rb') as fl:
        # Read Testing data
        fi.seek(16)
        fl.seek(8)
        Xtest = np.fromfile(fi, dtype=np.uint8) # pixels as unsigned interger (0 - 255)
        Ytestlabel = np.fromfile(fl, dtype=np.uint8) # labels as unsigned interger (0 - 255)
    return Xtrain, Xtest, Ytrainlabel, Ytestlabel

def prepareXY(Xtrain, Xtest, Ytrainlabel, Ytestlabel, mtrain, mtest, nX, nY):
    # Prepare training and testing data
    # Reshape and normalise X
    Xtrain = Xtrain.reshape((mtrain, nX)).T/255
    Xtest = Xtest.reshape((mtest, nX)).T/255
    # Use one hot representation for Y
    Ytrain = np.zeros((nY, mtrain))
    Ytrain[Ytrainlabel, np.arange(mtrain)] = 1
    Ytest = np.zeros((nY, mtest))
    Ytest[Ytestlabel, np.arange(mtest)] = 1
    return Xtrain, Xtest, Ytrain, Ytest