# -*- coding: utf-8 -*-
"""
Fully connected feed forward artificial neural network for hand written digits classification on the MNIST dataset

Weight Initialisation: He or Xavier
Activation: ReLU
Output Activation: Softmax
Optimiser: Adam (without bias correction)
Regularisation: Yes


Softmax output layer:
    dZ = 1/m * (Ypred - Y)
    
Back propagation:
    dAprev = np.dot(W.T, dZ)
    dZprev = dAprev * f'(Zprev)

Gradients:
    dW = np.dot(dZ, Aprev.T)
    db = np.sum(dZ, axis=1)

Cross entropy cost:
    1/m * np.sum(-Y * log(Ypred))
    
Adam:
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad**2
    params -= alpha * v/(np.sqrt(s)+ eps)
"""
import numpy as np
import matplotlib.pyplot as plt

from MNISTutil import readMNIST, prepareXY


######################################################################
##### Define utility functions
def countparams(layerdims):
    L = len(layerdims)
    count = 0
    for l in range(1, L):
        count += (layerdims[l-1] + 1) * layerdims[l]
    return count

def initparams(layerdims):
    L = len(layerdims)
    params = {}
    # Initialise weights and bias using He initialisation
    for l in range(1, L):
        params['W'+str(l)] = np.random.randn(layerdims[l], layerdims[l-1]) * np.sqrt(2/layerdims[l-1])
        params['b'+str(l)] = np.zeros((layerdims[l], 1))
    return params

def graddescent(params, grads, alpha):
    L = len(params)//2 + 1
    
    for l in range(1, L):   
        params['W'+str(l)] -= alpha * grads['dW'+str(l)]
        params['b'+str(l)] -= alpha * grads['db'+str(l)]

def initAdam(layerdims):
    L = len(layerdims)
    cache = {}
    # Initialise all exponential moving averages to be 0
    for l in range(1, L):
        cache['vdW'+str(l)] = np.zeros((layerdims[l], layerdims[l-1]))
        cache['vdb'+str(l)] = np.zeros((layerdims[l], 1))
        cache['sdW'+str(l)] = np.zeros((layerdims[l], layerdims[l-1]))
        cache['sdb'+str(l)] = np.zeros((layerdims[l], 1))
    
    return cache

def Adam(params, grads, cache, alpha, beta1 = 0.9, beta2 = 0.999):
    eps = 1e-8
    L = len(params)//2 + 1
    
    for l in range(1, L):
        vdW = cache['vdW'+str(l)]
        vdb = cache['vdb'+str(l)]
        sdW = cache['sdW'+str(l)]
        sdb = cache['sdb'+str(l)]
        dW = grads['dW'+str(l)]
        db = grads['db'+str(l)]
        
        # Compute exponential moving first moment
        vdW = beta1 * vdW + (1 - beta1) * dW
        vdb = beta1 * vdb + (1 - beta1) * db
        # Compute exponential moving second moment
        sdW = beta2 * sdW + (1 - beta2) * dW**2
        sdb = beta2 * sdb + (1 - beta2) * db**2
        
        # Update cache
        cache['vdW'+str(l)] = vdW
        cache['vdb'+str(l)] = vdb
        cache['sdW'+str(l)] = sdW
        cache['sdb'+str(l)] = sdb
        # Update parameters
        params['W'+str(l)] -= alpha * vdW/(np.sqrt(sdW) + eps)
        params['b'+str(l)] -= alpha * vdb/(np.sqrt(sdb) + eps)
    
def forwardlinear(A, W, b): 
    return W.dot(A) + b

def softmax(Z):
    shiftZ = Z - np.max(Z, axis=0) # For numerical stability
    exp = np.exp(shiftZ)
    return exp/(np.sum(exp, axis=0))
    
def relu(Z):
    return Z * (Z > 0)

def backwardlinear(dZ, W):
    return W.T.dot(dZ)

def backwardrelu(dA, Z):
    return dA * (Z > 0)

def gradW(dZ, Aprev):
    return dZ.dot(Aprev.T)

def gradb(dZ):
    return np.sum(dZ, axis=1, keepdims=True)

def crossentropyloss(Ypred, Y):
    eps = 1e-8
    return 1/Y.shape[1] * np.sum(-Y * np.log(Ypred + eps))

def forwardprop(X, params):
    L = len(params)//2
    cache = {}
    
    A = X
    cache['A0'] = A
    for l in range(1, L):
        W = params['W'+str(l)]
        b = params['b'+str(l)]
        Z = forwardlinear(A, W, b)
        A = relu(Z)
        cache['Z'+str(l)] = Z
        cache['A'+str(l)] = A
    WL = params['W'+str(L)]
    bL = params['b'+str(L)]
    ZL = forwardlinear(A, WL, bL)
    AL = softmax(ZL)
    cache['Z'+str(L)] = ZL
    cache['A'+str(L)] = AL
    
    return AL, cache

def backwardprop(Ypred, Y, lambd, params, cacheZA):
    L = len(params)//2 + 1
    m = Y.shape[1]
    grads = {}
    
    dZ = 1/m * (Ypred - Y)
    for l in reversed(range(2, L)):
        Zprev = cacheZA['Z'+str(l-1)]
        Aprev = cacheZA['A'+str(l-1)]
        W = params['W'+str(l)]
        grads['dW'+str(l)] = gradW(dZ, Aprev) + lambd/m * W
        grads['db'+str(l)] = gradb(dZ)
        dZ = backwardrelu(backwardlinear(dZ, W), Zprev)
    A0 = cacheZA['A0']
    W1 = params['W1']
    grads['dW1'] = gradW(dZ, A0) + lambd/m * W1
    grads['db1'] = gradb(dZ)
    
    return grads

######################################################################
##### Initialise and train neural network
np.random.seed(0)
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
epochs = 1
batchsize = 100

# Initialise trainable parameters
params = initparams(layerdims)

# Load X and Y
mtrain = 60000
mtest = 10000
Xtrain, Xtest, Ytrain, Ytest = prepareXY(*readMNIST(), mtrain, mtest, nX, nY)

# Train the network
batches = int(mtrain/batchsize)
Jbatches = []
cacheAdam = initAdam(layerdims)
for epoch in range(epochs):
    for batch in np.random.permutation(batches):
        Xbatch = Xtrain[:, batch*batchsize:(batch+1)*batchsize]
        Ybatch = Ytrain[:, batch*batchsize:(batch+1)*batchsize]
        
        # Forward Propagation
        Ybatchpred, cacheZA = forwardprop(Xbatch, params)
        
        # Compute cost
        Jbatches.append(crossentropyloss(Ybatchpred, Ybatch))
        
        # Backward propagation
        grads = backwardprop(Ybatchpred, Ybatch, lambd, params, cacheZA)
        
        # Gradient descent using Adam
        Adam(params, grads, cacheAdam, alpha, beta1, beta2)
        #graddescent(params, grads, alpha)
    
    if batches * batchsize != mtrain:
        Xbatch = Xtrain[:, batches*batchsize:]
        Ybatch = Ytrain[:, batches*batchsize:]
        Ybatchpred, cacheZA = forwardprop(Xbatch, params)
        Jbatches.append(crossentropyloss(Ybatchpred, Ybatch))
        grads = backwardprop(Ybatchpred, Ybatch, lambd, params, cacheZA)
        Adam(params, grads, cacheAdam, alpha, beta1, beta2)
        #graddescent(params, grads, alpha)
        
    print('Epoch:', epoch + 1, ', Cost:', Jbatches[-1])
        
plt.plot(Jbatches)
plt.xlabel('Number of updates')
plt.ylabel('Cross entropy loss')

print('Number of parameters:', countparams(layerdims))

Ytrainlabelpred = forwardprop(Xtrain, params)[0].argmax(0)
Ytrainlabel = Ytrain.argmax(0)
print('Train accuracy:', np.mean(Ytrainlabelpred == Ytrainlabel))

Ytestlabelpred = forwardprop(Xtest, params)[0].argmax(0)
Ytestlabel = Ytest.argmax(0)
print('Test accuracy:', np.mean(Ytestlabelpred == Ytestlabel))