import numpy as np
import pandas as pd
import copy
import random

class mnist_data(object):
    # defines a dataset class that makes it convenient to refer to the features and labels via methods mnist_data.X
    # and mnist_data.y
    def __init__(self,X,y):
        self.X = X
        self.y = y

def load_data():
    # reads data from csv files and return a dataset object containing the MNIST data
    data = np.loadtxt('mnist_train.csv',delimiter=',')
    n = len(data)
    # pixel intensity is normalized (divided by 255); this turns out to work much better vs un-normalized
    X = (data[:,1:])/255.0
    y = data[:,0]
    train = mnist_data(X,y)
    # test_data
    data = np.loadtxt('mnist_test.csv',delimiter=',')
    n = len(data)
    X = (data[:,1:])/255.0
    y = data[:,0]
    test = mnist_data(X,y)
    return (train,test)

def prepare_data(train,test):
    training_inputs = train.X
    training_old_outputs = train.y
    training_outputs = np.zeros((len(train.y),10),dtype=int)
    for i in range(len(train.y)):
        training_outputs[i,int(training_old_outputs[i])] = 1
    training_data = np.hstack([training_inputs,training_outputs]) 
    test_data = np.hstack([test.X,test.y.reshape(-1,1)])
    return (training_data,test_data)

class neural_net(object):
    # a neural network class

    def __init__(self, sizes=[784,30,10]):
    # initializes the architecture of the network, with 3 layers, 784 inputs (corresponding to 28x28 pixel values),
    # and 10 outputs (relative "probabilities" of the input being each of the 10 labels)
        self.num_layers = len(sizes)
        self.sizes = sizes # the default size is [784,30,10]. might want to change it while tuning
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
    
   
    def feedforward(self, a):
        # given a network weights and biases, calculate the outputs
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    
    def evaluate(self, test_data):
        # given a network weights and biases, calculate the fraction of times it predicts the correct label
        pred = self.feedforward(test_data[:,:784].T)
        pred = np.argmax(pred,axis=0)
        return (pred==test_data[:,-1]).sum() / len(test_data)
    
    
    def cost_derivative(self, output_activations, y):
        # the derivative at the final (output) layer
        return (output_activations-y)
    
   
    def nabla_avg(self, data):
        # backpropagation: compute deltas, then nabla_b and nabla_w
        # note the use of matrix calculations to speed up training
        # data is Bx784
        x = data[:,0:784].T
        y = data[:,784:].T
        deltas = [0]*(self.num_layers-1)
        # feedforward
        activation = x
        activations = [activation] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        deltas[-1] = self.cost_derivative(activations[-1], y)*sigmoid_prime(zs[-1])
        for index in range(2,self.num_layers):
            deltas[-index] = np.dot(self.weights[-index+1].T,deltas[-index+1]) * sigmoid_prime(zs[-index])
        nabla_b = copy.copy(deltas)
        nabla_b = [nb.mean(axis=1).reshape(bias.shape) for nb,bias in zip(nabla_b,self.biases)]
        nabla_w = [0]*(self.num_layers-1)
        for index in range(len(nabla_w)):
            nabla_w[index] = deltas[index].dot(activations[index].T) / len(data)
        return (nabla_b,nabla_w)

    def update(self,mini_batch,eta):
        # gradient descend: subtract derivative, using a learning rate eta
        nabla_b,nabla_w = self.nabla_avg(mini_batch)
        self.biases = [b - eta*nb for b,nb in zip(self.biases,nabla_b)]
        self.weights = [w - eta*nw for w,nw in zip(self.weights, nabla_w)]
            
    def fit_predict(self,train,test,epochs=10,mini_batch_size=10,eta=3):
        # prepare data to be in the appropriate shapes
        training_data,test_data = prepare_data(train,test)
        n = len(training_data)
        for j in range(epochs):
            # randomly select batches for training
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update(mini_batch, eta)
            print("Epoch {0}: {1}".format(j, self.evaluate(test_data)))

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))



training_data,test_data = load_data()
net = neural_net()
net.fit_predict(training_data,test_data)
