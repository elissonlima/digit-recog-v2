import numpy as np
import logging
import random

np.seterr( over='ignore' )

def ReLU(x):
    return x * (x > 0)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

class MLP:

    def __init__(self, layer_spec, random_state=0):
        self.num_layers = len(layer_spec)
        self.layer_spec = layer_spec
        np.random.seed(random_state)
        self.biases = [np.random.randn(y, 1) 
                        for y in layer_spec[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in 
                        zip(layer_spec[:-1], layer_spec[1:])]
    
    def feedforward(self, in_):
        sigmoid_vect = np.vectorize(sigmoid)
        for b, w in zip(self.biases, self.weights):
            print(w.shape, b.shape)
            in_ = sigmoid_vect(np.dot(w, in_) + b)
        return in_

    def backpropagation(self, in_, out_):
        
        #Feed forward
        a = [in_]
        z = []
        sigmoid_vect = np.vectorize(sigmoid)
        sigmoid_prime_vect = np.vectorize(sigmoid_prime)


        for b,w in zip(self.biases, self.weights):
            in_ = np.dot(w, in_) + b
            z.append(in_)
            in_ = sigmoid_vect(in_)
            a.append(in_)

        #Calc sensibility in out layer
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        delta = self.cost_derivative(a[-1],out_) \
             * sigmoid_prime_vect(z[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, a[-2].T)

        #Calc sensibility in all hidden layers
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].T, delta_b[-l+1]) \
               * sigmoid_prime_vect(z[-l]) 
            delta_b[-l] = delta
            delta_w[-l] = np.dot(delta, a[-l-1].T)
            
        return (delta_w, delta_b)

    def cost_derivative(self, a, y):
        return a - y

    def SGD(self, train_dataset, epochs, alpha=0.01,
            batch_size=10):
        for epoch in range(epochs):
            logging.info("Epoch {}: Started".format(epoch+1))
            random.shuffle(train_dataset)

            for x, y in train_dataset:

                delta_w, delta_b = self.backpropagation(x,y)

                self.weights =  [w-(alpha*nw)
                        for w, nw in zip(self.weights, delta_w)]
                self.biases = [b-(alpha*nb)
                        for b, nb in zip(self.biases, delta_b)]

            logging.info("Epoch {} Finished. Accuracy: {}"
                .format(epoch+1,self.evaluate(train_dataset)))

    def evaluate(self, dataset):
        test_results = [(np.argmax(self.feedforward(x)), 
                         np.argmax(y))
                        for x, y in dataset]
        return sum(int(x == y) for x, y in test_results) / len(dataset)

    def predict(self, in_):
        return np.argmax(self.feedforward(in_))




