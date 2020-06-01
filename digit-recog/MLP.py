import numpy as np
import logging

np.seterr( over='ignore' )

def ReLU(x):
    return x * (x > 0)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    y = sigmoid(z)
    return y*(1-y)

class MLP:

    def __init__(self, layer_spec=[784,250,10], random_state=4):
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

        delta_b[-1] = self.cost_function(a[-1],out_) \
             * sigmoid_prime_vect(z[-1])
        delta_w[-1] = np.dot(delta_b[-1], a[-2].T)

        #Calc sensibility in all hidden layers
        for l in range(2, self.num_layers):
            delta_b[-l] = ( np.dot(self.weights[-l+1].T, delta_b[-l+1])
               * sigmoid_prime_vect(z[-l]) )
            delta_w[-l] = np.dot(delta_b[-l], a[-l-1].T)
            
        return (delta_w, delta_b)

    def cost_function(self, a, y):
        return a - y

    def SGD(self, train_dataset, train_labels, epochs, alpha=0.1):
        train_length = len(train_dataset)

        for epoch in range(epochs):
            logging.info("Epoch {}: Started".format(epoch+1))
            err = 0
            for train_num in range(train_length):
                err += ( np.sum(self.feedforward(train_dataset[train_num]) - 
                    train_labels[train_num]) ) ** 2
                delta_w, delta_b = self.backpropagation(train_dataset[train_num], 
                    train_labels[train_num])
                self.weights =  [w-alpha*nw for w, nw in zip(self.weights, delta_w)]
                self.biases = [b-alpha*nb for b, nb in zip(self.biases, delta_b)]

                if(train_num % 1000 == 0 and train_num > 0):
                    logging.debug("Epoch {} - Training Num: {} - RMSE: {}"
                        .format(epoch+1,train_num,err/(train_num+1)))
            logging.info("Epoch {}] Finished. RMSE: {}".format(epoch+1,err/train_length))

    def predict(self, in_):
        return np.argmax(self.feedforward(in_))




