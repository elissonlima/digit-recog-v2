import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

class MLPKeras:

    def __init__(self, layer_spec=[784,250,10], random_state=0):
        self.model = Sequential()
        np.random.seed(random_state)
        self.model.add(Dense(layer_spec[1], activation='sigmoid', 
            input_shape=(layer_spec[0],)))
        for i in range(2, len(layer_spec)):
            self.model.add(Dense(layer_spec[i], activation='sigmoid'))

    def SGD(self, train_dataset, train_labels, epochs, alpha=0.1, batch_size=10):
        self.model.compile(loss="mse", optimizer=SGD(learning_rate=alpha),
                            metrics=['accuracy'])

        self.model.fit(train_dataset.reshape((train_dataset.shape[0],
                    train_dataset.shape[1])), 
                    train_labels.reshape((train_labels.shape[0],
                    train_labels.shape[1])), epochs=epochs,
                        verbose=1, batch_size=batch_size )