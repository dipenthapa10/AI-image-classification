#AI - MNIST digit recoggnizer
#Dipen Thapa

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier # mutlilayer perception
from sklearn.metrics import accuracy_score # calculates how accurate our model is
import tensorflow as tf

(X_train , y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# x train = images and y_train = answer

print("training images: ", X_train.shape)
print("training labels: ",y_train.shape)
print("test images: ", X_test.shape)
print("test labels: ", y_test.shape)


X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000,784)

print("after flattening: ")
print("X_train shape:", X_train.shape)
print("X_test shape: ",X_test.shape)


# normalzing pixels 

X_train = X_train / 255.0
X_test = X_test / 255.0
print("after normalising")
print("Minimum pixel value: ", X_train.min())
print("Maximum mixel value: ",X_train.max())

#neural networks 

model = MLPClassifier(hidden_layer_sizes=(128,64), # two hidden layers: layer 1 = 128 & 2 =64
                      activation = 'relu', #function each neurons uses to decide
                      solver='adam', # method to update the weights during learning
                      max_iter = 20,
                      random_state = 42,
                      verbose = True) # prints the progress during training 

print("Neural Network Built")