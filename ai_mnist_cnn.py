# CNN - MNIST Digit Recognizer
# Dipen Thapa

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Fix random seed 
# tf.random.set_seed(42)
# np.random.seed(42)

# load data
(X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()
print("Training Imagee: ", X_train.shape)
print("test Image: ", X_test.shape)

#prepare data
X_train = X_train.reshape(60000,28,28,1) # kept as 2D grid where 1 is number of color channels i.e just brightness
X_test = X_test.reshape(10000,28,28,1)

X_train = X_train/255.0
X_test = X_test / 255.0

print("after reshape: ",X_train.shape)
print("after normalising: min=", X_train.min(),"max= ", X_train.max())

# CNN
#build model
model = models.Sequential([
     # CONVOLUTIONAL LAYER 1
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),

#     32        →  number of filters
#              32 different windows slide across the image
#              each filter looks for a different pattern
#              filter 1 finds horizontal edges
#              filter 2 finds vertical edges
#              filter 3 finds curves
#              and so on...

# (3,3)     →  size of each filter window
#              3 pixels wide, 3 pixels tall
#              looks at a 3×3 group of pixels at a time

# activation='relu'  →  same relu you already know

# input_shape=(28,28,1)  →  tells the model what our
#                            input looks like
#                            28 tall, 28 wide, 1 channel

    # POOLING LAYER 1
    layers.MaxPooling2D((2,2)),

     # CONVOLUTIONAL LAYER 2
    layers.Conv2D(64, (3,3), activation='relu'),
    
    # POOLING LAYER 2
    layers.MaxPooling2D((2,2)),
    
    # FLATTEN
    layers.Flatten(),
    
    # FULLY CONNECTED LAYER
    layers.Dense(64, activation='relu'),
    
    # OUTPUT LAYER
    layers.Dense(10, activation='softmax')
])
model.summary()

#complie
model.compile( optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ["accuracy"]
              )

print('model compiled')

# .fit () --> for training the model
# train
history = model.fit(
    X_train, y_train,
                    epochs = 5,
                    validation_data = (X_test,y_test)
                    )

#evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("CNN accuracy: ", round(test_accuracy * 100, 2), "%")


# Manual accuracy check using for loop
print("Checking accuracy manually with for loop...")
correct = 0
total = len(X_test) #total number of test images
for i in range(total):
    # get one image and reshape for model
    single_image = X_test[i].reshape(1, 28, 28, 1)
    # make prediction
    prediction = model.predict(single_image, verbose=0) # verbose=0 means do not print anything
    # get predicted digit
    predicted_digit = np.argmax(prediction)
    # get correct answer
    true_digit = y_test[i]
    
    # check if correct
    if predicted_digit == true_digit:
        correct += 1
# calculate accuracy
manual_accuracy = correct / total * 100

print(f"Manual accuracy : {round(manual_accuracy, 2)}%")
print(f"Evaluate accuracy: {round(test_accuracy * 100, 2)}%")


# saving the trained model
model.save('mnist_cnn_model.keras')
print("Model saved!")