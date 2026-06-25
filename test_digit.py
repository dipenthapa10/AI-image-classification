
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

#loading the trained model
model = tf.keras.models.load_model('mnist_cnn_model.keras')
print("Model loaded!")

img = Image.open('7.png')
img = img.convert('L') #convert to grayscale (white or black)
img = img.resize((28,28))
# Convert to numpy array
img_array = np.array(img)

print("Image shape:", img_array.shape)
print("Pixel range:", img_array.min(), "to", img_array.max())

# Show the image
plt.imshow(img_array, cmap='gray')
plt.title('Your Handwritten Digit')
plt.savefig('my_digit_processed.png')
print("Saved processed image!")

img_array = 255 - img_array
# Reshape to match what model expects
img_array = img_array.reshape(1, 28, 28, 1)

# Make prediction
prediction = model.predict(img_array)

# The digit with highest probability wins
predicted_digit = np.argmax(prediction)
confidence = prediction[0][predicted_digit] * 100

print(f"\n AI THINKS YOUR DIGIT IS: {predicted_digit}")
print(f"CONFIDENCE: {confidence:.1f}%")