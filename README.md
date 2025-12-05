# image-double-digit-preditction
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=3)

# -----------------------------
# USER INPUT NUMBER
# -----------------------------
num = int(input("Enter a digit (0–9): "))

# Find the first test image with that digit
index = np.where(y_test == num)[0][0]

img = x_test[index]

# Show image
plt.imshow(img, cmap='gray')
plt.title("Actual Digit = " + str(num))
plt.axis('off')
plt.show()

# Predict
img_input = img.reshape(1, 28, 28)
pred = model.predict(img_input)

print("Predicted digit =", pred.argmax())
