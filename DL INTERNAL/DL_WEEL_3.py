import keras
import tensorflow
import numpy as np
from keras import layers
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Display the first 25 images
figure = plt.figure(figsize=(4, 6))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.axis('off')
    plt.title(str(y_train[i]))  # Convert label array to string for title
    plt.imshow(x_train[i])
plt.show()

# Normalize and reshape data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Note: The original file had expand_dims which is unnecessary for 3-channel (RGB) images.
# If the model expected a different shape, it would be added here.
# x_train = np.expand_dims(x_train, -1) # Original line
# x_test = np.expand_dims(x_test, -1)  # Original line

# Define the model
model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=(32, 32, 3)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Normalization(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

print(model.summary())

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_split=0.2)

# Evaluate the model
score = model.evaluate(x_test, y_test, batch_size=32)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Plot training history (Accuracy)
plt.plot(history.history["accuracy"], label="acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.show()

# Plot training history (Loss)
plt.plot(history.history["loss"], label='loss')
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.show()