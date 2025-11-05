import keras
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(x_train[0])
num_classes = 10

# Display the first 25 images
figure = plt.figure(figsize=(4, 6))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.axis('off')
    plt.title(str(y_train[i])) # Convert label to string
    plt.imshow(x_train[i])
plt.show()

# Normalize data, expand dimensions for CNN, and one-hot encode labels
# Note: The original file normalized y_train/y_test, which is incorrect.
# Labels should be one-hot encoded, not normalized.
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)
print(x_train[0])


# Define the model
model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(28, 28, 1)),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    keras.layers.Normalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
print(model.summary())

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train_cat, batch_size=256, epochs=5, validation_split=0.1)

# Evaluate the model
score = model.evaluate(x_test, y_test_cat, batch_size=32)
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