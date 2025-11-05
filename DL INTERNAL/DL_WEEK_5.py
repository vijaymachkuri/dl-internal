import keras
import matplotlib.pyplot as plt

# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

print("Number of training samples:", len(x_train))
print("Number of test samples:", len(x_test))
print("First training review (encoded):", x_train[0])
print("First training label:", y_train[0])

# Pad sequences to ensure equal length
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)

# Build the model
model = keras.Sequential([
    keras.layers.InputLayer(shape=(200,), batch_size=32),
    keras.layers.Embedding(10000, 1, input_length=200),
    keras.layers.Conv1D(32, 3, activation="relu"),
    keras.layers.MaxPool1D(3),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation="sigmoid")
])

# Display model summary
print(model.summary())

# Compile the model
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=5,
    validation_data=(x_test, y_test)
)

# ---- Plot Accuracy vs Validation Accuracy ----
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Accuracy vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ---- Plot Loss vs Validation Loss ----
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Loss vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()