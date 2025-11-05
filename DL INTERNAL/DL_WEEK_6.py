import keras
import matplotlib.pyplot as plt
import numpy as np

# Load IMDb dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
print(len(x_train))
print(len(x_test))
print(x_train[0])
print(y_train[0])

# Pad sequences
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)

# Define the model (Note: The notebook had two model definitions; this is the second one that was trained)
model = keras.Sequential([
    keras.layers.InputLayer(shape=(200,)),  # Maxlen is 200, not 32
    keras.layers.Embedding(10000, 64),
    keras.layers.SimpleRNN(32, return_sequences=True),
    keras.layers.LSTM(32),
    keras.layers.Dense(1, activation="sigmoid")
])

print(model.summary())

# Compile the model
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(x_train, y_train, batch_size=32, validation_split=0.2, epochs=5)

# Evaluate the model
result = model.evaluate(x_test, y_test, batch_size=32)
print("accuracy:", result[1])
print("loss:", result[0])

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