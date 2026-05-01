import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

print(tf.config.list_physical_devices('GPU'))

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Show one image
index = 10
plt.imshow(x_train[index])
plt.title(class_names[y_train[index][0]])
plt.axis("off")
plt.show()

# CNN
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),

    keras.layers.Conv2D(32, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(128, (3, 3), activation="relu"),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Summary
model.summary()

# Train
model.fit(x_train, y_train, epochs=10, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("\nTest accuracy:", test_acc)

# SAVE MODEL
model.save("../models/cifar10_cnn_tf.keras")

# Predict one
predictions = model.predict(x_test)

print("\nPredicted:", class_names[predictions[index].argmax()])
print("Actual:", class_names[y_test[index][0]])