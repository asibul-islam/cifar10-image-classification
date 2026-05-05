import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Check GPU
print(tf.config.list_physical_devices("GPU"))

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Data augmentation
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])

# Pretrained base model
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained model
base_model.trainable = False

# Build model
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Resizing(224, 224),
    data_augmentation,
    keras.layers.Rescaling(1./127.5, offset=-1),
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_split=0.1,
    batch_size=32
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("\nTest accuracy:", test_acc)

model.save("../models/cifar10_transfer_mobilenet_tf.keras")

# Predict one sample
index = 10
predictions = model.predict(x_test)

print("\nPredicted:", class_names[predictions[index].argmax()])
print("Actual:", class_names[y_test[index][0]])

# Plot accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("CIFAR-10 Transfer Learning Accuracy")
plt.show()