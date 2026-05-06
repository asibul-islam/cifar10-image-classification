import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Load CIFAR-10 test data
(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load trained model
model = keras.models.load_model("../models/cifar10_transfer_mobilenet_tf.keras")

# Predict
predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)
y_true = y_test.flatten()

# Find wrong predictions
wrong_indices = np.where(y_pred != y_true)[0]

print("Total wrong predictions:", len(wrong_indices))
print("Total test images:", len(y_true))

# Show first 9 wrong predictions
plt.figure(figsize=(10, 10))

for i, idx in enumerate(wrong_indices[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[idx])

    pred_label = class_names[y_pred[idx]]
    true_label = class_names[y_true[idx]]

    plt.title(f"P: {pred_label}\nA: {true_label}")
    plt.axis("off")

plt.tight_layout()
plt.show()