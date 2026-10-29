import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ==========================
# LOAD DATA
# ==========================

digits = load_digits()

X = digits.data
y = digits.target

print("X shape:", X.shape)
print("y shape:", y.shape)

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================
# BUILD MODEL
# ==========================

model = Sequential([
    Dense(25, activation="relu", input_shape=(64,)),
    Dense(15, activation="relu"),
    Dense(10, activation="softmax")
])

# ==========================
# COMPILE MODEL
# ==========================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# TRAIN MODEL
# ==========================

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    verbose=1
)

# ==========================
# EVALUATE MODEL
# ==========================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nAccuracy:", accuracy * 100, "%")

# ==========================
# MAKE PREDICTIONS
# ==========================

prediction = model.predict(X_test[:5])

print("\nPredictions:")

for i in range(5):
    predicted_digit = np.argmax(prediction[i])

    print(
        f"Actual: {y_test[i]} | "
        f"Predicted: {predicted_digit}"
    )
