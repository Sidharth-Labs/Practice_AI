import math
import numpy as np
import matplotlib.pyplot as plt

# Training data
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

# Cost Function
def compute_cost(x, y, w, b):
    m = len(x)
    cost = 0

    for i in range(m):
        prediction = w * x[i] + b
        cost += (prediction - y[i]) ** 2

    return cost / (2 * m)


# Gradient Function
def compute_gradient(x, y, w, b):
    m = len(x)

    dj_dw = 0
    dj_db = 0

    for i in range(m):
        prediction = w * x[i] + b

        dj_dw += (prediction - y[i]) * x[i]
        dj_db += (prediction - y[i])

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


# Gradient Descent
def gradient_descent(x, y, w, b, alpha, iterations):

    cost_history = []

    for i in range(iterations):

        dj_dw, dj_db = compute_gradient(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(x, y, w, b)
        cost_history.append(cost)

        if i % 1000 == 0:
            print(f"Iteration {i}, Cost = {cost:.2f}")

    return w, b, cost_history


# Initial values
w = 0
b = 0

# Hyperparameters
alpha = 0.01
iterations = 10000

# Train Model
w_final, b_final, cost_history = gradient_descent(
    x_train, y_train, w, b, alpha, iterations
)

print("\nFinal Parameters:")
print("w =", w_final)
print("b =", b_final)

# Predictions
print("\nPredictions:")
print("1000 sqft house:", w_final * 1.0 + b_final)
print("1200 sqft house:", w_final * 1.2 + b_final)
print("2000 sqft house:", w_final * 2.0 + b_final)

# Plot Cost
plt.plot(cost_history)
plt.title("Cost vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True)
plt.show()