import numpy as np

# ==========================
# SIGMOID
# ==========================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ==========================
# COST FUNCTION
# ==========================

def compute_cost(X, y, w, b):
    m = X.shape[0]

    z = np.dot(X, w) + b
    f_wb = sigmoid(z)

    cost = -(1 / m) * np.sum(
        y * np.log(f_wb) +
        (1 - y) * np.log(1 - f_wb)
    )

    return cost


# ==========================
# GRADIENT
# ==========================

def compute_gradient(X, y, w, b):

    m = X.shape[0]

    z = np.dot(X, w) + b
    f_wb = sigmoid(z)

    error = f_wb - y

    dj_dw = (1 / m) * np.dot(X.T, error)
    dj_db = (1 / m) * np.sum(error)

    return dj_db, dj_dw


# ==========================
# REGULARIZED COST
# ==========================

def compute_cost_reg(X, y, w, b, lambda_):

    m = X.shape[0]

    cost = compute_cost(X, y, w, b)

    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)

    return cost + reg_cost


# ==========================
# REGULARIZED GRADIENT
# ==========================

def compute_gradient_reg(X, y, w, b, lambda_):

    m = X.shape[0]

    dj_db, dj_dw = compute_gradient(X, y, w, b)

    dj_dw = dj_dw + (lambda_ / m) * w

    return dj_db, dj_dw


# ==========================
# GRADIENT DESCENT
# ==========================

def gradient_descent(
    X,
    y,
    w,
    b,
    alpha,
    num_iters,
    lambda_=0
):

    J_history = []

    for i in range(num_iters):

        if lambda_ == 0:
            dj_db, dj_dw = compute_gradient(X, y, w, b)
        else:
            dj_db, dj_dw = compute_gradient_reg(
                X, y, w, b, lambda_
            )

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if lambda_ == 0:
            cost = compute_cost(X, y, w, b)
        else:
            cost = compute_cost_reg(
                X, y, w, b, lambda_
            )

        J_history.append(cost)

        if i % 1000 == 0:
            print(f"Iteration {i}: Cost {cost:.6f}")

    return w, b, J_history


# ==========================
# PREDICTION
# ==========================

def predict(X, w, b):

    probabilities = sigmoid(np.dot(X, w) + b)

    return (probabilities >= 0.5).astype(int)


# ==========================
# EXAMPLE USAGE
# ==========================

X = np.array([
    [34, 78],
    [30, 43],
    [35, 72],
    [60, 86],
    [79, 75]
])

y = np.array([0, 0, 0, 1, 1])

m, n = X.shape

w = np.zeros(n)
b = 0

alpha = 0.001
iterations = 10000

w, b, J_history = gradient_descent(
    X,
    y,
    w,
    b,
    alpha,
    iterations
)

predictions = predict(X, w, b)

accuracy = np.mean(predictions == y) * 100

print("\nWeights:", w)
print("Bias:", b)
print("Predictions:", predictions)
print(f"Accuracy: {accuracy:.2f}%")