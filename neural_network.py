"""Scratchpad: the whole network on tiny numbers, plus a gradient check."""
import numpy as np


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)   # stable: avoids overflow in exp
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def loss_fn(probs, true_class):
    return -np.log(probs[true_class] + 1e-9)


# ---- 1. Forward pass by hand ----
x = np.array([4, -2])
w1 = np.array([[3, 1, 0],
               [1, 1, 5]])
b1 = np.array([0, 2, 1])
w2 = np.array([[1, 0],
               [2, 1],
               [0, 3]])
b2 = np.array([1, 0])

z1 = x @ w1 + b1
a1 = relu(z1)
z2 = a1 @ w2 + b2
probs = softmax(z2)
print("hidden pre-activation:", z1)      # [10  4 -9]
print("hidden after ReLU:    ", a1)      # [10  4  0]
print("output scores:        ", z2)      # [19  4]
print("probabilities:        ", probs)
print("loss (true class 1):  ", loss_fn(probs, 1))

# ---- 2. Gradient check: compare backprop with a numerical estimate ----
rng = np.random.default_rng(0)
X = rng.standard_normal((5, 4))
y = rng.integers(0, 3, size=5)
W1 = rng.standard_normal((4, 6)) * 0.5
B1 = np.zeros(6)
W2 = rng.standard_normal((6, 3)) * 0.5
B2 = np.zeros(3)


def batch_loss(W1_):
    p = softmax(relu(X @ W1_ + B1) @ W2 + B2)
    return -np.mean(np.log(p[np.arange(len(y)), y]))


# analytic gradient (same formulas as the training script)
Z1 = X @ W1 + B1
A1 = relu(Z1)
P = softmax(A1 @ W2 + B2)
dZ2 = P.copy()
dZ2[np.arange(len(y)), y] -= 1
dZ2 /= len(y)
dW1 = X.T @ ((dZ2 @ W2.T) * (Z1 > 0))

# numerical gradient: nudge each weight up/down by eps
eps = 1e-5
num = np.zeros_like(W1)
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        W1[i, j] += eps
        up = batch_loss(W1)
        W1[i, j] -= 2 * eps
        down = batch_loss(W1)
        W1[i, j] += eps
        num[i, j] = (up - down) / (2 * eps)

rel_err = np.max(np.abs(dW1 - num) / (np.abs(dW1) + np.abs(num) + 1e-12))
print("\nmax relative error, backprop vs numerical:", rel_err)
print("gradient check", "PASSED" if rel_err < 1e-5 else "FAILED")
