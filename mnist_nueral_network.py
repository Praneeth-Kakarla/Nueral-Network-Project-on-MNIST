"""Two-layer neural network on MNIST, written with NumPy only.

Improvements over the first version:
  * float32 everywhere (faster, matches the data type)
  * index-based mini-batches (no copying the dataset every epoch)
  * Adam optimizer + learning-rate decay
  * small random-shift data augmentation (helps with hand-drawn digits)
  * keeps the best epoch (by validation accuracy)
  * saves weights as .npz (safe to load, unlike pickle)
"""
import gzip
import pickle
import time

import numpy as np

# ---- configuration ----
DATA_PATH = "mnist.pkl.gz"
MODEL_PATH = "model.npz"
HIDDEN_SIZE = 128
EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
LR_DECAY = 0.9          # lr is multiplied by this after every epoch
AUGMENT = True          # random shifts of up to MAX_SHIFT pixels
MAX_SHIFT = 2
SEED = 0


# ---- data ----
def load_data(path=DATA_PATH):
    with gzip.open(path, "rb") as f:
        train, val, test = pickle.load(f, encoding="latin1")
    return tuple((X.astype(np.float32), y.astype(np.int64)) for X, y in (train, val, test))


def augment(X, rng, max_shift=MAX_SHIFT):
    """Shift every image by a random (dy, dx). MNIST borders are empty, so wrap-around is harmless."""
    imgs = X.reshape(-1, 28, 28)
    out = np.empty_like(imgs)
    shifts = rng.integers(-max_shift, max_shift + 1, size=(len(imgs), 2))
    for i, (dy, dx) in enumerate(shifts):
        out[i] = np.roll(imgs[i], (dy, dx), axis=(0, 1))
    return out.reshape(len(X), -1)


# ---- model ----
def init_params(input_size, hidden_size, output_size, rng):
    """He initialization, float32."""
    return {
        "W1": (rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)).astype(np.float32),
        "b1": np.zeros(hidden_size, dtype=np.float32),
        "W2": (rng.standard_normal((hidden_size, output_size)) * np.sqrt(2.0 / hidden_size)).astype(np.float32),
        "b2": np.zeros(output_size, dtype=np.float32),
    }


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def forward(p, X):
    z1 = X @ p["W1"] + p["b1"]
    a1 = relu(z1)
    probs = softmax(a1 @ p["W2"] + p["b2"])
    return z1, a1, probs


def cross_entropy(probs, y):
    n = probs.shape[0]
    return float(-np.mean(np.log(probs[np.arange(n), y] + 1e-9)))


def backward(p, X, y, z1, a1, probs):
    n = X.shape[0]
    dz2 = probs.copy()
    dz2[np.arange(n), y] -= 1.0          # softmax + cross-entropy gradient: probs - onehot
    dz2 /= n

    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)
    dz1 = (dz2 @ p["W2"].T) * (z1 > 0)   # ReLU derivative
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def accuracy(p, X, y):
    return float(np.mean(np.argmax(forward(p, X)[2], axis=1) == y))


# ---- optimizer ----
class Adam:
    def __init__(self, params, beta1=0.9, beta2=0.999, eps=1e-8):
        self.b1, self.b2, self.eps, self.t = beta1, beta2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, lr):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= (lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(np.float32)


# ---- training ----
def main():
    rng = np.random.default_rng(SEED)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    print(f"train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

    params = init_params(X_train.shape[1], HIDDEN_SIZE, 10, rng)
    opt = Adam(params)
    lr = LEARNING_RATE
    n = X_train.shape[0]
    best_val, best_params = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        perm = rng.permutation(n)
        total_loss, batches = 0.0, 0

        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            Xb, yb = X_train[idx], y_train[idx]
            if AUGMENT:
                Xb = augment(Xb, rng)

            z1, a1, probs = forward(params, Xb)
            total_loss += cross_entropy(probs, yb)
            grads = backward(params, Xb, yb, z1, a1, probs)
            opt.step(params, grads, lr)
            batches += 1

        val_acc = accuracy(params, X_val, y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_params = {k: v.copy() for k, v in params.items()}
        print(f"epoch {epoch:2d}/{EPOCHS}  loss: {total_loss / batches:.4f}  "
              f"val_acc: {val_acc:.4f}  lr: {lr:.5f}  ({time.time() - start_time:.1f}s)")
        lr *= LR_DECAY

    print(f"best val_acc: {best_val:.4f}")
    test_acc = accuracy(best_params, X_test, y_test)
    print(f"test accuracy (best epoch): {test_acc:.4f}")

    np.savez(MODEL_PATH, **best_params)
    print(f"model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
