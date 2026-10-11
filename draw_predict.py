"""Draw a digit with the mouse and let the trained network predict it.

Improvements over the first version:
  * strokes are connected (no dotted lines on fast mouse movement)
  * smooth area-averaged resizing (soft gray edges like real MNIST)
  * digit centered by center of mass, as in the original MNIST preprocessing
  * live prediction when you lift the mouse + bar chart of all 10 probabilities
  * model path is relative to this file, so it runs from any folder
"""
import pickle
import tkinter as tk
from pathlib import Path

import numpy as np

CANVAS_SIZE = 280        # 10x the 28x28 MNIST resolution
BRUSH_RADIUS = 10
BASE_DIR = Path(__file__).resolve().parent


# ---- model loading ----
def load_model():
    npz_path = BASE_DIR / "model.npz"
    pkl_path = BASE_DIR / "model.pkl"
    if npz_path.exists():
        data = np.load(npz_path)
        return {k: data[k] for k in ("W1", "b1", "W2", "b2")}
    if pkl_path.exists():  # fallback for the old format
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError("No model.npz or model.pkl found. Run mnist_nueral_network.py first.")


MODEL = load_model()
W1, b1, W2, b2 = MODEL["W1"], MODEL["b1"], MODEL["W2"], MODEL["b2"]


# ---- inference ----
def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def forward(X):
    return softmax(relu(X @ W1 + b1) @ W2 + b2)


# ---- preprocessing (mimics how MNIST was built) ----
def resize_area(img, out_h, out_w):
    """Box-filter resize: every output pixel is the mean of the input block it covers."""
    in_h, in_w = img.shape
    sat = np.pad(img, ((1, 0), (1, 0))).cumsum(0).cumsum(1)   # summed-area table

    ys = np.linspace(0, in_h, out_h + 1).round().astype(int)
    xs = np.linspace(0, in_w, out_w + 1).round().astype(int)
    y0 = np.minimum(ys[:-1], in_h - 1)                         # each block at least 1px
    x0 = np.minimum(xs[:-1], in_w - 1)
    y1 = np.minimum(np.maximum(ys[1:], y0 + 1), in_h)
    x1 = np.minimum(np.maximum(xs[1:], x0 + 1), in_w)

    total = (sat[np.ix_(y1, x1)] - sat[np.ix_(y0, x1)]
             - sat[np.ix_(y1, x0)] + sat[np.ix_(y0, x0)])
    area = np.outer(y1 - y0, x1 - x0)
    return (total / area).astype(np.float32)


def shift_image(img, dy, dx):
    """Shift without wrap-around (pads with zeros)."""
    pad = 10
    padded = np.pad(img, pad)
    return padded[pad - dy:pad - dy + 28, pad - dx:pad - dx + 28]


def to_mnist_format(grid, threshold=0.3):
    """Crop to the ink, fit into a 20x20 box (keeping aspect ratio), place in 28x28,
    then shift so the center of mass sits at the middle of the frame."""
    ys, xs = np.nonzero(grid > threshold)
    if ys.size == 0:
        return None

    cropped = grid[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = cropped.shape
    if h > w:
        new_h, new_w = 20, max(1, round(w * 20 / h))
    else:
        new_w, new_h = 20, max(1, round(h * 20 / w))
    resized = resize_area(cropped, new_h, new_w)

    out = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    out[top:top + new_h, left:left + new_w] = resized

    total = out.sum()
    if total > 0:
        cy = (out.sum(axis=1) * np.arange(28)).sum() / total
        cx = (out.sum(axis=0) * np.arange(28)).sum() / total
        out = shift_image(out, int(round(13.5 - cy)), int(round(13.5 - cx)))
    return out


# ---- GUI ----
class App:
    def __init__(self, root):
        self.root = root
        root.title("Draw a digit")
        self.grid = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
        self.last = None

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=2, padx=(5, 0), pady=5)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.release)

        self.bars = tk.Canvas(root, width=200, height=CANVAS_SIZE, bg="#f0f0f0", highlightthickness=0)
        self.bars.grid(row=0, column=2, padx=5, pady=5)
        self.draw_bars(np.zeros(10))

        self.result_label = tk.Label(root, text="Draw a digit", font=("Helvetica", 16))
        self.result_label.grid(row=1, column=0, columnspan=3, pady=8)

        tk.Button(root, text="Predict", command=self.predict, font=("Helvetica", 12)).grid(
            row=2, column=0, sticky="ew", padx=5, pady=5)
        tk.Button(root, text="Clear", command=self.clear, font=("Helvetica", 12)).grid(
            row=2, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

    def stamp(self, x, y):
        r = BRUSH_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        y0, y1 = max(0, y - r), min(CANVAS_SIZE, y + r)
        x0, x1 = max(0, x - r), min(CANVAS_SIZE, x + r)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        self.grid[y0:y1, x0:x1][mask] = 1.0

    def paint(self, event):
        x, y = event.x, event.y
        if self.last is not None:                       # connect to the previous point
            lx, ly = self.last
            steps = max(abs(x - lx), abs(y - ly)) // 3 + 1
            for t in np.linspace(0, 1, steps + 1):
                self.stamp(int(lx + (x - lx) * t), int(ly + (y - ly) * t))
        else:
            self.stamp(x, y)
        self.last = (x, y)

    def release(self, _event):
        self.last = None
        self.predict()                                   # live prediction

    def clear(self):
        self.canvas.delete("all")
        self.grid[:] = 0.0
        self.last = None
        self.result_label.config(text="Draw a digit")
        self.draw_bars(np.zeros(10))

    def draw_bars(self, probs, best=None):
        self.bars.delete("all")
        for d in range(10):
            y = 10 + d * 27
            self.bars.create_text(10, y + 10, text=str(d), anchor="w", font=("Helvetica", 11))
            color = "#2a7de1" if d == best else "#9db8d9"
            self.bars.create_rectangle(30, y + 2, 30 + probs[d] * 130, y + 18, fill=color, outline="")
            self.bars.create_text(32 + probs[d] * 130, y + 10, text=f"{probs[d] * 100:.0f}%",
                                  anchor="w", font=("Helvetica", 9))

    def predict(self):
        small = to_mnist_format(self.grid)
        if small is None:
            self.result_label.config(text="Draw something first!")
            return
        probs = forward(small.reshape(1, 784))[0]
        digit = int(np.argmax(probs))
        self.result_label.config(text=f"Prediction: {digit}  ({probs[digit] * 100:.1f}% confident)")
        self.draw_bars(probs, best=digit)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
