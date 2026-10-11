import os

import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------- Data ----------

def load_data(name):
    """Load "circles" or "mnist" as float tensors (Xtrain, ytrain, Xtest, ytest)"""
    d = loadmat(os.path.join(DATA_DIR, f"{name}.mat"))
    Xtrain = torch.from_numpy(d["Xtrain"]).float()
    Xtest = torch.from_numpy(d["Xtest"]).float()
    ytrain = torch.from_numpy(d["Ytrain"]).float()
    ytest = torch.from_numpy(d["Ytest"]).float()
    # scale MNIST pixels between [0, 1]
    if name == "mnist":
        Xtrain, Xtest = Xtrain / 255, Xtest / 255
    return Xtrain, ytrain, Xtest, ytest


def set_seed(seed=0):
    torch.manual_seed(seed)


# ---------- Training ----------

def evaluate(y_hat, y):
    """Cross-entropy loss and accuracy"""
    L = -torch.mean(torch.sum(y * torch.log(y_hat + 1e-12), dim=1))
    acc = torch.mean((y_hat.argmax(1) == y.argmax(1)).float())
    return L.item(), acc.item()


def train(step, predict, data, Nbatch, Nepoch, verbose=True):
    """Generic mini-batch training loop
    step(Xb, yb): does forward + backward + parameter update on one batch
    predict(X): returns the probabilities y_hat (N, ny)
    data: (Xtrain, ytrain, Xtest, ytest) as returned by load_data"""
    Xtrain, ytrain, Xtest, ytest = data
    N = Xtrain.shape[0]
    curves = {"loss_train": [], "loss_test": [], "acc_train": [], "acc_test": []}

    for epoch in range(Nepoch):
        perm = torch.randperm(N)
        Xs, ys = Xtrain[perm], ytrain[perm]
        for j in range(N // Nbatch):
            step(Xs[j * Nbatch:(j + 1) * Nbatch], ys[j * Nbatch:(j + 1) * Nbatch])

        with torch.no_grad():
            Ltr, atr = evaluate(predict(Xtrain), ytrain)
            Lte, ate = evaluate(predict(Xtest), ytest)
        for k, v in zip(curves, [Ltr, Lte, atr, ate]):
            curves[k].append(v)
        if verbose:
            print(f"epoch {epoch:3d} | train loss {Ltr:.4f} acc {atr:.3f} | test loss {Lte:.4f} acc {ate:.3f}")

    return curves


# ---------- Plots ----------

def plot_curves(curves, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(curves["loss_train"], label="train"); a1.plot(curves["loss_test"], label="test")
    a1.set_title("loss"); a1.legend()
    a2.plot(curves["acc_train"], label="train"); a2.plot(curves["acc_test"], label="test")
    a2.set_title("accuracy"); a2.legend()
    fig.suptitle(title)
    plt.show()


def show_mnist(X, y, predict=None, n=16):
    """Show the first n MNIST images with their label"""
    pred = None
    if predict is not None:
        with torch.no_grad():
            pred = predict(X[:n]).argmax(1)
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        label = y[i].argmax().item()
        if pred is None:
            ax.set_title(str(label), fontsize=9)
        else:
            ok = pred[i].item() == label
            ax.set_title(f"{pred[i].item()} ({label})", fontsize=9, color="green" if ok else "red")
    plt.tight_layout()
    plt.show()


def show_mnist_grid(X, y, predict=None, per_class=15):
    """One row per digit with random examples (misclassified digits are shown in red)"""
    labels = y.argmax(1)
    fig, axes = plt.subplots(10, per_class, figsize=(per_class * 0.6, 10 * 0.6))
    for c in range(10):
        idx = torch.nonzero(labels == c).flatten()
        idx = idx[torch.randperm(len(idx))[:per_class]]
        pred = None
        if predict is not None:
            with torch.no_grad():
                pred = predict(X[idx]).argmax(1)
        for j, ax in enumerate(axes[c]):
            ax.axis("off")
            if pred is not None and pred[j] != c:
                ax.imshow(X[idx[j]].reshape(28, 28), cmap="Reds")
                ax.set_title(str(pred[j].item()), fontsize=7, color="red", pad=1)
            else:
                ax.imshow(X[idx[j]].reshape(28, 28), cmap="gray_r")
    plt.tight_layout()
    plt.show()
