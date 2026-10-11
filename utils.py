import torch
import matplotlib.pyplot as plt

def plot_curves(curves, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(curves["loss_train"], label="train"); a1.plot(curves["loss_test"], label="test")
    a1.set_title("loss"); a1.legend()
    a2.plot(curves["acc_train"], label="train"); a2.plot(curves["acc_test"], label="test")
    a2.set_title("accuracy"); a2.legend()
    fig.suptitle(title)
