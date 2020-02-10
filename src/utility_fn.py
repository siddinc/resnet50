import os
import matplotlib.pyplot as plt


def mkdir(p):
    if not os.path.exists(p):
        os.mkdir(p)


def link(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst)


def plot_metrics(r):
    plt.subplot(121)
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(r.history['accuracy'], label='acc')
    plt.plot(r.history['val_accuracy'], label='val_acc')
    plt.legend()

    plt.show()
