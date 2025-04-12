
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps


def plot_s(b, sub_title=['', '', '', ''], size=40, vmin=None, vmax=None):
    k = len(b)
    fig, axs = plt.subplots(1, k, figsize=(6*k, 4))
    for i in range(k):
        if i == 0:
            ax = axs[i].imshow(b[i].reshape(size, size),
                               cmap=colormaps['viridis'], vmin=vmin, vmax=vmax)
        else:  # plt.cm.get_cmap('viridis', 21)
            ax = axs[i].imshow(b[i].reshape(size, size),
                               cmap=colormaps['viridis'], vmin=vmin, vmax=vmax)
        axs[i].set_title(sub_title[i], fontsize=16)
        fig.colorbar(ax, ax=axs[i])

        axs[i].set_xticks(np.arange(-0.5, 40, 5))
        axs[i].set_yticks(np.arange(-0.5, 40, 5))
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])

        axs[i].tick_params(axis='x', colors=(0, 0, 0, 0))
        axs[i].tick_params(axis='y', colors=(0, 0, 0, 0))
    plt.show()
