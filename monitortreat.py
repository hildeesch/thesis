import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt

def showpath(matrix, path):
    fig, ax = plt.subplots()
    colormap = cm.Oranges
    colormap.set_bad(color='black')
    im = ax.imshow(matrix,colormap,vmin=0, vmax=1)
    for step in path:
        ax.add_patch(mpl.patches.Rectangle((step[0]-0.5,step[1]-0.5), 1, 1, hatch='///////', fill=False, snap=False,edgecolor='r'))
    ax.set_title("Planned path")
    fig.tight_layout()
    plt.show()
