import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from copy import deepcopy

def show_map(matrix):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix,cmap=cm.Greys)
    ax.set_title("Spatial distribution of crops")
    fig.tight_layout()
    plt.show()

# plantdist: 1 = plant in every row/column, 2= one distance between them, etc.
def withrows(matrix,plantdist,rowdist,show=True):
    heatmatrix=deepcopy(matrix)
    for rows in range(0,100,rowdist):
        for columns in range(0,100,plantdist):
            if heatmatrix[rows, columns] == 0.0:
                heatmatrix[rows,columns]=1.0


    if show:
        fig, ax = plt.subplots()
        colormap = cm.Greens
        colormap.set_bad(color='black')
        im = ax.imshow(heatmatrix, colormap, vmin=0, vmax=1)
        ax.set_title("Spatial distribution of crops (with rows)")
        fig.tight_layout()
        plt.show()

    return heatmatrix

def norows(matrix,density,show=True):
    heatmatrix=deepcopy(matrix)
    for rows in range(0,100,density):
        for columns in range(0,100,density):
            if heatmatrix[rows, columns] == 0.0:
                heatmatrix[rows,columns]=1.0


    if show:
        fig, ax = plt.subplots()
        colormap = cm.Greens
        colormap.set_bad(color='black')
        im = ax.imshow(heatmatrix, colormap, vmin=0, vmax=1)
        ax.set_title("Spatial distribution of crops (no rows)")
        fig.tight_layout()
        plt.show()

    return heatmatrix
