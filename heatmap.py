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

# density: 1 = plant in every row/column, 2= one distance, etc.
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
# vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
#               "potato", "wheat", "barley"]
# farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
#            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
#
# harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
#                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
#                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
#                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
#                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
#                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
#                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
#
#
# fig, ax = plt.subplots()
# im = ax.imshow(harvest)
#
# # Show all ticks and label them with the respective list entries
# ax.set_xticks(np.arange(len(farmers)), labels=farmers)
# ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)
#
# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")
#
# # Loop over data dimensions and create text annotations.
# for i in range(len(vegetables)):
#     for j in range(len(farmers)):
#         text = ax.text(j, i, harvest[i, j],
#                        ha="center", va="center", color="w")
#
# ax.set_title("Harvest of local farmers (in tons/year)")
# fig.tight_layout()
# plt.show()