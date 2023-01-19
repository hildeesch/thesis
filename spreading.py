import random

import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from copy import deepcopy

def weedsspread(matrixstructure,matrixplants,show=True):
    weedmatrix=deepcopy(matrixstructure)
    patchnr=3
    patchsize=10
    spreadrange=2
    spreadspeed=1
    plantattach=True
    patches=0
    while patches!=patchnr:
        row=random.randrange(100)
        col=random.randrange(100)
        if not np.isnan(weedmatrix[row,col]):
            patches=patches+1
            weedmatrix[row,col]=1.0 #centre of the patch
            curspread=random.randrange(patchsize) #the current size is at most the patchsize
            for rowcount in range((curspread*2)+1):
                for colcount in range((curspread*2)+1):
                    rowadd=rowcount-curspread
                    coladd=colcount-curspread
                    if row+rowadd<100 and row+rowadd>-1 and col+coladd<100 and col+coladd>-1: #within bounds of grid
                        if np.sqrt(rowadd**2+coladd**2)<=curspread: #to create round patches
                            if not np.isnan(weedmatrix[row+rowadd,col+coladd]):
                                weedmatrix[row + rowadd, col + coladd] = 1.0


    # for rows in range(0,100):
    #     for columns in range(0,100):
    #         if matrixplants[rows, columns] == 1.0:
    #             matrixstructure[rows,columns]=1.0 #so if there is a plant, there is a weed (at this point)
    #

    uncertaintymatrix(matrixstructure,weedmatrix,spreadrange)
    if show:
        fig, ax = plt.subplots()
        colormap = cm.Oranges
        colormap.set_bad(color='black')
        im = ax.imshow(weedmatrix, colormap, vmin=0, vmax=1)
        ax.set_title("Spatial distribution of weeds")
        fig.tight_layout()
        plt.show()

    return weedmatrix

def uncertaintymatrix(matrixstructure,matrixweeds,spreadrange,show=True):
    uncertaintymatrix = deepcopy(matrixstructure)
    for row in range(100):
        for col in range(100):
            for dist in range(spreadrange):
                if row-dist>-1 and row+dist<100 and col-dist>-1 and col+dist<100: #contains mistake due to taking both the col and row
                    if matrixweeds[row+dist,col]>0.0 or matrixweeds[row-dist,col]>0.0 or matrixweeds[row,col+dist]>0.0 or matrixweeds[row,col-dist]>0.0:
                        if not np.isnan(matrixstructure[row,col]) and matrixweeds[row,col]==0.0:
                            uncertaintymatrix[row,col]=1.0

    if show:
        fig, ax = plt.subplots()
        colormap = cm.Blues
        colormap.set_bad(color='black')
        im = ax.imshow(uncertaintymatrix, colormap, vmin=0, vmax=1)
        ax.set_title("Spatial distribution of uncertainty")
        fig.tight_layout()
        plt.show()

    return uncertaintymatrix