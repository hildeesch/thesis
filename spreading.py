import random

import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable

def weedsspread(matrixstructure,matrixplants,show=True):
    weedmatrix=deepcopy(matrixstructure)
    patchnr=3
    patchsize=10
    spreadrange=3
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
                                if plantattach and matrixplants[row+rowadd,col+coladd]>0.0:
                                    weedmatrix[row + rowadd, col + coladd] = 1.0
                                elif plantattach and matrixplants[row+rowadd,col+coladd]==0.0:
                                    weedmatrix[row + rowadd, col + coladd] = 0.5
                                elif not plantattach and matrixplants[row+rowadd,col+coladd]>0.0:
                                    weedmatrix[row + rowadd, col + coladd] = 1.0
                                elif not plantattach and matrixplants[row+rowadd,col+coladd]==0.0:
                                    weedmatrix[row + rowadd, col + coladd] = 0.5

    uncertaintyweeds(matrixstructure,matrixplants,weedmatrix,spreadrange,plantattach)
    if show:
        fig, ax = plt.subplots()
        colormap = cm.Oranges
        colormap.set_bad(color='black')
        im = ax.imshow(weedmatrix, colormap, vmin=0, vmax=1)
        ax.set_title("Spatial distribution of weeds")
        fig.tight_layout()
        plt.show()

    return weedmatrix

def uncertaintyweeds(matrixstructure,matrixplants,matrixweeds,spreadrange,plantattach,show=True):
    uncertaintymatrix = deepcopy(matrixstructure)
    for row in range(100):
        for col in range(100):
            for dist in range(spreadrange):
                if row-dist>-1 and row+dist<100 and col-dist>-1 and col+dist<100: #contains mistake due to taking both the col and row
                    if matrixweeds[row+dist,col]>0.0 or matrixweeds[row-dist,col]>0.0 or matrixweeds[row,col+dist]>0.0 or matrixweeds[row,col-dist]>0.0:
                        if not np.isnan(matrixstructure[row,col]) and matrixweeds[row,col]==0.0:
                            if plantattach and matrixplants[row, col] > 0.0:
                                uncertaintymatrix[row,col]=1.0
                            elif plantattach and matrixplants[row, col] == 0.0:
                                uncertaintymatrix[row,col]=0.5
                            elif not plantattach and matrixplants[row, col] > 0.0:
                                uncertaintymatrix[row,col]=1.0
                            elif not plantattach and matrixplants[row, col] == 0.0:
                                uncertaintymatrix[row,col]=0.5

    if show:
        fig, ax = plt.subplots()
        colormap = cm.Blues
        colormap.set_bad(color='black')
        im = ax.imshow(uncertaintymatrix, colormap, vmin=0, vmax=1)
        ax.set_title("Spatial distribution of uncertainty")
        fig.tight_layout()
        plt.show()

    return uncertaintymatrix

def pathogenspread(matrixstructure,matrixplants,show=True):
    pathogenmatrix=deepcopy(matrixstructure)
    patchnr=3
    infectionduration=5 #days since the start of the infection
    spreadrange=3
    spreadspeed=1
    reproductionrate=0.2 #should be between 0 and 1 (fraction of the point that reproduces)
    saturation=3
    plantattach=True #always true for pathogen

    patches=0
    while patches!=patchnr:
        row=random.randrange(100)
        col=random.randrange(100)
        if not np.isnan(pathogenmatrix[row,col]):
            patches=patches+1
            pathogenmatrix[row,col]=1.0 #centre of the patch
            curspread=random.randrange(infectionduration) #the current patch duration is at most the duration
            for day in range(curspread):
                for row in range(100):
                    for col in range(100):
                        for dist in range(spreadrange):
                            if row - dist > -1 and row + dist < 100 and col - dist > -1 and col + dist < 100:  # contains mistake due to taking both the col and row
                                if pathogenmatrix[row + dist, col] > 0.0 or pathogenmatrix[row - dist, col] > 0.0 or \
                                        pathogenmatrix[row, col + dist] > 0.0 or pathogenmatrix[row, col - dist] > 0.0:
                                    if not np.isnan(matrixstructure[row, col]):
                                        maxpathogenvalue = np.nanmax([pathogenmatrix[row + dist, col],
                                                               pathogenmatrix[row - dist, col],
                                                               pathogenmatrix[row, col + dist],
                                                               pathogenmatrix[row, col - dist]])
                                        if matrixplants[row, col] > 0.0:
                                            pathogenmatrix[row, col] = min(saturation,pathogenmatrix[row, col]*(1+reproductionrate)+ reproductionrate * maxpathogenvalue * (
                                                        1 / spreadspeed))
                                        elif matrixplants[row, col] == 0.0:
                                            pathogenmatrix[row, col] = min(saturation,pathogenmatrix[row, col]*(1+reproductionrate)+reproductionrate * maxpathogenvalue * (
                                                        1 / spreadspeed) * 0.5)
    uncertaintypathogen(matrixstructure,matrixplants,pathogenmatrix,spreadrange,spreadspeed,reproductionrate,saturation)


    if show:
        fig, ax = plt.subplots()
        colormap = cm.Oranges
        colormap.set_bad(color='black')
        im = ax.imshow(pathogenmatrix, colormap, vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        ax.set_title("Spatial distribution of pathogen")
        fig.tight_layout()
        plt.show()

    return pathogenmatrix

def uncertaintypathogen(matrixstructure,matrixplants,matrixpathogen,spreadrange,spreadspeed,reproductionrate,saturation,show=True):
    uncertaintymatrix = deepcopy(matrixstructure)
    for row in range(100):
        for col in range(100):
            for dist in range(spreadrange):
                if row-dist>-1 and row+dist<100 and col-dist>-1 and col+dist<100: #contains mistake due to taking both the col and row
                    if matrixpathogen[row+dist,col]>0.0 or matrixpathogen[row-dist,col]>0.0 or matrixpathogen[row,col+dist]>0.0 or matrixpathogen[row,col-dist]>0.0:
                        if not np.isnan(matrixstructure[row,col]) and matrixpathogen[row,col]<saturation:
                            maxpathogenvalue = np.nanmax([matrixpathogen[row + dist, col],
                                                              matrixpathogen[row - dist, col],
                                                              matrixpathogen[row, col + dist],
                                                              matrixpathogen[row, col - dist]])

                            if matrixplants[row, col] > 0.0:
                                uncertaintymatrix[row,col]=uncertaintymatrix[row,col]+reproductionrate*maxpathogenvalue*(1/spreadspeed)
                            elif matrixplants[row, col] == 0.0:
                                uncertaintymatrix[row,col]=uncertaintymatrix[row,col]+reproductionrate*maxpathogenvalue*(1/spreadspeed)*0.5

    if show:
        fig, ax = plt.subplots()
        colormap = cm.Blues
        colormap.set_bad(color='black')
        im= ax.imshow(uncertaintymatrix, colormap, vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)

        ax.set_title("Spatial distribution of uncertainty")
        fig.tight_layout()
        plt.show()

    return uncertaintymatrix