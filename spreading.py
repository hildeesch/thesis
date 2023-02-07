import random

import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable

from weed import Weed
from pathogen import Pathogen

def weedsspread(matrixstructure,matrixplants,weed,show=True):
    weedmatrix=deepcopy(matrixstructure)
    patchnr = weed.patchnr
    patchsize = weed.patchsize
    spreadrange = weed.spreadrange
    spreadspeed = weed.spreadspeed
    plantattach = weed.plantattach

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

    uncertaintymatrix = uncertaintyweeds(matrixstructure,matrixplants,weedmatrix,weed, True)
    if show:
        fig, ax = plt.subplots()
        colormap = cm.Oranges
        colormap.set_bad(color='black')
        im = ax.imshow(weedmatrix, colormap, vmin=0, vmax=1, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        ax.set_title("Spatial distribution of weeds")
        fig.tight_layout()
        plt.show()

    return weedmatrix, uncertaintymatrix

def uncertaintyweeds(matrixstructure,matrixplants,matrixweeds,weed,show=True):
    spreadspeed = weed.spreadspeed
    spreadrange = weed.spreadrange
    plantattach = weed.plantattach
    uncertaintymatrix = deepcopy(matrixstructure)
    for row in range(100):
        for col in range(100):
            for rowdist in range(spreadrange+1):
                for coldist in range(spreadrange+1):
                    # options: straight above, straight below, straight left, straight right, and 4 diagonals between these
                    within_radius=False
                    if np.sqrt(coldist ** 2 + rowdist ** 2) <= spreadrange:  # to enforce a radius
                        within_radius=True
                    matrixweeds_around = []
                    # the whole part with if-s is just to check whether the elements within the spreadrange are still within the field
                    row_min, row_max, col_min, col_max = 0, 0, 0, 0
                    if row - rowdist < 0:
                        row_min = 1
                    if row + rowdist > 99:
                        row_max = 1
                    if col - coldist < 0:
                        col_min = 1
                    if col + coldist > 99:
                        col_max = 1
                    if row_min == 0:
                        matrixweeds_around.append(
                            matrixweeds[row - rowdist, col])  # left
                    if row_max == 0:
                        matrixweeds_around.append(
                            matrixweeds[row + rowdist, col])  # right
                    if col_max == 0:
                        matrixweeds_around.append(
                            matrixweeds[row, col + coldist])  # up
                    if col_min == 0:
                        matrixweeds_around.append(
                            matrixweeds[row, col - coldist])  # down
                    if row_min == 0 and col_max == 0 and within_radius:
                        matrixweeds_around.append(
                            matrixweeds[row - rowdist, col + coldist])  # left up
                    if row_min == 0 and col_min == 0 and within_radius:
                        matrixweeds_around.append(
                            matrixweeds[row - rowdist, col - coldist])  # left down
                    if row_max == 0 and col_max == 0 and within_radius:
                        matrixweeds_around.append(
                            matrixweeds[row + rowdist, col + coldist])  # right up
                    if row_max == 0 and col_min == 0 and within_radius:
                        matrixweeds_around.append(
                            matrixweeds[row + rowdist, col - coldist])  # right down

                    maxweedvalue = np.nanmax(matrixweeds_around)

                    if maxweedvalue>0.0 and not np.isnan(maxweedvalue):
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
        im = ax.imshow(uncertaintymatrix, colormap, vmin=0, vmax=1, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        ax.set_title("Spatial distribution of uncertainty")
        fig.tight_layout()
        plt.show()

    return uncertaintymatrix

def pathogenspread(matrixstructure,matrixplants,pathogen,show=True):
    pathogenmatrix=deepcopy(matrixstructure)
    patchnr = pathogen.patchnr
    infectionduration = pathogen.infectionduration
    spreadrange = pathogen.spreadrange
    spreadspeed = pathogen.spreadspeed
    reproductionrate= pathogen.reproductionrate
    saturation = pathogen.saturation
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
                pathogenmatrix_new = deepcopy(pathogenmatrix)
                for row in range(100):
                    for col in range(100):
                        for rowdist in range(spreadrange+1):
                            for coldist in range(spreadrange+1):
                                #options: straight above, straight below, straight left, straight right, and 4 diagonals between these
                                matrixpathogen_around = []
                                within_radius = False
                                if np.sqrt(coldist ** 2 + rowdist ** 2) <= spreadrange:  # to enforce a radius
                                    within_radius = True
                                matrixpathogen_around = []
                                # the whole part with if-s is just to check whether the elements within the spreadrange are still within the field
                                row_min, row_max, col_min, col_max = 0, 0, 0, 0
                                if row-rowdist<0:
                                    row_min = 1
                                if row+rowdist>99:
                                    row_max=1
                                if col-coldist<0:
                                    col_min = 1
                                if col+coldist>99:
                                    col_max = 1
                                if row_min==0:
                                    matrixpathogen_around.append(pathogenmatrix[row-rowdist,col]) #left
                                if row_max==0:
                                    matrixpathogen_around.append(pathogenmatrix[row+rowdist,col]) #right
                                if col_max==0:
                                    matrixpathogen_around.append(pathogenmatrix[row,col+coldist]) #up
                                if col_min==0:
                                    matrixpathogen_around.append(pathogenmatrix[row,col-coldist]) #down
                                if row_min==0 and col_max==0 and within_radius:
                                    matrixpathogen_around.append(pathogenmatrix[row-rowdist,col+coldist]) #left up
                                if row_min==0 and col_min==0 and within_radius:
                                    matrixpathogen_around.append(pathogenmatrix[row-rowdist,col-coldist]) #left down
                                if row_max == 0 and col_max == 0 and within_radius:
                                    matrixpathogen_around.append(pathogenmatrix[row+rowdist,col+coldist]) #right up
                                if row_max == 0 and col_min == 0 and within_radius:
                                    matrixpathogen_around.append(pathogenmatrix[row+rowdist,col-coldist]) #right down

                                # the pathogen value is only increased if it is part of the field (not edge/obstacle) and it has not reached the max value yet
                                if not np.isnan(matrixstructure[row,col]) and pathogenmatrix[row,col]<saturation:
                                    maxpathogenvalue = np.nanmax(matrixpathogen_around)
                                    if maxpathogenvalue>0.0 and not np.isnan(maxpathogenvalue):
                                        if matrixplants[row, col] > 0.0:
                                            pathogenmatrix_new[row, col] = min(saturation,pathogenmatrix[row, col]*(1+reproductionrate)+ reproductionrate * maxpathogenvalue * (
                                                        1 / spreadspeed))
                                        elif matrixplants[row, col] == 0.0:
                                            pathogenmatrix_new[row, col] = min(saturation,pathogenmatrix[row, col]*(1+reproductionrate)+reproductionrate * maxpathogenvalue * (
                                                        1 / spreadspeed) * 0.5)
                pathogenmatrix=deepcopy(pathogenmatrix_new)
    uncertaintymatrix= uncertaintypathogen(matrixstructure,matrixplants,pathogenmatrix,pathogen,True)


    if show:
        fig, ax = plt.subplots()
        colormap = cm.Oranges
        colormap.set_bad(color='black')
        im = ax.imshow(pathogenmatrix, colormap, vmin=0, vmax=1, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        ax.set_title("Spatial distribution of pathogen")
        fig.tight_layout()
        plt.show()

    return pathogenmatrix, uncertaintymatrix

def uncertaintypathogen(matrixstructure,matrixplants,matrixpathogen,pathogen,show=True):
    spreadrange = pathogen.spreadrange
    spreadspeed = pathogen.spreadspeed
    reproductionrate = pathogen.reproductionrate
    saturation = pathogen.saturation
    uncertaintymatrix = deepcopy(matrixstructure)
    for row in range(100):
        for col in range(100):
            for rowdist in range(spreadrange+1):
                for coldist in range(spreadrange+1):
                    #options: straight above, straight below, straight left, straight right, and 4 diagonals between these
                    matrixpathogen_around = []
                    within_radius = False
                    if np.sqrt(coldist ** 2 + rowdist ** 2) <= spreadrange:  # to enforce a radius
                        within_radius = True
                    # the whole part with if-s is just to check whether the elements within the spreadrange are still within the field
                    row_min, row_max, col_min, col_max = 0, 0, 0, 0
                    if row-rowdist<0:
                        row_min = 1
                    if row+rowdist>99:
                        row_max=1
                    if col-coldist<0:
                        col_min = 1
                    if col+coldist>99:
                        col_max = 1
                    if row_min==0:
                        matrixpathogen_around.append(matrixpathogen[row-rowdist,col]) #left
                    if row_max==0:
                        matrixpathogen_around.append(matrixpathogen[row+rowdist,col]) #right
                    if col_max==0:
                        matrixpathogen_around.append(matrixpathogen[row,col+coldist]) #up
                    if col_min==0:
                        matrixpathogen_around.append(matrixpathogen[row,col-coldist]) #down
                    if row_min==0 and col_max==0 and within_radius:
                        matrixpathogen_around.append(matrixpathogen[row-rowdist,col+coldist]) #left up
                    if row_min==0 and col_min==0 and within_radius:
                        matrixpathogen_around.append(matrixpathogen[row-rowdist,col-coldist]) #left down
                    if row_max == 0 and col_max == 0 and within_radius:
                        matrixpathogen_around.append(matrixpathogen[row+rowdist,col+coldist]) #right up
                    if row_max == 0 and col_min == 0 and within_radius:
                        matrixpathogen_around.append(matrixpathogen[row+rowdist,col-coldist]) #right down

                    # the pathogen value is only increased if it is part of the field (not edge/obstacle) and it has not reached the max value yet
                    if not np.isnan(matrixstructure[row,col]) and matrixpathogen[row,col]<saturation:
                        maxpathogenvalue = np.nanmax(matrixpathogen_around)
                        if maxpathogenvalue>0.0 and not np.isnan(maxpathogenvalue):
                            if matrixplants[row, col] > 0.0:
                                uncertaintymatrix[row,col]=uncertaintymatrix[row,col]+reproductionrate*maxpathogenvalue*(1/spreadspeed)
                            elif matrixplants[row, col] == 0.0:
                                uncertaintymatrix[row,col]=uncertaintymatrix[row,col]+reproductionrate*maxpathogenvalue*(1/spreadspeed)*0.5

    if show:
        fig, ax = plt.subplots()
        colormap = cm.Blues
        colormap.set_bad(color='black')
        im= ax.imshow(uncertaintymatrix, colormap, vmin=0, vmax=1, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)

        ax.set_title("Spatial distribution of uncertainty")
        fig.tight_layout()
        plt.show()

    return uncertaintymatrix