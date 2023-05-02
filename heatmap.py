import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from copy import deepcopy
from shapely.geometry import Point, Polygon


def show_map(matrix):
    fig, ax = plt.subplots()
    colormap = cm.Greys
    colormap.set_bad(color='black')
    im = ax.imshow(matrix,cmap=colormap, vmin=0, vmax=1, origin='lower')

    ax.set_title("Spatial distribution of crops")
    fig.tight_layout()
    plt.show()

# plantdist: 1 = plant in every row/column, 2= one distance between them, etc.
def withrows(matrix,plantdist,rowdist,show=True):
    heatmatrix=deepcopy(matrix)
    heatmatrix[heatmatrix==0.5]=0
    edgematrix=deepcopy(heatmatrix)
    row_nrs = [] # to keep track of where the rows are
    row_edges = [] # to keep track of the left and right edges of each row
    for rows in range(0,100,rowdist):
        for columns in range(0,100,plantdist):
            if matrix[rows, columns] == 0.0:
                heatmatrix[rows,columns]=1.0
                if rows not in row_nrs:
                    row_nrs.append(rows)
    for index, rows in enumerate(row_nrs):
        left=None
        for columns in range(0,100):
            if matrix[rows,columns]==0.5:
                if left ==None:
                    left=columns
                else:
                    right=columns
                    #break;
        edgematrix[rows,left]=0.5
        edgematrix[rows,right]=0.5
        row_edges.append([left,right])
    print(row_nrs)
    #print(row_edges)
    if show:
        fig, ax = plt.subplots()
        colormap = cm.Greens
        colormap.set_bad(color='black')
        im = ax.imshow(heatmatrix, colormap, vmin=0, vmax=1, origin='lower')
        ax.set_title("Spatial distribution of crops (with rows)")
        fig.tight_layout()
        plt.show()

        fig, ax = plt.subplots()
        colormap = cm.Reds
        colormap.set_bad(color='black')
        im = ax.imshow(edgematrix, colormap, vmin=0, vmax=1, origin='lower')
        ax.set_title("Edge points")
        fig.tight_layout()
        plt.show()

    return heatmatrix, row_nrs,row_edges

def norows(matrix,density,show=True):
    heatmatrix=deepcopy(matrix)
    heatmatrix[heatmatrix==0.5]=0
    for rows in range(0,100,density):
        for columns in range(0,100,density):
            if heatmatrix[rows, columns] == 0.0:
                heatmatrix[rows,columns]=1.0


    if show:
        fig, ax = plt.subplots()
        colormap = cm.Greens
        colormap.set_bad(color='black')
        im = ax.imshow(heatmatrix, colormap, vmin=0, vmax=1, origin='lower')
        ax.set_title("Spatial distribution of crops (no rows)")
        fig.tight_layout()
        plt.show()

    return heatmatrix

def polygon(shape,show=True):
    if shape == "rectangle":
        # Create Point objects
        p1 = (0, 0)
        p2 = (0,99)
        p3 = (99, 99)
        p4 = (99,0)

        # Create a Polygon
        coords = [p1,p2,p3,p4]

    if shape == "hexagon":
        # Create Point objects
        # convex:
        # p1 = (20, 0)
        # p2 = (0,50)
        # p3 = (20,99)
        # p4 = (79, 90)
        # p5 = (99,50)
        # p6 = (79,0)

        # convex small
        p1 = (40, 30)
        p2 = (20,50)
        p3 = (40,79)
        p4 = (69, 70)
        p5 = (79,50)
        p6 = (59,30)

        # concave:
        # p1 = (20, 0)
        # p2 = (40,50)
        # p3 = (20,99)
        # p4 = (79, 90)
        # p5 = (69,50)
        # p6 = (79,0)

        # concave small
        # p1 = (20, 25)
        # p2 = (40,50)
        # p3 = (20,79)
        # p4 = (79, 70)
        # p5 = (69,50)
        # p6 = (79,25)

        # Create a Polygon
        coords = [p1,p2,p3,p4,p5,p6]
    poly = Polygon(coords)

    matrix = np.zeros((100,100))
    for row in range(100):
        for column in range(100):
            newpoint=Point(column,row)
            if not newpoint.within(poly):
                matrix[row,column] = np.nan
            if newpoint.buffer(1).intersects(poly) and not newpoint.within(poly):
                matrix[row,column] = 0.5
    #matrix[30:40,45:60] = np.nan

    if show:
        show_map(matrix)
    return matrix