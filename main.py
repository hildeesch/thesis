# This is a sample Python script.
from heatmap import show_map
from heatmap import withrows
from heatmap import norows
from monitortreat import showpath
from spreading import weedsspread
from spreading import pathogenspread
import numpy as np

from weed import Weed
from pathogen import Pathogen

from Planner.Sampling_based_Planning.rrt_2D.informed_rrt_star_rig_spread import main as rig


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    matrix_obstacle = np.zeros((100,100))
    matrix_obstacle[10:20,35:50] = np.nan

    matrix_square = np.zeros((100,100))

    matrix_convex = np.zeros((100,100))
    matrix_convex[:,0] = np.nan
    matrix_convex[0:35,1] = np.nan
    matrix_convex[53:100,1] = np.nan
    matrix_convex[0:23,2] = np.nan
    matrix_convex[60:100,2] = np.nan
    matrix_convex[0:20,3]= np.nan
    matrix_convex[65:100,3] = np.nan
    matrix_convex[0:15,4] = np.nan
    matrix_convex[72:100,4] = np.nan
    matrix_convex[0:9,5] = np.nan
    matrix_convex[81:100,5] = np.nan
    matrix_convex[0:3,6] = np.nan
    matrix_convex[92:100,6] = np.nan
    matrix_convex[95:100,7] = np.nan
    matrix_convex[:,-1] = np.nan
    matrix_convex[0:35,-2] = np.nan
    matrix_convex[53:100,-2] = np.nan
    matrix_convex[0:23,-3] = np.nan
    matrix_convex[60:100,-3] = np.nan
    matrix_convex[0:20,-4]= np.nan
    matrix_convex[65:100,-4] = np.nan
    matrix_convex[0:15,-5] = np.nan
    matrix_convex[72:100,-5] = np.nan
    matrix_convex[0:9,-6] = np.nan
    matrix_convex[81:100,-6] = np.nan
    matrix_convex[0:3,-7] = np.nan
    matrix_convex[92:100,-7] = np.nan
    matrix_convex[95:100,-8] = np.nan

    matrix_nonconvex = np.zeros((100,100))
    matrix_nonconvex[:,0] = np.nan
    matrix_nonconvex[2:88,1] = np.nan
    matrix_nonconvex[5:80,2] = np.nan
    matrix_nonconvex[12:76,3] = np.nan
    matrix_nonconvex[22:69,4] = np.nan
    matrix_nonconvex[29:61,5] = np.nan
    matrix_nonconvex[34:55,6] = np.nan

    matrix_nonconvex[:,-1] = np.nan
    matrix_nonconvex[:,-2] = np.nan
    matrix_nonconvex[:,-3] = np.nan
    matrix_nonconvex[2:88,-4] = np.nan
    matrix_nonconvex[5:80,-5] = np.nan
    matrix_nonconvex[12:76,-6] = np.nan
    matrix_nonconvex[22:69,-7] = np.nan
    matrix_nonconvex[29:61,-8] = np.nan
    matrix_nonconvex[34:55,-9] = np.nan

    #show_map(matrix_nonconvex)
    heat_matrix = withrows(matrix_convex,2,4,False)

    pathogen1 = Pathogen(patchnr=5,infectionduration=3,spreadrange=5, spreadspeed=1, reproductionrate=0.2, saturation=3)
    spread_matrix, uncertainty_matrix = pathogenspread(matrix_convex,heat_matrix,pathogen1, False)
    weed1 = Weed(patchnr=4,patchsize=7,spreadrange=3,spreadspeed=1,saturation=3,plantattach=False)
    #spread_matrix,uncertainty_matrix = weedsspread(matrix_convex,heat_matrix,weed1)
    path=[[0,0],[1,0],[2,0],[2,1],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],[2,8],[2,9],
    [2,10],[3,10],[4,10],[5,10],[6,11],[6,12],[7,12],[8,13],[9,14],[10,14],[11,15]]
    #showpath(spread_matrix,path)

    rig(uncertainty_matrix)

if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
