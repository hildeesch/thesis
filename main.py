# This is a sample Python script.
from heatmap import show_map
from heatmap import withrows
from heatmap import norows
from monitortreat import showpath
from spreading import weedsspread
from spreading import pathogenspread
import numpy as np

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    matrix1 = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                        [2.4, 0.0, 4.0, 2.0, 2.7, 0.0, 0.0],
                        [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                        [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                        [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                        [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                        [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
    matrix2 = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    matrix_obstacle = np.zeros((100,100))
    matrix_obstacle[10:20,35:50] = np.nan

    matrix_square = np.zeros((100,100))

    matrix_convex = np.zeros((100,100))
    matrix_convex[:,0] = 2.0
    matrix_convex[0:35,1] = 2.0
    matrix_convex[53:100,1] = 2.0
    matrix_convex[0:23,2] = 2.0
    matrix_convex[60:100,2] = 2.0
    matrix_convex[0:20,3]= 2.0
    matrix_convex[65:100,3] = 2.0
    matrix_convex[0:15,4] = 2.0
    matrix_convex[72:100,4] = 2.0
    matrix_convex[0:9,5] = 2.0
    matrix_convex[81:100,5] = 2.0
    matrix_convex[0:3,6] = 2.0
    matrix_convex[92:100,6] = 2.0
    matrix_convex[95:100,7] = 2.0
    matrix_convex[:,-1] = 2.0
    matrix_convex[0:35,-2] = 2.0
    matrix_convex[53:100,-2] = 2.0
    matrix_convex[0:23,-3] = 2.0
    matrix_convex[60:100,-3] = 2.0
    matrix_convex[0:20,-4]= 2.0
    matrix_convex[65:100,-4] = 2.0
    matrix_convex[0:15,-5] = 2.0
    matrix_convex[72:100,-5] = 2.0
    matrix_convex[0:9,-6] = 2.0
    matrix_convex[81:100,-6] = 2.0
    matrix_convex[0:3,-7] = 2.0
    matrix_convex[92:100,-7] = 2.0
    matrix_convex[95:100,-8] = 2.0

    matrix_nonconvex = np.zeros((100,100))
    matrix_nonconvex[:,0] = 2.0
    matrix_nonconvex[2:88,1] = 2.0
    matrix_nonconvex[5:80,2] = 2.0
    matrix_nonconvex[12:76,3] = 2.0
    matrix_nonconvex[22:69,4] = 2.0
    matrix_nonconvex[29:61,5] = 2.0
    matrix_nonconvex[34:55,6] = 2.0

    matrix_nonconvex[:,-1] = 2.0
    matrix_nonconvex[:,-2] = 2.0
    matrix_nonconvex[:,-3] = 2.0
    matrix_nonconvex[2:88,-4] = 2.0
    matrix_nonconvex[5:80,-5] = 2.0
    matrix_nonconvex[12:76,-6] = 2.0
    matrix_nonconvex[22:69,-7] = 2.0
    matrix_nonconvex[29:61,-8] = 2.0
    matrix_nonconvex[34:55,-9] = 2.0

    #show_map(matrix_nonconvex)
    heat_matrix = withrows(matrix_obstacle,2,4,False)
    spread_matrix = pathogenspread(matrix_obstacle,heat_matrix)
    path=[[0,0],[1,0],[2,0],[2,1],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],[2,8]]
    #showpath(spread_matrix,path)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
