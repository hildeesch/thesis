# This is a sample Python script.
from heatmap import show_map
from heatmap import withrows
from heatmap import norows
from heatmap import polygon
from monitortreat import showpath
from monitortreat import updatematrix
from spreading import weedsspread
from spreading import pathogenspread
import numpy as np
import time
from copy import deepcopy


from weed import Weed
from pathogen import Pathogen

from Planner.Sampling_based_Planning.rrt_2D.informed_rrt_star_rig_spread import main as rig
from Planner.Sampling_based_Planning.rrt_2D.informed_rrt_star_rig_spread_rows import main as rig_rows
from Planner.Sampling_based_Planning.rrt_2D.informed_rrt_star_rig_spread_rows_v2 import main as rig_rows_v2

from Planner.Sampling_based_Planning.rrt_2D.informed_rrt_star_rig_spread_matrix import main as rig_matrix
from Planner.Sampling_based_Planning.rrt_2D.informed_rrt_star_rig_spread_rows_matrix import main as rig_rows_matrix


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # matrix_obstacle = np.zeros((100,100))
    # matrix_obstacle[10:20,35:50] = np.nan
    #
    # matrix_square = np.zeros((100,100))
    #
    # matrix_convex = np.zeros((100,100))
    # matrix_convex[:,0] = np.nan
    # matrix_convex[0:35,1] = np.nan
    # matrix_convex[53:100,1] = np.nan
    # matrix_convex[0:23,2] = np.nan
    # matrix_convex[60:100,2] = np.nan
    # matrix_convex[0:20,3]= np.nan
    # matrix_convex[65:100,3] = np.nan
    # matrix_convex[0:15,4] = np.nan
    # matrix_convex[72:100,4] = np.nan
    # matrix_convex[0:9,5] = np.nan
    # matrix_convex[81:100,5] = np.nan
    # matrix_convex[0:3,6] = np.nan
    # matrix_convex[92:100,6] = np.nan
    # matrix_convex[95:100,7] = np.nan
    # matrix_convex[:,-1] = np.nan
    # matrix_convex[0:35,-2] = np.nan
    # matrix_convex[53:100,-2] = np.nan
    # matrix_convex[0:23,-3] = np.nan
    # matrix_convex[60:100,-3] = np.nan
    # matrix_convex[0:20,-4]= np.nan
    # matrix_convex[65:100,-4] = np.nan
    # matrix_convex[0:15,-5] = np.nan
    # matrix_convex[72:100,-5] = np.nan
    # matrix_convex[0:9,-6] = np.nan
    # matrix_convex[81:100,-6] = np.nan
    # matrix_convex[0:3,-7] = np.nan
    # matrix_convex[92:100,-7] = np.nan
    # matrix_convex[95:100,-8] = np.nan
    #
    # matrix_nonconvex = np.zeros((100,100))
    # matrix_nonconvex[:,0] = np.nan
    # matrix_nonconvex[2:88,1] = np.nan
    # matrix_nonconvex[5:80,2] = np.nan
    # matrix_nonconvex[12:76,3] = np.nan
    # matrix_nonconvex[22:69,4] = np.nan
    # matrix_nonconvex[29:61,5] = np.nan
    # matrix_nonconvex[34:55,6] = np.nan
    #
    # matrix_nonconvex[:,-1] = np.nan
    # matrix_nonconvex[:,-2] = np.nan
    # matrix_nonconvex[:,-3] = np.nan
    # matrix_nonconvex[2:88,-4] = np.nan
    # matrix_nonconvex[5:80,-5] = np.nan
    # matrix_nonconvex[12:76,-6] = np.nan
    # matrix_nonconvex[22:69,-7] = np.nan
    # matrix_nonconvex[29:61,-8] = np.nan
    # matrix_nonconvex[34:55,-9] = np.nan


    time_start = time.time()
    field_matrix = polygon("hexagon",False)
    [plant_matrix,row_nrs,row_edges] = withrows(field_matrix,2,1,True)
    #plant_matrix = norows(field_matrix,2,False)
    #show_map(matrix_nonconvex)
    #heat_matrix = withrows(matrix_square,2,4,True)

    pathogen1 = Pathogen(patchnr=2,infectionduration=4,spreadrange=3, reproductionfraction=0.5, reproductionrate=2, standarddeviation=0.3, saturation=5)
    #pathogen1 = Pathogen(patchnr=1,infectionduration=6,spreadrange=6, reproductionfraction=0.5, reproductionrate=2, standarddeviation=0.1, saturation=3) # one big patch
    spread_matrix, worldmodel_matrix, uncertainty_matrix = pathogenspread(field_matrix,plant_matrix,pathogen1, False)
    #weed1 = Weed(patchnr=4,patchsize=7,spreadrange=3,reproductionrate=2,standarddeviation=1, saturation=1,plantattach=False)
    #spread_matrix,worldmodel_matrix, uncertainty_matrix = weedsspread(field_matrix,plant_matrix,weed1, True)
    #del plant_matrix
    #del spread_matrix #to save memory
    #np.save('uncertainty_matrixfile.npy',uncertainty_matrix)
    #np.save('uncertainty_matrixfile_small.npy', uncertainty_matrix)
    #uncertainty_matrix= np.load('uncertainty_matrixfile.npy')
    #uncertainty_matrix= np.load('uncertainty_matrixfile_small.npy')
    #print(np.nansum(uncertainty_matrix))
    #uncertainty_matrix[uncertainty_matrix==0]=0.001 # little bit of uncertainty all over the map

    #half uniform matrix:
    #uncertainty_matrix=deepcopy(field_matrix)
    #uncertainty_matrix[uncertainty_matrix==0.5]=0
    #(uncertainty_matrix[:,0:50])[uncertainty_matrix[:,0:50]==0]=0.5

    #show_map(uncertainty_matrix)
    print(np.nansum(uncertainty_matrix))
    #rig(uncertainty_matrix)
    #[finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration] = rig_matrix(uncertainty_matrix)
    [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration] = rig_rows_matrix(uncertainty_matrix,row_nrs,row_edges)
    sensoruncertainty=0
    [spread_matrix_updated,worldmodel_updated,uncertainty_matrix_updated] = updatematrix(pathogen1,plant_matrix,spread_matrix,worldmodel_matrix,uncertainty_matrix,infopath, sensoruncertainty,True)

    #print("sum entropy = "+str(np.nansum(uncertainty_matrix)))

    time_end = time.time()
    time_total = time_end-time_start
    print("Time taken = "+str(time_total)+" seconds. This is more than "+str(time_total//60)+" minutes")

    #showpath(uncertainty_matrix,finalpath,finalcost,finalinfo,budget, steplength, searchradius, iteration,True,True)

    tests=True
    results=[]
    if tests:
        scenario=1
        print("Without uncertainty all over")
        while scenario<=7:
            [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration] = rig_matrix(uncertainty_matrix,scenario)
            print("Scen.= "+str(scenario)+" Cost= "+str(finalcost)+" Info= "+str(finalinfo))
            results.append("Scen.= "+str(scenario)+" Cost= "+str(finalcost)+" Info= "+str(finalinfo))
            scenario+=1
        scenario=1
        uncertainty_matrix[uncertainty_matrix == 0] = 0.001  # little bit of uncertainty all over the map
        print("With uncertainty all over")
        while scenario<=6:
            [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration] = rig_matrix(uncertainty_matrix,scenario)
            print("Scen.= "+str(scenario)+" Cost= "+str(finalcost)+" Info= "+str(finalinfo))
            results.append("Scen.= "+str(scenario)+" Cost= "+str(finalcost)+" Info= "+str(finalinfo))
            scenario+=1

        print(results)



if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
