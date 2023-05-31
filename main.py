# This is a sample Python script.
from heatmap import show_map
from heatmap import withrows
from heatmap import norows
from heatmap import polygon
from monitortreat import showpath
from monitortreat import showpathlong
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
    time_start = time.time()

    # Choose the field shape:
    [field_matrix,field_vertex] = polygon("hexagon",False)

    rowsbool = False #if you want without rows, set to False:
    if rowsbool:
        [plant_matrix,row_nrs,row_edges,field_vertex] = withrows(field_matrix,2,2,field_vertex,False)
    else:
        plant_matrix = norows(field_matrix,2,False)
    #show_map(matrix_nonconvex)
    pathogenbool = True # for weed, set to False
    if pathogenbool:
        # Configure the spreading characteristics of the pathogen
        # more aggressive
        pathogen1 = Pathogen(patchnr=3,infectionduration=4,spreadrange=5, reproductionfraction=0.5, reproductionrate=2, standarddeviation=0.3, saturation=5)
        # two patches
        pathogen1 = Pathogen(patchnr=2,infectionduration=4,spreadrange=3, reproductionfraction=0.5, reproductionrate=2, standarddeviation=0.3, saturation=5)
        # one patch:
        #pathogen1 = Pathogen(patchnr=1,infectionduration=6,spreadrange=6, reproductionfraction=0.5, reproductionrate=2, standarddeviation=0.1, saturation=3) # one big patch
        # Start with no info (i.e., blank world_model):
        # worldmodel_matrix = deepcopy(field_matrix)
        # worldmodel_matrix[worldmodel_matrix == 0.5] = 0.001
        # show_map(worldmodel_matrix)

        spread_matrix, worldmodel_matrix, uncertainty_matrix = pathogenspread(field_matrix,plant_matrix,pathogen1, False)
    else:
        weed1 = Weed(patchnr=4,patchsize=7,spreadrange=3,reproductionrate=2,standarddeviation=1, saturation=1,plantattach=False)
        # # Start with no info:
        # worldmodel_matrix = deepcopy(field_matrix)
        # worldmodel_matrix[worldmodel_matrix == 0.5] = 0.001
        # # show_map(worldmodel_matrix)

        spread_matrix,worldmodel_matrix, uncertainty_matrix = weedsspread(field_matrix,plant_matrix,weed1, True)
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
    scenario = 1
    #rig(uncertainty_matrix)
    total_days=20
    costmatrix=None #initialize for first day
    for day in range(1,total_days+1):
        if rowsbool:
            [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,costmatrix] = rig_rows_matrix(
                uncertainty_matrix, row_nrs, row_edges, field_vertex,scenario,costmatrix)
        else:
            [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,costmatrix] = rig_matrix(uncertainty_matrix,scenario,costmatrix)

        # Set the sensor uncertainty and update the world model for the next day:
        sensoruncertainty=0
        [spread_matrix_updated,worldmodel_updated,uncertainty_matrix_updated] = updatematrix(pathogen1,plant_matrix,spread_matrix,worldmodel_matrix,uncertainty_matrix,infopath, sensoruncertainty,False)

        #print("sum entropy = "+str(np.nansum(uncertainty_matrix)))

        time_end = time.time()
        time_total = time_end-time_start
        print("Time taken = "+str(time_total)+" seconds. This is more than "+str(time_total//60)+" minutes")

        # Saving the figure:
        boolsave=True
        if boolsave:
            #showpath(uncertainty_matrix,finalpath,finalcost,finalinfo,budget, steplength, searchradius, iteration,True,True)
            if day==1:
                figlong=None
                axlong=None
            [figlong,axlong]=showpathlong(day,total_days,figlong,axlong,uncertainty_matrix,finalpath,finalcost,finalinfo,budget, steplength, searchradius, iteration,True,True)

        spread_matrix=spread_matrix_updated
        worldmodel_matrix=worldmodel_updated
        uncertainty_matrix=uncertainty_matrix_updated
    # tests=True
    # results=[]
    # if tests:
    #     scenario=1
    #     print("Without uncertainty all over")
    #     while scenario<=7:
    #         [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration] = rig_matrix(uncertainty_matrix,scenario)
    #         print("Scen.= "+str(scenario)+" Cost= "+str(finalcost)+" Info= "+str(finalinfo))
    #         results.append("Scen.= "+str(scenario)+" Cost= "+str(finalcost)+" Info= "+str(finalinfo))
    #         scenario+=1
    #     scenario=1
    #     uncertainty_matrix[uncertainty_matrix == 0] = 0.001  # little bit of uncertainty all over the map
    #     print("With uncertainty all over")
    #     while scenario<=6:
    #         [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration] = rig_matrix(uncertainty_matrix,scenario)
    #         print("Scen.= "+str(scenario)+" Cost= "+str(finalcost)+" Info= "+str(finalinfo))
    #         results.append("Scen.= "+str(scenario)+" Cost= "+str(finalcost)+" Info= "+str(finalinfo))
    #         scenario+=1
    #
    #     print(results)



if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
