from heatmap import show_map
from heatmap import withrows
from heatmap import norows
from heatmap import polygon
from monitortreat import showpath
from monitortreat import showpathlong
from monitortreat import updatematrix
from spreading import weedsspread
from spreading import pathogenspread
from budgeted_coverage import max_coverage, animation_max_coverage

import numpy as np
import time
from copy import deepcopy
import os
import pickle
import matplotlib as mpl
### new
from infomap import create_random_infomap
###

from weed import Weed
from pathogen import Pathogen

from Planner.Sampling_based_Planning.rrt_2D.informed_rrt_star_rig_spread import main as rig
from Planner.Sampling_based_Planning.rrt_2D.informed_rrt_star_rig_spread_rows import main as rig_rows
from Planner.Sampling_based_Planning.rrt_2D.informed_rrt_star_rig_spread_rows_v2 import main as rig_rows_v2

from Planner.Sampling_based_Planning.rrt_2D.informed_rrt_star_rig_spread_matrix_restructured import main as rig_matrix
from Planner.Sampling_based_Planning.rrt_2D.informed_rrt_star_rig_spread_rows_matrix import main as rig_rows_matrix

def getDefaultSettings(informed=True,rewiring=True,
step_len=10,search_radius=20,budget=350,
stopsetting="strict",multirobot=False,rowsbool=False):
    # setting the defaults:
    informed=informed
    rewiring = rewiring
    step_len=step_len
    search_radius=search_radius
    budget = budget
    stopsetting = stopsetting
    multirobot = multirobot

    # setting default step_len and search_radius based on the rowsbool
    # assume no rows:
    rowsbool =rowsbool
    # if not step_len and not rowsbool:
    #     step_len = 40
    #     search_radius=40
    # if not step_len and rowsbool:
    #     step_len=200
    #     search_radius=200
    # if not budget and not rowsbool:
    #     budget = 350
    # if not budget and rowsbool:
    #     budget = 500
    # returning all the settings
    return [rowsbool, budget, informed, rewiring, step_len, search_radius, stopsetting, multirobot]

def getSettings(scenario):
    # setting the defaults:
    rowsbool = False
    informed=True
    rewiring = True
    step_len = 30
    search_radius=30
    budget = None
    stopsetting = "strict"
    multirobot = False

    match scenario[0]: # budget
        case 1:
            budget = round((9999/100)*1) # 1 percent
        case 2:
            budget = round((9999/100)*5) # 5 percent
        case 3:
            budget = round((9999/100)*10) # 10 percent

    match scenario[1]: # nr of gaussians
        case 1:
            gaussians_nr=1
        case 2:
            gaussians_nr=10
        case 3: 
            gaussians_nr=40
    
    match scenario[2]: # gaussians size
        case 1: 
            gaussians_size = 1
        case 2: 
            gaussians_size = 5
        case 3: 
            gaussians_size = 10

    match scenario[3]: # robot nr
        case 1:
            multirobot = 1
        case 2:
            multirobot = 2
        case 3:
            multirobot = 5
    return [rowsbool, budget, informed, rewiring, step_len, search_radius, stopsetting, multirobot],[gaussians_nr,gaussians_size]

def prepandtest():
    scenariolist = [[1, 1, 1], [1, 1, 2], [1, 1, 3], 
                    [1, 2, 1], [1, 2, 2], [1, 2, 3], 
                    [1, 3, 1], [1, 3, 2], [1, 3, 3],
                    [2, 1, 1], [2, 1, 2], [2, 1, 3], 
                    [2, 2, 1], [2, 2, 2], [2, 2, 3], 
                    [2, 3, 1], [2, 3, 2], [2, 3, 3],
                    [3, 1, 1], [3, 1, 2], [3, 1, 3], 
                    [3, 2, 1], [3, 2, 2], [3, 2, 3], 
                    [3, 3, 1], [3, 3, 2], [3, 3, 3]]

    for scenario in scenariolist:
        settings,[gaussians_nr,gaussians_size] = getSettings(scenario)
        if gaussians_size==1:
            uncertainty_matrix = create_random_infomap(type="point",source_nr=gaussians_nr)
        else:
            uncertainty_matrix = create_random_infomap(source_nr=gaussians_nr, source_size=gaussians_size)

        # Our method
        pathname = str("../Result_files/") + str(scenario) + str("_method/")
        [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,
                            matrices,samplelocations] = rig_matrix(uncertainty_matrix,settings)
        animation_max_coverage(uncertainty_matrix,finalpath,finalinfo,"Proposed Method",False,pathname)

        # Uninformed method
        pathname = str("../Result_files/") + str(scenario) + str("_uninformed/")
        uniform_matrix = deepcopy(uncertainty_matrix)
        uniform_matrix[uniform_matrix >= 0.0] = 0.001
        [finalpath, infopath, finalcost, uninf_finalinfo, budget, steplength, searchradius, iteration,
                            matrices,samplelocations] = rig_matrix(uniform_matrix,settings)
        nodelist = []
        finalinfo = 0
        for gridpoint in infopath:
            if gridpoint not in nodelist:
                finalinfo += uncertainty_matrix[gridpoint[1],gridpoint[0]]
        animation_max_coverage(uncertainty_matrix,finalpath,finalinfo,"Uninformed Method",False,pathname)

        # Budgeted coverage
        pathname = str("../Result_files/") + str(scenario) + str("_coverage/")
        startpos=[50,0]
        [finalpath, finalinfo, finalcost] = max_coverage(uncertainty_matrix, settings[1], startpos, settings[7],False)
        animation_max_coverage(uncertainty_matrix,finalpath,finalinfo,"Coverage Method",False,pathname)

def default():
    time_start = time.process_time()

    # Choose the field shape:
    #[field_matrix,field_vertex] = polygon("hexagon_small",True)
    [field_matrix,field_vertex] = polygon("rectangle",False)
    #[field_matrix,field_vertex] = polygon("hexagon_convex",True)
    #[field_matrix,field_vertex] = polygon("hexagon_concave",True)
    #[field_matrix,field_vertex] = polygon("rectangle_obstacle",True)
    rowsbool = True
    if rowsbool:
        [plant_matrix,row_nrs,row_edges,field_vertex] = withrows(field_matrix,2,3,field_vertex,False)

    else:
        plant_matrix = norows(field_matrix,2,False)
    #show_map(matrix_nonconvex)
    weedbool = False # for pathogen, set to False
    if not weedbool:
        # Configure the spreading characteristics of the pathogen
        # more aggressive
        pathogen1 = Pathogen(patchnr=3,infectionduration=4,spreadrange=5, reproductionfraction=0.5, reproductionrate=2, standarddeviation=0.3, saturation=5)
        # two patches
        pathogen1 = Pathogen(patchnr=2,infectionduration=6,spreadrange=3, reproductionfraction=0.5, reproductionrate=2, standarddeviation=0.3, saturation=5)
        # one patch:
        #pathogen1 = Pathogen(patchnr=1,infectionduration=6,spreadrange=6, reproductionfraction=0.5, reproductionrate=2, standarddeviation=0.1, saturation=3) # one big patch

        spread_matrix, worldmodel_matrix, uncertainty_matrix = pathogenspread(field_matrix,plant_matrix,pathogen1, True)

        # Start with no info (i.e., blank world_model):
        # worldmodel_matrix = deepcopy(field_matrix)
        # worldmodel_matrix[worldmodel_matrix == 0.0] = 0.001
        # Start with uniform uncertainty matrix (no info on uncertainty):
        # uncertainty_matrix = deepcopy(field_matrix)
        # uncertainty_matrix[uncertainty_matrix == 0.0] = 0.001
        # show_map(worldmodel_matrix)
    else:
        weed1 = Weed(patchnr=4,patchsize=7,spreadrange=3,reproductionrate=2,standarddeviation=1, saturation=1,plantattach=False)

        spread_matrix,worldmodel_matrix, uncertainty_matrix = weedsspread(field_matrix,plant_matrix,weed1, True)

        # # Start with no info:
        # worldmodel_matrix = deepcopy(field_matrix)
        # worldmodel_matrix[worldmodel_matrix == 0.0] = 0.001
        # # show_map(worldmodel_matrix)
    #del plant_matrix
    #del spread_matrix #to save memory
    #np.save('uncertainty_matrixfile.npy',uncertainty_matrix)
    #np.save('Testing_files/uncertainty_matrixfile_test.npy',uncertainty_matrix)
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
    #scenario = 1
    #rig(uncertainty_matrix)
    total_days=1
    matrices=None #initialize for first day
    scenariosettings=None
    for day in range(1,total_days+1):
        if rowsbool:
            [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,matrices,samplelocations] = rig_rows_matrix(
                uncertainty_matrix, row_nrs, row_edges, field_vertex,scenariosettings,matrices)
        else:
            [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,
                            matrices,samplelocations] = rig_matrix(uncertainty_matrix,scenariosettings,matrices)

        # Set the sensor uncertainty and update the world model for the next day:
        sensoruncertainty=0
        [spread_matrix_updated,worldmodel_updated,uncertainty_matrix_updated] = updatematrix(pathogen1,plant_matrix,spread_matrix,worldmodel_matrix,uncertainty_matrix,infopath, sensoruncertainty,False)

        #print("sum entropy = "+str(np.nansum(uncertainty_matrix)))

        #time_end = time.time()
        time_end = time.process_time()
        time_total = time_end-time_start
        print("Time taken = "+str(time_total)+" seconds. This is more than "+str(time_total//60)+" minutes")

        # Saving the figure:
        boolsave=False
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
def minimum_example():
    #uncertainty_matrix = create_random_infomap()
    uncertainty_matrix = create_random_infomap(type="point",source_nr=30)
    # default: informed=True,rewiring=True,step_len=10,search_radius=20,budget=350,
    #    stopsetting="strict",multirobot=False,rowsbool=False
    default_scenario = getDefaultSettings(stopsetting="mild",step_len=40,budget=600)
    pathname = [] # empty pathname
    default_scenario.append(pathname) # such that files can also be saved within the algorithm
    [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,
                            matrices,samplelocations] = rig_matrix(uncertainty_matrix,default_scenario)
    showpath(uncertainty_matrix,finalpath,finalcost,finalinfo,budget, steplength, searchradius, iteration,True,False)

def multi_robot(): # NOT FUNCTIONAL YET AT ALL
    total_robots=2
    uncertainty_matrix = create_random_infomap(type="point",source_nr=30)
    default_scenario = getDefaultSettings(stopsetting="mild",step_len=30,budget=400, multirobot=total_robots)
    #pathname = [] # empty pathname
    pathname = str("Figures/Savefolder/")
    default_scenario.append(pathname) # such that files can also be saved within the algorithm

    # for day in range(1,total_robots+1):
        
    [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,
                            matrices,samplelocations] = rig_matrix(uncertainty_matrix,default_scenario)



def coverage():
    startpos=[50,0]
    uncertainty_matrix = create_random_infomap()
    #uncertainty_matrix = create_random_infomap(type="point",source_nr=30)
    max_coverage(uncertainty_matrix, 300, startpos,3)


if __name__ == '__main__':
    #default()
    #prepandtest()
    #minimum_example()
    #multi_robot()
    coverage()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
