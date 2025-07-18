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
    step_len = 30/5
    search_radius=30/5
    budget = None
    stopsetting = "strict"
    multirobot = False

    match scenario[0]: # budget
        case 1:
            budget = round((399/100)*10) # 10 percent 
        case 2:
            budget = round((399/100)*15) # 15 percent
        case 3:
            budget = round((399/100)*25) # 25 percent

    match scenario[1]: # nr of gaussians
        case 1:
            gaussians_nr=1
        case 2:
            gaussians_nr=3
        case 3: 
            gaussians_nr=10
    
    match scenario[2]: # gaussians size
        case 1: 
            gaussians_size = 1
        case 2: 
            gaussians_size = 2
        case 3: 
            gaussians_size = 5

    match scenario[3]: # robot nr
        case 1:
            multirobot = 1
        case 2:
            multirobot = 2
        case 3:
            multirobot = 5
    pathname = []
    return [rowsbool, budget, informed, rewiring, step_len, search_radius, stopsetting, multirobot,pathname],[gaussians_nr,gaussians_size]
from itertools import product

def prepandtest(test="comparison"):
    if test=="comparison":
        scenariolist = list(product([1, 2, 3], repeat=4))
        # for it in range(2,20):
        for it in [2,5,6,8,9,12,13,15,16,17]:
            pathname = str("Result_files/small/comparison/") + str(it) 
            if not os.path.exists(pathname):
                os.makedirs(str("Result_files/small/comparison/") + str(it))



            for scenario in scenariolist:
                settings,[gaussians_nr,gaussians_size] = getSettings(scenario)
                # Total info in the map:
                scenario_basepath = str("Result_files/small/comparison/")+str(it)+"/"+ str(scenario) +str("/") 
                uncertainty_map_path = os.path.join(scenario_basepath, 'uncertainty_matrix.npy')

                if os.path.exists(uncertainty_map_path):
                    uncertainty_matrix = np.load(uncertainty_map_path)
                else:
                    if gaussians_size==1:
                        uncertainty_matrix = create_random_infomap(type="point",size=(20,20),source_nr=gaussians_nr)
                    else:
                        uncertainty_matrix = create_random_infomap(source_nr=gaussians_nr,size=(20,20), source_size=gaussians_size)
                    # Ensure base path exists and save the matrix
                    os.makedirs(scenario_basepath, exist_ok=True)
                    np.save(uncertainty_map_path, uncertainty_matrix)
                    np.save(scenario_basepath + 'totalinfomatrix.npy', np.nansum(uncertainty_matrix))
        
                # Our method
                informed_matrix = deepcopy(uncertainty_matrix)
                pathname = str("Result_files/small/comparison/")+str(it)+"/"+ str(scenario) + str("_method")+str("/")
                if not os.path.exists(pathname):
                    os.makedirs(pathname)
                    time_start=time.process_time()
                    [path, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,
                                        matrices,samplelocations_saved] = rig_matrix(uncertaintymatrix=informed_matrix,scenario=settings,iterations=200,samplelocations=[])
                    np.save(scenario_basepath + 'samplelocations_saved.npy', samplelocations_saved)
                    finalpath=path[0]
                    rounds=path[1]
                    time_end = time.process_time()
                    time_total = time_end-time_start
                    print("Time taken = "+str(time_total)+" seconds. This is more than "+str(time_total//60)+" minutes")
                    animation_max_coverage(uncertainty_matrix,finalpath,finalinfo,"Proposed Method",False,pathname,rounds,(20,20))
                    np.save(pathname + 'finalinfo.npy', finalinfo)
                    np.save(pathname + 'finalcosts.npy', finalcost)
                    np.save(pathname + 'computingtime.npy', time_total)

                    np.save(pathname + 'finalpath.npy', finalpath) # to change viz if needed
                    np.save(pathname + 'rounds.npy', rounds) # to change viz if needed
                    np.save(pathname + 'infopath.npy', infopath) # to change viz if needed
                    np.save(pathname + 'totalinfomatrix.npy', np.nansum(uncertainty_matrix))

                # else skip to next method
                
                # Uninformed method
                pathname = str("Result_files/small/comparison/")+str(it)+"/"+ str(scenario) + str("_uninformed")+str("/")
                samplelocations_path = os.path.join(scenario_basepath, 'samplelocations_saved.npy')
                if not os.path.exists(pathname) and os.path.exists(samplelocations_path):
                    samplelocations_saved = np.load(samplelocations_path, allow_pickle=True)
                    os.makedirs(pathname)
                    uniform_matrix = deepcopy(uncertainty_matrix)
                    uniform_matrix[uniform_matrix >= 0.0] = 0.001
                    time_start=time.process_time()

                    # Define a reusable key based on robot count and budget
                    reuse_key = f"robots{settings[7]}_budget{settings[1]}"
                    reuse_dir = str("Result_files/small/comparison/")+str(it)+str("uninformed_cache/")
                    reuse_file = os.path.join(reuse_dir, f"{reuse_key}.npz")
                    #[path, infopath, finalcost, uninf_finalinfo, budget, steplength, searchradius, iteration,
                    #                    matrices,samplelocations] = rig_matrix(uniform_matrix,settings)
                    if not os.path.exists(reuse_file):
                    # No cached version, run and save
                        [path, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,
                                            matrices,samplelocations] = rig_matrix(uncertaintymatrix=uniform_matrix,scenario=settings,samplelocations=samplelocations_saved,iterations=200)
                        finalpath=path[0]
                        rounds=path[1]
                    
                        # Save reusable data
                        if not os.path.exists(reuse_dir):
                            os.makedirs(reuse_dir)
                        time_end = time.process_time()
                        time_total = time_end-time_start
                        np.savez(reuse_file, path=finalpath, infopath=infopath, finalcost=finalcost, rounds=rounds, time_total=time_total)

                    else:
                        # Load cached path
                        print(f"Loading cached uninformed path from {reuse_file}")
                        data = np.load(reuse_file, allow_pickle=True)
                        finalpath = data['path']
                        infopath = data['infopath'].tolist()
                        finalcost = data['finalcost']
                        rounds = data['rounds']
                        time_total = data['time_total']

                    
                    nodelist = []
                    finalinfo = 0
                    for gridpoint in infopath:
                        if gridpoint not in nodelist:
                            finalinfo += uncertainty_matrix[gridpoint[1],gridpoint[0]]
                            nodelist.append(gridpoint)

                    print("Time taken = "+str(time_total)+" seconds. This is more than "+str(time_total//60)+" minutes")
                    animation_max_coverage(uncertainty_matrix,finalpath,finalinfo,"Uninformed Method",False,pathname,rounds, (20,20))
                    np.save(pathname + 'finalinfo.npy', finalinfo)
                    np.save(pathname + 'finalcosts.npy', finalcost)
                    np.save(pathname + 'computingtime.npy', time_total)

                    np.save(pathname + 'finalpath.npy', finalpath) # to change viz if needed
                    np.save(pathname + 'rounds.npy', rounds) # to change viz if needed
                    np.save(pathname + 'infopath.npy', infopath) # to change viz if needed
                    np.save(pathname + 'totalinfomatrix.npy', np.nansum(uncertainty_matrix))
                    
                # Budgeted coverage
                budget_matrix = deepcopy(uncertainty_matrix)
                pathname = str("Result_files/small/comparison/")+str(it)+"/"+ str(scenario) + str("_coverage")+str("/")
                if not os.path.exists(pathname):
                    os.makedirs(pathname)
                    startpos=[50,40]
                    time_start=time.process_time()
                    [finalpath, finalinfo, finalcost] = max_coverage(budget_matrix, settings[1], startpos, settings[7],False)
                    time_end = time.process_time()
                    time_total = time_end-time_start
                    print(finalpath)
                    print(finalinfo)
                    print(finalcost)
                    print("Time taken = "+str(time_total)+" seconds. This is more than "+str(time_total//60)+" minutes")
                    animation_max_coverage(uncertainty_matrix,finalpath,finalinfo,"Coverage Method",False,pathname,[],(20,20))
                    np.save(pathname + 'finalinfo.npy', finalinfo)
                    np.save(pathname + 'finalcosts.npy', finalcost)
                    np.save(pathname + 'computingtime.npy', time_total)
                    
                    np.save(pathname + 'finalpath.npy', finalpath) # to change viz if needed
                    #np.save(pathname + 'infopath.npy', infopath) # to change viz if needed
                    np.save(pathname + 'totalinfomatrix.npy', np.nansum(uncertainty_matrix))

                # Overview of scenarios in text file:
                with open(("Result_files/small/comparison/"+str(it)+"/"+"scenario_overview.txt"), "a") as file:
                    wr_str = str(scenario)+" Budget: "+ str(settings[1])+ ", Gaussians nr: "+str( gaussians_nr)+", Gaussian size: "+ str(gaussians_size)+ ", Robots: "+ str(settings[7])+"\n"
                    file.write(wr_str)
                    print("SCENARIO \n")
                    print(wr_str)
    elif test=="increasing_iterations":
        scenario = [1,2,2,1]
        settings,[gaussians_nr,gaussians_size] = getSettings(scenario)
                    
        for it in range(41):
            pathname = str("Result_files/small/increasing_iterations/") + str(it) + '/uncertainty_matrix.npy'

            if not os.path.exists(pathname):
                if gaussians_size==1:
                    uncertainty_matrix = create_random_infomap(type="point",size=(20,20),source_nr=gaussians_nr)
                else:
                    uncertainty_matrix = create_random_infomap(source_nr=gaussians_nr, size=(20,20),source_size=gaussians_size)
                os.makedirs(str("Result_files/small/increasing_iterations/") + str(it))

                np.save(pathname,uncertainty_matrix)
                samplelocations_saved=[]
                samplelocations_loaded=False
            else:
                uncertainty_matrix = np.load(pathname)
                pathname = str("Result_files/small/increasing_iterations/") + str(it) + '/samplelocations.npy'
                samplelocations_saved = np.load(pathname)
                samplelocations_loaded = True

            for budget in [round((399/100)*10),round((399/100)*25)]:
                for robots in [5,2,1]:
                # for robots in [1]:
                    for iterations in [200,150,125,100,75,50,25]:
                        settings[1] = budget
                        settings[7] = robots

                        # Our method
                        informed_matrix = deepcopy(uncertainty_matrix)
                        pathname = str("Result_files/small/increasing_iterations/") +str(it)+str("/") + str(scenario) + str("_method_b")+str(budget)+"_r"+str(robots)+"_"+str(iterations)+str("/")
                        if not os.path.exists(pathname):
                            os.makedirs(pathname)
                        else:
                            continue # skip to the next scenario if we had this one already
                        print(pathname)
                        time_start=time.process_time()
                        [path, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,
                                            matrices,samplelocations] = rig_matrix(uncertaintymatrix=informed_matrix,scenario=settings,samplelocations=samplelocations_saved, iterations=iterations)
                        finalpath=path[0]
                        rounds=path[1]
                        if samplelocations_loaded==False:
                            samplelocations_saved=samplelocations
                            pathname = str("Result_files/small/increasing_iterations/") + str(it) + '/samplelocations.npy'
                            np.save(pathname, samplelocations_saved)
                            pathname = str("Result_files/small/increasing_iterations/") +str(it)+str("/") + str(scenario) + str("_method_b")+str(budget)+"_r"+str(robots)+"_"+str(iterations)+str("/")
                            samplelocations_loaded = True
                        time_end = time.process_time()
                        time_total = time_end-time_start
                        print("Time taken = "+str(time_total)+" seconds. This is more than "+str(time_total//60)+" minutes")
                        animation_max_coverage(uncertainty_matrix,finalpath,finalinfo,"Proposed Method",False,pathname,rounds,(20,20))
                        np.save(pathname + 'finalinfo.npy', finalinfo)
                        np.save(pathname + 'finalcosts.npy', finalcost)
                        np.save(pathname + 'computingtime.npy', time_total)
                        np.save(pathname + 'finalpath.npy', finalpath) # to change viz if needed
                        np.save(pathname + 'rounds.npy', rounds) # to change viz if needed
                        np.save(pathname + 'infopath.npy', infopath) # to change viz if needed
                        np.save(pathname + 'totalinfomatrix.npy', np.nansum(uncertainty_matrix))



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
    showpath(uncertainty_matrix,finalpath[0],finalcost,finalinfo,budget, steplength, searchradius, iteration,True,False)

def multi_robot(): # NOT FUNCTIONAL YET AT ALL
    total_robots=3
    #uncertainty_matrix = create_random_infomap(type="point",source_nr=30)
    uncertainty_matrix = create_random_infomap(type="fixed",source_nr=30)
    default_scenario = getDefaultSettings(stopsetting="mild",step_len=30,budget=400, multirobot=total_robots)
    pathname = [] # empty pathname
    #pathname = str("Figures/Savefolder/")
    default_scenario.append(pathname) # such that files can also be saved within the algorithm

    # for day in range(1,total_robots+1):
        
    [path, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,
                            matrices,samplelocations] = rig_matrix(uncertainty_matrix,default_scenario)
    finalpath=path[0]
    rounds=path[1]
    animation_max_coverage(uncertainty_matrix,finalpath,finalinfo,"Proposed Method",True,rounds)




def coverage():
    startpos=[50,0]
    uncertainty_matrix = create_random_infomap()
    test_matrix = deepcopy(uncertainty_matrix)
    #uncertainty_matrix = create_random_infomap(type="point",source_nr=30)
    sum_best_coords, sum_info, sum_costs = max_coverage(test_matrix, 300, startpos,3)
    animation_max_coverage(uncertainty_matrix,sum_best_coords,sum_info,"Coverage Method",True)


if __name__ == '__main__':
    #default()
    #prepandtest()
    #prepandtest("increasing_iterations")
    prepandtest("comparison")
    #minimum_example()
    #multi_robot()
    #coverage()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
