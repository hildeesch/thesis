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
import os
import pickle


from weed import Weed
from pathogen import Pathogen

from Planner.Sampling_based_Planning.rrt_2D.informed_rrt_star_rig_spread import main as rig
from Planner.Sampling_based_Planning.rrt_2D.informed_rrt_star_rig_spread_rows import main as rig_rows
from Planner.Sampling_based_Planning.rrt_2D.informed_rrt_star_rig_spread_rows_v2 import main as rig_rows_v2

from Planner.Sampling_based_Planning.rrt_2D.informed_rrt_star_rig_spread_matrix import main as rig_matrix
from Planner.Sampling_based_Planning.rrt_2D.informed_rrt_star_rig_spread_rows_matrix import main as rig_rows_matrix


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def getSettings(scenario, rowsbool):
    # setting the defaults:
    informed=True
    rewiring = True
    step_len = None
    budget = None
    stopsetting = "strict"
    horizonplanning = False

    # setting the scenario based on the tested variable
    match scenario[0]:
        case 1:
            print("Var: budget") #TODO decide on the budgets (percentage or absolute?)
            match scenario[1]:
                case 1:
                    print("Setting: 10 procent - no rows")
                    rowsbool=False
                    budget = 200
                case 2:
                    print("Setting: 20 procent - no rows")
                    rowsbool=False
                    budget = 350
                case 3:
                    print("Setting: 35 procent - no rows")
                    rowsbool=False
                    budget = 500
                case 4:
                    print("Setting: 50 procent - no rows")
                    rowsbool=False
                    budget = 650
                case 5:
                    print("Setting: 10 procent - rows")
                    rowsbool=True
                    budget = 300
                case 6:
                    print("Setting: 20 procent - rows")
                    rowsbool=True
                    budget = 500
                case 7:
                    print("Setting: 35 procent - rows")
                    rowsbool=True
                    budget = 700
                case 8:
                    print("Setting: 50 procent - rows")
                    rowsbool=True
                    budget = 900
        case 2:
            print("Var: Informed vs uninformed")
            match scenario[1]:
                case 1:
                    print("Informed - no rows")
                    rowsbool=False
                    informed = True
                case 2:
                    print("Uninformed - no rows")
                    rowsbool=False
                    informed = False
                case 3:
                    print("Informed - rows")
                    rowsbool=True
                    informed = True
                case 4:
                    print("Uninformed - rows")
                    rowsbool=True
                    informed = False
        case 3:
            print("Var: Rewiring vs no rewiring")
            match scenario[1]:
                case 1:
                    print("Rewiring - no rows")
                    rowsbool=False
                    rewiring = True
                case 2:
                    print("No rewiring - no rows")
                    rowsbool=False
                    rewiring = False
                case 3:
                    print("Rewiring - rows")
                    rowsbool=True
                    rewiring = True
                case 4:
                    print("No rewiring - rows")
                    rowsbool=True
                    rewiring = False
        case 4:
            print("Var: Step length and search radius")
            match scenario[1]:
                case 1:
                    print("No rows - step = 20, radius = 20")
                    rowsbool=False
                    step_len=20
                    search_radius=20
                case 2:
                    print("No rows - step = 20, radius = 40")
                    rowsbool=False
                    step_len=20
                    search_radius=40
                case 3:
                    print("No rows - step = 40, radius = 20")
                    rowsbool=False
                    step_len=40
                    search_radius=20
                case 4:
                    print("No rows - step = 40, radius = 40")
                    rowsbool=False
                    step_len=40
                    search_radius=40
                case 5:
                    print("Rows - step = 100, radius = 100")
                    rowsbool=True
                    step_len=100
                    search_radius=100
                case 6:
                    print("Rows - step = 100, radius = 200")
                    rowsbool=True
                    step_len=100
                    search_radius=200
                case 7:
                    print("Rows - step = 200, radius = 100")
                    rowsbool=True
                    step_len=200
                    search_radius=100
                case 8:
                    print("Rows - step = 200, radius = 200")
                    rowsbool=True
                    step_len=200
                    search_radius=200
        case 5:
            print("Var: Stopping criteria ")
            match scenario[1]:
                case 1:
                    print("Mild - no rows")
                    rowsbool=False
                    stopsetting="mild"
                case 2:
                    print("Strict - no rows")
                    rowsbool=False
                    stopsetting="strict"
                case 3:
                    print("Mild - rows")
                    rowsbool=True
                    stopsetting="mild"
                case 4:
                    print("Strict - rows")
                    rowsbool=True
                    stopsetting="strict"
        case 6:
            print("Var: Long simulations")
            match scenario[1]:
                case 1:
                    print("Horizon - no rows")
                    rowsbool=False
                    horizonplanning=True
                case 2:
                    print("Single path - no rows")
                    rowsbool=False
                    horizonplanning=False
                case 3:
                    print("Horizon - rows")
                    rowsbool=True
                    horizonplanning=True
                case 4:
                    print("Single path - rows")
                    rowsbool=True
                    horizonplanning=False

    # setting default step_len and search_radius based on the rowsbool
    if not step_len and not rowsbool:
        step_len = 40
        search_radius=40
    if not step_len and rowsbool:
        step_len=200
        search_radius=200
    if not budget and not rowsbool:
        budget = 350
    if not budget and rowsbool:
        budget = 500

    # returning all the settings
    return [rowsbool, budget, informed, rewiring, step_len, search_radius, stopsetting, horizonplanning]
def getDisease(name):
    if name == "pathogen1":
        disease = Pathogen(patchnr=3, infectionduration=4, spreadrange=5, reproductionfraction=0.5, reproductionrate=2,
                            standarddeviation=0.3, saturation=5)
    elif name=="pathogen2":
        # two patches
        disease = Pathogen(patchnr=2, infectionduration=4, spreadrange=3, reproductionfraction=0.5,
                            reproductionrate=2, standarddeviation=0.3, saturation=5)

    elif name == "weed1":
        disease = Weed(patchnr=4, patchsize=7, spreadrange=3, reproductionrate=2, standarddeviation=1,
                    saturation=2, plantattach=False)
    else:
        disease = Weed(patchnr=2, patchsize=4, spreadrange=5, reproductionrate=4, standarddeviation=1,
                    saturation=1, plantattach=False)
    return disease
def prepandtest():
    #time_start = time.time()
    scenariosettings=None
    #time_start = time.process_time()

    # Choose the field shape:
    #[field_matrix,field_vertex] = polygon("hexagon_small",True)
    #[field_matrix,field_vertex] = polygon("hexagon_convex",True)
    #[field_matrix,field_vertex] = polygon("hexagon_concave",True)
    #[field_matrix,field_vertex] = polygon("rectangle_obstacle",True)

    scenariovariants = [["convex", False, "pathogen1"], ["convex", False, "pathogen2"], ["convex", True, "weed1"],
                        ["convex", True, "weed2"],
                        ["nonconvex", False, "pathogen1"], ["nonconvex", False, "pathogen2"],
                        ["nonconvex", True, "weed1"], ["nonconvex", True, "weed2"],
                        ["obstacle", False, "pathogen1"], ["obstacle", False, "pathogen2"], ["obstacle", True, "weed1"],
                        ["obstacle", True, "weed2"], ["rectangle", False, "pathogen1"]]
    # variant = [field, weedbool, weed/pathogen type]
    preparing = True
    if preparing:
        # variant = [field, weedbool, weed/pathogen type]
        for rowsbool in [True,False]:
           #for variant in scenariovariants:
           for variant in scenariovariants[-1]:
               # Choose the field shape:
               # Receive: field_matrix, field_vertex
               if variant[0]=="convex":
                   [field_matrix, field_vertex] = polygon("hexagon_convex", False)
               elif variant[0]=="concave":
                   [field_matrix,field_vertex] = polygon("hexagon_concave",False)
               elif variant[0]=="obstacle":
                   [field_matrix,field_vertex] = polygon("rectangle_obstacle",False)
               elif variant[0]=="rectangle":
                   [field_matrix,field_vertex] = polygon("rectangle",False)

               # Define the plant locations
               # Receive plant_matrix, row_nrs, row_edges
               if rowsbool:
                   [plant_matrix, row_nrs, row_edges, field_vertex] = withrows(field_matrix, 2, 1, field_vertex, False)

               else:
                   plant_matrix = norows(field_matrix, 2, False)

               # Compute the spread and uncertainty
               # Receive: spread_matrix, worldmodel_matrix, uncertainty_matrix
               rates=[]
               weedbool = variant[1]
               if not weedbool:
                   # Configure the spreading characteristics of the pathogen
                   # more aggressive
                   if variant[2] == "pathogen1":
                       pathogen = Pathogen(patchnr=3, infectionduration=4, spreadrange=5, reproductionfraction=0.5,reproductionrate=2, standarddeviation=0.3, saturation=5)
                   else:
                       # two patches
                       pathogen = Pathogen(patchnr=2, infectionduration=4, spreadrange=3, reproductionfraction=0.5,
                                        reproductionrate=2, standarddeviation=0.3, saturation=5)

                   spread_matrix, worldmodel_matrix, uncertainty_matrix = pathogenspread(field_matrix, plant_matrix,
                                                                                         pathogen, False)
                   for i in range(1, 13):
                       reproductionrate = np.random.normal(pathogen.reproductionrate, pathogen.reproductionrateSTD)
                       rates.append(reproductionrate)
               else:
                   if variant[2] == "weed1":
                       weed = Weed(patchnr=4, patchsize=7, spreadrange=3, reproductionrate=2, standarddeviation=1,
                                saturation=2, plantattach=False)
                   else:
                       weed = Weed(patchnr=2, patchsize=4, spreadrange=5, reproductionrate=4, standarddeviation=1,
                                    saturation=1, plantattach=False)

                   spread_matrix, worldmodel_matrix, uncertainty_matrix = weedsspread(field_matrix, plant_matrix, weed,
                                                                                      False)

                   for i in range(1,13):
                       reproductionrate= np.random.normal(weed.reproductionrate, weed.reproductionrateSTD)
                       rates.append(reproductionrate)

               # Execute planning
               # Receive: samplelocations
               if scenariovariants.index(variant)%4==0:

                   matrices = None  # initialize for first day
                   scenariosettings = None
                   total_days=1
                   for day in range(1, total_days + 1):
                       if rowsbool:
                           [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,
                            matrices,samplelocations] = rig_rows_matrix(
                               uncertainty_matrix, row_nrs, row_edges, field_vertex, scenariosettings, matrices)
                       else:
                           [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,
                            matrices,samplelocations] = rig_matrix(uncertainty_matrix, scenariosettings, matrices)

               # Save everything to the correct scenario location:
               # for variant2 in scenariovariants:
               #     if variant[0]==variant2[0]:
               if rowsbool:
                   pathname = str("../Testing_files/rows/") + str(variant[0]) + str(variant[2])
                   #pathname = str("../Testing_files/rows/") + str(variant2[0]) + str(variant2[2])
                   if not os.path.exists(pathname):
                       os.makedirs(pathname)
                   # with open(pathname+'/row_nrs.pickle', 'wb') as handle:
                   #     pickle.dump(row_nrs,handle,protocol=pickle.HIGHEST_PROTOCOL)
                   # with open(pathname+'/row_edges.pickle', 'wb') as handle:
                   #     pickle.dump(row_edges,handle,protocol=pickle.HIGHEST_PROTOCOL)
                   # with open(pathname+'/field_vertex.pickle', 'wb') as handle:
                   #     pickle.dump(field_vertex,handle,protocol=pickle.HIGHEST_PROTOCOL)
                   np.save(pathname + '/row_nrs.npy', row_nrs)
                   np.save(pathname + '/row_edges.npy', row_edges)
                   np.save(pathname + '/field_vertex.npy', field_vertex)
               else:
                   pathname = str("../Testing_files/norows/") + str(variant[0]) + str(variant[2])
                   #pathname = str("../Testing_files/norows/") + str(variant2[0]) + str(variant2[2])
                   if not os.path.exists(pathname):
                       os.makedirs(pathname)

               np.save(pathname + '/matrices0.npy', matrices[0])
               np.save(pathname + '/matrices1.npy', matrices[1])
               # np.save(pathname + '/matrices2.npy', matrices[2])
               # np.save(pathname + '/matrices3.npy', matrices[3])

               np.save(pathname + '/spread_matrix.npy',spread_matrix)
               np.save(pathname + '/worldmodel_matrix.npy',worldmodel_matrix)
               np.save(pathname + '/uncertainty_matrix.npy',uncertainty_matrix)
               show_map(spread_matrix, False, True, pathname,"/spread_matrix")
               del spread_matrix
               show_map(worldmodel_matrix, False, True, pathname,"/worldmodel_matrix")
               del worldmodel_matrix
               show_map(uncertainty_matrix, False, True, pathname,"/uncertainty_matrix")
               del uncertainty_matrix

               np.save(pathname + '/samplelocations.npy',samplelocations)
               np.save(pathname + '/reproductionrates.npy', rates)


               print("Finished saving for this scenario/ field")


    testing=False
    if testing:
        scenariolist = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8],
                        [2, 1], [2, 2], [2, 3], [2, 4],
                        [3, 1], [3, 2], [3, 3], [3, 4],
                        [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8],
                        [5, 1], [5, 2], [5, 3], [5, 4],
                        [6, 1], [6, 2], [6, 3], [6, 4]]
        for scenario in scenariolist:
            scenariosettings = getSettings(scenario)
            # scenariosettings = [rowsbool, budget, informed, rewiring, step_len, search_radius, stopsetting, horizonplanning]

            # variant = [field, weedbool, weed/pathogen type]
            rowsbool = scenariosettings[0]
            #for variant in scenariovariants:
            for variant in scenariovariants[-1]:

                if rowsbool:
                    pathname = str("Testing_files/rows/")+str(variant[0])+str(variant[2])
                else:
                    pathname = str("Testing_files/norows/") + str(variant[0]) + str(variant[2])
                    row_nrs = np.load(pathname+'/row_nrs.npy')
                    row_edges = np.load(pathname+'/row_edges.npy')
                    field_vertex = np.load(pathname+'/field_vertex.npy')
                spread_matrix= np.load(pathname+'/spread_matrix.npy')
                worldmodel_matrix= np.load(pathname+'/worldmodel_matrix.npy')
                uncertainty_matrix =  np.load(pathname+'/uncertainty_matrix.npy')

                matrices0 = np.load(pathname+'/matrices0.npy')
                matrices1 = np.load(pathname+'/matrices1.npy')
                # matrices2 = np.load(pathname+'/matrices2.npy')
                # matrices3 = np.load(pathname+'/matrices3.npy')
                matrices = [matrices0,matrices1]


                samplelocations = np.load(pathname+ '/samplelocations.npy')
                reproductionrates = np.load(pathname+ '/reproductionrates.npy')


                time_start = time.process_time()
                if scenario[0]==6:
                    for dailyuncertainty in [0,0.001,0.01]:
                        print("Long simulations")
                        time_start = time.process_time()
                        total_days = 12
                        for day in range(1, total_days + 1):
                            reproductionrate=reproductionrates[day-1]
                            if rowsbool:
                                [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,
                                 matrices_new, samplelocations_new] = rig_rows_matrix(
                                    uncertainty_matrix, row_nrs, row_edges, field_vertex, scenariosettings, matrices,samplelocations)
                            else:
                                [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,
                                 matrices_new, samplelocations_new] = rig_matrix(uncertainty_matrix, scenariosettings, matrices,samplelocations)

                            time_end = time.process_time()
                            totaltime = time_end - time_start
                            # Saving the results per day
                            if rowsbool:
                                pathname = str("Result_files/rows/") + str(scenario) + str("/") + str(variant[0]) + str(
                                    variant[2])
                            else:
                                pathname = str("Result_files/norows/") + str(scenario) + str("/") + str(
                                    variant[0]) + str(variant[2])
                            if not os.path.exists(pathname):
                                os.makedirs(pathname)
                            np.save(pathname + '/finalpath_'+str(day)+'.npy', finalpath)
                            np.save(pathname + '/infopath_'+str(day)+'.npy', infopath)
                            np.save(pathname + '/finalcost_'+str(day)+'.npy', finalcost)
                            np.save(pathname + '/finalinfo_'+str(day)+'.npy', finalinfo)
                            np.save(pathname + '/iteration_'+str(day)+'.npy', iteration)
                            np.save(pathname + '/runtime_'+str(day)+'.npy', totaltime)

                            np.save(pathname + '/spread_matrix_'+str(day)+'.npy', spread_matrix)
                            np.save(pathname + '/worldmodel_matrix_'+str(day)+'.npy', worldmodel_matrix)
                            np.save(pathname + '/uncertainty_matrix_'+str(day)+'.npy', uncertainty_matrix)
                            show_map(spread_matrix, False, True, pathname, "/spread_matrix_"+str(day))
                            show_map(worldmodel_matrix, False, True, pathname, "/worldmodel_matrix"+str(day))
                            show_map(uncertainty_matrix, False, True, pathname, "/uncertainty_matrix"+str(day))

                            disease = getDisease(variant[2])
                            [spread_matrix, worldmodel_matrix, uncertainty_matrix] = updatematrix(
                                disease, plant_matrix, spread_matrix, worldmodel_matrix, uncertainty_matrix, infopath,
                                0,dailyuncertainty, reproductionrate,False)

                else:
                    print("Single simulations")
                    # Running the simulation
                    if rowsbool:
                        [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,
                         matrices, samplelocations] = rig_rows_matrix(
                            uncertainty_matrix, row_nrs, row_edges, field_vertex, scenariosettings, matrices,samplelocations)
                    else:
                        [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,
                         matrices, samplelocations] = rig_matrix(uncertainty_matrix, scenariosettings, matrices,samplelocations)
                    time_end = time.process_time()
                    totaltime = time_end-time_start
                    # Saving the results
                    if rowsbool:
                        pathname = str("Result_files/rows/") + str(scenario) +str("/") + str(variant[0]) + str(variant[2])
                    else:
                        pathname = str("Result_files/norows/") + str(scenario) +str("/") + str(variant[0]) + str(variant[2])
                    if not os.path.exists(pathname):
                        os.makedirs(pathname)
                    np.save(pathname + '/finalpath.npy', finalpath)
                    np.save(pathname + '/infopath.npy', infopath)
                    np.save(pathname + '/finalcost.npy', finalcost)
                    np.save(pathname + '/finalinfo.npy', finalinfo)
                    np.save(pathname + '/iteration.npy', iteration)
                    np.save(pathname + '/runtime.npy', totaltime)



def default():
    time_start = time.process_time()

    # Choose the field shape:
    #[field_matrix,field_vertex] = polygon("hexagon_small",True)
    [field_matrix,field_vertex] = polygon("rectangle",False)
    #[field_matrix,field_vertex] = polygon("hexagon_convex",False)
    #[field_matrix,field_vertex] = polygon("hexagon_concave",True)
    #[field_matrix,field_vertex] = polygon("rectangle_obstacle",True)
    rowsbool = False
    if rowsbool:
        [plant_matrix,row_nrs,row_edges,field_vertex] = withrows(field_matrix,2,1,field_vertex,False)

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
    default()
    #prepandtest()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
