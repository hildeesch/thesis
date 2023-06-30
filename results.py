import numpy as np
import time
from copy import deepcopy
import os
import pickle
from main import getSettings


def analyze_results():
    scenariolist = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8],
                        [2, 1], [2, 2], [2, 3], [2, 4],
                        [3, 1], [3, 2], [3, 3], [3, 4],
                        [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8],
                        [5, 1], [5, 2], [5, 3], [5, 4],
                        [6, 1], [6, 2], [6, 3], [6, 4]]
    variants = [["..","",""]]
    results_overview=[]
    results_overview_long=[]
    for variant in variants:
        for scenario in scenariolist:
            scenariosettings=getSettings(scenario)
            rowsbool = scenariosettings[0]

            if rowsbool:
                pathname_var = str("../Testing_files/rows/") + str(variant[0]) + str(variant[2])
                pathname_res = str("Result_files/rows/") + str(scenario) + str("/") + str(variant[0]) + str(
                        variant[2])

            else:
                pathname_var = str("../Testing_files/norows/") + str(variant[0]) + str(variant[2])
                pathname_res = str("Result_files/norows/") + str(scenario) + str("/") + str(
                    variant[0]) + str(variant[2])

            if scenario[0]!=6: # single sim
                results_overview.append([])
                runtime = np.load(pathname_res+'/runtime.npy')
                info = np.load(pathname_res + '/totalinfo.npy')
                uncertainty_matrix = np.load(pathname_var + '/uncertainty_matrix.npy')
                infomap = np.nansum(uncertainty_matrix)
                performance = info/infomap

                # results_overview = [[[setting,performance,runtime],[setting,performance,runtime]],[
                results_overview[scenario[0]].append([scenario[1],rowsbool,performance,runtime])
            else: # long sim
                runtime = np.load(pathname_res + '/runtime_12.npy')
                info = np.load(pathname_res + '/totalinfo_12.npy')
                uncertainty_matrix = np.load(pathname_var + '/uncertainty_matrix_12.npy')
                infomap = np.nansum(uncertainty_matrix)
                performance = info / infomap
                worldmodel_matrix = np.load(pathname_var + '/worldmodel_matrix_12.npy')
                spread_matrix = np.load(pathname_var + '/spread_matrix_12.npy')
                worldmodel_accuracy = np.nansum(np.absolute(worldmodel_matrix-spread_matrix))
                # results_overview = [[[setting,performance,runtime],[setting,performance,runtime]],[
                results_overview_long.append([scenario, rowsbool, performance, runtime,worldmodel_accuracy])


    # printing the results
    # each result is a tested variable, with each entry a different setting
    for result in results_overview:
        for outcomes in result:
            if outcomes[0,1]==1:
                print("Scenario "+str(outcomes[0]))
                getSettings(outcomes[0])
                print("Runtime: "+str(outcomes[3])+" Performance: "+str(outcomes[2]))
    #for result in results_overview_long: (only one result, nr 6)
    for outcomes in results_overview_long:
        print("Scenario " + str(outcomes[0]))
        getSettings(outcomes[0])
        print("Runtime: " + str(outcomes[3]) + " Performance: " + str(outcomes[2])+" Worldmodel inaccuracy "+str(outcomes[4]))


