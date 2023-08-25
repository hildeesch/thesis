import numpy as np
import time
from copy import deepcopy
import os
import pickle
from main import getSettings
from heatmap import show_map
from monitortreat import updatematrix
from monitortreat import showpathlong

def visualize_results():
    scenariolist = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8],
                    [2, 1], [2, 2], [2, 3], [2, 4],
                    [3, 1], [3, 2], [3, 3], [3, 4],
                    [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8],
                    [5, 1], [5, 2], [5, 3], [5, 4],
                    [6, 1], [6, 2], [6, 3], [6, 4]]
    # variants = [["..","",""]]
    variants = [["rectangle", "something", "pathogen1"]]
    results_overview = []
    results_overview_long = []
    for variant in variants:
        for scenario in scenariolist[-4:]:
        #for scenario in [[7,1],[7,2]]:
            if scenario[0] == 7 or scenario[0] ==8:
                scenariosettings = getSettings([6, 2])
            else:
                scenariosettings = getSettings(scenario)
            rowsbool = scenariosettings[0]

            if rowsbool:
                pathname_var = str("../../Results/Testing_files/rows/") + str(variant[0]) + str(variant[2])
                pathname_res = str("../../Results/Result_files/rows/") + str(scenario) + str("/") + str(
                    variant[0]) + str(
                    variant[2])

            else:
                pathname_var = str("../../Results/Testing_files/norows/") + str(variant[0]) + str(variant[2])
                pathname_res = str("../../Results/Result_files/norows/") + str(scenario) + str("/") + str(
                    variant[0]) + str(variant[2])

            if scenario[0] != 6 and scenario[0]!= 7 and scenario[0]!=8:  # single sim
                spread_matrix = np.load(pathname_var + '/spread_matrix.npy')
                uncertainty_matrix = np.load(pathname_var + '/uncertainty_matrix.npy')
                path = np.load(pathname_res + '/finalpath.npy')

                #show_map(spread_matrix,True,False)
                plt = show_map(uncertainty_matrix,False,False)
                for i in range(len(path)-1):
                    plt.plot([path[i,0], path[i+1,0]], [path[i,1], path[i+1,1]], "-r")
                plt.show()


            else:  # long sim
                for day in range(1, 13):
                    pathname_res_day = pathname_res + '/' + str(day) + '_'
                    spread_matrix = np.load(pathname_res_day + 'spread_matrix.npy')
                    uncertainty_matrix = np.load(pathname_res_day + 'uncertainty_matrix.npy')
                    path = np.load(pathname_res_day + 'finalpath.npy')

                    #show_map(spread_matrix,True,False)
                    # plt = show_map(uncertainty_matrix,False,False)
                    # for i in range(len(path)-1):
                    #     plt.plot([path[i,0], path[i+1,0]], [path[i,1], path[i+1,1]], "-r")
                    # plt.show()
                figlong = None
                axlong = None
                for day in range(1,13):
                    pathname_res_day = pathname_res + '/' + str(day) + '_'
                    uncertainty_matrix = np.load(pathname_res_day + 'uncertainty_matrix.npy')
                    worldmodel_matrix = np.load(pathname_res_day + 'worldmodel_matrix.npy')
                    spread_matrix=np.load(pathname_res_day+'spread_matrix.npy')
                    path = np.load(pathname_res_day + 'finalpath.npy')
                    [figlong, axlong] = showpathlong(day, 12, figlong, axlong, uncertainty_matrix, path,
                                                     None, None, None, None, None, None,
                                                     True, False)


def analyze_results():
    scenariolist = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8],
                        [2, 1], [2, 2], [2, 3], [2, 4],
                        [3, 1], [3, 2], [3, 3], [3, 4],
                        [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8],
                        [5, 1], [5, 2], [5, 3], [5, 4],
                        [6, 1], [6, 2], [6, 3], [6, 4]]
    #variants = [["..","",""]]
    variants = [["rectangle","something","pathogen1"]]
    variants = [["rectangle","something","pathogen2"]]
    results_overview=[]
    results_overview_long=[]
    for variant in variants:
        #for scenario in scenariolist[:12]:
        for scenario in [[8,1],[8,2]]:
        #for scenario in [[9,1],[9,2],[9,3],[9,4]]:
            if scenario[0]==7 or scenario[0]==8:
                scenariosettings = getSettings([6,2])
            else:
                scenariosettings = getSettings(scenario)
            rowsbool = scenariosettings[0]


            if rowsbool:
                pathname_var = str("../../Results/Testing_files/rows/") + str(variant[0]) + str(variant[2])
                pathname_res = str("../../Results/Result_files/rows/") + str(scenario) + str("/") + str(variant[0]) + str(
                        variant[2])

            else:
                pathname_var = str("../../Results/Testing_files/norows/") + str(variant[0]) + str(variant[2])
                pathname_res = str("../../Results/Result_files/norows/") + str(scenario) + str("/") + str(
                    variant[0]) + str(variant[2])

            if scenario[0]!=6 and scenario[0]!=7 and scenario[0]!=8: # single sim
                # results_overview.append([])
                runtime = np.load(pathname_res+'/runtime.npy')
                info = np.load(pathname_res + '/finalinfo.npy')
                cost = np.load(pathname_res + '/finalcost.npy')
                if scenario[0]==9:
                    plant_matrix = np.load(str("../Testing_files/rows/")+str(variant[0])+str('pathogen1') + '/plant_matrix.npy', allow_pickle=True)
                    uncertainty_matrix = deepcopy(plant_matrix)
                    uncertainty_matrix[uncertainty_matrix >= 0.0] = 0.001
                else:
                    uncertainty_matrix = np.load(pathname_var + '/uncertainty_matrix.npy')
                infomap = np.nansum(uncertainty_matrix)
                if scenario==[5,1] or scenario==[5,2]:
                    i_list = np.load(pathname_res+'/k_list.npy')
                    print(len(i_list))
                    print(i_list)
                    print("Performance before rewiring: "+str(i_list[-1]/infomap))
                if scenario == [2, 2] or scenario==[2,4]:
                    infopathsteps = np.load(pathname_res + '/infopath.npy')
                    infopath=[]
                    info=0
                    for step in infopathsteps:
                        if [step[0],step[1]] not in infopath:
                            print(step)
                            print(infopath)
                            infopath.append([step[0],step[1]])
                            info+=uncertainty_matrix[step[1],step[0]]

                performance = info/infomap

                # results_overview = [[[setting,performance,runtime],[setting,performance,runtime]],[
                #results_overview[scenario[0]].append([scenario[1],rowsbool,performance,runtime])
                results_overview.append([scenario, rowsbool, performance, runtime,cost])
            else: # long sim
                # reproductionrates = np.load(pathname_var + '/reproductionrates.npy')
                # print("Reproductionrates:")
                # print(reproductionrates)
                performancelist=[]
                runtimelist=[]
                totaldays=12
                for day in range(1,totaldays+1):
                    pathname_res_day = pathname_res+'/'+str(day)+'_'
                    samplelocations = np.load(pathname_res_day + 'samplelocations.npy')
                    print("Day "+str(day)+" Samplelocations: ")
                    print(samplelocations)
                    runtimelist.append(np.load(pathname_res_day + 'runtime.npy'))
                    info = np.load(pathname_res_day + 'finalinfo.npy')
                    cost = np.load(pathname_res_day + 'finalcost.npy')
                    if scenario == [7, 2] or scenario == [8, 2]:
                        pathname_res_inf = str("../../Results/Result_files/norows/") + str([scenario[0],1]) + str("/") + str(
                            variant[0]) + str(variant[2])+'/'+str(day)+'_'
                        uncertainty_matrix = np.load(pathname_res_inf + 'uncertainty_matrix.npy')
                        infopathsteps = np.load(pathname_res_day + 'infopath.npy')
                        infopath = []
                        info = 0
                        for step in infopathsteps:
                            if [step[0], step[1]] not in infopath:
                                print(step)
                                print(infopath)
                                infopath.append([step[0], step[1]])
                                info += uncertainty_matrix[step[1], step[0]]
                        infomap = np.nansum(uncertainty_matrix)
                    else:
                        uncertainty_matrix = np.load(pathname_res_day + 'uncertainty_matrix.npy')
                        infomap = np.nansum(uncertainty_matrix)
                    performancelist.append(info / infomap)
                    worldmodel_matrix = np.load(pathname_res_day + 'worldmodel_matrix.npy')
                    spread_matrix = np.load(pathname_res_day + 'spread_matrix.npy')
                    runtime=np.sum(runtimelist)/totaldays
                    performance=np.sum(performancelist)/totaldays
                print(runtimelist)
                worldmodel_accuracy = np.nansum(np.absolute(worldmodel_matrix-spread_matrix))/np.nansum(spread_matrix)
                # results_overview = [[[setting,performance,runtime],[setting,performance,runtime]],[
                results_overview_long.append([scenario, rowsbool, [performance,infomap], runtime,worldmodel_accuracy,cost])

    # printing the results
    # each result is a tested variable, with each entry a different setting
    #for result in results_overview:
    if True:
        result = results_overview
        for outcomes in result:
            if outcomes[0][1]==1:
                print("NEW TESTING VAR")
            print("Scenario "+str(outcomes[0]))
            getSettings(outcomes[0])
            print("Runtime: "+str(outcomes[3])+" Performance: "+str(outcomes[2])+" Cost: "+str(outcomes[-1]))
    #for result in results_overview_long: (only one result, nr 6)
    for outcomes in results_overview_long:
        print("Scenario " + str(outcomes[0]))
        if outcomes[0][0]!=7 and outcomes[0][0]!=8:
            getSettings(outcomes[0])
        print("Runtime: " + str(outcomes[3]) + " Performance: " + str(outcomes[2])+" Worldmodel inaccuracy "+str(outcomes[4]))


if __name__ == '__main__':
    #analyze_results()
    visualize_results()