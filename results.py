import numpy as np
import time
from copy import deepcopy
import os
import pickle
from main import getSettings
from heatmap import show_map
from monitortreat import updatematrix
from monitortreat import showpathlong
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

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


from itertools import product
def load_comparison_results(base_path="Result_files/small/comparison/", n_iterations=20):
    all_data = []
    methods = ['_method', '_uninformed', '_coverage']
    scenario_combinations = list(product([1, 2, 3], repeat=4))  # 81 combinations

    # for it in range(n_iterations):
    i=1
    for it in [2,5,6,8,9,12,13,15,16,17]:
        for scenario in scenario_combinations:
            for method in methods:
                scenario_path = os.path.join(base_path, str(it), str(scenario) + method)
                try:
                    finalinfo = np.load(os.path.join(scenario_path, 'finalinfo.npy')).item()
                    finalcost = np.load(os.path.join(scenario_path, 'finalcosts.npy')).item()
                    time_total = np.load(os.path.join(scenario_path, 'computingtime.npy')).item()
                    totalinfo = np.load(os.path.join(scenario_path, 'totalinfomatrix.npy')).item()
                except FileNotFoundError:
                    print(i, " File not found for scenario: ", scenario_path)
                    i+=1
                    continue  # Incomplete run

                # Decode scenario params
                budget_map = {1: 10, 2: 15, 3: 25}
                gauss_nr_map = {1: 1, 2: 3, 3: 10}
                gauss_size_map = {1: 1, 2: 2, 3: 5}
                robot_map = {1: 1, 2: 2, 3: 5}

                budget = budget_map[scenario[0]]
                gauss_nr = gauss_nr_map[scenario[1]]
                gauss_size = gauss_size_map[scenario[2]]
                robots = robot_map[scenario[3]]

                all_data.append({
                    "iteration": it,
                    "method": method.strip('_'),
                    "budget_percent": budget,
                    "gaussians_nr": gauss_nr,
                    "gaussians_size": gauss_size,
                    "robots": robots,
                    "finalinfo": finalinfo,
                    "finalcost": finalcost,
                    "normalized_info": finalinfo / totalinfo if totalinfo > 0 else np.nan,
                    "normalized_cost": (finalcost / ((399/100)*budget*robots)),
                    "runtime_sec": time_total
                })

    df = pd.DataFrame(all_data)
    return df

def analyze_results_new(test="comparison"):
    # Optional: Use a cleaner font and style
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 6,
    })
    # Color palette
    palette = sns.color_palette("deep")


    if test=="comparison":
        scenariolist = list(product([1, 2, 3], repeat=4))
        for it in range(10):
            file =  open("Result_files/scenario_overview.txt", "r")

            for scenario in scenariolist:
                scenario_description = file.readline()
                print(scenario_description)


                # Our method
                pathname = str("Result_files/") + str(scenario) + str("_method_")+str(it)+str("/")
                finalinfo = np.load(pathname + 'finalinfo.npy')
                finalcost = np.load(pathname + 'finalcosts.npy')
                time_total = np.load(pathname + 'computingtime.npy')
                print("Our method, info: ", finalinfo, " Cost: ",finalcost, " Time: ",time_total)

                # Uninformed method
                pathname = str("Result_files/") + str(scenario) + str("_uninformed_")+str(it)+str("/")
                finalinfo = np.load(pathname + 'finalinfo.npy')
                finalcost = np.load(pathname + 'finalcosts.npy')
                time_total = np.load(pathname + 'computingtime.npy')
                print("Uninformed, info: ", finalinfo, " Cost: ",finalcost, " Time: ",time_total)

                # Budgeted coverage
                pathname = str("Result_files/") + str(scenario) + str("_coverage_")+str(it)+str("/")
                finalinfo = np.load(pathname + 'finalinfo.npy')
                finalcost = np.load(pathname + 'finalcosts.npy')
                time_total = np.load(pathname + 'computingtime.npy')
                print("Coverage, info: ", finalinfo, " Cost: ",finalcost, " Time: ",time_total)

                # Total info in the map:
                pathname = str("Result_files/") + str(scenario) +str("_")+str(it)+str("/") 
                uncertainty_matrix_sum = np.load(pathname + 'totalinfomatrix.npy')    
                print("Totalinfo: ", uncertainty_matrix_sum,)
                print("") #whiteline between scenarios
            file.close()
    elif test=="increasing_iterations":
        scenario = [1,2,2,1]                            
        for it in range(10):
            for budget in [200,500]:
                for robots in [1,2,5]:
                    print("Scenario - Budget: ",budget," Robots: ",robots)
                    for iterations in [200,250,300,350,400,450,500]:
                        pathname = str("Result_files/increasing_iterations/") + str(scenario) + str("_method_b")+str(budget)+"_r"+str(robots)+"_it"+str(iterations)+"_"+str(it)+str("/")
                        finalinfo = np.load(pathname + 'finalinfo.npy')
                        finalcost = np.load(pathname + 'finalcosts.npy')
                        time_total = np.load(pathname + 'computingtime.npy')
                        print("it: ",iterations," Info: ", finalinfo, " Cost: ",finalcost, " Time: ",time_total)

                    # Total info in the map:
                    uncertainty_matrix_sum = np.load(pathname + 'totalinfomatrix.npy')    
                    print("Totalinfo: ", uncertainty_matrix_sum,)
                    print("") #whiteline between scenarios
    elif test == "comparison_small":
        analysis_options = {0:"avg",1:"scenario",2:"top"}
        analysis = analysis_options[1]
        print("Analysis style: ", analysis)
        df = load_comparison_results()
        # Average performance per method over all scenarios
        if analysis=="avg":
            analysis_outcome = df.groupby('method')[['normalized_info', 'finalcost', 'runtime_sec']].describe()
            print(analysis_outcome)

        # Or by scenario configuration
        if analysis=="scenario":
            # Pivot table to show methods side-by-side
            # Normal sort
            #pivot = df.groupby(['budget_percent', 'gaussians_nr', 'gaussians_size', 'robots', 'method'])['normalized_info'].mean().unstack('method')
            # Sort by robot nr first
            pivot = df.groupby(['robots','budget_percent', 'gaussians_nr', 'gaussians_size', 'method'])['normalized_info'].mean().unstack('method')

            # Determine best method per scenario
            pivot['best_method'] = pivot.apply(determine_best_method, axis=1)

            # Reset index to convert MultiIndex to columns
            analysis_outcome = pivot.reset_index()

            # --- Run and append per-scenario ANOVA results ---
            analysis_outcome = add_per_scenario_anova(df, analysis_outcome)


            print(analysis_outcome.to_markdown(index=False))
            # analysis_outcome = df.groupby(['method', 'budget_percent', 'gaussians_nr', 'gaussians_size', 'robots'])['normalized_info'].mean().unstack(level=0)
            # print(analysis_outcome.to_markdown())

            # Count best_method values (excluding ties)
            # Assuming ties contain '+' or are not exact matches to a single method name
            individual_wins = analysis_outcome[
                ~analysis_outcome['best_method'].str.contains(r'\+', na=False)
            ]['best_method'].value_counts()

            # Print summary
            print("\nBest method counts (ties excluded):")
            print(individual_wins.to_string())


        # Visualizing top scenarios:
        if analysis=="top":
            top = df.groupby(['method', 'budget_percent', 'gaussians_nr', 'gaussians_size', 'robots']) \
            ['normalized_info'].mean().reset_index().sort_values("normalized_info", ascending=False)
            print(top)
            sns.barplot(data=top.head(10), x="normalized_info", y="method", hue="robots")
            plt.title("Top 10 Configurations by Normalized Info")
            plt.tight_layout()
            plt.show()


    elif test == "increasing_iterations_small":
        results = []
        # ---------- LOAD DATA ----------
        scenario = [1,2,2,1]                            
        for it in range(0,40):
            for budget in [round((399/100)*10),round((399/100)*25)]:
                for robots in [1,2,5]:
                    print("Iteration: ", it)
                    print("Scenario - Budget: ",budget," Robots: ",robots)
                    for iterations in [200,150,125,100,75,50,25]:
                        pathname = str("Result_files/small/increasing_iterations/") +str(it)+"/"+ str(scenario) + str("_method_b")+str(budget)+"_r"+str(robots)+"_"+str(iterations)+str("/")
                        finalinfo = np.load(pathname + 'finalinfo.npy')
                        finalcost = np.load(pathname + 'finalcosts.npy')
                        time_total = np.load(pathname + 'computingtime.npy')
                        totalinfo = np.load(pathname + 'totalinfomatrix.npy')
                        print("it: ",iterations," Info: ", finalinfo, " Cost: ",finalcost, " Time: ",time_total)

                        path = np.load(pathname + "finalpath.npy")

                        
                        results.append({
                            "budget": budget,
                            "robots": robots,
                            "iterations": iterations,
                            "finalinfo": finalinfo,
                            "finalcost": float(finalcost.item()),
                            "normalized_cost": (finalcost / (budget*robots)),
                            "time_total": float(time_total.item()),
                            "normalized_info": (finalinfo / totalinfo)
                        })
                        #print(path)

                    # Total info in the map:
                    uncertainty_matrix_sum = np.load(pathname + 'totalinfomatrix.npy')    
                    print("Totalinfo: ", uncertainty_matrix_sum,)
                    print("") #whiteline between scenarios

        # ---------- CREATE DATAFRAME ----------
        df = pd.DataFrame(results)

        # Convert numerical budget back to percentages (optional)
        df["budget_percent"] = df["budget"] / 399 * 100

        # ---------- PLOTTING ----------
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        y_metric = "normalized_info"
        # y_metric = "normalized_cost"
        # y_metric = "time_total"
        for ax, budget_level in zip(axes, [10, 25]):
            df_budget = df[round(df["budget_percent"]) == budget_level] # filter by budget level

            # Lines:
            # for robot_count in sorted(df_budget["robots"].unique()):
            #     sub = df_budget[df_budget["robots"] == robot_count]
            #     grouped = sub.groupby("iterations")[y_metric]
            #     means = grouped.mean()
            #     stds = grouped.std()

            #     ax.plot(means.index, means.values, label=f"{robot_count} robots")
            #     ax.fill_between(means.index, means - stds, means + stds, alpha=0.2)

            # Points: 
            for idx, robot_count in enumerate(sorted(df_budget["robots"].unique())):

                sub = df_budget[df_budget["robots"] == robot_count]
                means = sub.groupby("iterations")[y_metric].mean()
                stds = sub.groupby("iterations")[y_metric].std()

                ax.errorbar(
                    means.index,
                    means,
                    yerr=stds,
                    label=f"{robot_count} robots",
                    marker='o',           # shows points
                    capsize=4,            # small horizontal line at top of error bar
                    linestyle='-',        # optional: line connecting the points
                    elinewidth=1.5,        # width of error bar lines
                    color=palette[idx]
                )
                ax.fill_between(means.index, means - stds, means + stds, alpha=0.2, color=palette[idx])

            ax.set_title(f"Budget {budget_level}%")
            ax.set_xlabel("Iterations")
            ax.grid(True)
            #ax.legend(title="Robots")
            if y_metric != "time_total":
                ax.set_ylabel(y_metric.replace("_", " ").capitalize())
                ax.set_ylim(0, 1)
            else:
                ax.set_ylabel("Time (s)")
            ax.legend(title="Robots", frameon=True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        if y_metric=="normalized_info":
            fig.suptitle("Performance over Iterations", fontsize=16, weight="bold")
        elif y_metric == "normalized_cost":
            fig.suptitle("Path Costs over Iterations", fontsize=16, weight="bold")
        elif y_metric=="time_total":
            fig.suptitle("Computing Time over Iterations", fontsize=16, weight="bold")
        else:
            fig.suptitle("??? over Iterations", fontsize=16, weight="bold")
        plt.tight_layout()
        plt.show()

    # Reading out results of one case
    # rounds = np.load(str("Result_files/small/increasing_iterations/[1, 2, 2, 1]_method_b40_r5_it200_1/rounds.npy"))
    # path = np.load(str("Result_files/small/increasing_iterations/[1, 2, 2, 1]_method_b40_r5_it200_1/finalpath.npy"))
    # for i in range(len(rounds)):
    #     print(path[i],rounds[i])
    # time = np.load(str("Result_files/small/increasing_iterations/[1, 2, 2, 1]_method_b40_r2_it200_0/computingtime.npy"))
    # print("Time: ", time)

from scipy.stats import f_oneway

def add_per_scenario_anova(df, scenario_df, scenario_columns=['robots','budget_percent', 'gaussians_nr', 'gaussians_size']):
    """
    Performs one-way ANOVA per scenario and appends results (p-value, significance) to scenario_df.
    """
    pvals = []
    significance = []

    for _, row in scenario_df.iterrows():
        # Extract scenario values
        scenario_values = {col: row[col] for col in scenario_columns}

        # Subset original data to match scenario
        mask = (df[scenario_columns] == pd.Series(scenario_values)).all(axis=1)
        group = df[mask]

        # Group normalized_info by method
        grouped = group.groupby("method")["normalized_info"].apply(list)

        # Only proceed if at least 2 methods have values
        if grouped.count() < 2:
            pvals.append(np.nan)
            significance.append(False)
            continue

        try:
            f_stat, p_val = f_oneway(*grouped)
            pvals.append(p_val)
            significance.append(p_val < 0.05)
        except Exception:
            pvals.append(np.nan)
            significance.append(False)

    scenario_df['anova_pvalue'] = pvals
    scenario_df['significant'] = significance
    return scenario_df

# Function to determine best method, handling ties
def determine_best_method(row):
    max_val = row.max()
    best_methods = row[row == max_val].index.tolist()
    if len(best_methods) == 1:
        return best_methods[0]
    else:
        return '+'.join(best_methods)  # or return 'tie' or None if preferred

def recompute_uninformed_finalinfo(base_path="Result_files/small/comparison", iterations=range(1, 20)):
    scenariolist = list(product([1, 2, 3], repeat=4))

    for it in iterations:
        for scenario in scenariolist:
            scenario_str = str(scenario)
            scenario_basepath = os.path.join(base_path, str(it), scenario_str)

            if os.path.exists(scenario_basepath):
                print(scenario_basepath)
                uncertainty_path = os.path.join(scenario_basepath, 'uncertainty_matrix.npy')
                uninformed_path = os.path.join(base_path, str(it), scenario_str + "_uninformed")

                # Paths for existing files
                infopath_file = os.path.join(uninformed_path, 'infopath.npy')
                finalinfo_file = os.path.join(uninformed_path, 'finalinfo.npy')

                # Check required files exist
                if not (os.path.exists(uncertainty_path) and os.path.exists(infopath_file)):
                    print("Required files do not exist: ", uncertainty_path)
                    continue

                finalinfo_prev = np.load(finalinfo_file).item()
                # Load files
                uncertainty_matrix = np.load(uncertainty_path)
                infopath = np.load(infopath_file, allow_pickle=True)

                # Recalculate finalinfo
                nodelist = []
                finalinfo = 0
                for gridpoint in infopath:
                    if tuple(gridpoint) not in nodelist:
                        finalinfo += uncertainty_matrix[gridpoint[1], gridpoint[0]]
                        nodelist.append(tuple(gridpoint))

                # Save corrected finalinfo
                np.save(finalinfo_file, finalinfo)
                print(f"Updated finalinfo for {uninformed_path} = {finalinfo:.2f} instead of {finalinfo_prev:.2f}")


if __name__ == '__main__':
    #analyze_results()
    # visualize_results()
    #analyze_results_new()
    #analyze_results_new("increasing_iterations_small")
    analyze_results_new("comparison_small")
    #recompute_uninformed_finalinfo(iterations=[2,5,8,12,16])