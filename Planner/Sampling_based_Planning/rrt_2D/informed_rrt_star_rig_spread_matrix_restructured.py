"""
INFORMED_RRT_STAR 2D
@author: huiming zhou
"""
import time
import os
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import matplotlib.patches as patches
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from datetime import datetime


#from Sampling_based_Planning.rrt_2D import env, plotting, utils
from Planner.Sampling_based_Planning.rrt_2D import env, plotting, utils
from Planner.Sampling_based_Planning.rrt_2D import dubins_path as dubins
from Planner.Sampling_based_Planning.rrt_2D import reeds_shepp as reedsshepp

from copy import deepcopy

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.cost = 0 # cost upto this node
        self.info = 0
        self.totalcost = 0 # including the distance to the goal point
        self.totalinfo = 0 # including the last part to the goal point
        # multi-robot:
        self.round=1
        self.prevroundcost=0

    def __eq__(self, other):
        return isinstance(other, Node) and self.x == other.x and self.y == other.y and self.parent == other.parent

    def __hash__(self):
        return hash((self.x, self.y, self.parent))
    
    def has_same_position(self,other):
        return (self.x==other.x and self.y==other.y)

class IRrtStar:
    def __init__(self, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max,uncertaintymatrix,scenario,matrices,samplelocations):
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max

        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils(uncertaintymatrix)

        self.delta = self.utils.delta
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.V = [self.x_start]
        self.X_soln = [self.x_start]
        self.path = None

        self.inforadius = 0 # monitor radius around the robot (0 = no radius, so only at the location of the robot)

        # scenario = [rowsbool, budget, informed, rewiring, step_len, search_radius, stopsetting, horizonplanning]
        if scenario:
            self.budget = scenario[1]
            self.boolrewiring=scenario[3]
            self.stopsetting=scenario[6]
            self.multirobot=scenario[7] # either False (=single), or nr of robots (2+)
            self.pathname = scenario[-1]
        else:
            self.budget=350
            self.boolrewiring=True
            self.stopsetting="mild"
            self.multirobot=False
            self.pathname=[]
        #self.scenario=scenario
        self.samplelocations=samplelocations
        #self.samplelocations_add=False

        if len(samplelocations)>0:
            self.samplelocations_add=False
            #print(self.samplelocations)
        else:
            self.samplelocations_add = True
            self.samplelocations=[]

        self.uncertaintymatrix = uncertaintymatrix
        self.max_info = np.nansum(uncertaintymatrix)

        self.x_best = self.x_start # just for now, remove later
        self.i_best = 0


        # for saving of cost/info matrices
        self.infopathmatrix = np.empty((100 * 100, 100 * 100), dtype=object)
        self.infomatrix = np.empty((100 * 100, 100 * 100)) 
        if matrices is None:
            self.costmatrix = np.empty((100 * 100, 100 * 100))
            self.anglematrix = np.empty((100 * 100, 100 * 100))
        else:
            self.costmatrix = matrices[0]
            self.anglematrix=matrices[1]


        self.time = np.zeros(8) # for debugging
        # 0 = sample, 1 = nearest, 2 = steer, 3 = near, 4 = rewiring, 5 = lastpath, 6 = pruning, 7 = totaltime


    def init(self):
        self.x_best = self.x_start
        self.show = False
        self.print = False
        self.visualizationmode ="False" #steps, nosteps or False
        self.rewiringafter = True #TODO: why is this there? doesn't seem intuitive
        return 

    def planning(self):
        self.init()
        totalstarttime=time.time()

        #self.x_best = self.init()
        x_best = self.x_best
        c_best = np.inf
        i_best = 0.001
        startlen=0 # for checking node increase
        sampled_locations = set()

        k=0
        double=False # for viz purposes it is defined here
        while k<self.iter_max:
            k+=1
            if k==25 and self.visualizationmode=="steps":
                self.visualizationmode="nosteps" # change it after x iterations for less visualizations
            if k>=3-3: #only evaluate from when we might want it to stop
                cost = {node: node.totalcost for node in self.X_soln}
                info = {node: node.totalinfo for node in self.X_soln}
                #x_best = min(cost, key=cost.get)
                if len(info)>0 and not double:
                    self.x_best = max(info, key=info.get)
                    x_best = max(info, key=info.get)
                    #c_best = cost[x_best]
                    i_last_best = i_best
                    self.i_best = info[x_best]
                    if self.i_best == self.max_info: # stop if we reached all the info in the map
                        break



                    #print("i_best: "+str(i_best)+" i_last_best: "+str(i_last_best)+" Criterion value: "+str(((i_best-i_last_best)*100/i_last_best)))
                    # if ((i_best-i_last_best)/i_last_best)<0.001: #smaller than 1% improvement
                    #     count_down-=1
                    # else:
                    #     count_down=20 #reset
                    #     print("reset countdown")
            # if k==302 and "[5," not in self.pathname: # to test up to certain iteration
            #     #count_down=0
            #     stopcriterion=True

            if k%50==0 and not double:
                print("ATTENTION!!! ATTENTION!!! ATTENTION!!! AGAIN FIFTY CYCLES FURTHER, CURRENT CYCLE ="+str(k)) # to know how far we are
            endlen=len(self.V)
            if self.print:
                print("Nr of nodes added: "+str(endlen-startlen))
            # if (endlen-startlen)==0 and not double:
            #     k-=1
            #     print("Len X_Near was: "+str(len(self.Near(self.V,x_new))))
            startlen = len(self.V)

            if (k-1)%25==0 and self.visualizationmode=="nosteps" and k>1 and not double:  # visualize all new connections with near ndoes
                self.fig, self.ax = plt.subplots()
                self.animation_new(k-1, x_new)
                #self.animation(k-1, x_new)
                if self.pathname:
                    plt.savefig(self.pathname + "animation_"+str(k-1))
                if self.show:
                    plt.show()
                else:
                    plt.close()


            timestart=time.time()
            #x_rand = self.SampleFreeSpace()
            if self.samplelocations_add:
                x_rand = self.SampleFreeSpace()
                self.samplelocations.append([x_rand.x, x_rand.y])
            else:
                x_rand = Node((self.samplelocations[k-1]))
                if double: # otherwise we keep trying the same location again and again
                    x_rand = self.SampleFreeSpace()
            if self.visualizationmode=="steps" and not double:  # visualize all new connections with near ndoes
                self.fig, self.ax = plt.subplots()
                # self.animation(k, x_rand)
                self.animation_new(k, x_rand)
                if self.pathname:
                    plt.savefig(self.pathname + "animation_"+str(k-1)+"_1_nearnodes")                
                if self.show:
                    plt.show()
                else:
                    plt.close()
            timeend=time.time()
            self.time[0]+=(timeend-timestart)
            timestart=time.time()
            x_nearest = self.Nearest(self.V, x_rand)
            timeend = time.time()
            self.time[1] += (timeend - timestart)
            timestart=time.time()
            x_new = self.Steer(x_nearest, x_rand) #so that we only generate one new node, not multiple
            timeend = time.time()
            self.time[2] += (timeend - timestart)
            #if self.Cost(x_nearest) + self.Line(x_nearest, x_rand) + self.Line(x_rand, self.x_goal) > self.budget:
                #just for debugging purposes (for now)
                #print("Past the budget")
            double=False
            if (x_new.x,x_new.y) in sampled_locations:  # co-located nodes
                double=True #there is already a node at this location, so we skip it
                # print("double")
                k-=1
                if self.samplelocations_add:
                    self.samplelocations.pop()
            #if x_nearest.cost + self.Line(x_nearest, x_new) + self.Line(x_new, self.x_goal) < self.budget and not double:  # budget check for nearest parent (to make it more efficient)
            if not double:  # budget check for nearest parent (to make it more efficient)
                # print(x_nearest.cost + self.Line(x_nearest, x_new) + self.Line(x_new, self.x_goal))
                node_new=[]
                sampled_locations.add((x_new.x,x_new.y))

                # doubleroundstrategy:
                # if day == 1:
                #     print(len(self.Near(self.V, x_new)))

                #for x_near in self.Near(self.X_soln,x_new):
                for x_near in self.Near(self.V,x_new):
                    node_new = Node((x_new.x, x_new.y))
                    node_new.parent = x_near  # added
                    dist = self.get_distance_and_angle(x_near,node_new)[0]
                    c_min = x_near.cost + dist
                    #endcost = self.get_distance_and_angle(node_new,self.x_goal)[0] #TODO: doesn't seem used
                    
                    node_new.cost = c_min 
                    node_new.info = self.Info_cont(node_new)
                    self.V.append(node_new) #generate a "node"/trajectory to each near point
                    # multirobot strategy
                    if node_new.parent.round>1:
                        node_new.round = node_new.parent.round
                        node_new.prevroundcost=node_new.parent.prevroundcost

                    node_new.totalcost=node_new.cost+ self.Line(node_new,self.x_goal)
                    # Simplified condition for both single and multirobot:
                    if (node_new.totalcost-node_new.prevroundcost) <= (self.budget):
                        timestart = time.time()
                        self.LastPath(node_new)
                        timeend = time.time()
                        self.time[5] += (timeend - timestart)
                        self.X_soln.append(node_new)
                    # if node_new.totalcost <= self.budget:  # extra check for budget for actual parent
                    #     self.X_soln.append(node_new)
                    # # tworoundstrategy2:
                    # if self.multirobot:
                    #     if node_new.round > 1 and (node_new.totalcost-node_new.prevroundcost) <= (self.budget):
                    #         # if a node_new is in round 2, it is allowed to have a double budget
                    #         # to make sure we don't surpass the original budget in round 1, we add that every parent also has to be a solution
                    #         self.X_soln.append(node_new)
                # Visualization before pruning
                if self.visualizationmode=="steps" and not double: # visualize all new connections with near ndoes
                    self.fig, self.ax = plt.subplots()
                    # self.animation(k,x_new,3)
                    self.animation_new(k,x_new,3)
                    if self.pathname:
                        plt.savefig(self.pathname + "animation_"+str(k-1)+"_2_preprune")
                    if self.show:
                        plt.show()
                    else:
                        plt.close()
                # Pruning:
                if self.visualizationmode=="steps" and not double: # make them red to see which ones go away
                    self.fig, self.ax = plt.subplots()
                    # self.animation(k,x_new,1)
                    self.animation_new(k,x_new,1)
                if node_new!=[]: # so it has actually been assigned
                    timestart= time.time()
                    self.Pruning(node_new) 
                    timeend = time.time()
                    self.time[6] += (timeend - timestart)                #tworoundstrategy2: first prune, then create a start of a new round for each node of the first round
                if self.multirobot:
                    #self.RoundTwoAdd(node_new)
                    self.MultiRobotAdd(x_new)
                timestart = time.time()
                self.Rewiring_new(x_new)
                timeend= time.time()
                self.time[4] += (timeend-timestart)
                #print("node_new: ("+str(node_new.x)+","+str(node_new.y)+")")


                if node_new!=[]: # so it has actually been assigned
                    # self.Pruning(node_new)
                    if self.visualizationmode=="steps" and not double: # show all connections after pruning
                        # self.animation(k, x_new,2)
                        self.animation_new(k, x_new,2)
                        if self.pathname:
                            plt.savefig(self.pathname + "animation_"+str(k-1)+"_3_postprune")
                        if self.show:
                            plt.show()
                        else:
                            plt.close()
            if k % 50 == 0 and not double:
                if self.show and not self.visualizationmode:
                #    self.animation()
                   self.animation_new()
                self.time[7] = time.time()-totalstarttime
                print(self.time)
                if k>0:
                    print("It.: ", k, " Time: " ,self.time[7], " Info: ",x_best.info, " Tot. info: ",x_best.totalinfo, " Cost: ",x_best.cost, " Totalcost: ",x_best.totalcost," Round: ",x_best.round," Nodes: ",len(self.V))
                    #print("It.: " + str(k) + " Time: " + str(self.time[7]) + " Info: " + str(x_best.info) + " Tot. info: "+str(x_best.totalinfo) + " Cost: " + str(x_best.cost) + " Totalcost: "+str(x_best.totalcost) +" Nodes: "+str(len(self.V)))
                    #print("Check check: i_best = ", i_best)


        # Rewiring in Hindsight:
        top10info = [] #to see effect of rewiring
        if self.rewiringafter or not self.multirobot and self.boolrewiring: #TODO: these ifs don't make sense to me
            #info = {node: node.totalinfo for node in self.X_soln}
            info = {node: node.totalinfo for node in self.X_soln if node.round == self.multirobot}
            if len(info)==0:
                info = {node: node.totalinfo for node in self.X_soln}
            topinfo = sorted(info, key=info.get)[-(1)].totalinfo

            for i in range(min(200,len(info))):  # rewire the x best nodes
                # self.x_best = max(info, key=info.get)
                curnode = sorted(info, key=info.get)[-(i + 1)]
                previnfo=curnode.totalinfo
                if curnode.totalinfo*1.2<topinfo:
                    print("Stop rewiring nodes at i = "+str(i))
                    break
                curnode = self.rewiring_afterv2(curnode, self.multirobot)
                top10info.append([round(previnfo),round(curnode.totalinfo),round(((curnode.totalinfo-previnfo)/previnfo),2)])
                # ,round(curnode.totalinfo-previnfo),round(((curnode.totalinfo-previnfo)/previnfo),2)
            if self.print:
                print("Rewiring, info changes:")
                print(top10info)
            info = {node: node.totalinfo for node in self.X_soln}
            self.x_best = max(info, key=info.get)
            x_best = max(info, key=info.get)
            print("Best node after rewiring: tot. info: " + str(x_best.totalinfo) + " Cost: " + str(x_best.totalcost))
        if self.multirobot and not self.rewiringafter and self.boolrewiring:
            multirobot=False # to make sure it rewires in the correct way
            top20nodes=[] #top 10 nodes split in two rounds each
            #info = {node: node.totalinfo for node in self.X_soln}
            info = {node: node.totalinfo for node in self.X_soln if node.round == self.multirobot}
            topinfo = sorted(info, key=info.get)[-(1)].totalinfo
            for i in range(min(200,len(info))):  # rewire the 10 best nodes
                # self.x_best = max(info, key=info.get)
                curnode = sorted(info, key=info.get)[-(i + 1)]
                if curnode.totalinfo*1.2<topinfo:
                    print("Stop rewiring nodes at i = "+str(i))
                    break
                [nodefirstround, nodesecondround] = self.splitDoublePath(curnode)
                if nodefirstround not in top20nodes:
                    nodefirstround = self.rewiring_afterv2(nodefirstround,self.multirobot)
                    top20nodes.append(nodefirstround)
                if nodesecondround: # not None
                    nodesecondround = self.rewiring_afterv2(nodesecondround,self.multirobot)
                    top20nodes.append(nodesecondround)
                else:
                    print("No second round to be rewired")
            info = {node: node.totalinfo for node in top20nodes}
            self.x_best = max(info, key=info.get)
            x_best = max(info, key=info.get)
            print("Best node after rewiring: tot. info: " + str(x_best.totalinfo) + " Cost: " + str(x_best.totalcost))
            multirobot=True # set it back again

        # Extracting the path
        #self.path = self.ExtractPath(x_best)
        #[self.path,nodes] = self.ExtractPath(x_best)
        ## ADDED (TODO: sort out this part)
        # multi_info = {node: node.totalinfo for node in self.X_soln if (node.round == self.multirobot and node.totalcost>node.prevroundcost)}
        # x_check = max(multi_info, key=multi_info.get)
        # print("Multiround: Best node after rewiring: tot. info: " + str(x_check.totalinfo) + " Cost: " + str(x_check.totalcost)+" Prevcost: "+str(x_check.prevroundcost)+ " Pos: "+ str(x_check.x)+str(x_check.y)+" Parent round: "+str(x_check.parent.round))
        # #
        # Changed: for final node selection, select the path with highest info and lowest cost (in case there are multiple with same info)
        # Find the maximum totalinfo value
        max_info = max(node.totalinfo for node in self.X_soln)
        
        # Select nodes with the maximum totalinfo
        best_candidates = [node for node in self.X_soln if node.totalinfo == max_info]
        
        # Among these, select the node with the lowest totalcost
        x_best = min(best_candidates, key=lambda node: node.totalcost)

        # Old approach (not considering lowest cost:)
        # info = {node: node.totalinfo for node in self.X_soln}
        # x_best = max(info, key=info.get)

        # as it was:
        [[self.path,path_rounds],infopath] = self.ExtractPath(x_best)

        node = x_best
        ## BEGIN OF VIZ
        # self.animation()
        if self.print:
            self.fig, self.ax = plt.subplots()
            self.animation_new()
            plt.plot(x_best.x, x_best.y, "bs", linewidth=3)
            plt.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
            # making the second part of two-day paths another colour
            secondround = False
            for index, cell in enumerate(self.path):
                if secondround:
                    plt.plot([self.path[index - 1][0], self.path[index][0]],
                                [self.path[index - 1][1], self.path[index][1]], "-m")
                if [cell[0], cell[1]] == [self.x_start.x, self.x_start.y] and index > 0 and [self.path[index - 1][0],
                                                                                                self.path[index - 1][
                                                                                                    1]] != [self.x_start.x,
                                                                                                            self.x_start.y] and secondround == False:
                    secondround = True
                    # print("Second round starts at index "+str(index))

            #doubleroundstrategy:
            # if path_firstround!=self.path:
            #     print("First round is not equal to the final path")
            #     plt.plot([x for x, _ in path_firstround], [y for _, y in path_firstround], '-c')
            # else:
            #     print("First round is equal to the final path")
            #     print(len(self.X_soln),len(self.X_soln_prev))




            #plt.plot([x for x, _ in x_best.infopath], [y for _, y in x_best.infopath], '-b')
            #plt.plot([x for x, _ in x_best.lastinfopath], [y for _, y in x_best.lastinfopath], '-c')
            #plt.plot([x for x, _ in self.path[:2]],[y for _, y in self.path[:2]], '-k') # to see whether the path actually ends at the goal
            plt.pause(0.01)
            if self.show:
                plt.show()
            if self.pathname:
                plt.savefig(self.pathname + "final_bestpath")

            fig, ax = plt.subplots()
            colormap = cm.Blues
            colormap.set_bad(color='black')
            im= ax.imshow(self.uncertaintymatrix, colormap, vmin=0, vmax=3, origin='lower')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            # for node in nodes:
            #     ax.plot(node[0], node[1], marker=(8, 2, 0),color="blue", linewidth=3, markersize=20)
            for cell in infopath:
                ax.plot(cell[0],cell[1],marker="o",markersize=1,color="blue")
            #ax.plot(x_best.x, x_best.y, marker=(8, 2, 0), color="green", linewidth=3, markersize=20)
            ax.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
            # making the second part of two-day paths another colour
            secondround = False
            if self.multirobot:
                for index, cell in enumerate(self.path):
                    if secondround:
                        plt.plot([self.path[index - 1][0], self.path[index][0]],
                                    [self.path[index - 1][1], self.path[index][1]], "-m")
                    if [cell[0], cell[1]] == [self.x_start.x, self.x_start.y] and index > 0 and [self.path[index - 1][0],
                                                                                                    self.path[index - 1][
                                                                                                        1]] != [self.x_start.x,
                                                                                                                self.x_start.y]:
                        secondround = True

            #doubleroundstrategy:
            # if path_firstround!=self.path:
            #     ax.plot([x for x, _ in path_firstround], [y for _, y in path_firstround], '-c')


            ax.set_title("Spatial distribution of uncertainty and final path")
            #fig.tight_layout()
            if self.show:
                plt.show()
            if self.pathname:
                plt.savefig(self.pathname + "final_distribution")
            # Note: now that we removed stopcriterion, we don't have the k_list anymore

        # k_list_avg_der_neg=[]
        # i_list_avg_der_neg=[]
        # k_list_avg_der2_neg = []
        # i_list_avg_der2_neg = []
        # for index in range(len(self.k_list)):
        #     if self.i_list_avg_der[index]<=0:
        #         k_list_avg_der_neg.append(self.k_list[index])
        #         i_list_avg_der_neg.append(self.i_list_avg_der[index])
        #     if self.i_list_avg_der2[index]<=0:
        #         k_list_avg_der2_neg.append(self.k_list[index])
        #         i_list_avg_der2_neg.append(self.i_list_avg_der2[index])

        # fig, ax = plt.subplots(2, 2)
        # ax[0, 0].scatter(self.k_list, self.i_list, s=0.5)
        # ax[0, 0].set_title("Best node info")
        # ax[0, 1].scatter(self.k_list, self.i_list_avg, s=0.5)
        # ax[0, 1].set_title("10 best nodes info (avg)")
        # ax[1, 0].scatter(self.k_list, self.i_list_avg_der, s=0.5)
        # ax[1,0].scatter(k_list_avg_der_neg, i_list_avg_der_neg,s=0.5,c='r')
        # ax[1, 0].set_title("10 best nodes info increase 10 it")
        # ax[1, 1].scatter(self.k_list, self.i_list_avg_der2, s=0.5)
        # ax[1, 1].set_title("10 best nodes info increase 10 der2")
        # plt.show()

        # fig, ax = plt.subplots(4,3)
        # ax[0,0].scatter(k_list, i_list,s=0.5)
        # ax[0,0].set_title("Best node info")
        # ax[0,1].scatter(k_list, i_list_inc,s=0.5)
        # ax[0,1].set_title("Best node info increase")
        # ax[0,2].scatter(k_list, i_list_perc,s=0.5)
        # ax[0,2].set_title("Best node info increase %")
        # ax[1,0].scatter(k_list, i_list_10,s=0.5)
        # ax[1,0].set_title("10 best nodes info (avg)")
        # #ax[1,1].scatter(k_list, i_list_inc_10)
        # #ax[1,1].set_title("10 best nodes info increase (avg)")
        # #ax[1,2].scatter(k_list, i_list_perc_10)
        # #ax[1,2].set_title("10 best nodes info increase %")
        # ax[2, 1].scatter(k_list, i_list_inc_10_avg,s=0.5)
        # ax[2, 1].set_title("10 best nodes info increase 10 it")
        # #ax[2, 2].scatter(k_list, i_list_perc_10_avg)
        # #ax[2, 2].set_title("10 best nodes info increase % 10 it")
        # ax[3, 1].scatter(k_list, i_list_inc_10_avg_der,s=0.5)
        # ax[3, 1].set_title("10 best nodes info increase 10 der2")
        # #ax[3, 2].scatter(k_list, i_list_perc_10_avg_der)
        # #ax[3, 2].set_title("10 best nodes info increase % 10 der2")
        # #ax.legend(["Best node info","Best node increase","Best node increase %","10 nodes info","10 nodes increase","10 nodes increase %"])
        # #ax.grid()
        # plt.show()


        # fig, ax = plt.subplots(2, 1)
        # ax[0].scatter(self.k_list, self.i_list_avg_der,s=0.5)
        # ax[0].scatter(k_list_avg_der_neg, i_list_avg_der_neg,s=0.5,c='r')
        # ax[0].set_title("10 best nodes info increase 10 it")
        # ax[0].grid()
        # ax[0].set_ylim((-0.05,0.05))
        # ax[1].scatter(self.k_list, self.i_list_avg_der2,s=0.5)
        # ax[1].scatter(k_list_avg_der2_neg, i_list_avg_der2_neg,s=0.5,c='r')
        # ax[1].set_title("10 best nodes info increase 10 der2")
        # ax[1].grid()
        # ax[1].set_ylim((-0.05,0.05))

        # plt.show()
        # END OF VIZ


        #doubleroundstrategy:
        # node = x_best
        # while node.parent:
        #     if [node.parent.x,node.parent.y]==[self.x_goal.x,self.x_goal.y]:
        #         x_best=node.parent
        #         break
        #     node = node.parent
        # [self.path,infopath] = self.ExtractPath(x_best)
        # self.budget=self.budget/2

        #tworoundstrategy2:

        #check recalculating functionality (just for debugging)
        # node = x_best
        # while node.parent:
        #     node=node.parent
        #     if node.parent==self.x_start and node.parent.cost==0:
        #         break
        # self.Recalculate(node)
        # print("Recalculating check: totalinfo double path: "+str(x_best.totalinfo))

        #x_best=nodefirstround

        #check the info score of the second part only:
        # if self.multirobot and self.rewiringafter:

        #     [nodefirstround,nodesecondround]=self.splitDoublePath(node)

        #     # node = nodesecondround
        #     # while node.parent:
        #     #     print(node.info)
        #     #     node=node.parent
        #     if nodefirstround.totalinfo>nodesecondround.totalinfo or nodesecondround==None: # round 1 is better or there is no actual second round
        #         x_best=nodefirstround
        #         [self.path, infopath] = self.ExtractPath(nodefirstround)
        #         print("Executed path is round 1")

        #     else:
        #         x_best=nodesecondround
        #         [self.path, infopath] = self.ExtractPath(nodesecondround)
        #         print("Executed path is round 2")
            # print("node first round: "+str(nodefirstround.x),str(nodefirstround.y)+" node second round: "+str(nodesecondround.x),str(nodesecondround.y))
            # print(len(infopathfirst))
            # print(len(infopathsecond))
            # print(len(infopath))
            # infopathsecondcheck=[]
            # infosecondcheck=0
            # for cell in infopathsecond:
            #     if cell not in infopathsecondcheck:
            #         infopathsecondcheck.append(cell)
            #         infosecondcheck+=self.uncertaintymatrix[cell[1],cell[0]]
            # print("Second info check = "+str(infosecondcheck))
        # final return
        matrices= [self.costmatrix,self.anglematrix]
        # if self.pathname:
        #     np.save(self.pathname + 'k_list.npy', self.k_list)
        #     np.save(self.pathname + 'i_list.npy', self.i_list)
        #     np.save(self.pathname + 'i_list_avg_der.npy', self.i_list_avg_der)
        #     np.save(self.pathname + 'i_list_avg_der2.npy', self.i_list_avg_der2)
        return [self.path,path_rounds], infopath, x_best.totalcost, x_best.totalinfo, self.budget, self.step_len, self.search_radius, k, matrices, self.samplelocations


    def splitDoublePath(self,initial_node):
        if initial_node.round==1 or ((initial_node.totalcost-initial_node.prevroundcost)==0): # there is no second round
            print("No second round: Initial node totalcost: ", initial_node.totalcost," Prev round cost: ",initial_node.prevroundcost, "Pos: ",initial_node.x, initial_node.y, "Round: ", initial_node.round, "Round parent: ", initial_node.parent.round)

            return initial_node,None
        node=initial_node
        prev_copynode = None
        print("Initial node totalcost: ", initial_node.totalcost," Prev round cost: ",initial_node.prevroundcost, "Pos: ",initial_node.x, initial_node.y, "Round: ", initial_node.round, "Round parent: ", initial_node.parent.round)
        while node.parent:
            # copynode = deepcopy(node) # just now
            copynode = Node((node.x, node.y))
            copynode.info = node.info
            copynode.cost = node.cost
            copynode.totalinfo = node.totalinfo
            copynode.totalcost = node.totalcost
            if prev_copynode:
                prev_copynode.parent = copynode
                if ([copynode.x,copynode.y]==[self.x_goal.x,self.x_goal.y] and copynode.totalcost>0):
                    startsecondround = copynode
                if ([prev_copynode.x, prev_copynode.y] == [self.x_goal.x, self.x_goal.y] and prev_copynode.cost>0):
                    nodefirstround = copynode
            # copynode.parent = copyparent
            if node==initial_node:
                copyinitial_node=copynode
            self.V.append(copynode)
            self.X_soln.append(copynode)

            prev_copynode = copynode
            node = node.parent

        prev_copynode.parent = self.x_start  # for the last one

        nodesecondround=copyinitial_node
        startsecondround.parent=self.x_start

        startsecondround.cost = startsecondround.cost - initial_node.prevroundcost
        startsecondround.info = self.Info_cont(startsecondround)
        self.LastPath(startsecondround)
        startsecondround.prevroundcost=0
        self.Recalculate(startsecondround)
        if nodefirstround.cost==0:
            return nodesecondround,None
        elif nodesecondround.cost==0:
            return nodefirstround,None
        print("Total info first round: " + str(nodefirstround.totalinfo) + " Info: " + str(
            nodefirstround.info) + " Cost: " + str(nodefirstround.totalcost))
        print("Total info second round (on its own): " + str(nodesecondround.totalinfo) + " Info: " + str(
             nodesecondround.info) + " Cost: " + str(nodesecondround.totalcost))
        return nodefirstround,nodesecondround

    def FindInfo(self, node_end_x,node_end_y,node_start_x,node_start_y,node,distance,totalpath=True):
        #node_end = the goal or new node
        #node_start = the (potential) parent
        #currentinfopath = the infopath of the parent (node_start)
        #distance = the distance between the nodes (e.g. self.step_len or search_radius)
        idx_start = node_start_y * 100 + node_start_x
        idx_end = node_end_y * 100 + node_end_x

        info = self.infomatrix[idx_start,idx_end]
        infopath = self.infopathmatrix[idx_start,idx_end]
        infopath_set = set()
        #infopath=None
        if [node_start_x,node_start_y]==[node_end_x,node_end_y]:
            infopath = [[node_start_x,node_start_y]]
            info=self.uncertaintymatrix[node_start_y,node_start_x]
            self.infomatrix[idx_start, idx_end] = info
            self.infomatrix[
                idx_end, idx_start] = info  # mirror the matrix
            self.infopathmatrix[idx_start, idx_end] = infopath[::]
            self.infopathmatrix[idx_end, idx_start] = infopath[
                                                                                                    ::-1]  # mirror the matrix

        if infopath is None:

            dt = 1 / (2 * distance)
            t = 0
            info = 0
            infopath = []
            while t < 1.0:
                xline = node_end_x - node_start_x
                yline = node_end_y - node_start_y
                xpoint = round(node_start_x + t * xline)
                ypoint = round(node_start_y + t * yline)
                if not (xpoint,ypoint) in infopath_set:
                    value = self.uncertaintymatrix[ypoint, xpoint]
                    if not np.isnan(value):
                        info += value
                        infopath.append([xpoint, ypoint])
                        infopath_set.add((xpoint, ypoint))
                    if np.isnan(value):  # to prevent going through edges and/or obstacles
                        self.costmatrix[idx_start,idx_end] = np.inf
                if self.inforadius>0:
                    for rowdist in range(-self.inforadius,self.inforadius+1):
                        for coldist in range(-self.inforadius,self.inforadius+1):
                            if (coldist**2+rowdist**2)<=self.inforadius**2: #radius
                                xpoint_=xpoint+coldist
                                ypoint_=ypoint+rowdist
                                if not (xpoint_, ypoint_) in infopath and not np.isnan(self.uncertaintymatrix[ypoint_, xpoint_]):
                                    info += self.uncertaintymatrix[ypoint_, xpoint_]
                                    infopath.append([xpoint_, ypoint_]) #TODO: decide if we want to add the nan points to the infopath or not (in that case we need to change some stuff below)
                                    infopath_set.add((xpoint_,ypoint_))
                t += dt


            self.infomatrix[idx_start,idx_end] = info
            self.infomatrix[idx_end,idx_start] = info # mirror the matrix
            self.infopathmatrix[idx_start,idx_end] = infopath[::]
            self.infopathmatrix[idx_end,idx_start] = infopath[::-1] # mirror the matrix
        #if totalpath: #if we want to append the current path to the new path
        infonode = 0
        currentinfopath=[]
        curnode = node

        while curnode.parent:
            #print("index 1 = " + str(curnode.parent.y*100+curnode.parent.x) + " index 2 = " + str(
            #    curnode.y*100+curnode.x))
            currentinfopath.extend(self.infopathmatrix[curnode.parent.y*100+curnode.parent.x,curnode.y*100+curnode.x])
            curnode= curnode.parent
        #if True: # just for debugging purposes now
        if not any(element in currentinfopath for element in infopath): # the whole infopath is new
            infonode+=info
        else: #if some infopoints overlap
            for element in infopath:
                if not element in currentinfopath:
                    infonode+=self.uncertaintymatrix[element[1],element[0]]

        return infonode
        # else:
        #     return info

    def Recalculate(self,parent):
        for node in self.V:  # to recalculate the cost and info for nodes further down the line
            if node.parent == parent:

                if node==self.x_best:
                    previnfo=node.info
                    prevtotalinfo=node.totalinfo

                dist = self.Line(parent, node)
                node.info = parent.info + self.FindInfo(node.x, node.y, parent.x, parent.y, parent,
                                                            dist, True)
                # if (parent.cost+dist-parent.prevroundcost) > self.budget and (node in self.X_soln):
                #     print("ERROR ERROR ERROR: Recalculating - old cost: ",node.cost," New cost: ",(parent.cost+dist))
                node.cost = parent.cost + dist
                self.LastPath(node)
                if node.round>1:
                    if [node.x,node.y]==[self.x_start.x,self.x_start.y]: # start of new round
                        node.prevroundcost=node.totalcost
                    else:
                        node.prevroundcost=node.parent.prevroundcost

                if (node not in self.X_soln) and ((node.totalcost-node.prevroundcost)<=self.budget):
                    self.X_soln.append(node)
                # else: # only nodes in x_soln can have children
                #     self.Recalculate(node)  # recalculates the cost and info for nodes further down the path
                self.Recalculate(node)  # recalculates the cost and info for nodes further down the path

    def Recalculate_selected(self,parent):
        # print("Recalculate inside for node ", parent.x,parent.y)
        for node in self.candidates:  # to recalculate the cost and info for nodes further down the line
            if node.parent == parent: 
                dist = self.Line(parent, node)
                node.info = parent.info + self.FindInfo(node.x, node.y, parent.x, parent.y, parent,
                                                            dist, True)
                node.cost = parent.cost + dist
                self.LastPath(node)
                if node.round>1:
                    if [node.x,node.y]==[self.x_start.x,self.x_start.y]: # start of new round
                        node.prevroundcost=node.totalcost
                    else:
                        node.prevroundcost=node.parent.prevroundcost

                if (node not in self.X_soln) and ((node.totalcost-node.prevroundcost)<=self.budget):
                    self.X_soln.append(node)
                # else: # only nodes in x_soln can have children
                #     self.Recalculate(node)  # recalculates the cost and info for nodes further down the path
                self.Recalculate_selected(node)  # recalculates the cost and info for nodes further down the path
    def Rewiring_new(self,x_new): # adapt for multi-robot
        for x_near in self.Near(self.V, x_new, self.search_radius,False):
            if x_near!=self.x_start and not x_near.parent.has_same_position(self.x_start): # not the start of the route and not the start of the next round
                c_near = x_near.cost
                c_new = x_near.parent.parent.cost + self.Line(x_near.parent.parent, x_new) + self.Line(x_new,x_near)            
                # I think this doesn't apply in our case:
                # if x_new.parent.x==x_near.x and x_new.parent.y==x_near.y:
                #     return # if the parent of x_new = x_near, we don't want to make the parent of x_near = x_new (because then we create a loose segment

                # determine the gained info in the rewiring segment
                i_cur_segment = x_near.info - x_near.parent.parent.info

                if c_new < c_near and i_cur_segment==0: #note: this is different than the condition in pruning
                    # Create the new node at the newly sampled location
                    newnode = Node((x_new.x, x_new.y))
                    newnode.parent = x_near.parent.parent
                    newnode.info = x_near.parent.parent.info + self.FindInfo(x_new.x, x_new.y,
                                                                            x_near.parent.parent.x,
                                                                                x_near.parent.parent.y, 
                                                                                x_near.parent.parent,
                                                                                self.search_radius,
                                                                                True)
                    newnode.cost = x_near.parent.parent.cost + self.Line(x_near.parent.parent, x_new)
                    self.V.append(newnode)
                    newnode.totalcost=newnode.cost+ self.Line(newnode,self.x_goal)
                    if ((newnode.totalcost-newnode.prevroundcost) <= (self.budget)):
                        self.LastPath(newnode)
                        self.X_soln.append(newnode)

                    info = newnode.info + self.FindInfo(x_near.x, x_near.y, newnode.x, newnode.y, newnode,
                                                        self.search_radius, True)
                    # Rewire the near node to the new node
                    x_near.cost = c_new
                    x_near.info = info
                    x_near.parent = newnode
                    self.LastPath(x_near)
                    if x_near.round>1: # TODO: check if this logic is sound and if there are more cases like this
                        if [x_near.x,x_near.y]==[self.x_start.x,self.x_start.y]: # start of new round
                            x_near.prevroundcost=x_near.totalcost
                            newnode.prevroundcost = x_near.prevroundcost
                        else:
                            x_near.prevroundcost=x_near.parent.parent.prevroundcost
                            newnode.round=x_near.parent.parent.round
                            newnode.prevroundcost=x_near.parent.parent.prevroundcost
                    if (x_near not in self.X_soln) and (x_near.totalcost<=self.budget): #TODO: add second round thing (budget)
                        self.X_soln.append(x_near)
                    # else: # only nodes in x_soln can have children
                    #     self.Recalculate(x_near)  # recalculates the cost and info for nodes further down the path
                    self.Recalculate(x_near)  # recalculates the cost and info for nodes further down the path

                    # print("Rewiring took place!! New cost: ", x_near.cost, " Old cost: ", c_near)        

    def rewiring_afterv2(self, best_node, doubleround): #rewiring afterwards
        # goal: gain more info while remaining within the budget
        if self.print:
            print("Start rewiring after v2")
            print("Position: ["+str(best_node.x),str(best_node.y)+"] Info: " + str(best_node.info) + " Tot. info: " + str(
                best_node.totalinfo) + " Cost: "+str(best_node.cost)+" Total cost: " + str(best_node.totalcost)+" First round cost:"+ str(best_node.prevroundcost))

        bestpath=[]
        infosteps=[]
        node=best_node

        # multirobot:
        #costfirstround=best_node.prevroundcost
        prevroundcosts={}
        prevroundcosts[best_node.round+1]=best_node.totalcost
        prev_copynode=None
        self.candidates=[]

        #TODO i don't understand what we do in this while loop (why add all the nodes again as solutions )

        while node!=self.x_start:
            #copynode = deepcopy(node) # just now
            if node.round not in prevroundcosts:
                prevroundcosts[node.round]=node.prevroundcost
            copynode = Node((node.x, node.y))
            if prev_copynode:
                prev_copynode.parent=copynode
            #copynode.parent = copyparent
            copynode.info = node.info
            copynode.cost = node.cost
            copynode.totalinfo = node.totalinfo
            copynode.totalcost = node.totalcost
            copynode.round = node.round
            copynode.prevroundcost = node.prevroundcost
            if node==best_node:
                copybest_node=copynode
            self.V.append(copynode)
            self.X_soln.append(copynode)
            self.candidates.append(copynode)
            # if len(bestpath)>0:
            #     bestpath[-1].parent=copynode # just now
            bestpath.append(copynode) # just now
            #bestpath.append(node)
            #infosteps.append(node.info-node.parent.info) # increase of info
            costincrease = node.cost - node.parent.cost
            if costincrease == 0:
                infosteps.append(0)
            else:
                infosteps.append((node.info-node.parent.info)/(node.cost-node.parent.cost)) # density of increase of info

            # the infosteps contain the added info for the parent to the node (of that node)
            # tworoundstrategy2:
            # if self.multirobot:
            #     if costfirstround==0 and [node.x,node.y]==[self.x_goal.x,self.x_goal.y]:
            #         costfirstround=node.totalcost
            prev_copynode=copynode
            node=node.parent

        prev_copynode.parent=self.x_start # for the last one
        best_node=copybest_node
        if node.round not in prevroundcosts: # in case round 1 == 0
            prevroundcosts[node.round]=node.prevroundcost
        roundcosts={}
        for i in range(max(prevroundcosts)-1):
            round=i+1
            roundcosts[round] = prevroundcosts[round+1]-prevroundcosts[round]

        # if prevroundcosts==0 or not doubleround:
        #     #print("Rewiring: updated first round cost")      
        #     costfirstround=best_node.totalcost


        pastindexes = []
        totalinfo = best_node.totalinfo
        notfinished = True
        sortedinfo = sorted(infosteps)

        i=1
        while notfinished:
            bestindex = infosteps.index(sortedinfo[-1])
            sortindex=1
            while (bestindex in pastindexes) or (bestindex==len(bestpath)-1):
                sortindex+=1
                if sortindex >= len(bestpath):  # we have had every piece
                    notfinished = False
                    if self.print:
                        print("Completed rewiring v2")
                    break
                bestindex = infosteps.index(sortedinfo[-sortindex])

            if not notfinished: # to prevent an extra loop
                break
            pastindexes.append(bestindex)
            node = bestpath[bestindex]
            # so we rewire the part that is at the moment most profitable

            #print("Best index: "+str(bestindex))
            #print(node.parent.parent.x, node.parent.parent.y)
            checked_locations=set()

            for x_near in self.Near(self.V, node, self.search_radius*2,False):
            #for x_near in self.Near(self.V, node, 10):
                if not (x_near.has_same_position(node)) and not ((x_near.x,x_near.y) in checked_locations) and not (node.parent.has_same_position(self.x_goal)):
                    checked_locations.add((x_near.x,x_near.y))
                    x_temp = Node((node.x, node.y))
                    x_temp.parent = node.parent
                    # self.child_map[node.parent] = x_temp
                    x_temp.info = node.info
                    x_temp.cost = node.cost
                    x_temp.totalinfo = node.totalinfo
                    x_temp.totalcost = node.totalcost
                    x_temp.prevroundcost = node.prevroundcost
                    x_temp.round = node.round

                    #if node.parent != self.x_start and node != self.x_start:  # because otherwise there's no "old" path to go back to
                    c_old = node.cost
                    c_new = node.parent.parent.cost + self.Line(node.parent.parent, x_near) + self.Line(x_near,node)

                    # if x_new.parent.x==x_near.x and x_new.parent.y==x_near.y:
                    #     return # if the parent of x_new = x_near, we don't want to make the parent of x_near = x_new (because then we create a loose segment
                    #if (c_new-c_old) < (self.budget-best_node.totalcost): # still within budget
                    # tworoundstrategy2:
                    #costsecondround = best_node.totalcost-costfirstround
                    if ((c_new-c_old)<(self.budget-roundcosts[node.round])): # still within budget

                        newnode = Node((x_near.x, x_near.y))
                        newnode.parent = node.parent.parent

                        newnode.info = node.parent.parent.info + self.FindInfo(node.parent.parent.x,
                                                                                    node.parent.parent.y, x_near.x,
                                                                                    x_near.y,
                                                                                    node.parent.parent,
                                                                                    self.search_radius,
                                                                                    True)
                        dist = self.Line(node.parent.parent, x_near)
                        newnode.cost = node.parent.parent.cost + dist
                        newnode.round = newnode.parent.round
                        newnode.prevroundcost = newnode.parent.prevroundcost
                        self.LastPath(newnode)
                        info = newnode.info + self.FindInfo(x_near.x, x_near.y, node.x, node.y, newnode,
                                                            self.search_radius, True)


                        info_old = node.info

                        info_new = info
                        if info_new > info_old:  # note: this is different than the condition in pruning
                            self.V.append(newnode)
                            self.candidates.append(newnode)
                            if ((newnode.totalcost-newnode.prevroundcost)<=self.budget):
                                #self.X_soln.add(newnode)
                                self.X_soln.append(newnode)

                            # rewiring:

                            node.parent = newnode
                            node.info = info_new
                            node.cost = c_new

                            self.LastPath(node)  # also recalculate the last info part


                            # Scenario 1:
                            # else:
                            # print("Recalculate for node ", node.x,node.y,bestindex )
                            self.Recalculate_selected(node)  # recalculates the cost and info for nodes further down the path
                            if totalinfo>=best_node.totalinfo:
                                # reverse rewiring
                                node.parent = x_temp.parent
                                node.info = x_temp.info
                                node.cost = x_temp.cost
                                node.totalinfo = x_temp.totalinfo
                                node.totalcost = x_temp.totalcost
                                node.prevroundcost = x_temp.prevroundcost
                                node.round = x_temp.round
                                # print("Recalculate reverse")
                                self.Recalculate_selected(node)
                            else:
                                bestpath[bestindex + 1] = newnode
                                # tworoundstrategy2:
                                # if self.multirobot:
                                #     if node.round == 1:
                                #         costfirstround += (c_new - c_old)
                                #     else:  # round 2:
                                #         costsecondround += (c_new - c_old)
                                # else:
                                #     costfirstround += (c_new - c_old)
                                roundcosts[node.round] += (c_new - c_old)

                                if self.print:
                                    print("Improved path through hindsight rewiring with increase in info: "+str(best_node.totalinfo-totalinfo))
                                totalinfo=best_node.totalinfo
                            # # bit of debugging:
                            # thisnode=x_new
                            # while thisnode.parent:
                            #     thisnode=thisnode.parent
                            # if not (thisnode.x==self.x_start.x and thisnode.y==self.x_start.y):
                            #     print("WARNING WARNING WARNING LOOSE END LOOSE END LOOSE END LOOSE END LOOSE END LOOSE END AT X_NEAR = ("+str(x_near.x)+","+str(x_near.y)+")")

                            # x_near.infopath = infopath
                            # print("Rewiring took place!!")
                        else:
                            del newnode
                    # else:
                    #     notfinished=False
                # i+=1
                # if i==len(bestpath):
                #     notfinished=False
        #print("End rewiring after v2")
        if self.print:
            print("Info: " + str(best_node.info) + " Tot. info: " + str(
                best_node.totalinfo) + " Cost: " + str(best_node.totalcost))
        # if best_node.totalcost>=self.budget:
        #     print("costfirstround var = "+str(costfirstround))
        return best_node


    # def Pruning(self, x_new):
    #     # print("Pre pruning we have ", len(self.V)," nodes")
    #     pos_key = (x_new.x, x_new.y)
    #     nodelist_complete = [node for node in self.V if (node.x, node.y) == pos_key]
    #     to_remove = set()

    #     for round in range(max(1,self.multirobot)): # once if for single robot, else for x nr robots
    #         nodelist = [n for n in nodelist_complete if n.round == (round + 1)]
    #         # Sort: lowest cost, then highest info (optimization of the iterations)
    #         nodelist.sort(key=lambda n: (n.cost, -n.info))
    #         # now the pruning
    #         for i, node1 in enumerate(nodelist):
    #             if node1 in to_remove:
    #                 continue
    #             for j in range(i + 1, len(nodelist)):
    #                 node2 = nodelist[j]
    #                 if node2 in to_remove:
    #                     continue

    #                 #if (node2.cost<=node1.cost and node2.info>node1.info): #prune lesser paths or doubles
    #                 # most recent optimized:
    #                 same_parent = (node1.parent == node2.parent)
    #                 dominates = (
    #                     (node1.cost <= node2.cost and node1.info > node2.info) or
    #                     (node1.cost < node2.cost and node1.info == node2.info) or
    #                     (same_parent)
    #                 )
    #                 if (dominates): #prune lesser paths or doubles
    #                     to_remove.add(node2)
    #     for node in to_remove:
    #         if node == self.x_best:
    #             print("[PRUNING] Best node is removed, info: " + str(self.x_best.totalinfo))
    #         if node in self.V:
    #             self.V.remove(node)
    #         if node in self.X_soln:
    #             self.X_soln.remove(node)
    #     # print("Post pruning we have ", len(self.V)," nodes")
    def Pruning(self, x_new):
        pos_key = (x_new.x, x_new.y)
        nodelist_complete = [node for node in self.V if (node.x, node.y) == pos_key]

        # Sort once: by round, then cost (asc), then info (desc)
        nodelist_complete.sort(key=lambda n: (n.round, n.cost, -n.info))

        # Group nodes by round
        by_round = defaultdict(list)
        for node in nodelist_complete:
            by_round[node.round].append(node)

        # Parallelizable pruning logic per round
        def prune_round(nodelist):
            to_remove = set()
            for i, node1 in enumerate(nodelist):
                if node1 in to_remove:
                    continue
                for j in range(i + 1, len(nodelist)):
                    node2 = nodelist[j]
                    if node2 in to_remove:
                        continue

                    same_parent = (node1.parent == node2.parent)
                    dominates = (
                        (node1.cost <= node2.cost and node1.info > node2.info) or
                        (node1.cost < node2.cost and node1.info == node2.info) or
                        same_parent
                    )
                    if dominates:
                        to_remove.add(node2)
            return to_remove

        with ThreadPoolExecutor() as executor:
            removals = list(executor.map(prune_round, by_round.values()))

        # Combine all nodes to be removed
        to_remove = set().union(*removals)

        # Use sets for fast deletion
        V_set = set(self.V)
        X_soln_set = set(self.X_soln)

        for node in to_remove:
            if node == self.x_best:
                print("[PRUNING] Best node is removed, info: " + str(self.x_best.totalinfo))
            V_set.discard(node)
            X_soln_set.discard(node)

        # Update lists
        self.V = list(V_set)
        self.X_soln = list(X_soln_set)

    # def RoundTwoAdd(self, x_new):
    def MultiRobotAdd(self, x_new): 
        for node in self.X_soln:
            if node.has_same_position(x_new): #all nodes at the new sampled location
                # tworoundstrategy2:
                if node.round < self.multirobot:
                    NextRoundStartNode = Node((self.x_goal.x, self.x_goal.y))
                    NextRoundStartNode.cost = node.totalcost
                    NextRoundStartNode.totalcost = node.totalcost
                    NextRoundStartNode.info = node.info
                    NextRoundStartNode.totalinfo = node.totalinfo
                    NextRoundStartNode.parent = node
                    NextRoundStartNode.round = node.round+1
                    NextRoundStartNode.prevroundcost= node.totalcost # prevroundcost = costs of all previous rounds
                    #NextRoundStartNode.prevroundcost= node.totalcost-node.prevroundcost
                    self.V.append(NextRoundStartNode)
                    #self.X_soln.add(Round2StartNode)
                    self.X_soln.append(NextRoundStartNode)

            # end tworoundstrategy2

    def Steer(self, x_start, x_goal):
        dist, theta = self.get_distance_and_angle(x_start, x_goal)
        dist = min(self.step_len, dist)
        notfinished=True
        while notfinished and dist>=0:
            xpos = min(99,(math.floor(x_start.x + dist * math.cos(theta))))
            ypos = min(99,(math.floor(x_start.y + dist * math.sin(theta))))
            node_new = Node((xpos,ypos))
            if not np.isnan(self.uncertaintymatrix[node_new.y,node_new.x]) and self.get_distance_and_angle(x_start,node_new)[0]<=(self.step_len):
                dist = self.Line(x_start,node_new)
                notfinished=False # to prevent sampling in obstacle and sampling too far due to rounding
            dist-=1
        #node_new.parent = x_start
        if self.print:
            print("nearest=("+str(x_start.x)+","+str(x_start.y)+") - x_rand=("+str(x_goal.x)+","+str(x_goal.y)+") - dist = "+str(dist+1)+" - x_new=("+str(node_new.x)+","+str(node_new.y)+")")
        return node_new

    def Near(self, nodelist, node, max_dist=0, reduction=True):
        timestart=time.time()
        if max_dist==0:
            max_dist = self.step_len
        X_near = []
        for nd in nodelist:
            if nd == node:
                continue  # skip self
            dist = self.get_distance_and_angle(nd, node)[0]
            if 0.0 < dist <= max_dist:
                X_near.append(nd)
        # dist_table = [self.get_distance_and_angle(nd,node)[0] for nd in nodelist]
        # X_near = [nodelist[ind] for ind in range(len(dist_table)) if (dist_table[ind] <= max_dist and dist_table[ind] > 0.0)]
        timeend = time.time()
        self.time[3] += (timeend - timestart)
        limit = 500
        if len(X_near)>limit and max_dist>=5 and reduction: # if it returns many results, we decrease the radius (when reduction is set to True)
            X_near_reducted = self.Near(X_near,node,max_dist-1)
            if len(X_near_reducted)>0:
                return X_near_reducted
            else:
                return X_near

        return X_near



    @staticmethod
    def SampleUnitBall():
        while True:
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if x ** 2 + y ** 2 < 1:
                return np.array([[x], [y], [0.0]])

    def SampleFreeSpace(self):      
        xpoint = np.random.random_integers(int(self.x_range[0]), int(self.x_range[1]))
        ypoint = np.random.random_integers(int(self.x_range[0]), int(self.x_range[1]))
        # making sure we actually sample in the free space (and not in edge or obstacle):
        while np.isnan(self.uncertaintymatrix[ypoint,xpoint]):
            xpoint = np.random.random_integers(int(self.x_range[0]), int(self.x_range[1]))
            ypoint = np.random.random_integers(int(self.x_range[0]), int(self.x_range[1]))
        return Node((xpoint,ypoint))

    def ExtractPath(self, node):
        if node==self.x_best:
            print("Final cost: "+str(node.totalcost))
            print("Final info value: "+str(node.totalinfo))
        path=[]
        
        # to visualize radius of infopath
        curnode = node
        currentinfopath=[]
        while curnode.parent:
            # print("index 1 = " + str(curnode.parent.y*100+curnode.parent.x) + " index 2 = " + str(
            #    curnode.y*100+curnode.x))
            currentinfopath.extend(
                self.infopathmatrix[curnode.parent.y * 100 + curnode.parent.x, curnode.y * 100 + curnode.x])
            curnode = curnode.parent
        currentinfopath.extend(self.infopathmatrix[self.x_goal.y * 100 + self.x_goal.x, node.y * 100 + node.x])

        # extracting the path
        path = [[self.x_goal.x, self.x_goal.y]]
        rounds=[node.round]
        while node.parent:
            path.append([node.x, node.y])
            rounds.append(node.round)
            node = node.parent
        path.append([node.x, node.y])  # this should be the start
        rounds.append(1) # round 1 at start

        path.reverse() # front to back instead of back to front
        rounds.reverse()
        currentinfopath.reverse()

        return [path,rounds], currentinfopath



    @staticmethod
    def RotationToWorldFrame(x_start, x_goal, L):
        a1 = np.array([[(x_goal.x - x_start.x) / L],
                       [(x_goal.y - x_start.y) / L], [0.0]])
        e1 = np.array([[1.0], [0.0], [0.0]])
        M = a1 @ e1.T
        U, _, V_T = np.linalg.svd(M, True, True)
        C = U @ np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T

        return C

    #@staticmethod
    def Nearest(self,nodelist, n):
        min_node = None
        min_dist = float('inf')
        for nd in nodelist:
            dist = self.get_distance_and_angle(nd, n)[0]
            if dist < min_dist:
                min_dist = dist
                min_node = nd
        return min_node
        # return nodelist[int(np.argmin([self.get_distance_and_angle(nd, n)[0] for nd in nodelist]))]



    #@staticmethod
    def Line(self,x_start, x_goal):
        dist,angle = self.get_distance_and_angle(x_start,x_goal)
        return dist

    def LastPath(self,node):
        if node.totalcost==0:
            node.totalcost=node.cost+ self.Line(node,self.x_goal)
        info = self.FindInfo(self.x_goal.x,self.x_goal.y,node.x,node.y,node,node.totalcost-node.cost,False)



        #node.lastinfopath=infopath
        # print(self.FindInfo(self.x_goal.x,self.x_goal.y,node.x,node.y,node,node.totalcost-node.cost,False))
        # info2=self.FindInfo(self.x_goal.x,self.x_goal.y,node.x,node.y,node,node.totalcost-node.cost,False)
        # print("info: "+str(info))
        # print("info2: "+str(info2))
        node.totalinfo=node.info+info

    def Cost(self, node):
        if node == self.x_start:
            return 0.0

        if node.parent is None:
            return np.inf

        cost = 0.0
        while node.parent:
            #print("node.x: "+str(node.x)+"node.parent.x: "+str(node.parent.x))
            #print("node.y: "+str(node.y)+"node.parent.y: "+str(node.parent.y))
            cost += math.hypot(node.x - node.parent.x, node.y - node.parent.y)
            node = node.parent

        return cost
    def Info(self,node):
        if node == self.x_start:
            return 0.0
        if node.parent is None:
            return 0.0
        info = self.uncertaintymatrix[int(node.y),int(node.x)]

        while node.parent:
            info += node.parent.info
            node = node.parent
        return info

    def Info_cont(self,node):
        if node == self.x_start:
            return 0.0
        if node.parent is None:
            return 0.0

        info = self.FindInfo(node.x,node.y,node.parent.x,node.parent.y,node.parent,self.step_len,True)

        #node.infopath=infopath

        info+=node.parent.info
        return info

    #@staticmethod
    def get_distance_and_angle(self,node_start, node_end):
        idx_start = node_start.y * 100 + node_start.x
        idx_end = node_end.y * 100 + node_end.x
        distance = self.costmatrix[idx_start,idx_end]
        angle = self.anglematrix[idx_start,idx_end]

        if not distance: # element is empty
            #print("calculating distance for entry in matrix, x = " + str(node_start.x) + ", y = " + str(
            #    node_start.y) + ", x = " + str(node_end.x) + ", y = " + str(node_end.y))
            #print("index 1 = " + str(node_start.y * 100 + node_start.x) + " index 2 = " + str(
            #    node_end.y * 100 + node_end.x))
            dx = node_end.x - node_start.x
            dy = node_end.y - node_start.y
            [distance,angle] = math.hypot(dx, dy), math.atan2(dy, dx)
            self.costmatrix[idx_start,idx_end]=distance
            self.costmatrix[idx_end,idx_start]=distance # mirror the matrix
            self.anglematrix[idx_start,idx_end]=angle
            self.anglematrix[idx_end,idx_start]=angle-math.pi # mirror the matrix
            if distance > 0:
                self.FindInfo(node_end.x,node_end.y,node_start.x,node_start.y,node_start,distance,False) # to check whether the path passes through obstacles/edges



        return distance, angle
    
    @staticmethod
    def roundAngle(angle):
        while angle<0:
            angle+=2*math.pi
        while angle>=2*math.pi:
            angle-=2*math.pi
        int = round(angle*8/(2*math.pi))
        rounded = int*2*math.pi/8
        #print(rounded/math.pi)

        return rounded, int
    def animation_new(self, k=None, x_new=None, pruningstep=False):
        """
        Improved visualization for the RIG algorithm.
        """
        if pruningstep != 2:
            self.ax.clear()  # Clears only the plot, keeping the figure

        # Set modern dark background
        self.ax.set_facecolor("#1E1E1E")  # Dark gray background

        # Define line colors
        path_color = "red" if pruningstep == 1 else "gray"
        path_color_2 = "violet" # second round
        path_color_3 = "green" # third round (for now ) #TODO
        # Define transparency
        opacity = 0.8 if (pruningstep == 1 or pruningstep == 2 or pruningstep == 3) else 0.2

        # Title update
        if k and pruningstep != 2:
            self.plot_grid_new(f"RIG, k = {k}, new node = ({x_new.x},{x_new.y})")
        elif not k:
            self.plot_grid_new(f"RIG, k_max = {self.iter_max}")

        
        if pruningstep == 2:
            for node in self.V:
                if node.parent:
                    plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-w", alpha=0.8)
        
        for node in self.V:
            if node.parent:
                if node.parent.round==2: #TODO change for multiround
                    plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-", color=path_color_2, alpha=opacity, linewidth=0.5)
                elif node.parent.round==3: #TODO change for multiround
                    plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-", color=path_color_3, alpha=opacity, linewidth=0.5)
                else:
                    plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-", color=path_color, alpha=opacity, linewidth=0.5)
            elif node != self.x_start:
                plt.scatter(node.x, node.y, color="blue", s=15)  # Small nodes

        # Draw the best path in bold blue
        if self.x_best != self.x_start:
            node = self.x_best
            if node.parent.round==2:
                plt.plot([node.x, self.x_goal.x], [node.y, self.x_goal.y], "-m")
            else:
                plt.plot([node.x, self.x_goal.x], [node.y, self.x_goal.y], "-b")
            while node.parent:
                if node.parent.round==2:
                    plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-m")
                else:
                    plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-b")
                node = node.parent

        # Highlight new node
        if x_new:
            plt.scatter(x_new.x, x_new.y, color="yellow", s=40, edgecolors="black", linewidth=1.2, zorder=3)

        # Set limits, styling
        plt.xlim([0, 100])
        plt.ylim([0, 100])
        self.ax.set_xlim([0, 100])
        self.ax.set_ylim([0, 100])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        # Draw a grid overlay (every 5 units, adjust as needed)
        #self.ax.set_xticks(range(0, 100))
        #self.ax.set_yticks(range(0, 100))
        #self.ax.grid(True, color="grey", linestyle="--", linewidth=0.5, alpha=0.3)

        # Apply layout
        self.fig.tight_layout()
        plt.pause(0.01)  # Smooth real-time updates

    def plot_grid_new(self, name):
        """
        Improved grid visualization with better color mapping.
        """
        colormap = cm.Blues
        colormap.set_bad(color='black')

        self.ax.imshow(self.uncertaintymatrix, cmap=colormap, vmin=0, vmax=1, origin='lower', extent=[-0.5, 99.5, -0.5, 99.5])  # Aligns pixel centers correctly)

        # Mark start and goal positions
        plt.scatter(self.x_start.x, self.x_start.y, color="cyan", s=80, edgecolors="black", linewidth=2, label="Start")
        plt.scatter(self.x_goal.x, self.x_goal.y, color="red", s=80, edgecolors="black", linewidth=2, label="Goal")

        plt.title(name, fontsize=12, color="white")
        # plt.legend(facecolor="#1E1E1E", edgecolor="white", fontsize=10)

    def animation(self, k=None,x_new=None, pruningstep=False):
        if pruningstep!=2:
            plt.cla()

        if pruningstep==1: #before pruning
            color_line = "-r"
        else:
            color_line = "-g"
        if k and pruningstep!=2:
            self.plot_grid("RIG, k = " + str(k)+", new node = ("+str(x_new.x)+","+str(x_new.y)+")")
        elif not k:
            self.plot_grid("RIG, k_max = " + str(self.iter_max))
        if pruningstep!=2:
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

        
        if pruningstep==2:
            for node in self.V:
                if node.parent:
                    plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-w")
        for node in self.V:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], color_line)
            elif not node.parent and not (node.x==self.x_start.x and node.y==self.x_start.y):
                plt.plot(node.x, node.y, "bs", linewidth=3)
        #if not x_new:
        if self.x_best!=self.x_start:
            node = self.x_best
            while node.parent:
                if node.parent.round==2:
                    plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-m")
                else:
                    plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-b")
                node = node.parent

        # for node in self.V:
        #     if node.parent:
        #         #reachedparent=False
        #         prevpoint=[node.x,node.y]
        #         for point in node.infopath[::-1]:
        #             reachedparent= (point[0]==node.parent.x and point[1]==node.parent.y)
        #
        #             plt.plot([point[0], prevpoint[0]], [point[1], prevpoint[1]], "-g")
        #             prevpoint=point
        #             # if reachedparent:
        #             #     break

        # if c_best != np.inf:
        #     self.draw_ellipse(x_center, c_best, dist, theta)
        if x_new:
            plt.plot(x_new.x, x_new.y, "bs", linewidth=3)

        plt.xlim([0, 100])
        plt.ylim([0, 100])
        self.ax.set_xlim([0, 100])
        self.ax.set_ylim([0, 100])

        # Draw a grid overlay (every 5 units, adjust as needed)
        # self.ax.set_xticks(range(0, 100))
        # self.ax.set_yticks(range(0, 100))
        # self.ax.grid(True, color="grey", linestyle="--", linewidth=0.5, alpha=0.3)


        self.fig.tight_layout()
        plt.pause(0.01)


        # if x_new and pruningstep!=1:
        #     #now = datetime.now()
        #     #now_string = now.strftime("%m_%d_%H_%M")
        #     #now_string=str(time.time())
        #     now_string=str(k)
        #     if pruningstep!=False:
        #         if pruningstep==3: #workaround such that 0 and False are not identified as the same
        #             pruningstep=0
        #         now_string=now_string+"_"+str(pruningstep)
        #     path = "../../Documents/Thesis_project/Figures/GIF_figures/"
        #     dirname = os.path.dirname(path)
        #     filename = "/Visualization_" + now_string + ".png"
        #     # plt.savefig(os.path.join(dirname,filename))
        #     plt.savefig(path + filename)

    def plot_grid(self, name):

        # for (ox, oy, w, h) in self.obs_boundary:
        #     self.ax.add_patch(
        #         patches.Rectangle(
        #             (ox, oy), w, h,
        #             edgecolor='black',
        #             facecolor='black',
        #             fill=True
        #         )
        #     )

        # for (ox, oy, w, h) in self.obs_rectangle:
        #     self.ax.add_patch(
        #         patches.Rectangle(
        #             (ox, oy), w, h,
        #             edgecolor='black',
        #             facecolor='gray',
        #             fill=True
        #         )
        #     )
        #
        # for (ox, oy, r) in self.obs_circle:
        #     self.ax.add_patch(
        #         patches.Circle(
        #             (ox, oy), r,
        #             edgecolor='black',
        #             facecolor='gray',
        #             fill=True
        #         )
        #     )
        #
        # #added to visualize the edges of the field and the obstacles:
        # for row in range(100):
        #     for col in range(100):
        #         if np.isnan(self.uncertaintymatrix[row,col]):
        #             self.ax.add_patch(patches.Rectangle((col-0.5,row-0.5),1,1,
        #                                                 edgecolor='black',
        #                                                 facecolor='gray',
        #                                                 fill=True
        #                                                 ))

        colormap = cm.Blues
        colormap.set_bad(color='black')
        im = self.ax.imshow(self.uncertaintymatrix, cmap=colormap, vmin=0, vmax=1, origin='lower',extent=[0, 100, 0, 100])

        plt.plot(self.x_start.x, self.x_start.y, "bs", linewidth=3)
        plt.plot(self.x_goal.x, self.x_goal.y, "rs", linewidth=3)

        plt.title(name)
        #plt.axis("equal")

    @staticmethod
    def draw_ellipse(x_center, c_best, dist, theta):
        a = math.sqrt(c_best ** 2 - dist ** 2) / 2.0
        b = c_best / 2.0
        angle = math.pi / 2.0 - theta
        cx = x_center[0]
        cy = x_center[1]
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        rot = Rot.from_euler('z', -angle).as_matrix()[0:2, 0:2]
        fx = rot @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(cx, cy, ".b")
        plt.plot(px, py, linestyle='--', color='darkorange', linewidth=2)



def main(uncertaintymatrix,scenario=None,matrices=None,samplelocations=[],iterations=250):
    # mid:
    x_start = (50, 50)  # Starting node
    #x_goal = (37, 18)  # Goal node
    x_goal = (50,50)
    # edge:
    x_start = (50,40)
    x_goal = (50,40)
    # scenario = [rowsbool, budget, informed, rewiring, step_len, search_radius, stopsetting, horizonplanning]
    if scenario:
        rrt_star = IRrtStar(x_start, x_goal, scenario[4], 0.0, scenario[5], iterations+1, uncertaintymatrix, scenario, matrices, samplelocations)
    else:
        rrt_star = IRrtStar(x_start, x_goal, 15, 0.0, 15, 300,uncertaintymatrix,scenario,matrices,samplelocations)
    [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,matrices,samplelocations]=rrt_star.planning()

    return finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration, matrices,samplelocations


if __name__ == '__main__':
    main()
