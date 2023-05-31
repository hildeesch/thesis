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

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

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
        #self.infopath = [] # all grid points that are monitored
        self.totalcost = 0 # including the distance to the goal point
        self.totalinfo = 0 # including the last part to the goal point
        #self.lastinfopath = [] # the monitored grid elements from the node to the goal
        #for dubins:
        self.angle = -1


class IRrtStar:
    def __init__(self, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max,uncertaintymatrix,scenario=3,costmatrix=None):
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max

        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils(uncertaintymatrix)

        #self.fig, self.ax = plt.subplots()
        self.delta = self.utils.delta
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.V = [self.x_start]
        self.X_soln = set()
        self.path = None

        self.reductioncount = 0 # to count the shortening of range in Near()
        self.reduction = 0

        self.inforadius = 1 # monitor radius around the robot (0 = no radius, so only at the location of the robot)

        self.budget=200
        self.scenario=scenario
        self.kinematic = "none" # kinematic constraint
        # choices: "none", "dubins", "reedsshepprev", "reedsshepp", "ranger", "limit"
        if self.kinematic == "ranger" or self.kinematic=="limit":
            self.angularcost=1 # set the turning cost (in rad)
        #self.dubinsmatrix = np.empty((3, 100000), dtype=object)
        #self.dubinsmat = np.empty((3, 100000), dtype=object)
        self.dubinsmat = {}

        self.uncertaintymatrix = uncertaintymatrix

        self.x_best = self.x_start # just for now, remove later


        if self.kinematic=="dubins" or self.kinematic=="reedsshepp" or self.kinematic=="reedsshepprev":
            print("dubins/ reedsshepp")
            self.costmatrix=None
            # self.costmatrix = np.empty((100 * 100 * 8, 100 * 100))
            # self.anglematrix = np.empty((100 * 100 * 8, 100 * 100))
            # self.infopathmatrix = np.empty((100 * 100 * 8, 100 * 100), dtype=object)
            # self.infomatrix = np.empty((100 * 100 * 8, 100 * 100))
            self.anglematrix = np.empty((100 * 100, 100 * 100))
        else:
            if costmatrix is None:
                self.costmatrix = np.empty((100 * 100, 100 * 100))
            else:
                self.costmatrix=costmatrix
            self.anglematrix = np.empty((100 * 100, 100 * 100))
            self.infopathmatrix = np.empty((100 * 100, 100 * 100), dtype=object)
            self.infomatrix = np.empty((100 * 100, 100 * 100))

        self.time = np.zeros(8) # for debugging
        # 0 = sample, 1 = nearest, 2 = steer, 3 = near, 4 = rewiring, 5 = lastpath, 6 = pruning, 7 = totaltime


    def init(self):
        # cMin, theta = self.get_distance_and_angle(self.x_start, self.x_goal)
        # #C = self.RotationToWorldFrame(self.x_start, self.x_goal, cMin)
        # xCenter = np.array([[(self.x_start.x + self.x_goal.x) / 2.0],
        #                     [(self.x_start.y + self.x_goal.y) / 2.0], [0.0]])
        x_best = self.x_start

        #return theta, cMin, xCenter, C, x_best
        return x_best

    def planning(self):
        show = False
        #theta, dist, x_center, C, x_best = self.init()
        self.x_best = self.init()
        x_best = self.init()
        c_best = np.inf
        count_down=20
        i_best = 0.001
        totalstarttime=time.time()
        startlen=0 # for checking node increase

        k_list=[]
        i_list=[]
        k=0
        while k<self.iter_max:
            k+=1
            #time.sleep(0.1)
            if k>=100-3: #only evaluate from when we might want it to stop #TODO make 400 a variable
                cost = {node: node.totalcost for node in self.X_soln}
                info = {node: node.totalinfo for node in self.X_soln}
                #x_best = min(cost, key=cost.get)
                if len(info)>0:
                    self.x_best = max(info, key=info.get)
                    x_best = max(info, key=info.get)
                    #c_best = cost[x_best]
                    i_last_best = i_best
                    i_best = info[x_best]

                    k_list.append(k)
                    i_list.append(i_best)
                    #print("i_best: "+str(i_best)+" i_last_best: "+str(i_last_best)+" Criterion value: "+str(((i_best-i_last_best)*100/i_last_best)))
                    if ((i_best-i_last_best)/i_last_best)<0.001: #smaller than 1% improvement
                        count_down-=1
                    else:
                        count_down=20 #reset
                        print("reset countdown")
            if k==201: # to test up to certain iteration
                count_down=0
            if count_down<=0 and (k>200 or k>self.iter_max-3):
                print("Reached stopping criterion at iteration "+str(k))
                break # we stop iterating if the best score is not improving much anymore and we already passed at least ... cycles

            if k%50==0:
                print("ATTENTION!!! ATTENTION!!! ATTENTION!!! AGAIN FIFTY CYCLES FURTHER, CURRENT CYCLE ="+str(k)) # to know how far we are
            endlen=len(self.V)
            print("Nr of nodes added: "+str(endlen-startlen))
            if (endlen-startlen)==0 and not double:
                k-=1
            #     print("Len X_Near was: "+str(len(self.Near(self.V,x_new))))
            startlen = len(self.V)

            timestart=time.time()
            x_rand = self.SampleFreeSpace()
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
            for node in self.V:
                if node.x == x_new.x and node.y == x_new.y:  # co-located nodes
                    double=True #there is already a node at this location, so we skip it
                    print("double")
                    k-=1
                    break
            #if x_nearest.cost + self.Line(x_nearest, x_new) + self.Line(x_new, self.x_goal) < self.budget and not double:  # budget check for nearest parent (to make it more efficient)
            if not double:  # budget check for nearest parent (to make it more efficient)
                # print(x_nearest.cost + self.Line(x_nearest, x_new) + self.Line(x_new, self.x_goal))
                node_new=[]
                # if self.scenario!=5:
                #     for x_near in self.Near(self.V, x_new, self.search_radius):
                #         timestart = time.time()
                #         self.Rewiringv2(x_near, x_new)
                for x_near in self.Near(self.V,x_new):
                    node_new = Node((x_new.x, x_new.y))
                    node_new.parent = x_near  # added

                    if self.kinematic=="dubins" or self.kinematic=="reedsshepp" or self.kinematic=="reedsshepprev":
                        #[dubinscost,dubinsinfo] = self.dubins(x_near,x_new)
                        [dubinscost,dubinsinfo,infopath] = self.dubinsnomatrix(x_near,x_new)
                        c_min = x_near.cost + dubinscost
                        #endcost = self.dubins(x_new,self.x_goal,True)
                        [endcost,endangle] = self.dubinsnomatrix(x_new,self.x_goal,True)
                    else:
                        #c_min = x_near.cost + self.Line(x_near, x_new)
                        #endcost = self.Line(x_new, self.x_goal)
                        c_min = x_near.cost + self.get_distance_and_angle(x_near,node_new)[0]
                        endcost = self.get_distance_and_angle(node_new,self.x_goal)[0]

                    # if c_min+self.Line(x_new, self.x_goal) > self.budget:
                    #     print("past budget (step 2): "+str(c_min+self.Line(x_new, self.x_goal)))

                    node_new.cost = c_min #+self.Line(x_new, self.x_goal)
                    #node_new.info = self.Info(node_new)
                    if self.kinematic=="dubins" or self.kinematic=="reedsshepp" or self.kinematic=="reedsshepprev":
                        node_new.info = x_near.info+ dubinsinfo
                    else:
                        node_new.info = self.Info_cont(node_new)
                    self.V.append(node_new) #generate a "node"/trajectory to each near point


                    timestart = time.time()
                    self.LastPath(node_new)
                    timeend = time.time()
                    self.time[5] += (timeend - timestart)
                    if node_new.totalcost <= self.budget:  # extra check for budget for actual parent
                      self.X_soln.add(node_new)

                #print("node_new: ("+str(node_new.x)+","+str(node_new.y)+")")
                if node_new!=[]: # so it has actually been assigned
                    timestart=time.time()
                    self.Pruning(node_new)
                    timeend = time.time()
                    self.time[6] += (timeend - timestart)

            if k % 50 == 0 and show:
                self.animation()
                self.time[7] = time.time()-totalstarttime
                #print(self.time)
                if k>0:
                    print("It.: " + str(k) + " Time: " + str(self.time[7]) + " Info: " + str(x_best.info) + " Tot. info: "+str(x_best.totalinfo) + " Cost: " + str(x_best.cost) + " Totalcost: "+str(x_best.totalcost) +" Nodes: "+str(len(self.V)))
                    # if k==200:
                    #     for i in range(10): # rewire the 10 best nodes
                    #         info = {node: node.totalinfo for node in self.X_soln}
                    #         #self.x_best = max(info, key=info.get)
                    #         curnode = sorted(info, key=info.get)[-(i+1)]
                    #         self.Rewiring_afterv2(curnode)
                    #     #self.Rewiring_after(self.x_best)
                    #     info = {node: node.totalinfo for node in self.X_soln}
                    #     self.x_best = max(info, key=info.get)
                    #     print("Best node after rewiring: tot. info: "+str(x_best.totalinfo)+" Cost: "+str(x_best.totalcost))
        # Rewiring in Hindsight:
        info = {node: node.totalinfo for node in self.X_soln}
        for i in range(10):  # rewire the 10 best nodes
            # self.x_best = max(info, key=info.get)
            curnode = sorted(info, key=info.get)[-(i + 1)]
            self.Rewiring_afterv2(curnode)
        # self.Rewiring_after(self.x_best)
        info = {node: node.totalinfo for node in self.X_soln}
        self.x_best = max(info, key=info.get)
        x_best = max(info, key=info.get)
        print("Best node after rewiring: tot. info: " + str(x_best.totalinfo) + " Cost: " + str(x_best.totalcost))

        # Extracting the path
        #self.path = self.ExtractPath(x_best)
        #[self.path,nodes] = self.ExtractPath(x_best)
        [self.path,infopath] = self.ExtractPath(x_best)

        #for point in reversed(x_best.infopath):
        #    print("infopoint (x,y)=("+str(point[0])+","+str(point[1])+")")
        #print("length of infopath: "+str(len(x_best.infopath)+len(x_best.lastinfopath)))

        node = x_best
        if self.kinematic!="dubins" and self.kinematic!="reedsshepp" and self.kinematic!="reedsshepprev":
            infopathlength = len(self.infopathmatrix[node.y * 100 + node.x, self.x_goal.y * 100 + self.x_goal.x])
            finalpath = self.infopathmatrix[node.y * 100 + node.x, self.x_goal.y * 100 + self.x_goal.x]
            checkcostscore = self.Line(node, self.x_goal)
            while node.parent:
                # path.append([node.x, node.y])
                checkcostscore += self.Line(node.parent, node)
                infopathlength += len(self.infopathmatrix[node.parent.y * 100 + node.parent.x, node.y * 100 + node.x])
                finalpath.extend(self.infopathmatrix[node.parent.y * 100 + node.parent.x, node.y * 100 + node.x])
                node = node.parent
            # print("Total info path length: "+str(infopathlength))

            # print("Info path length without duplicates: "+str(len([list(t) for t in set(tuple(element) for element in finalpath)])))
            checkinfoscore = 0
            # checkinfoscore2=0
            difflist = []
            # for element in list(set(tuple(element) for element in finalpath)):
            for element in finalpath:
                if not element in difflist:
                    checkinfoscore += self.uncertaintymatrix[int(element[1]), int(element[0])]
                    difflist.append(element)
                # checkinfoscore2+=self.uncertaintymatrix[int(element[1]),int(element[0])]

            print("Node info (without last part): " + str(x_best.info) + " Node cost (without last part): " + (
                str(x_best.cost)))
            print("Info node to end: " + str(
                self.infomatrix[x_best.y * 100 + x_best.x, self.x_start.y * 100 + self.x_start.x]))
            print(" Check info: " + str(checkinfoscore) + " Check costs: " + str(checkcostscore))
            # print("Check info2: "+str(checkinfoscore2))
            # print("len finalpath: "+str(len(finalpath))+" len without duplicates: "+str(len(difflist))+" difflist: "+str(difflist))
            # print("Last path info: "+str(self.FindInfo(self.x_goal.x,self.x_goal.y,x_best.x,x_best.y,self.x_goal,x_best.totalcost-x_best.cost,False)))
            print("Total number of nodes: " + str(len(self.V)))

        if show:
            self.animation()
            plt.plot(x_best.x, x_best.y, "bs", linewidth=3)
            plt.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
            #plt.plot([x for x, _ in x_best.infopath], [y for _, y in x_best.infopath], '-b')
            #plt.plot([x for x, _ in x_best.lastinfopath], [y for _, y in x_best.lastinfopath], '-c')
            #plt.plot([x for x, _ in self.path[:2]],[y for _, y in self.path[:2]], '-k') # to see whether the path actually ends at the goal
            plt.pause(0.01)
            plt.show()


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
            ax.set_title("Spatial distribution of uncertainty and final path")
            #fig.tight_layout()
            plt.show()

            fig, ax = plt.subplots()
            ax.scatter(k_list, i_list)
            ax.grid()
            plt.show()

        return self.path, infopath, x_best.totalcost, x_best.totalinfo, self.budget, self.step_len, self.search_radius, k, self.costmatrix

    def FindInfo(self, node_end_x,node_end_y,node_start_x,node_start_y,node,distance,totalpath=True):
        #node_end = the goal or new node
        #node_start = the (potential) parent
        #currentinfopath = the infopath of the parent (node_start)
        #distance = the distance between the nodes (e.g. self.step_len or search_radius)

        info = self.infomatrix[node_start_y * 100 + node_start_x,node_end_y * 100 + node_end_x]
        infopath = self.infopathmatrix[node_start_y * 100 + node_start_x,node_end_y * 100 + node_end_x]
        if infopath == None:
            dt = 1 / (2 * distance)
            t = 0
            info = 0
            infopath = []
            while t < 1.0:
                xline = node_end_x - node_start_x
                yline = node_end_y - node_start_y
                xpoint = round(node_start_x + t * xline)
                ypoint = round(node_start_y + t * yline)
                if not [xpoint,ypoint] in infopath:
                    if not np.isnan(self.uncertaintymatrix[ypoint, xpoint]):
                        info += self.uncertaintymatrix[ypoint, xpoint]
                        infopath.append([xpoint, ypoint])
                    if np.isnan(self.uncertaintymatrix[ypoint, xpoint]):  # to prevent going through edges and/or obstacles
                        self.costmatrix[node_start_y * 100 + node_start_x,node_end_y * 100 + node_end_x] = np.inf
                if self.inforadius>0:
                    for rowdist in range(-self.inforadius,self.inforadius+1):
                        for coldist in range(-self.inforadius,self.inforadius+1):
                            if (coldist**2+rowdist**2)<=self.inforadius**2: #radius
                                xpoint_=xpoint+coldist
                                ypoint_=ypoint+rowdist
                                if not [xpoint_, ypoint_] in infopath and not np.isnan(self.uncertaintymatrix[ypoint_, xpoint_]):
                                    info += self.uncertaintymatrix[ypoint_, xpoint_]
                                    infopath.append([xpoint_, ypoint_]) #TODO: decide if we want to add the nan points to the infopath or not (in that case we need to change some stuff below)
                t += dt


            self.infomatrix[node_start_y * 100 + node_start_x,node_end_y * 100 + node_end_x] = info
            self.infomatrix[node_end_y * 100 + node_end_x,node_start_y * 100 + node_start_x] = info # mirror the matrix
            self.infopathmatrix[node_start_y * 100 + node_start_x,node_end_y * 100 + node_end_x] = infopath
            self.infopathmatrix[node_end_y * 100 + node_end_x,node_start_y * 100 + node_start_x] = infopath[::-1] # mirror the matrix
        #if totalpath: #if we want to append the current path to the new path
        infopath=infopath
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

    def Recalculate(self,parent, prevparent=None):
        for node in self.V:  # to recalculate the cost and info for nodes further down the line
            if node.parent == parent:
                # Scenario 3:
                if self.scenario==3 or self.scenario==6:
                    if not prevparent==None:
                        # saving temporary node to compare the total info values (TODO: check if deepcopy is faster than assigning all the node parameters)
                        x_temp = Node((node.x, node.y))
                        x_temp.parent = prevparent
                        x_temp.info = node.info
                        x_temp.cost = node.cost
                        x_temp.totalinfo = node.totalinfo
                        x_temp.totalcost = node.totalcost


                # Scenario 2:
                if self.scenario==2:
                # if the old one had a higher total value, we change the parent to the old (the copy) #TODO check how valid this is
                    if not prevparent==None:
                        node.parent=prevparent

                #if prevparent==None:
                #if True: #adapted scenario 3
                if self.scenario==3 or self.scenario==4 or self.scenario==1: #recalculating
                    if node==self.x_best:
                        previnfo=node.info
                        prevtotalinfo=node.totalinfo
                    if self.kinematic=="dubins" or self.kinematic=="reedsshepp" or self.kinematic=="reedsshepprev":
                        #[dist,info] = self.dubins(parent,node)
                        [dist,info] = self.dubinsnomatrix(parent,node,True)
                        node.info = parent.info + info
                    else:
                        dist = self.Line(parent, node)
                        node.info = parent.info + self.FindInfo(node.x, node.y, parent.x, parent.y, parent,
                                                                dist, True)

                    node.cost = parent.cost + dist

                    self.LastPath(node)

                    # if node == self.x_best and node.totalinfo<prevtotalinfo:
                    #     print("[RECALCULATE] Best node is removed, prev totinfo: " + str(
                    #                 prevtotalinfo) + " Previnfo: " + str(previnfo) + " New totinfo: " + str(
                    #                 node.totalinfo) + " Newinfo: " + str(node.info))
                oldparent=None
                # scenario 3
                if self.scenario==3:
                    if not prevparent==None:
                        #if x_temp.totalinfo>node.totalinfo:
                        self.V.append(x_temp)
                        # possibly: tab next line
                        oldparent=x_temp
                    #     print("Totalinfo recalculating not increased, copy of old node added")
                    # else:
                    #     print("Totalinfo recalculating increased, copy of old node not added")
                #only continue if the rewiring increased the totalinfo (prevparent==None)
                if self.scenario==6:
                    if not prevparent==None:
                        if x_temp.totalinfo>node.totalinfo:
                            self.V.append(x_temp)
                        # possibly: tab next line
                        oldparent=x_temp

                # scneario 7:
                if self.scenario==7: # always saving both copies and taking the oldparent with us
                    x_temp = Node((node.x, node.y))
                    x_temp.parent = prevparent
                    x_temp.info = node.info
                    x_temp.cost = node.cost
                    x_temp.totalinfo = node.totalinfo
                    x_temp.totalcost = node.totalcost
                    self.V.append(x_temp)
                    oldparent = x_temp
                self.Recalculate(node,oldparent)

    def Rewiring_after(self, best_node): #rewiring afterwards
        # goal: gain more info while remaining within the budget
        print("Start rewiring after")
        print(" Info: " + str(self.x_best.info) + " Tot. info: " + str(
            self.x_best.totalinfo) + " Cost: " + str(self.x_best.totalcost))
        bestpath=[]
        node=best_node
        while node!=self.x_start:
            bestpath.append(node)
            node=node.parent



        totalinfo = self.x_best.totalinfo
        notfinished = True
        i=2
        while notfinished:
            node = bestpath[-i]
            # print(i)
            # print(len(bestpath))

            #for x_near in self.Near(self.V, node, self.search_radius):
            for x_near in self.Near(self.V, node, 10):
                x_temp = Node((node.x, node.y))
                x_temp.parent = node.parent
                x_temp.info = node.info
                x_temp.cost = node.cost
                x_temp.totalinfo = node.totalinfo
                x_temp.totalcost = node.totalcost

                #if node.parent != self.x_start and node != self.x_start:  # because otherwise there's no "old" path to go back to
                c_old = node.cost
                if self.kinematic == "dubins" or self.kinematic=="reedsshepp":
                    #[cost, info] = self.dubins(node, x_near, False)
                    c_new = node.parent.parent.cost + self.dubinsnomatrix(node.parent.parent,x_near,True)[0] + self.dubinsnomatrix(node,x_near,True)[0]

                else:
                    c_new = node.parent.parent.cost + self.Line(node.parent.parent, x_near) + self.Line(node,
                                                                                                           x_near)

                # if x_new.parent.x==x_near.x and x_new.parent.y==x_near.y:
                #     return # if the parent of x_new = x_near, we don't want to make the parent of x_near = x_new (because then we create a loose segment
                if (c_new-c_old) < (self.budget-best_node.totalcost): # still within budget
                    if not self.kinematic == "dubins" and not self.kinematic=="reedsshepp":
                        newnode = Node((x_near.x, x_near.y))
                        newnode.parent = node.parent.parent
                        addnew = True
                        # for node in self.V[::-1]: # to prevent adding doubles
                        #     if node.parent==newnode.parent and node.x==newnode.x and node.y==newnode.y:
                        #         newnode=node
                        #         addnew = False
                        #         break
                        if addnew:
                            newnode.info = node.parent.parent.info + self.FindInfo(node.parent.parent.x,
                                                                                     node.parent.parent.y, x_near.x,
                                                                                     x_near.y,
                                                                                     node.parent.parent,
                                                                                     self.search_radius,
                                                                                     True)
                            newnode.cost = node.parent.parent.cost + self.Line(node.parent.parent, x_near)
                            self.LastPath(newnode)
                        info = newnode.info + self.FindInfo(x_near.x, x_near.y, node.x, node.y, newnode,
                                                            self.search_radius, True)
                    if self.kinematic=="dubins" or self.kinematic=="reedsshepp" or self.kinematic=="reedsshepprev":
                        newnode = Node((x_near.x, x_near.y))
                        newnode.parent = node.parent.parent
                        addnew = True
                        if addnew:
                            [cost, info, infopath] = self.dubinsnomatrix(node.parent.parent, x_near, False)
                            newnode.info = node.parent.parent.info + info
                            newnode.cost = node.parent.parent.cost + cost
                            self.LastPath(newnode)
                        info = newnode.info + self.dubinsnomatrix(newnode, node, False)[1]
                    # info += x_near.parent.parent.info

                    info_old = node.info

                    info_new = info
                    if info_new >= info_old:  # note: this is different than the condition in pruning
                        if addnew:
                            self.V.append(newnode)

                        # rewiring:

                        node.parent = newnode
                        node.info = info_new
                        node.cost = c_new
                        self.LastPath(node)  # also recalculate the last info part


                        # Scenario 1:
                        # else:
                        self.Recalculate(node,
                                         None)  # recalculates the cost and info for nodes further down the path
                        if totalinfo>=best_node.totalinfo:
                            #print("Total info not increased, was: "+(str(totalinfo))+" is now: "+str(best_node.totalinfo)+", reverse rewiring")
                            # reverse rewiring
                            node.parent = x_temp.parent
                            node.info = x_temp.info
                            node.cost = x_temp.cost
                            node.totalinfo = x_temp.totalinfo
                            node.totalcost = x_temp.totalcost
                            self.Recalculate(node,None)
                        else:
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
            i+=1
            if i==len(bestpath):
                notfinished=False
        print("End rewiring after")
        print(" Info: " + str(self.x_best.info) + " Tot. info: " + str(
            self.x_best.totalinfo) + " Cost: " + str(self.x_best.totalcost))

    def Rewiring_afterv2(self, best_node): #rewiring afterwards
        # goal: gain more info while remaining within the budget
        print("Start rewiring after v2")
        print("Position: ["+str(best_node.x),str(best_node.y)+"] Info: " + str(best_node.info) + " Tot. info: " + str(
            best_node.totalinfo) + " Cost: " + str(best_node.totalcost))
        bestpath=[]
        infosteps=[]
        node=best_node
        while node!=self.x_start:
            # copynode = deepcopy(node) # just now
            # self.V.append(copynode)
            # if len(bestpath)>0:
            #     bestpath[-1].parent=copynode # just now
            # bestpath.append(copynode) # just now
            bestpath.append(node)
            infosteps.append(node.info-node.parent.info)
            # the infosteps contain the added info for the parent to the node (of that node)

            node=node.parent
        # bestpath[-1].parent=self.x_start # just now


        pastindexes = []
        totalinfo = best_node.totalinfo
        notfinished = True
        sortedinfo = sorted(infosteps)

        i=1
        print("Len path: "+str(len(bestpath)))
        while notfinished:
            bestindex = infosteps.index(sortedinfo[-1])
            sortindex=1
            while (bestindex in pastindexes) or (bestindex==len(bestpath)-1):
                sortindex+=1
                if sortindex >= len(bestpath):  # we have had every piece
                    notfinished = False
                    print("Completed rewiring v2")
                    break;
                bestindex = infosteps.index(sortedinfo[-sortindex])

            if not notfinished: # to prevent an extra loop
                break;
            pastindexes.append(bestindex)
            node = bestpath[bestindex]
            # so we rewire the part that is at the moment most profitable

            print("Best index: "+str(bestindex))
            #print(node.parent.parent.x, node.parent.parent.y)
            checked_locations=[]
            for x_near in self.Near(self.V, node, self.search_radius):
            #for x_near in self.Near(self.V, node, 10):
                if not (x_near.x==node.x and x_near.y==node.y) and not ([x_near.x,x_near.y] in checked_locations):
                    checked_locations.append([x_near.x,x_near.y])
                    x_temp = Node((node.x, node.y))
                    x_temp.parent = node.parent
                    x_temp.info = node.info
                    x_temp.cost = node.cost
                    x_temp.totalinfo = node.totalinfo
                    x_temp.totalcost = node.totalcost

                    #if node.parent != self.x_start and node != self.x_start:  # because otherwise there's no "old" path to go back to
                    c_old = node.cost
                    if self.kinematic == "dubins" or self.kinematic=="reedsshepp" or self.kinematic=="reedsshepprev":
                        #[cost, info] = self.dubins(node, x_near, False)
                        c_new = node.parent.parent.cost + self.dubinsnomatrix(node.parent.parent,x_near,True)[0] + self.dubinsnomatrix(node,x_near,True)[0]

                    else:
                        c_new = node.parent.parent.cost + self.Line(node.parent.parent, x_near) + self.Line(node,
                                                                                                               x_near)

                    # if x_new.parent.x==x_near.x and x_new.parent.y==x_near.y:
                    #     return # if the parent of x_new = x_near, we don't want to make the parent of x_near = x_new (because then we create a loose segment
                    if (c_new-c_old) < (self.budget-best_node.totalcost): # still within budget
                        if (not self.kinematic == "dubins" and not self.kinematic=="reedsshepp" and not self.kinematic=="reedsshepprev"):
                            newnode = Node((x_near.x, x_near.y))
                            newnode.parent = node.parent.parent

                            newnode.info = node.parent.parent.info + self.FindInfo(node.parent.parent.x,
                                                                                     node.parent.parent.y, x_near.x,
                                                                                     x_near.y,
                                                                                     node.parent.parent,
                                                                                     self.search_radius,
                                                                                     True)
                            newnode.cost = node.parent.parent.cost + self.Line(node.parent.parent, x_near)
                            self.LastPath(newnode)
                            info = newnode.info + self.FindInfo(x_near.x, x_near.y, node.x, node.y, newnode,
                                                                self.search_radius, True)
                        elif (self.kinematic=="dubins" or self.kinematic=="reedsshepp" or self.kinematic=="reedsshepprev"):
                            newnode = Node((x_near.x, x_near.y))
                            newnode.parent = node.parent.parent
                            [cost, info, infopath] = self.dubinsnomatrix(node.parent.parent, x_near, False)
                            newnode.info = node.parent.parent.info + info
                            newnode.cost = node.parent.parent.cost + cost
                            self.LastPath(newnode)
                            info = newnode.info + self.dubinsnomatrix(newnode, node, False)[1]
                        # info += x_near.parent.parent.info

                        info_old = node.info

                        info_new = info
                        if info_new > info_old:  # note: this is different than the condition in pruning
                            self.V.append(newnode)

                            # rewiring:

                            node.parent = newnode
                            node.info = info_new
                            node.cost = c_new
                            self.LastPath(node)  # also recalculate the last info part


                            # Scenario 1:
                            # else:
                            self.Recalculate(node,
                                             None)  # recalculates the cost and info for nodes further down the path
                            if totalinfo>=best_node.totalinfo:
                                # reverse rewiring
                                node.parent = x_temp.parent
                                node.info = x_temp.info
                                node.cost = x_temp.cost
                                node.totalinfo = x_temp.totalinfo
                                node.totalcost = x_temp.totalcost
                                self.Recalculate(node,None)
                            else:
                                bestpath[bestindex + 1] = newnode

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
        print("End rewiring after v2")
        print(" Info: " + str(best_node.info) + " Tot. info: " + str(
            best_node.totalinfo) + " Cost: " + str(best_node.totalcost))


    def Pruning(self, x_new):
        nodelist=[]
        costlist=[]
        infolist=[]
        for node in self.V:
            if node.x==x_new.x and node.y==x_new.y: #co-located nodes
                nodelist.append(node) # so we know which nodes are colocated
                costlist.append(node.cost) #to compare the costs
                infolist.append(node.info) #to compare the info values
        # now the pruning
        for index, node1 in enumerate(nodelist):
            for index2,node2 in enumerate(nodelist):
                #if (node2.cost<=node1.cost and node2.info>node1.info): #prune lesser paths or doubles
                if (node2.cost<=node1.cost and node2.info>node1.info) or (node2.cost<node1.cost and node2.info==node1.info) or (node1.parent==node2.parent and index!=index2): #prune lesser paths or doubles
                #if (node2.cost <= node1.cost and node2.info > node1.info) or (node1.parent == node2.parent and index != index2):  # prune lesser paths or doubles

                        # print("node 1 =("+str(node1.x)+","+str(node1.y)+") parent =("+str(node1.parent.x)+","+str(node1.parent.y)+")")
                        # print("node 1 cost = "+str(node1.cost)+" node 2 cost ="+str(node2.cost)+"node 1 info = "+str(node1.info)+" node 2 info ="+str(node2.info))
                        # print("node 2 =(" + str(node2.x) + "," + str(node2.y) + ") parent =(" + str(node2.parent.x) + "," + str(node2.parent.y) + ")")
                # else:
                    #     print("Alternative pruned")
                    # prune the node from all nodes
                    nochildren=True
                    for allnode in self.V:
                        if allnode.parent==node1: # in this case we can't prune the node because other nodes depend on it
                            nochildren=False
                    if nochildren:
                        # if (node1.cost == node2.cost and node1.info == node2.info and index != index2 and node1.parent == node2.parent):
                        #     print("Double detected, now pruned")
                        if node1 in self.V: #still have to figure out why this is needed (TODO)
                            if node1 == self.x_best:
                                print("[PRUNING] Best node is removed, info: " + str(
                                    self.x_best.totalinfo))
                            self.V.remove(node1)
                        if node1 in self.X_soln:
                            self.X_soln.discard(node1) #remove from solutions

    def Steer(self, x_start, x_goal):
        dist, theta = self.get_distance_and_angle(x_start, x_goal)
        dist = min(self.step_len-self.reduction, dist)
        notfinished=True
        while notfinished and dist>=0:
            xpos = min(99,(math.floor(x_start.x + dist * math.cos(theta))))
            ypos = min(99,(math.floor(x_start.y + dist * math.sin(theta))))
            node_new = Node((xpos,ypos))
            if not np.isnan(self.uncertaintymatrix[node_new.y,node_new.x]) and self.get_distance_and_angle(x_start,node_new)[0]<=(self.step_len-self.reduction):
                dist = self.Line(x_start,node_new)
                notfinished=False # to prevent sampling in obstacle and sampling too far due to rounding
            dist-=1
        #node_new.parent = x_start

        if self.kinematic=="dubins" or self.kinematic=="reedsshepp" or self.kinematic=="reedsshepprev":
            #dist = self.dubins(x_start,node_new,True)
            [dist,angle] = self.dubinsnomatrix(x_start,node_new,True)

        print("nearest=("+str(x_start.x)+","+str(x_start.y)+") - x_rand=("+str(x_goal.x)+","+str(x_goal.y)+") - dist = "+str(dist+1)+" - x_new=("+str(node_new.x)+","+str(node_new.y)+")")
        return node_new

    def Near(self, nodelist, node, max_dist=0):
        timestart=time.time()
        if max_dist==0:
            max_dist = self.step_len
        if max_dist==self.step_len or max_dist==self.search_radius:
            max_dist-=self.reduction
        dist_table = [self.get_distance_and_angle(nd,node)[0] for nd in nodelist]
        X_near = [nodelist[ind] for ind in range(len(dist_table)) if (dist_table[ind] <= max_dist and dist_table[ind] > 0.0)]
        timeend = time.time()
        self.time[3] += (timeend - timestart)
        if len(X_near)>500 and max_dist>=5:
            self.reductioncount+=1
            #print("Shortening the range for Near: "+str(max_dist-1))
            #print("Current step length ="+str(self.step_len-self.reduction))
            if self.reductioncount==1000 and (self.step_len-self.reduction>=5):
                self.reductioncount=0
                self.reduction+=1
                print("Range is reducted by "+str(self.reduction))
            return self.Near(nodelist,node,max_dist-1)
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
        if not self.kinematic=="dubins" and not self.kinematic=="reedsshepp" and not self.kinematic=="reedsshepprev":
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

            while node.parent:
                path.append([node.x, node.y])
                node = node.parent
            path.append([node.x, node.y])  # this should be the start



        if (self.kinematic=="dubins" or self.kinematic=="reedsshepp" or self.kinematic=="reedsshepprev") and node!=self.x_start:
            path = []
            totalcost = 0 # just for checking/debugging now
            [cost, angle, dubins_x, dubins_y, infopath] = self.getDubins(node, self.x_goal)
            totalcost+=cost
            currentinfopath=[] # to visualize radius around infopath
            currentinfopath.extend(infopath)

            for i in range(len(dubins_x)):
                #path.append([dubins_x[-(i+1)],dubins_y[-(i+1)]])
                path.append([dubins_x[i], dubins_y[i]])

            #print(dubins_x)
            # infopathnode = []
            # for node in infopathrel:
            #     infopathnode.append([node[0] + gx, node[1] + gy])



            nodes = []
            nodes.append([node.x,node.y])


            node_child=node
            node=node.parent
            lastangle=0
            last= False
            #print(path)
            while node.parent or last:
                nodes.append([node.x, node.y])

                print(node.x,node.y,node_child.x,node_child.y)
                [cost, angleparent, dubins_x, dubins_y, infopath] = self.getDubins(node, node_child,last,lastangle)
                currentinfopath.extend(infopath)

                node_child=node
                totalcost += cost

                for i in range(len(dubins_x)):
                    #path.append([dubins_x[-(i+1)], dubins_y[-(i+1)]])
                    path.append([dubins_x[i],dubins_y[i]])
                #print(path)

                if not last:
                    lastangle = self.anglematrix[node.parent.y * 100 + node.parent.x, node.y * 100 + node.x] + math.pi

                node = node.parent
                if last:
                    break
                elif not node.parent:
                    last = True

                #node = Node((0,0))
            print("Totalcost: "+str(totalcost))

        #return path
        return path, currentinfopath



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
        return nodelist[int(np.argmin([self.get_distance_and_angle(nd, n)[0] for nd in nodelist]))]

        # return nodelist[int(np.argmin([(nd.x - n.x) ** 2 + (nd.y - n.y) ** 2
        #                                for nd in nodelist]))]

    #@staticmethod
    def Line(self,x_start, x_goal):
        dist,angle = self.get_distance_and_angle(x_start,x_goal)
        return dist

    def LastPath(self,node):
        #node.totalinfo=node.info+self.FindInfo(node.x,node.y,self.x_start.x,self.x_start.y,node,self.step_len,True)
        if self.kinematic=="dubins" or self.kinematic=="reedsshepp" or self.kinematic=="reedsshepprev":
            [cost,info,infopath] = self.dubinsnomatrix(node,self.x_goal)

            node.totalcost=node.cost+cost
            node.totalinfo=node.info+info
        else:
            node.totalcost=node.cost+ self.Line(node,self.x_goal)

            info = self.FindInfo(self.x_goal.x,self.x_goal.y,node.x,node.y,node,node.totalcost-node.cost,False)

            # to recheck whether we can make the angle and don't pass through obstacles etc:
            node.totalcost = node.cost + self.get_distance_and_angle(node,self.x_goal)[0]

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
        if self.kinematic=="dubins" or self.kinematic=="reedsshepp" or self.kinematic=="reedsshepprev":
            #return self.dubins(node_start,node_end,False)
            return self.dubinsnomatrix(node_start,node_end,True)
        distance = self.costmatrix[node_start.y*100+node_start.x,node_end.y*100+node_end.x]
        angle = self.anglematrix[node_start.y*100+node_start.x,node_end.y*100+node_end.x]

        if not distance: # element is empty
            #print("calculating distance for entry in matrix, x = " + str(node_start.x) + ", y = " + str(
            #    node_start.y) + ", x = " + str(node_end.x) + ", y = " + str(node_end.y))
            #print("index 1 = " + str(node_start.y * 100 + node_start.x) + " index 2 = " + str(
            #    node_end.y * 100 + node_end.x))
            dx = node_end.x - node_start.x
            dy = node_end.y - node_start.y
            [distance,angle] = math.hypot(dx, dy), math.atan2(dy, dx)
            self.costmatrix[node_start.y*100+node_start.x,node_end.y*100+node_end.x]=distance
            self.costmatrix[node_end.y*100+node_end.x,node_start.y*100+node_start.x]=distance # mirror the matrix
            self.anglematrix[node_start.y*100+node_start.x,node_end.y*100+node_end.x]=angle
            self.anglematrix[node_end.y*100+node_end.x,node_start.y*100+node_start.x]=angle-math.pi # mirror the matrix
            if distance > 0:
                self.FindInfo(node_end.x,node_end.y,node_start.x,node_start.y,node_start,distance,False) # to check whether the path passes through obstacles/edges

        # Ranger kinematic constraints: taking time to turn
        if self.kinematic=="ranger":
          if node_start.parent:
                dangle = (self.anglematrix[node_start.parent.y*100 + node_start.parent.x,node_start.y*100 + node_start.x]-angle)**2# squared difference in angle between the line segments
                distance+=np.sqrt(dangle)*self.angularcost

        # Limited angle kinematic constraint
        if self.kinematic=="limit":
            anglelimit = 2*math.pi/4 #90 degrees
            if node_start.parent:
                dangle = (self.anglematrix[node_start.parent.y*100 + node_start.parent.x,node_start.y*100 + node_start.x]-angle)**2# squared difference in angle between the line segments
                if (dangle > (anglelimit ** 2)):
                    distance += np.inf  # if the angle exceeds the limit, inf is added
                else:
                    distance+=np.sqrt(dangle)*self.angularcost



        return distance, angle
    def dubinsnomatrix(self, node_start, node_end, costOnly=False): # to get info and cost for dubins kinematic constraints (=smooth trajectory)
        [cost,angle,dubins_x,dubins_y,infopathnode] = self.getDubins(node_start,node_end)
        path = []


        info=0
        if not node_start.parent==self.x_start and not node_start.parent==None:
            node=node_start.parent
            node_child=node_start
            #print(node.x,node.y)
            last= False
            lastangle=0
            while node.parent or last:
                [costparent, angleparent, dubins_xparent, dubins_yparent, infopath] = self.getDubins(node, node_child,last,lastangle)
                node_child=node
                path.extend(infopath)

                if not last:
                    lastangle = self.anglematrix[node.parent.y * 100 + node.parent.x, node.y * 100 + node.x] + math.pi

                node = node.parent
                if last:
                    #print("last break")
                    break
                elif not node.parent:
                    last = True


            #node = Node((0,0))
        for infocell in infopathnode:
            if infocell not in path:
                if np.isnan(self.uncertaintymatrix[infocell[1], infocell[0]]):
                    cost=np.inf
                    break;
                info += self.uncertaintymatrix[infocell[1], infocell[0]]

            path.append(infocell)

        if costOnly:
            return cost,angle
        #print(info)
        # print(path)
        return cost,info,path

    def getDubins(self,node_start,node_end,last=False,lastangle=0):
        #print("getDubinsfunction")
        #print(last)
        if last:
            angle=lastangle
        else:
            angle = self.anglematrix[node_start.y * 100 + node_start.x, node_end.y * 100 + node_end.x]

            if not angle:
                # if node_end.angle==-1 or (not costOnly):
                # print("starting dubins cost")
                dx = node_end.x - node_start.x
                dy = node_end.y - node_start.y
                [syaw, int] = self.roundAngle(math.atan2(dy, dx))
                self.anglematrix[node_start.y * 100 + node_start.x, node_end.y * 100 + node_end.x] = syaw
                angle = syaw
        maxc = 0.5

        sx = node_end.x
        sy = node_end.y
        if last:
            syaw=lastangle
        else:
            syaw = self.anglematrix[node_start.y * 100 + node_start.x, node_end.y * 100 + node_end.x] + math.pi
        if syaw >= 2 * math.pi:
            syaw = syaw - 2 * math.pi
        # [syaw,int] = self.roundAngle(syaw)
        gx = node_start.x
        gy = node_start.y
        if (not node_start.parent) or last:
            gyaw = syaw
        else:
            gyaw = self.anglematrix[node_start.parent.y * 100 + node_start.parent.x, node_start.y * 100 + node_start.x] + math.pi
        # [gyaw,int] = self.roundAngle(gyaw)
        #print(syaw,gyaw,last, (not node_start.parent))
        ## new part (21/4)
        # [node_end.x-node_start.x, node_end.y-node_start.y, gyaw]
        if (gx - sx, gy - sy, gyaw) not in self.dubinsmat:  # relative position + end angle
        #if [gx - sx, gy - sy, gyaw] not in self.dubinsmat[0].tolist():  # relative position + end angle
            if self.kinematic=="dubins":
                [dubinspath, self.dubinsmat, infopathrel] = dubins.calc_dubins_path(sx, sy, syaw, gx, gy, gyaw, maxc,
                                                                            self.dubinsmat)
            elif self.kinematic=="reedsshepprev":
                if gyaw<math.pi:
                    if (syaw-gyaw)**2> (syaw-(gyaw+math.pi))**2:
                        #print("reverse angle is smaller")
                        gyaw=gyaw+math.pi
                        [dubinspath, infopathrel] = reedsshepp.calc_optimal_path(sx, sy, syaw, gx, gy, gyaw,
                                                                                            maxc)
                    else:
                        [dubinspath, infopathrel] = reedsshepp.calc_optimal_path(sx, sy, syaw, gx, gy, gyaw,
                                                                                            maxc)
                else:
                    if (syaw-gyaw)**2> (syaw-(gyaw-math.pi))**2:
                        #print("reverse angle is smaller")
                        gyaw=gyaw-math.pi
                        [dubinspath, infopathrel] = reedsshepp.calc_optimal_path(sx, sy, syaw, gx, gy, gyaw,
                                                                                            maxc)
                    else:
                        [dubinspath, infopathrel] = reedsshepp.calc_optimal_path(sx, sy, syaw, gx, gy, gyaw,
                                                                                            maxc)
            elif self.kinematic=="reedsshepp":
                [dubinspath, infopathrel] = reedsshepp.calc_optimal_path(sx, sy, syaw, gx, gy, gyaw, maxc)
            cost = dubinspath.L
            dubins_rel_x = dubinspath.x
            dubins_rel_y = dubinspath.y
            # for i in range(len(self.dubinsmat[0])):
            #     if self.dubinsmat[0, i] == None:
            #         self.dubinsmat[0, i] = [gx - sx, gy - sy, gyaw]
            #         self.dubinsmat[1, i] = [dubinspath.L, dubinspath.x, dubinspath.y, infopathrel]
            #         # add = True
            #         # index = i
            #         # print("new index added in dubinsmatrix")
            #         break;
            self.dubinsmat[(gx - sx, gy - sy, gyaw)]=[dubinspath.L, dubinspath.x, dubinspath.y, infopathrel]
        else:
            #print("reusing the dubinsmat")
            #index = self.dubinsmat[0].tolist().index([gx - sx, gy - sy, gyaw])
            #index = self.dubinsmat.index([gx - sx, gy - sy, gyaw])
            # index = np.where(self.dubinsmat[0] == [gx - sx, gy - sy, gyaw])

            [cost, dubins_rel_x, dubins_rel_y, infopathrel] = self.dubinsmat[(gx - sx, gy - sy, gyaw)]

        # if costOnly:
        #     return cost, gyaw
        # else:
        #dubins_x = dubins_rel_x + sx
        dubins_x = [x + sx for x in dubins_rel_x]
        dubins_y = [y + sy for y in dubins_rel_y]
        #dubins_y = dubins_rel_y + sy # note: the algorithm adds and detracts sx,sy instead of gx,gy
        infopath = []
        for cell in infopathrel:
            infopath.append([cell[0] + sx, cell[1] + sy])
            if self.inforadius > 0:
                for rowdist in range(-self.inforadius, self.inforadius+1):
                    for coldist in range(-self.inforadius, self.inforadius+1):
                        if (coldist ** 2 + rowdist ** 2) <= self.inforadius ** 2:  # radius
                            xpoint_ = cell[0] + sx + coldist
                            ypoint_ = cell[1] + sy + rowdist
                            if not [xpoint_, ypoint_] in infopath and not np.isnan(self.uncertaintymatrix[ypoint_, xpoint_]):
                                #info += self.uncertaintymatrix[ypoint_, xpoint_]
                                infopath.append([xpoint_, ypoint_])
        return cost,angle,dubins_x,dubins_y,infopath

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
    def animation(self):
        plt.cla()
        self.plot_grid("Informed rrt*, N = " + str(self.iter_max))
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        if self.kinematic=="dubins" or self.kinematic=="reedsshepp" or self.kinematic=="reedsshepprev":
            for node in self.V:
                #path = self.ExtractPath(node)
                if node!=self.x_start:
                    [cost, angleparent, dubins_x, dubins_y, infopath] = self.getDubins(node.parent, node)
                    path=[]
                    for i in range(len(dubins_x)):
                        path.append([dubins_x[i], dubins_y[i]])
                    # if path!=[]:
                    #     print(path)
                    for i in range(0,len(path)-10,10):
                        plt.plot([path[i][0], path[i+10][0]], [path[i][1], path[i+10][1]], "-g")
        elif not self.kinematic=="dubins" and not self.kinematic=="reedsshepp" and not self.kinematic=="reedsshepprev":
            for node in self.V:
                if node.parent:
                    plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")
                elif not node.parent and not (node.x==self.x_start.x and node.y==self.x_start.y):
                    plt.plot(node.x, node.y, "bs", linewidth=3)

            if self.x_best!=self.x_start:
                node = self.x_best
                while node.parent:
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

        plt.pause(0.01)

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

        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        for (ox, oy, r) in self.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        #added to visualize the edges of the field and the obstacles:
        for row in range(100):
            for col in range(100):
                if np.isnan(self.uncertaintymatrix[row,col]):
                    self.ax.add_patch(patches.Rectangle((col-0.5,row-0.5),1,1,
                                                        edgecolor='black',
                                                        facecolor='gray',
                                                        fill=True
                                                        ))

        plt.plot(self.x_start.x, self.x_start.y, "bs", linewidth=3)
        plt.plot(self.x_goal.x, self.x_goal.y, "rs", linewidth=3)

        plt.title(name)
        plt.axis("equal")

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



def main(uncertaintymatrix,scenario=3,costmatrix=None):
    x_start = (50, 50)  # Starting node
    #x_goal = (37, 18)  # Goal node
    x_goal = (50,50)

    rrt_star = IRrtStar(x_start, x_goal, 15, 0.0, 15, 2000,uncertaintymatrix,scenario,costmatrix)
    [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,costmatrix]=rrt_star.planning()

    return finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration, costmatrix


if __name__ == '__main__':
    main()
