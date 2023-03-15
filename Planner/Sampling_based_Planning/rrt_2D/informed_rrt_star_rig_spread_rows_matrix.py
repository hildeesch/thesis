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
from copy import deepcopy
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from Sampling_based_Planning.rrt_2D import env, plotting, utils
from Planner.Sampling_based_Planning.rrt_2D import env, plotting, utils

class Node:
    __slots__ = ['x','y','parent','cost','info','totalcost','totalinfo']
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.cost = 0
        self.info = 0
        #self.infopath = []
        self.totalcost = 0 # including the distance to the goal point
        self.totalinfo = 0 # including the last part to the goal point
        #self.lastinfopath = [] # the monitored grid elements from the node to the goal


class IRrtStar:
    def __init__(self, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max,uncertaintymatrix,row_nrs,row_edges):
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max

        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils(uncertaintymatrix)

        self.fig, self.ax = plt.subplots()
        self.delta = self.utils.delta
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.V = [self.x_start]
        self.X_soln = set()
        self.path = None

        self.budget=200
        self.uncertaintymatrix = uncertaintymatrix
        self.row_nrs = row_nrs
        self.row_edges = row_edges
        self.cost_left = np.zeros((len(self.row_nrs),len(self.row_nrs)))
        self.cost_right = np.zeros((len(self.row_nrs),len(self.row_nrs)))
        self.infopath_left = np.empty((len(self.row_nrs),len(self.row_nrs)),dtype = object )
        self.infopath_right = np.empty((len(self.row_nrs),len(self.row_nrs)),dtype = object )
        self.info_left = np.zeros((len(self.row_nrs),len(self.row_nrs)))
        self.info_right = np.zeros((len(self.row_nrs),len(self.row_nrs)))

        self.costmatrix = np.empty((100*100,100*100) )
        self.directionmatrix = np.empty((100*100,100*100),dtype = object )
        self.infopathmatrix = np.empty((100*100,100*100),dtype = object )
        self.infomatrix = np.empty((100*100,100*100) )

        self.allpoints=[] # all the points that can be sampled for nodes
        for index,row in enumerate(row_nrs):
            for column in range(self.row_edges[index][0],self.row_edges[index][1]+1):
                self.allpoints.append([column,row])

        self.rowdist=4

    def init(self):
        #cMin, theta = self.get_distance_and_angle(self.x_start, self.x_goal)
        #C = self.RotationToWorldFrame(self.x_start, self.x_goal, cMin)
        #xCenter = np.array([[(self.x_start.x + self.x_goal.x) / 2.0],
        #                    [(self.x_start.y + self.x_goal.y) / 2.0], [0.0]])
        x_best = self.x_start

        #return theta, cMin, xCenter, C, x_best

        self.EdgeCostInfo()
        return x_best

    def planning(self):
        show=True
        #theta, dist, x_center, C, x_best = self.init()
        #theta, dist, x_center, x_best = self.init()
        x_best = self.init()
        #c_best = np.inf
        count_down=3
        i_best = 0
        for k in range(self.iter_max):
            #time.sleep(0.1)
            if k>=150-3: #only evaluate from when we might want it to stop
                cost = {node: node.totalcost for node in self.X_soln}
                info = {node: node.totalinfo for node in self.X_soln}
                #x_best = min(cost, key=cost.get)
                x_best = max(info, key=info.get)
                #c_best = cost[x_best]
                i_last_best = i_best
                i_best = info[x_best]
                if i_last_best>0: # to prevent division by zero
                    if ((i_best-i_last_best)/i_last_best)<0.01: #smaller than 1% improvement
                        count_down-=1
                    else:
                        count_down=10 #reset
            if count_down<=0 and k>150:
                print("Reached stopping criterion at iteration "+str(k))
                break # we stop iterating if the best score is not improving much anymore and we already passed at least ... cycles

            if k%50==0:
                print("ATTENTION!!! ATTENTION!!! ATTENTION!!! AGAIN FIFTY CYCLES FURTHER, CURRENT CYCLE ="+str(k)) # to know how far we are

            #x_rand = self.Sample(c_best, dist, x_center, C)
            x_rand = self.Sample()
            x_nearest = self.Nearest(self.V, x_rand)
            x_new = self.Steer_section(x_nearest, x_rand) #so that we only generate one new node, not multiple
            #if self.Cost(x_nearest) + self.Line(x_nearest, x_rand) + self.Line(x_rand, self.x_goal) > self.budget:
                #just for debugging purposes (for now)
                #print("Past the budget")
            double=False
            for node in self.V:
                if node.x == x_new.x and node.y == x_new.y:  # co-located nodes
                    double=True #there is already a node at this location, so we skip it
                    print("double")
                    break
            #if x_nearest.cost + self.Line(x_nearest, x_new) + self.Line(x_new, self.x_goal) < self.budget and not double:  # budget check for nearest parent (to make it more efficient)
            if not double:  # budget check for nearest parent (to make it more efficient)
                # print(x_nearest.cost + self.Line(x_nearest, x_new) + self.Line(x_new, self.x_goal))
                for x_near in self.Near(self.V,x_new):

                    if x_new and not self.utils.is_collision(x_near, x_new):
                        c_min = x_near.cost+self.FindCostInfo(x_new.x,x_new.y,x_near.x,x_near.y,x_near,False,True) #cost from near node to new node

                        if c_min+self.FindCostInfo(self.x_goal.x,self.x_goal.y,x_new.x,x_new.y,x_near,False,True) <=self.budget: #extra check for budget for actual parent (cmin+ cost to goal node)

                            node_new = Node((x_new.x,x_new.y))
                            node_new.parent = x_near #added
                            node_new.cost = self.Cost(node_new) #+self.Line(x_new, self.x_goal)
                            node_new.info = self.Info_cont(node_new)
                            self.V.append(node_new) #generate a "node"/trajectory to each near point

                            #rewire
                            for x_near in self.Near(self.V,x_new,self.search_radius):
                                self.Rewiring(x_near,node_new)

                            if self.InGoalRegion(node_new):
                                if not self.utils.is_collision(node_new, self.x_goal):
                                    self.X_soln.add(node_new)
                                    self.LastPath(node_new)

                self.Pruning(x_new)

            if k % 20 == 0 and show:
                self.animation()

        self.path = self.ExtractPath(x_best)
        node = x_best
        infopathlength=len(self.infopathmatrix[node.y*100+node.x,self.x_goal.y*100+self.x_goal.x])
        finalpath= self.infopathmatrix[node.y*100+node.x,self.x_goal.y*100+self.x_goal.x]
        checkcostscore=self.Line(node,self.x_goal)
        while node.parent:
            checkcostscore+=self.Line(node.parent,node)
            infopathlength += len(self.infopathmatrix[node.parent.y*100+node.parent.x,node.y*100+node.x])
            finalpath.extend(self.infopathmatrix[node.parent.y*100+node.parent.x,node.y*100+node.x])
            node = node.parent

        checkinfoscore=0
        difflist=[]
        for element in finalpath:
            if not element in difflist:
                checkinfoscore+=self.uncertaintymatrix[int(element[1]),int(element[0])]
                difflist.append(element)

        print("Node info (without last part): "+str(x_best.info)+" Node cost (without last part): "+(str(x_best.cost)))
        print("Info node to end: "+str(self.infomatrix[x_best.y*100+x_best.x,self.x_start.y*100+self.x_start.x]))
        print(" Check info: "+str(checkinfoscore)+" Check costs: "+str(checkcostscore))
        print("Total number of nodes: "+str(len(self.V)))


        #print("length of infopath: "+str(len(x_best.infopath)+len(x_best.lastinfopath)))

        #self.animation(x_center=x_center, c_best=c_best, dist=dist, theta=theta)
        if show:
            self.animation()
            plt.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
            #plt.plot([x for x, _ in self.infopathmatrix[x_best.y*100+x_best.x][self.x_goal.y*100+self.x_goal.x]],[y for _, y in self.infopathmatrix[x_best.y*100+x_best.x][self.x_goal.y*100+self.x_goal.x]],'-b')
            #plt.plot(x_best.x, x_best.y, "bs", linewidth=3)

            #plt.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
            #plt.plot([x for x, _ in x_best.infopath], [y for _, y in x_best.infopath], '-r')
            #plt.plot([x for x, _ in x_best.lastinfopath], [y for _, y in x_best.lastinfopath], '-b')
            #print(x_best.lastinfopath)
            #plt.plot([x for x, _ in self.path[:2]],[y for _, y in self.path[:2]], '-b') # to see whether the path actually ends at the goal
            plt.pause(0.01)
            plt.show()
            #plt.close()

            fig, ax = plt.subplots()
            colormap = cm.Blues
            colormap.set_bad(color='black')
            im = ax.imshow(self.uncertaintymatrix, colormap, vmin=0, vmax=3, origin='lower')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            #ax.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
            ax.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')

            #ax.plot([x for x, _ in x_best.infopath], [y for _, y in x_best.infopath], '-r')
            #ax.plot([x for x, _ in x_best.lastinfopath], [y for _, y in x_best.lastinfopath], '-r')
            ax.set_title("Spatial distribution of uncertainty and final path")
            fig.tight_layout()
            plt.show()
            #plt.close()

        return self.path, x_best.totalcost, x_best.totalinfo, self.budget, self.steplength, self.searchradius, k

    def FindInfo(self, node_end_x, node_end_y, node_start_x, node_start_y, currentinfopath, distance, totalpath=True):
        # node_end = the goal or new node
        # node_start = the (potential) parent
        # currentinfopath = the infopath of the parent (node_start)
        # distance = the distance between the nodes (e.g. self.step_len or search_radius)
        dt = 1 / (2 * distance)
        t = 0
        info = 0
        infopath = []
        if totalpath:  # if we want to append the current path to the new path
            # for point in currentinfopath:
            #     infopath.append(point)
            infopath.extend(currentinfopath)  # extend = append but for a list instead of 1 element

        while t <= 1.0:
            xline = node_end_x - node_start_x
            yline = node_end_y - node_start_y
            xpoint = math.floor(node_start_x + t * xline)
            ypoint = math.floor(node_start_y + t * yline)
            if [xpoint, ypoint] not in infopath:  # only info value when the point is not already monitored before
                info += self.uncertaintymatrix[ypoint, xpoint]
                infopath.append([xpoint, ypoint])
            t += dt

        return infopath, info

    def EdgeCostInfo(self):
        # index_start = the index of the edge node from which we come
        # index_end = the index of the edge node to which we go
        for index in range(len(self.row_nrs)-1):
            self.cost_left[index][index+1] = math.hypot(self.row_nrs[index + 1] - self.row_nrs[index],
                         self.row_edges[index + 1][0] - self.row_edges[index][0])  # note: this distance is not really in the y-direction, but from edge to edge point
            self.cost_left[index+1][index] = self.cost_left[index][index+1]

            [infopath, info] = self.FindInfo(self.row_edges[index + 1][0], self.row_nrs[index + 1], self.row_edges[index][0],
                                             self.row_nrs[index], [], self.cost_left[index+1][index],
                                             False)
            self.infopath_left[index][index+1] = infopath
            self.infopath_left[index+1][index] = infopath[::-1]
            self.info_left[index][index+1] = info
            self.info_left[index+1][index] = info

            self.cost_right[index][index+1] = math.hypot(self.row_nrs[index + 1] - self.row_nrs[index],
                         self.row_edges[index + 1][1] - self.row_edges[index][1])  # note: this distance is not really in the y-direction, but from edge to edge point
            self.cost_right[index+1][index] = self.cost_right[index][index+1]

            [infopath, info] = self.FindInfo(self.row_edges[index + 1][1], self.row_nrs[index + 1], self.row_edges[index][1],
                                             self.row_nrs[index], [], self.cost_right[index+1][index],
                                             False)
            self.infopath_right[index][index+1] = infopath
            self.infopath_right[index+1][index] = infopath[::-1]
            self.info_right[index][index+1] = info
            self.info_right[index+1][index] = info

        for index1 in range(len(self.row_nrs)):
            for index2 in range(len(self.row_nrs)):
                if not abs(index2-index1)==1 and index1<index2:

                    index_add=0
                    self.infopath_left[index1][index2]=[]
                    self.infopath_right[index1][index2]=[]
                    while index1+index_add<index2:
                        self.cost_left[index1][index2]+=self.cost_left[index1+index_add][index1+index_add+1]
                        self.info_left[index1][index2]+=self.info_left[index1+index_add][index1+index_add+1]
                        # for infopoint in self.infopath_left[index1+index_add][index1+index_add+1]:
                        #     self.infopath_left[index1][index2].append(infopoint)
                        self.infopath_left[index1][index2].extend(self.infopath_left[index1+index_add][index1+index_add+1])

                        self.cost_right[index1][index2]+=self.cost_right[index1+index_add][index1+index_add+1]
                        self.info_right[index1][index2]+=self.info_right[index1+index_add][index1+index_add+1]
                        # for infopoint in self.infopath_right[index1+index_add][index1+index_add+1]:
                        #     self.infopath_right[index1][index2].append(infopoint)
                        self.infopath_right[index1][index2].extend(self.infopath_right[index1+index_add][index1+index_add+1])

                        index_add+=1
                    self.cost_left[index2][index1] = self.cost_left[index1][index2]
                    self.info_left[index2][index1] = self.info_left[index1][index2]
                    self.infopath_left[index2][index1] = self.infopath_left[index1][index2][::-1]
                    self.cost_right[index2][index1] = self.cost_right[index1][index2]
                    self.info_right[index2][index1] = self.info_right[index1][index2]
                    self.infopath_right[index2][index1] = self.infopath_right[index1][index2][::-1]
#        print(self.cost_right[1][24])
#        print(self.cost_right[24][1])
#        print(self.infopath_left[1][2])
#        print(self.infopath_left[2][1])

    def FindCostInfo(self, node_end_x, node_end_y, node_start_x, node_start_y, node, totalpath=True, costOnly=False):
        # node_end = the goal or new node
        # node_start = the (potential) parent
        # currentcost = cost of the parent (node_start)
        # currentinfopath = the infopath of the parent (node_start)
        # costOnly = shortcut in case we only want the distance and not the info

        cost = self.costmatrix[node_start_y*100+node_start_x, node_end_y*100+node_end_x]
        node_direction = self.directionmatrix[node_start_y*100+node_start_x, node_end_y*100+node_end_x]

        info = self.infomatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x]
        infopath = self.infopathmatrix[node_start_y * 100 + node_start_x,node_end_y * 100 + node_end_x]

        if not cost or (costOnly==False and infopath==None): # element is empty or we need the variables for finding the infopath later
            #print("calculating distance for entry in matrix, x = " + str(node_start.x) + ", y = " + str(
            #    node_start.y) + ", x = " + str(node_end.x) + ", y = " + str(node_end.y))
            #print("index 1 = " + str(node_start.y * 100 + node_start.x) + " index 2 = " + str(
            #    node_end.y * 100 + node_end.x))
            if node_end_y == node_start_y:  # in the same row
                dxright = abs(node_end_x - node_start_x)
                if node_end_x < node_start_x:
                    direction = "left"
                    dx = dxright
                else:
                    direction = "right"
                    dx = dxright  # makes no difference for left or right
                dy = 0
                node_direction = [direction, "none"]
            else:  # not in same row
                index_start = self.row_nrs.index(node_start_y)
                index_end = self.row_nrs.index(node_end_y)
                dxleft = node_end_x - self.row_edges[index_end][0] + node_start_x - self.row_edges[index_start][0]
                dxright = self.row_edges[index_end][1] - node_end_x + self.row_edges[index_start][1] - node_start_x
                if dxright < dxleft:
                    direction = "right"
                    # print("direction RIGHT")
                    start_edge_x = self.row_edges[index_start][1]
                    end_edge_x = self.row_edges[index_end][1]
                    dx = dxright
                else:
                    direction = "left"
                    # print("direction LEFT")
                    start_edge_x = self.row_edges[index_start][0]
                    end_edge_x = self.row_edges[index_end][0]
                    dx = dxleft
                if index_end > index_start:
                    updown = "up"
                else:
                    updown = "down"
                dy = 0
                edgepath = []
                # for index,edgepoint in enumerate(self.row_edges[index_start:index_end+1]):
                #     if direction=="left":
                #         #dy+= math.hypot(self.row_nrs[index+1] - self.row_nrs[index], self.row_edges[index+1][0]-self.row_edges[index][0]) #note: this distance is not really in the y-direction, but from edge to edge point
                #         dy+= self.cost_left[index][index+1]
                #         #edgepath.append([self.row_edges[index][0],self.row_nrs[index]])
                #     elif direction=="right":
                #         #dy+= math.hypot(self.row_nrs[index+1] - self.row_nrs[index], self.row_edges[index+1][1]-self.row_edges[index][1]) #note: this distance is not really in the y-direction, but from edge to edge point
                #         dy += self.cost_right[index][index + 1]
                #         #edgepath.append([self.row_edges[index][1],self.row_nrs[index]])
                if direction == "left":
                    # dy+= math.hypot(self.row_nrs[index+1] - self.row_nrs[index], self.row_edges[index+1][0]-self.row_edges[index][0]) #note: this distance is not really in the y-direction, but from edge to edge point
                    dy += self.cost_left[index_start][index_end]
                    # edgepath.append([self.row_edges[index][0],self.row_nrs[index]])
                elif direction == "right":
                    # dy+= math.hypot(self.row_nrs[index+1] - self.row_nrs[index], self.row_edges[index+1][1]-self.row_edges[index][1]) #note: this distance is not really in the y-direction, but from edge to edge point
                    dy += self.cost_right[index_start][index_end]
                    # edgepath.append([self.row_edges[index][1],self.row_nrs[index]])

                # dy = math.hypot(self.row_nrs[index_end] - self.row_nrs[index_start], end_edge_x-start_edge_x) #note: this distance is not really in the y-direction, but from edge to edge point

                node_direction = [direction, updown]
            cost = dx + dy
            self.costmatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x] = cost
            self.costmatrix[node_end_y * 100 + node_end_x, node_start_y * 100 + node_start_x] = cost #mirror the matrix
            self.directionmatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x] = node_direction
            self.directionmatrix[node_end_y * 100 + node_end_x, node_start_y * 100 + node_start_x] = node_direction


        if costOnly:
            return cost
        #infopath = []
        # if totalpath:
        #     # for point in currentinfopath:
        #     #     infopath.append(point)
        #     infopath.extend(currentinfopath)
        #info = 0
        # node.parent --> node
        # node --> self.x_goal


        if infopath ==None:
            infopath=[]
            info=0
            if node_direction == ["left", "none"]:
                for xpoint in range(node_start_x, node_end_x - 1, -1):
                    ypoint = node_start_y
                    # if [xpoint,
                    #     ypoint] not in currentinfopath:  # only info value when the point is not already monitored before
                    info += self.uncertaintymatrix[ypoint, xpoint]
                    if np.isnan(self.uncertaintymatrix[ypoint, xpoint]):
                        self.costmatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x] = np.inf
                        self.costmatrix[node_end_y * 100 + node_end_x, node_start_y * 100 + node_start_x] = np.inf # mirror the matrix
                    infopath.append([xpoint, ypoint])
            if node_direction == ["right", "none"]:
                for xpoint in range(node_start_x, node_end_x + 1):
                    ypoint = node_start_y
                    # if [xpoint,
                    #     ypoint] not in currentinfopath:  # only info value when the point is not already monitored before
                    info += self.uncertaintymatrix[ypoint, xpoint]
                    if np.isnan(self.uncertaintymatrix[ypoint, xpoint]):
                        self.costmatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x] = np.inf
                        self.costmatrix[node_end_y * 100 + node_end_x, node_start_y * 100 + node_start_x] = np.inf # mirror the matrix

                    infopath.append([xpoint, ypoint])
            if node_direction == ["left", "up"] or node_direction == ["left","down"]:
                for xpoint in range(node_start_x, start_edge_x-1, -1):
                    ypoint = node_start_y
                    # if [xpoint,
                    #     ypoint] not in currentinfopath:  # only info value when the point is not already monitored before
                    info += self.uncertaintymatrix[ypoint, xpoint]
                    if np.isnan(self.uncertaintymatrix[ypoint, xpoint]):
                        self.costmatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x] = np.inf
                        self.costmatrix[node_end_y * 100 + node_end_x, node_start_y * 100 + node_start_x] = np.inf  # mirror the matrix
                    infopath.append([xpoint, ypoint])

                infopoints = self.infopath_left[index_start][index_end]
                for infopoint in infopoints:
                    #if infopoint not in currentinfopath:
                        # infopath.append(infopoint)
                        # info += self.info_left[index][index+1]
                        # break
                    info += self.uncertaintymatrix[infopoint[1],infopoint[0]]
                    #if np.isnan(self.uncertaintymatrix[infopoint[1],infopoint[0]]):
                    #    self.costmatrix[node_start_y * 100 + node_start_x][node_end_y * 100 + node_end_x] = np.inf
                    infopath.append(infopoint)

                for xpoint in range(end_edge_x, node_end_x + 1):
                    ypoint = node_end_y
                    # if [xpoint,
                    #     ypoint] not in currentinfopath:  # only info value when the point is not already monitored before
                    info += self.uncertaintymatrix[ypoint, xpoint]
                    if np.isnan(self.uncertaintymatrix[ypoint, xpoint]):
                        self.costmatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x] = np.inf
                        self.costmatrix[node_end_y * 100 + node_end_x, node_start_y * 100 + node_start_x] = np.inf # mirror the matrix
                    infopath.append([xpoint, ypoint])

            if node_direction == ["right", "up"] or node_direction==["right","down"]:
                for xpoint in range(node_start_x, start_edge_x):
                    ypoint = node_start_y
                    #if [xpoint,ypoint] not in currentinfopath:  # only info value when the point is not already monitored before
                    info += self.uncertaintymatrix[ypoint, xpoint]
                    if np.isnan(self.uncertaintymatrix[ypoint, xpoint]):
                        self.costmatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x] = np.inf
                        self.costmatrix[node_end_y * 100 + node_end_x, node_start_y * 100 + node_start_x] = np.inf # mirror the matrix
                    infopath.append([xpoint, ypoint])

                infopoints = self.infopath_right[index_start][index_end]
                for infopoint in infopoints:
                    #if infopoint not in currentinfopath:
                        # infopath.append(infopoint)
                        # info += self.info_right[index][index+1]
                        # break
                    info += self.uncertaintymatrix[infopoint[1],infopoint[0]]
                    #if np.isnan(self.uncertaintymatrix[infopoint[1],infopoint[0]]):
                    #    self.costmatrix[node_start_y * 100 + node_start_x][node_end_y * 100 + node_end_x] = np.inf
                    infopath.append(infopoint)

                for xpoint in range(end_edge_x, node_end_x - 1, -1):
                    ypoint = node_end_y
                    #if [xpoint,
                    #    ypoint] not in currentinfopath:  # only info value when the point is not already monitored before
                    info += self.uncertaintymatrix[ypoint, xpoint]
                    if np.isnan(self.uncertaintymatrix[ypoint, xpoint]):
                        self.costmatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x] = np.inf
                        self.costmatrix[node_end_y * 100 + node_end_x, node_start_y * 100 + node_start_x] = np.inf # mirror the matrix
                    infopath.append([xpoint, ypoint])

            self.infomatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x] = info
            self.infomatrix[node_end_y * 100 + node_end_x, node_start_y * 100 + node_start_x] = info # mirror the matrix
            self.infopathmatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x] = infopath
            self.infopathmatrix[node_end_y * 100 + node_end_x, node_start_y * 100 + node_start_x] = infopath[::-1]

        infopath=infopath
        infonode = 0
        currentinfopath=[]
        curnode = node

        while curnode.parent:
            #print("index 1 = " + str(curnode.parent.y*100+curnode.parent.x) + " index 2 = " + str(
            #    curnode.y*100+curnode.x))
            currentinfopath.extend(self.infopathmatrix[curnode.parent.y*100+curnode.parent.x, curnode.y*100+curnode.x])
            curnode= curnode.parent
        if not any(element in currentinfopath for element in infopath): # the whole infopath is new
            infonode+=info
        else: #if some infopoints overlap
            for element in infopath:
                if not element in currentinfopath:
                    infonode+=self.uncertaintymatrix[element[1],element[0]]
        return cost,infonode
        #return [cost,infopath,info]

    def Rewiring(self, x_near, x_new):
        if x_new.parent.x == x_near.x and x_new.parent.y == x_near.y:
            return  # if the parent of x_new = x_near, we don't want to make the parent of x_near = x_new (because then we create a loose segment

        #cost_heuristic = abs(x_new.x-x_near.x)+abs(x_new.y-x_near.y) #manhattan distance
        #info_heuristic = cost_heuristic*(cost_heuristic/x_new.cost)*x_new.info #weighted average of x_new info
        cost_heuristic_x = (x_new.x-x_near.x)^2
        cost_heuristic_y = (x_new.y-x_near.y)^2
        c_near = x_near.cost

        if cost_heuristic_x < ((c_near-x_new.cost)**2) and cost_heuristic_y < ((c_near-x_new.cost)**2): #first do heuristic and then the actual calculation

            [cost,info] = self.FindCostInfo(x_near.x,x_near.y,x_new.x,x_new.y,x_new,True)

            info_new = x_new.info + info
            info_near = x_near.info

            c_new = x_new.cost+cost
            if c_new < c_near and info_new >= info_near:  # note: this is different than the condition in pruning
                x_near.parent = x_new
                x_near.info = info_new
                x_near.cost = c_new
                #x_near.infopath = infopath
                #print("Rewiring took place!!")
                self.Recalculate(x_near)  # recalculates the cost and info for nodes further down the path

    def Recalculate(self,parent):
        for node in self.V:  # to recalculate the cost and info for nodes further down the line
            if node.parent == parent:
                [dist,info] = self.FindCostInfo(node.x,node.y,node.parent.x,node.parent.y,parent,True)

                node.cost = parent.cost + dist
                node.info = parent.info + info
                self.LastPath(node)
                self.Recalculate(node)
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
                if (node2.cost<=node1.cost and node2.info>node1.info) or (node1.parent==node2.parent and index!=index2): #prune lesser paths or doubles
                    # if (node1.cost==node2.cost and node1.info==node2.info and index!=index2):
                    #     print("Double detected, now pruned")
                    # else:
                    #     print("Alternative pruned")
                    # prune the node from all nodes
                    nochildren = True
                    for allnode in self.V:
                        if allnode.parent == node1:  # in this case we can't prune the node because other nodes depend on it
                            nochildren = False
                    if nochildren:
                        if node1 in self.V: #still have to figure out why this is needed (TODO)
                            self.V.remove(node1)
                        #nodelist.pop(index)
                        #costlist.pop(index)
                        #infolist.pop(index)
                        #costlist[index]=np.nan
                        #infolist[index]=np.nan
                        # TODO still: how to "pop" or "remove" from the list we iterate over? to speed up pruningz
                        if node1 in self.X_soln:
                            self.X_soln.remove(node1)  # also from the solutions

    def Steer(self, x_start, x_goal):
        #dist, theta = self.get_distance_and_angle(x_start, x_goal)
        dist = self.FindCostInfo(x_goal.x,x_goal.y,x_start.x,x_start.y,x_start,False,True)

        # closer=False
        # if dist>self.step_len:
        #     #just for now for debugging purposes
        #     closer=True
        if dist<=self.step_len:
            print("x_rand close enough, x_rand = x_new --> x_rand=("+str(x_goal.x)+","+str(x_goal.y)+")")
            #return x_goal
            return Node((int(x_goal.x), int(x_goal.y)))
        distleft = self.FindCostInfo(self.row_edges[self.row_nrs.index(x_goal.y)][0], self.row_nrs[self.row_nrs.index(x_goal.y)], x_start.x, x_start.y,
                                     x_start, False, True)
        distright = self.FindCostInfo(self.row_edges[self.row_nrs.index(x_goal.y)][1], self.row_nrs[self.row_nrs.index(x_goal.y)], x_start.x, x_start.y,
                                      x_start, False, True)
        boolright = 0
        if distright<distleft:
            boolright=1 #bool right means right is shorter if boolright=1
        #sampling within the end row
        withinbounds=True
        distance=5
        while withinbounds and dist>self.step_len:
            if boolright==1:
                xpoint = x_goal.x+distance
                ypoint = self.row_nrs[self.row_nrs.index(x_goal.y)]
                if xpoint>=self.row_edges[self.row_nrs.index(x_goal.y)][boolright]:
                    withinbounds=False
                else:
                    dist = self.FindCostInfo(xpoint,ypoint, x_start.x, x_start.y,
                                         x_start, False, True)

            else:
                xpoint = x_goal.x-distance
                ypoint = self.row_nrs[self.row_nrs.index(x_goal.y)]
                if xpoint<=self.row_edges[self.row_nrs.index(x_goal.y)][boolright]:
                    withinbounds=False
                else:
                    dist = self.FindCostInfo(xpoint, ypoint, x_start.x, x_start.y,
                                             x_start, False, True)
            distance+=5

        index = self.row_nrs.index(x_goal.y)
        while dist>self.step_len:
            xpoint = self.row_edges[index][boolright]
            ypoint = self.row_nrs[index]
            dist= self.FindCostInfo(xpoint,ypoint, x_start.x, x_start.y,
                                     x_start, False, True)


            if index==self.row_nrs.index(x_start.y):
                break
            if self.row_nrs.index(x_goal.y) > self.row_nrs.index(x_start.y):
                index-=1
            else:
                index+=1


        # if dist>self.step_len: #left edge end
        #     dist = self.FindCostInfo(self.row_edges[self.row_nrs.index(x_goal.y)][0], x_goal.y, x_start.x, x_start.y, x_start.infopath, False, True)
        #     xpoint=self.row_edges[self.row_nrs.index(x_goal.y)][0]
        #     ypoint=x_goal.y
        # if dist>self.step_len: # right edge end
        #     dist = self.FindCostInfo(self.row_edges[self.row_nrs.index(x_goal.y)][1], x_goal.y, x_start.x, x_start.y, x_start.infopath, False, True)
        #     xpoint=self.row_edges[self.row_nrs.index(x_goal.y)][1]
        #     ypoint=x_goal.y
        # if dist>self.step_len: # left edge start
        #     dist = self.FindCostInfo(self.row_edges[self.row_nrs.index(x_start.y)][0], x_start.y, x_start.x, x_start.y, x_start.infopath, False, True)
        #     xpoint=self.row_edges[self.row_nrs.index(x_start.y)][0]
        #     ypoint=x_start.y
        # if dist>self.step_len: # right edge start
        #     dist = self.FindCostInfo(self.row_edges[self.row_nrs.index(x_start.y)][1], x_start.y, x_start.x, x_start.y, x_start.infopath, False, True)
        #     xpoint=self.row_edges[self.row_nrs.index(x_start.y)][1]
        #     ypoint=x_start.y
        if dist>self.step_len:
            xpoint=random.choice([x_start.x-self.step_len,x_start.x+self.step_len])
            ypoint=x_start.y
            dist=self.step_len
            print("last resort sampling: in own row")
        #node_new.parent = x_start
        #print("WE NEED TO SAMPLE CLOSER TO THE NEAREST: ("+str(x_start.x)+","+str(x_start.y)+")")
        print("x_nearest=("+str(x_start.x)+","+str(x_start.y)+")")
        print("x_rand=("+str(x_goal.x)+","+str(x_goal.y)+") - dist = "+str(dist)+" - x_new=("+str(xpoint)+","+str(ypoint)+")")
        return Node((int(xpoint),int(ypoint)))

    def Steer_section(self, x_start, x_goal): # with sectioning
        xpoint=x_start.x
        ypoint = x_start.y # just for debugging now

        #dist, theta = self.get_distance_and_angle(x_start, x_goal)
        dist = self.FindCostInfo(x_goal.x,x_goal.y,x_start.x,x_start.y,x_start,False,True)

        # closer=False
        # if dist>self.step_len:
        #     #just for now for debugging purposes
        #     closer=True
        if dist<=self.step_len:
            print("x_nearest=(" + str(x_start.x) + "," + str(x_start.y) + ")")
            print("x_rand close enough, x_rand = x_new --> x_rand=("+str(x_goal.x)+","+str(x_goal.y)+")")
            #return x_goal
            return Node((int(x_goal.x), int(x_goal.y)))
        distleft = self.FindCostInfo(self.row_edges[self.row_nrs.index(x_goal.y)][0], self.row_nrs[self.row_nrs.index(x_goal.y)], x_start.x, x_start.y,
                                     x_start, False, True)
        distright = self.FindCostInfo(self.row_edges[self.row_nrs.index(x_goal.y)][1], self.row_nrs[self.row_nrs.index(x_goal.y)], x_start.x, x_start.y,
                                      x_start, False, True)
        boolright = 0
        direction=-1
        if distright<distleft:
            boolright=1 #bool right means right is shorter if boolright=1
            direction=1
        # steps:
        # 1) is edgepoint on start row too far?
        # 2) is edgepoint of end row too far?
        # 3) is final point too far?

        # step 1:
        dist = self.FindCostInfo(self.row_edges[self.row_nrs.index(x_start.y)][boolright], self.row_nrs[self.row_nrs.index(x_start.y)], x_start.x, x_start.y,
                                     x_start, False, True)
        if dist>=self.step_len:
            #print("step 1")
            # find point within start row
            step=0
            while dist>self.step_len:
                xpoint = self.row_edges[self.row_nrs.index(x_start.y)][boolright]-direction*step #note the - sign because we're going backwards on the path
                ypoint = x_start.y
                dist = self.FindCostInfo(xpoint, ypoint, x_start.x, x_start.y,
                                     x_start, False, True)
                step+=1
            print("x_rand=(" + str(x_goal.x) + "," + str(x_goal.y) + ") - dist = " + str(dist) + " - x_new=(" + str(
                xpoint) + "," + str(ypoint) + ")")

            return Node((int(xpoint),int(ypoint)))
        # step 2:

        dist = self.FindCostInfo(self.row_edges[self.row_nrs.index(x_goal.y)][boolright], self.row_nrs[self.row_nrs.index(x_goal.y)], x_start.x, x_start.y,
                                     x_start, False, True)
        if dist>=self.step_len:
           #print("step 2")
           # find point on edge between start and end edge
           updown=-1
           if x_goal.y>x_start.y:
               updown=1
           step=0
           while dist > self.step_len:
               xpoint = self.row_edges[self.row_nrs.index(x_goal.y)-updown*step][
                     boolright]
               ypoint = self.row_nrs[self.row_nrs.index(x_goal.y)-updown*step]
               dist = self.FindCostInfo(xpoint,ypoint,x_start.x,x_start.y,x_start,False,True)
               step+=1
           print("x_rand=(" + str(x_goal.x) + "," + str(x_goal.y) + ") - dist = " + str(dist) + " - x_new=(" + str(
               xpoint) + "," + str(ypoint) + ")")

           return Node((int(xpoint),int(ypoint)))


        # step 3:
        # find point on end row
        #print("step 3")
        dist = self.FindCostInfo(x_goal.x,x_goal.y,x_start.x,x_start.y,x_start,False,True)
        step = 0
        while dist > self.step_len:
            xpoint = x_goal.x + direction * step  # note the - sign because we're going backwards on the path
            ypoint = x_goal.y
            dist = self.FindCostInfo(xpoint, ypoint, x_start.x, x_start.y, x_start, False, True)
            step+=1

        if xpoint==x_start.x and ypoint==x_start.y:
            print("Something went wrong, no new location found")
        print("x_rand=("+str(x_goal.x)+","+str(x_goal.y)+") - dist = "+str(dist)+" - x_new=("+str(xpoint)+","+str(ypoint)+")")
        return Node((int(xpoint),int(ypoint)))


    def Near(self, nodelist, node,max_dist=0):
        if max_dist==0:
            max_dist = self.step_len
        #heuristic:
        nodelist_new = nodelist[:]
        for nd in nodelist: #TODO check if this actually speeds things up
            if (nd.x-node.x)**2>max_dist**2 and (nd.y-node.y)**2>max_dist**2:
                nodelist_new.remove(nd)
        #actual calculation
        dist_table = [self.FindCostInfo(nd.x, nd.y, node.x, node.y, node, False, True) for nd in nodelist]
        X_near = [nodelist[ind] for ind in range(len(dist_table)) if (dist_table[ind] <= max_dist and dist_table[ind] > 0.0
                                                      and not self.utils.is_collision(nodelist[ind], node)==True)]
        #print("number of near nodes: "+str(len(X_near)))
        return X_near

    #def Sample(self, c_max, c_min, x_center, C):
    def Sample(self):

        # c_max=np.inf
        # if c_max < np.inf:
        #     print("not random sampling")
        #     r = [c_max / 2.0,
        #          math.sqrt(c_max ** 2 - c_min ** 2) / 2.0,
        #          math.sqrt(c_max ** 2 - c_min ** 2) / 2.0]
        #     L = np.diag(r)
        #
        #     while True:
        #         x_ball = self.SampleUnitBall()
        #         #x_rand = np.dot(np.dot(C, L), x_ball) + x_center
        #         if self.x_range[0] + self.delta <= x_rand[0] <= self.x_range[1] - self.delta and \
        #                 self.y_range[0] + self.delta <= x_rand[1] <= self.y_range[1] - self.delta:
        #             break
        #     x_rand = Node((x_rand[(0, 0)], x_rand[(1, 0)]))
        # else:
        #     x_rand = self.SampleFreeSpace()
        x_rand = self.SampleFreeSpace()
        return x_rand

    @staticmethod
    def SampleUnitBall():
        while True:
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if x ** 2 + y ** 2 < 1:
                return np.array([[x], [y], [0.0]])

    def SampleFreeSpace(self):
        location = random.choice(self.allpoints)
        return Node((location[0],location[1]))

    def ExtractPath(self, node):
        print("Final cost: "+str(node.totalcost))
        print("Final info value: "+str(node.totalinfo))
        #path = [[self.x_goal.x, self.x_goal.y]]
        path= []
        path.extend((self.infopathmatrix[node.y*100+node.x, self.x_goal.y*100+self.x_goal.x])[::-1])
        while node.parent:
            #path.append([node.x, node.y])
            path.extend((self.infopathmatrix[node.parent.y*100+node.parent.x, node.y*100+node.x])[::-1])
            node = node.parent

        #path.append([self.x_start.x, self.x_start.y])

        return path

    def InGoalRegion(self, node):
        #if self.Line(node, self.x_goal) < self.step_len:
        if node.cost+self.FindCostInfo(self.x_goal.x, self.x_goal.y, node.x, node.y, node, False, True) <= self.budget:
            return True

        return False

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
        #return nodelist[int(np.argmin([(nd.x - n.x) ** 2 + (nd.y - n.y) ** 2
        #                               for nd in nodelist]))]
        return nodelist[int(np.argmin([self.FindCostInfo(nd.x, nd.y, n.x, n.y, n, False, True) for nd in nodelist]))]

    def LastPath(self, node):
        [cost,info] = self.FindCostInfo(self.x_goal.x,self.x_goal.y,node.x,node.y,node,False)
        node.totalcost = node.cost + cost
        node.totalinfo = node.info + info
        #node.lastinfopath = lastinfopath

    @staticmethod
    def Line(x_start, x_goal):
        return math.hypot(x_goal.x - x_start.x, x_goal.y - x_start.y)

    def Cost(self, node):
        if node == self.x_start:
            return 0.0

        if node.parent is None:
            return np.inf

        cost = self.FindCostInfo(node.x, node.y, node.parent.x, node.parent.y, node.parent, False, True)


        #while node.parent:
        #    #print("node.x: "+str(node.x)+"node.parent.x: "+str(node.parent.x))
        #    #print("node.y: "+str(node.y)+"node.parent.y: "+str(node.parent.y))
        #    cost += node.parent.cost
        #    node = node.parent
        cost+=node.parent.cost
        return cost
    def Info(self,node):
        if node.parent is None:
            return 0.0
        info = self.uncertaintymatrix[int(node.y),int(node.x)]

        while node.parent:
            info += node.parent.info
            node = node.parent
        return info

    def Info_cont(self,node):
        if node.parent is None:
            return 0.0
        [cost,info] = self.FindCostInfo(node.x,node.y,node.parent.x,node.parent.y,node.parent,True)
        #node.infopath = infopath
        info+=node.parent.info

        return info

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    #def animation(self, x_center=None, c_best=None, dist=None, theta=None):
    def animation(self):

        plt.cla()
        self.plot_grid("Informed rrt*, N = " + str(self.iter_max))
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        # for node in self.V:
        #     if node.parent:
        #         #reachedparent=False
        #         prevpoint=[node.x,node.y]
        #         for point in node.infopath[::-1]:
        #             reachedparent= (point[0]==node.parent.x and point[1]==node.parent.y)
        #
        #             plt.plot([point[0], prevpoint[0]], [point[1], prevpoint[1]], "-g")
        #             prevpoint=point
        #             if reachedparent:
        #                 break
        for node in self.V:
            if node.parent:
                #reachedparent=False
                prevpoint=[node.x,node.y]
                infopath = self.infopathmatrix[node.parent.y*100+node.parent.x, node.y*100+node.x]
                for point in infopath[::-1]:
                    #reachedparent= (point[0]==node.parent.x and point[1]==node.parent.y)

                    plt.plot([point[0], prevpoint[0]], [point[1], prevpoint[1]], "-g")
                    prevpoint=point
                    #if reachedparent:
                    #    break

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



def main(uncertaintymatrix,row_nrs,row_edges):
    x_start = (50, 48)  # Starting node
    #x_goal = (37, 18)  # Goal node
    x_goal = (50,48)

    rrt_star = IRrtStar(x_start, x_goal, 25, 0.0, 25, 2000,uncertaintymatrix,row_nrs,row_edges)
    [finalpath, finalcost, finalinfo, budget, steplength, searchradius, iteration]=rrt_star.planning()

    return finalpath, finalcost, finalinfo, budget, steplength, searchradius, iteration


if __name__ == '__main__':
    main()
