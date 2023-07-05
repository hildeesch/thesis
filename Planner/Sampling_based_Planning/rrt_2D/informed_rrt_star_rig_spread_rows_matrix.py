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
from Planner.Sampling_based_Planning.rrt_2D import dubins_path as dubins
from Planner.Sampling_based_Planning.rrt_2D import reeds_shepp as reedsshepp

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")
from copy import deepcopy
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from Sampling_based_Planning.rrt_2D import env, plotting, utils
from Planner.Sampling_based_Planning.rrt_2D import env, plotting, utils
import time
class Node:
    __slots__ = ['x','y','parent','cost','info','totalcost','totalinfo','round','prevroundcost']
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
        # tworoundstrategy2:
        self.round = 1
        self.prevroundcost = 0

class NodeA:
    """
        A node class for A* Pathfinding
        parent is parent of the current Node
        position is current position of the Node in the maze
        g is cost from start to current Node
        h is heuristic based estimated cost for current Node to end Node
        f is total cost of present node i.e. :  f = g + h
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
class IRrtStar:
    def __init__(self, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max,uncertaintymatrix,row_nrs,row_edges,field_vertex,scenario,matrices,samplelocations):
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
        #self.X_soln = set()
        self.X_soln = []
        self.path = None

        self.x_best = self.x_start
        self.i_best = 0

        # for stopping criterion:
        self.k_list, self.i_list, self.i_list_avg, self.i_list_avg_der, self.i_list_avg_der2 = [], [], [], [], []

        self.inforadius=0
        self.scenario = scenario
        self.samplelocations = samplelocations
        self.samplelocations_add = False
        if not samplelocations:
            self.samplelocations = []
            self.samplelocations_add = True
        # scenario = [rowsbool, budget, informed, rewiring, step_len, search_radius, stopsetting, horizonplanning]
        if scenario:
            self.budget = scenario[1]
            self.boolrewiring=scenario[3]
            self.stopsetting=scenario[6]
            self.doubleround=scenario[7]
            self.pathname = scenario[-1]
        else:
            self.budget=500
            self.boolrewiring=True
            self.stopsetting="mild"
            self.doubleround=False
            self.pathname=None

        self.kinematic = "none"  # kinematic constraint
        # choices: "none", "dubins", "reedsshepprev", "reedsshepp", "ranger", "limit"
        if self.kinematic == "ranger" or self.kinematic=="limit":
            self.angularcost=1 # set the turning cost (in rad)
        self.dubinsmat = {}

        # for saving of cost/info matrices
        self.uncertaintymatrix = uncertaintymatrix
        self.row_nrs = row_nrs
        self.row_edges = row_edges
        self.field_vertex = field_vertex # cornerpoints of the field
        for vertex in self.field_vertex: #if the vertex is within a row, the xposition is made equal to the edge of the row (for simplicity)
            for index,row in enumerate(self.row_nrs):
                if vertex[1]==row:
                    vertex[0]=self.row_edges[index][vertex[3]]
                    #print("Vertex position changed")
        self.cost_left = np.zeros((len(self.row_nrs),len(self.row_nrs)))
        self.cost_right = np.zeros((len(self.row_nrs),len(self.row_nrs)))
        self.infopath_left = np.empty((len(self.row_nrs),len(self.row_nrs)),dtype = object )
        self.infopath_right = np.empty((len(self.row_nrs),len(self.row_nrs)),dtype = object )
        self.info_left = np.zeros((len(self.row_nrs),len(self.row_nrs)))
        self.info_right = np.zeros((len(self.row_nrs),len(self.row_nrs)))

        self.infopathmatrix = np.empty((100 * 100, 100 * 100), dtype=object)
        self.infomatrix = np.empty((100 * 100, 100 * 100))
        if matrices is None:
            self.costmatrix = np.empty((100*100,100*100) )
            # self.infopathmatrix = np.empty((100 * 100, 100 * 100), dtype=object)
            # self.infomatrix = np.empty((100 * 100, 100 * 100))
            self.anglematrix = np.empty((100 * 100, 100 * 100))
        else:
            self.costmatrix = matrices[0]
            # self.infopathmatrix = matrices[1]
            # self.infomatrix= matrices[2]
            self.anglematrix = matrices[1]
        #self.directionmatrix = np.empty((100*100,100*100),dtype = object )

        self.allpoints=[] # all the points that can be sampled for nodes
        if self.kinematic=="dubins" or self.kinematic=="reedsshepp":
            for index,row in enumerate(row_nrs):
                self.allpoints.append([self.row_edges[index][0],row])
                self.allpoints.append([self.row_edges[index][1],row])

        else:
            for index,row in enumerate(row_nrs):
                for column in range(self.row_edges[index][0],self.row_edges[index][1]+1):
                    self.allpoints.append([column,row])
        if len(self.allpoints)<self.iter_max:
            self.iter_max=len(self.allpoints)
            print("Number of iterations reducted to the maximal amount of points to be sampled: "+str(self.iter_max))

        self.rowdist=4

        self.time = np.zeros(8) # for debugging
        # 0 = sample, 1 = nearest, 2 = steer, 3 = near, 4 = rewiring, 5 = lastpath, 6 = pruning, 7 = total time

        self.maze = np.ones((100,100) ) # for A star
        #self.edgemaze = np.ones((100,100)) # for A star (to know where we can go up/down)


    def init(self):
        #cMin, theta = self.get_distance_and_angle(self.x_start, self.x_goal)
        #C = self.RotationToWorldFrame(self.x_start, self.x_goal, cMin)
        #xCenter = np.array([[(self.x_start.x + self.x_goal.x) / 2.0],
        #                    [(self.x_start.y + self.x_goal.y) / 2.0], [0.0]])
        x_best = self.x_start

        #return theta, cMin, xCenter, C, x_best

        self.EdgeCostInfo()
        # for row in range(100):
        #     for col in range(100):
        #         if [col,row] in self.allpoints:
        #             self.maze[row,col]=0

        # fig, ax1 = plt.subplots(1,1)
        # colormap = cm.Greys
        # colormap.set_bad(color='black')
        # im = ax1.imshow(self.maze, cmap=colormap, vmin=0, vmax=1, origin='lower')
        # ax1.set_title("A star maze")
        # fig.tight_layout()
        # plt.show()

        # fig, ax = plt.subplots()
        # colormap = cm.Greys
        # colormap.set_bad(color='black')
        # im = ax.imshow(self.edgemaze, cmap=colormap, vmin=0, vmax=1, origin='lower')
        #
        # ax.set_title("A star maze")
        # fig.tight_layout()
        # plt.show()
        #
        # fig, ax = plt.subplots()
        # tempmatrix = self.uncertaintymatrix
        # tempmatrix[tempmatrix>0]=0
        # for point in self.allpoints:
        #     tempmatrix[point[1],point[0]]=0.5
        # colormap = cm.Oranges
        # colormap.set_bad(color='black')
        # im = ax.imshow(tempmatrix, cmap=colormap, vmin=0, vmax=1, origin='lower')
        #
        # ax.set_title("Points that can be sampled")
        # fig.tight_layout()
        # plt.show()
        return x_best

    def planning(self):
        show=False
        doubleround=self.doubleround
        rewiringafter=False
        visualizationmode="nosteps" #steps, nosteps or False
        #theta, dist, x_center, C, x_best = self.init()
        #theta, dist, x_center, x_best = self.init()
        x_best = self.init()
        #c_best = np.inf
        #count_down=3
        i_best = 0
        startlen=0 # for checking node increase
        totalstarttime=time.time()


        if self.kinematic not in ["none","limit","ranger","dubins","reedsshepp","reedsshepprev"]:
            print("Kinematic constraint setting unknown. Set one of the following: none,limit,ranger,dubins,reedsshepp,reedsshepprev")
            return None


        k = 0
        iter_min=0
        if iter_min>self.iter_max:
            iter_min=0
        stopcriterion = False
        while k < self.iter_max:
            k += 1
            # time.sleep(0.1)
            if k >= 3 - 3:  # only evaluate from when we might want it to stop
                cost = {node: node.totalcost for node in self.X_soln}
                info = {node: node.totalinfo for node in self.X_soln}
                # x_best = min(cost, key=cost.get)
                if len(info) > 0 and not double:
                    self.x_best = max(info, key=info.get)
                    x_best = max(info, key=info.get)
                    # c_best = cost[x_best]
                    i_last_best = i_best
                    self.i_best = info[x_best]

                    i_best = info[x_best]

                    stopcriterion = self.StopCriterion(k)
                    if stopcriterion:
                        print("Stop criterion = True at it. "+str(k))
                        stopcriterion=False

                    #if i_last_best>0: # to prevent division by zero
                    # if ((i_best-i_last_best)/i_last_best)<0.01: #smaller than 1% improvement
                    #     count_down-=1
                    # else:
                    #     count_down=20 #reset
                    #     print("Reset countdown")
            if k==501: # to test up to certain iteration
                #count_down=0
                stopcriterion=True
            #if count_down<=0 and (k>200 or k>(self.iter_max-3)):
            if stopcriterion and (k>iter_min or k>(self.iter_max-3)):
                print("Reached stopping criterion at iteration "+str(k))
                break # we stop iterating if the best score is not improving much anymore and we already passed at least ... cycles

            if k%50==0 and not double:
                print("ATTENTION!!! ATTENTION!!! ATTENTION!!! AGAIN FIFTY CYCLES FURTHER, CURRENT CYCLE ="+str(k)) # to know how far we are
            endlen=len(self.V)
            print("Nr of nodes added: "+str(endlen-startlen))
            if (endlen-startlen)==0 and not double:
                k-=1
            startlen = len(self.V)

            if visualizationmode=="nosteps" and k>1 and not double:  # visualize all new connections with near ndoes
                self.fig, self.ax = plt.subplots()
                self.animation(k-1, x_new)
                if self.pathname and k%25==0:
                    plt.savefig(self.pathname + "animation_"+str(k))
                if show:
                    plt.show()
                else:
                    plt.close()


            timestart = time.time()
            if self.samplelocations_add:
                x_rand = self.SampleFreeSpace()
                self.samplelocations.append(x_rand)
            else:
                x_rand=self.samplelocations[k]
                if double: # otherwise we keep trying the same location again and again
                    x_rand = self.SampleFreeSpace()
            timeend= time.time()
            self.time[0]+=(timeend-timestart)
            timestart=time.time()
            x_nearest = self.Nearest(self.V, x_rand)
            timeend=time.time()
            self.time[1] += (timeend - timestart)
            timestart=time.time()
            #x_new = self.Steer_section(x_nearest, x_rand) #so that we only generate one new node, not multiple
            x_new = self.SteerAstar(x_nearest, x_rand) #so that we only generate one new node, not multiple
            timeend=time.time()
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
                    if self.samplelocations_add:
                        self.samplelocations.pop()
                    break
            #if x_nearest.cost + self.Line(x_nearest, x_new) + self.Line(x_new, self.x_goal) < self.budget and not double:  # budget check for nearest parent (to make it more efficient)
            if not double:  # budget check for nearest parent (to make it more efficient)
                # print(x_nearest.cost + self.Line(x_nearest, x_new) + self.Line(x_new, self.x_goal))
                for x_near in self.Near(self.V,x_new):

                    c_min = x_near.cost+self.FindCostInfoA(x_new.x,x_new.y,x_near.x,x_near.y,x_near,False,True) #cost from near node to new node


                    node_new = Node((x_new.x,x_new.y))
                    node_new.parent = x_near #added
                    node_new.cost = self.Cost(node_new) #+self.Line(x_new, self.x_goal)
                    node_new.info = self.Info_cont(node_new)
                    self.V.append(node_new) #generate a "node"/trajectory to each near point
                    # tworoundstrategy2
                    if doubleround:
                        if node_new.parent.round == 2:  # or ([node_new.x,node_new.y]==[self.x_goal.x,self.x_goal.y] and node_new.cost!=0):
                            node_new.round = 2
                            node_new.prevroundcost = node_new.parent.prevroundcost


                    timestart = time.time()
                    self.LastPath(node_new)
                    timeend = time.time()
                    self.time[5] += (timeend - timestart)
                    if node_new.totalcost <= self.budget:  # extra check for budget for actual parent (cmin+ cost to goal node)
                        self.X_soln.append(node_new)
                    # tworoundstrategy2:
                    if doubleround:
                        if node_new.round == 2 and (node_new.totalcost - node_new.prevroundcost) <= (self.budget):
                            # if a node_new is in round 2, it is allowed to have a double budget
                            # to make sure we don't surpass the original budget in round 1, we add that every parent also has to be a solution
                            # self.X_soln.add(node_new)
                            self.X_soln.append(node_new)


                timestart=time.time()
                self.Pruning(x_new)
                timeend = time.time()
                self.time[6] += (timeend - timestart)
                # tworoundstrategy2:
                if doubleround:
                    self.RoundTwoAdd(node_new)

            if (k-1) % 50 == 0 and not double:
                if show:
                    self.animation(k-1,x_new)
                self.time[7] = time.time()-totalstarttime
                print(self.time)
                if k>0:
                    print("It.: " + str(k) + " Time: " + str(self.time[7]) + " Info: " + str(x_best.info) + " Tot. info: "+str(x_best.totalinfo) + " Cost: " + str(x_best.totalcost) + " Nodes: "+str(len(self.V)))
                    # for i in range(10): # rewire the 10 best nodes
                    #     info = {node: node.totalinfo for node in self.X_soln}
                    #     #self.x_best = max(info, key=info.get)
                    #     curnode = sorted(info, key=info.get)[-(i+1)]
                    #     self.Rewiring_afterv2(curnode)
                    # #self.Rewiring_after(self.x_best)
                    # info = {node: node.totalinfo for node in self.X_soln}
                    # self.x_best = max(info, key=info.get)
                    # print("Best node after rewiring: tot. info: "+str(x_best.totalinfo)+" Cost: "+str(x_best.totalcost))

        # Rewiring in Hindsight:
        if rewiringafter or not doubleround and self.boolrewiring:
            info = {node: node.totalinfo for node in self.X_soln}
            for i in range(10):  # rewire the 10 best nodes
                # self.x_best = max(info, key=info.get)
                curnode = sorted(info, key=info.get)[-(i + 1)]
                self.Rewiring_afterv2(curnode, doubleround)
            # self.Rewiring_after(self.x_best)
            info = {node: node.totalinfo for node in self.X_soln}
            self.x_best = max(info, key=info.get)
            x_best = max(info, key=info.get)
            print("Best node after rewiring: tot. info: " + str(x_best.totalinfo) + " Cost: " + str(
                x_best.totalcost))
        if doubleround and not rewiringafter and self.boolrewiring:
            doubleround = False  # to make sure it rewires in the correct way
            top20nodes = []  # top 10 nodes split in two rounds each
            info = {node: node.totalinfo for node in self.X_soln}
            for i in range(10):  # rewire the 10 best nodes
                # self.x_best = max(info, key=info.get)
                curnode = sorted(info, key=info.get)[-(i + 1)]
                print("Top 10 nr "+str(i)+" node: "+str(curnode.x),str(curnode.y)+" Cost,info: "+str(curnode.totalcost),str(curnode.totalinfo))
                [nodefirstround, nodesecondround] = self.splitDoublePath(curnode)
                if nodefirstround not in top20nodes:
                    nodefirstround = self.Rewiring_afterv2(nodefirstround, doubleround)
                    top20nodes.append(nodefirstround)
                elif nodefirstround in top20nodes and (not nodesecondround or nodesecondround in top20nodes): # just for now (debug)
                    print("Nodefirstround seems to be double somehow")
                if nodesecondround and nodesecondround not in top20nodes:  # not None
                    nodesecondround = self.Rewiring_afterv2(nodesecondround, doubleround)
                    top20nodes.append(nodesecondround)
                else:
                    print("No second round to be rewired")
            info = {node: node.totalinfo for node in top20nodes}
            self.x_best = max(info, key=info.get)
            x_best = max(info, key=info.get)
            print("Best node after rewiring: tot. info: " + str(x_best.totalinfo) + " Cost: " + str(
                x_best.totalcost))
            doubleround = True  # set it back again


        # Extracting the final path (best path)
        #self.path = self.ExtractPath(x_best)
        [self.path,infopathradius] = self.ExtractPath(x_best)
        print(infopathradius)
        node = x_best
        #infopathlength=len(self.infopathmatrix[node.y*100+node.x,self.x_goal.y*100+self.x_goal.x])
        check = False
        if check:
            finalpath= self.infopathmatrix[node.y*100+node.x,self.x_goal.y*100+self.x_goal.x]
            checkcostscore=self.Line(node,self.x_goal)
            while node.parent:
                checkcostscore+=self.Line(node.parent,node)
                #infopathlength += len(self.infopathmatrix[node.parent.y*100+node.parent.x,node.y*100+node.x])
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

        if show:
            self.animation()
            plt.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
            node = x_best
            while node.parent:
                prevpoint = [node.x, node.y]
                # reachedparent=False
                if self.kinematic!="dubins" and self.kinematic!="reedsshepp":
                    infopath = self.infopathmatrix[node.parent.y * 100 + node.parent.x, node.y * 100 + node.x]
                else:
                    start = [node.parent.x, node.parent.y]
                    end = [node.x, node.y]
                    [infoastar, costastar] = self.search(start, end)  # A star steps
                    infopath = self.getInfopath(node.x, node.y, node.parent.x, node.parent.y, node.parent,infoastar)[0]
                for point in infopath[::-1]:
                    # reachedparent= (point[0]==node.parent.x and point[1]==node.parent.y)

                    plt.plot([point[0], prevpoint[0]], [point[1], prevpoint[1]], "-b")
                    prevpoint = point
                node = node.parent
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
            for row in range(100):
                for col in range(100):
                    if self.maze[row, col] == 0:
                        ax.add_patch(patches.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                                            edgecolor='black',
                                                            facecolor='0.5',
                                                            fill=True
                                                            ))
            #ax.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
            for cell in infopathradius: #TODO: something goes wrong here with the kinematics
                ax.plot(cell[0],cell[1],marker="o",markersize=1,color="blue")
            ax.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')

            # [currentinfopath, cost] = self.search(self.maze, self.edgemaze, [x_best.x,x_best.y], [self.x_goal.x,self.x_goal.y])
            # ax.plot([x for x, _ in currentinfopath], [y for _, y in currentinfopath], '-g')
            node=x_best
            i=0
            while node.parent:
                ax.plot(node.x, node.y, marker='o', markersize=6, color="pink")
                ax.text(node.x-1,node.y-1,i,color="blue",fontsize=8)
                node=node.parent
                i+=1
            ax.plot(node.x, node.y, marker='o', markersize=6, color="pink")
            ax.text(node.x - 1, node.y - 1, i, color="blue", fontsize=8)

            ax.plot(x_best.x, x_best.y, marker='o',markersize=6,color="green")

            #ax.plot([x for x, _ in x_best.infopath], [y for _, y in x_best.infopath], '-r')
            #ax.plot([x for x, _ in x_best.lastinfopath], [y for _, y in x_best.lastinfopath], '-r')
            ax.set_title("Spatial distribution of uncertainty and final path")
            fig.tight_layout()
            plt.show()
            #plt.close()

            k_list_avg_der_neg=[]
            i_list_avg_der_neg=[]
            k_list_avg_der2_neg = []
            i_list_avg_der2_neg = []
            for index in range(len(self.k_list)):
                if self.i_list_avg_der[index]<=0:
                    k_list_avg_der_neg.append(self.k_list[index])
                    i_list_avg_der_neg.append(self.i_list_avg_der[index])
                if self.i_list_avg_der2[index]<=0:
                    k_list_avg_der2_neg.append(self.k_list[index])
                    i_list_avg_der2_neg.append(self.i_list_avg_der2[index])

            fig, ax = plt.subplots(2, 2)
            ax[0, 0].scatter(self.k_list, self.i_list, s=0.5)
            ax[0, 0].set_title("Best node info")
            ax[0, 1].scatter(self.k_list, self.i_list_avg, s=0.5)
            ax[0, 1].set_title("10 best nodes info (avg)")
            ax[1, 0].scatter(self.k_list, self.i_list_avg_der, s=0.5)
            ax[1,0].scatter(k_list_avg_der_neg, i_list_avg_der_neg,s=0.5,c='r')
            ax[1, 0].set_title("10 best nodes info increase 10 it")
            ax[1, 1].scatter(self.k_list, self.i_list_avg_der2, s=0.5)
            ax[1, 1].set_title("10 best nodes info increase 10 der2")
            # ax[3, 2].scatter(k_list, i_list_perc_10_avg_der)
            # ax[3, 2].set_title("10 best nodes info increase % 10 der2")
            # ax.legend(["Best node info","Best node increase","Best node increase %","10 nodes info","10 nodes increase","10 nodes increase %"])
            # ax.grid()
            plt.show()


            fig, ax = plt.subplots(2, 1)
            ax[0].scatter(self.k_list, self.i_list_avg_der,s=0.5)
            ax[0].scatter(k_list_avg_der_neg, i_list_avg_der_neg,s=0.5,c='r')
            ax[0].set_title("10 best nodes info increase 10 it")
            ax[0].grid()
            ax[0].set_ylim((-0.05,0.05))
            ax[1].scatter(self.k_list, self.i_list_avg_der2,s=0.5)
            ax[1].scatter(k_list_avg_der2_neg, i_list_avg_der2_neg,s=0.5,c='r')
            ax[1].set_title("10 best nodes info increase 10 der2")
            ax[1].grid()
            ax[1].set_ylim((-0.05,0.05))

            plt.show()

        if doubleround and rewiringafter:

            [nodefirstround,nodesecondround]=self.splitDoublePath(node)

            # node = nodesecondround
            # while node.parent:
            #     print(node.info)
            #     node=node.parent
            if nodefirstround.totalinfo>nodesecondround.totalinfo or nodesecondround==None: # round 1 is better or there is no actual second round
                x_best=nodefirstround
                [self.path, infopath] = self.ExtractPath(nodefirstround)
                print("Executed path is round 1")

            else:
                x_best=nodesecondround
                [self.path, infopath] = self.ExtractPath(nodesecondround)
                print("Executed path is round 2")
        # final return
        matrices= [self.costmatrix,self.anglematrix]
        if self.pathname:
            np.save(self.pathname + 'i_list.npy', self.k_list)
            np.save(self.pathname + 'k_list.npy', self.i_list)
            np.save(self.pathname + 'i_list_avg_der.npy', self.i_list_avg_der)
            np.save(self.pathname + 'i_list_avg_der2.npy', self.i_list_avg_der2)
        return self.path, infopathradius, x_best.totalcost, x_best.totalinfo, self.budget, self.step_len, self.search_radius, k, matrices, self.samplelocations

    def StopCriterion(self, k):
        stopcriterion = False
        # self.StopCriterion(k_list, i_list, i_list_avg, i_list_avg_der, i_list_avg_der2)
        info = {node: node.totalinfo for node in self.X_soln}  # TODO: maybe see if can prevent computing this twice

        self.k_list.append(k)
        self.i_list.append(self.i_best)

        i_list_avg_cur = []
        # if len(self.X_soln)>=10:
        avg_amount=20 # over how many nodes do we average (the best x nodes)
        for i in range(1, min(avg_amount+1, len(info) + 1)):  # average over the 10 best nodes (or less if there are not 10 yet)
            # for i in range(int(np.ceil(len(self.X_soln)/10))): # average over the top 10% of nodes
            # self.x_best = max(info, key=info.get)
            i_list_avg_cur.append(info[sorted(info, key=info.get)[-i]])
        self.i_list_avg.append(sum(i_list_avg_cur) / len(i_list_avg_cur))
        # derivatives: #note that we use the derivative of the actual info values, not the increase
        if self.stopsetting=="mild":
            der_iterations = 25
        else:
            der_iterations = 50
        prevnr = min(len(self.i_list_avg), der_iterations + 1)
        if prevnr < 2:
            self.i_list_avg_der.append(1)
        else:
            self.i_list_avg_der.append((self.i_list_avg[-1] - self.i_list_avg[-(prevnr)]) / (prevnr - 1))
        # prevnr = min(len(i_list_inc_10_avg), 26)
        prevnr = min(len(self.i_list_avg_der), 2)
        if prevnr < 2:
            self.i_list_avg_der2.append(1)
        else:
            self.i_list_avg_der2.append((self.i_list_avg_der[-1] - self.i_list_avg_der[-(prevnr)]) / (prevnr - 1))

        if self.i_list_avg_der[-1]==0 and k>der_iterations:
            stopcriterion=True

        return stopcriterion
    def splitDoublePath(self,initial_node):
        if initial_node.round==1 or ((initial_node.totalcost-initial_node.prevroundcost)==0): # there is no second round
            return initial_node,None
        node=initial_node
        prev_copynode = None
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
        startsecondround.cost = self.Cost(startsecondround)
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
            self.maze[self.row_nrs[index],self.row_edges[index][0]]=0
            self.maze[self.row_nrs[index],self.row_edges[index][1]]=0
            # self.edgemaze[self.row_nrs[index],self.row_edges[index][0]]=0
            # self.edgemaze[self.row_nrs[index],self.row_edges[index][1]]=0

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

            #for A star:
            #if self.kinematic!="dubins" and self.kinematic!="reedsshepp":
            if True:  # for now

                for gridpoint in infopath:
                    self.maze[gridpoint[1],gridpoint[0]]=0
                    #self.edgemaze[gridpoint[1],gridpoint[0]]=0
                    path_width=2
                    for i in range(path_width+1):
                        self.maze[gridpoint[1], gridpoint[0] - (i)] = 0
                        self.maze[gridpoint[1], gridpoint[0] + (i)] = 0
                        #self.maze[gridpoint[1], gridpoint[0]+(i+1)] = 0
                        #self.maze[gridpoint[1], gridpoint[0] - (i + 1)] = 0

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

            #for A star:
            #if self.kinematic!="dubins" and self.kinematic!="reedsshepp":
            if True: # for now
                for gridpoint in infopath:
                    self.maze[gridpoint[1],gridpoint[0]]=0

                    for i in range(path_width):
                        self.maze[gridpoint[1], gridpoint[0] + (i)] = 0
                        self.maze[gridpoint[1], gridpoint[0] - (i)] = 0
                        #self.maze[gridpoint[1], gridpoint[0]-(i+1)] = 0
                        #self.maze[gridpoint[1], gridpoint[0] + (i + 1)] = 0

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


    # This function return the path of the search

    def return_path(self,current_node, maze):
        path = []
        no_rows, no_columns = np.shape(maze)
        # here we create the initialized result maze with -1 in every position
        result = [[-1 for i in range(no_columns)] for j in range(no_rows)]
        current = current_node
        while current is not None:
            path.append(current.position)
            current = current.parent
        # Return reversed path as we need to show from start to end path
        path = path[::-1]
        start_value = 0
        # we update the path of start to end found by A-star serch with every step incremented by 1
        for i in range(len(path)):
            result[path[i][0]][path[i][1]] = start_value
            start_value += 1
        return path, current_node.g

    def search(self,start, end):
        maze = self.maze
        """
            Returns a list of tuples as a path from the given start to the given end in the given maze
            :param maze:
            :param cost
            :param start:
            :param end:
            :return:
        """

        # Create start and end node with initized values for g, h and f
        start_node = NodeA(None, tuple(start))
        start_node.g = start_node.h = start_node.f = 0
        end_node = NodeA(None, tuple(end))
        end_node.g = end_node.h = end_node.f = 0

        #added:
        start_index = self.row_nrs.index(start_node.position[1])
        end_index = self.row_nrs.index(end_node.position[1])

        # Initialize both yet_to_visit and visited list
        # in this list we will put all node that are yet_to_visit for exploration.
        # From here we will find the lowest cost node to expand next
        yet_to_visit_list = []
        # in this list we will put all node those already explored so that we don't explore it again
        visited_list = []

        # Add the start node
        yet_to_visit_list.append(start_node)

        # Adding a stop condition. This is to avoid any infinite loop and stop
        # execution after some reasonable number of steps
        outer_iterations = 0
        max_iterations = (len(maze) // 2) ** 10

        # what squares do we search . serarch movement is left-right-top-bottom
        # (4 movements) from every positon

        move = [[1, 0,1.0],  # go up
                [0, -1,1.0],  # go left
                [-1, 0,1.0],  # go down
                [0, 1,1.0],  # go right
                [1, 1,math.sqrt(2)], # up right
                [-1, 1,math.sqrt(2)], # down right
                [1, -1,math.sqrt(2)], # up left
                [-1,-1,math.sqrt(2)] # down left
                ]

        """
            1) We first get the current node by comparing all f cost and selecting the lowest cost node for further expansion
            2) Check max iteration reached or not . Set a message and stop execution
            3) Remove the selected node from yet_to_visit list and add this node to visited list
            4) Perofmr Goal test and return the path else perform below steps
            5) For selected node find out all children (use move to find children)
                a) get the current postion for the selected node (this becomes parent node for the children)
                b) check if a valid position exist (boundary will make few nodes invalid)
                c) if any node is a wall then ignore that
                d) add to valid children node list for the selected parent

                For all the children node
                    a) if child in visited list then ignore it and try next node
                    b) calculate child node g, h and f values
                    c) if child in yet_to_visit list then ignore it
                    d) else move the child to yet_to_visit list
        """
        # find maze has got how many rows and columns
        no_rows, no_columns = np.shape(maze)

        # Loop until you find the end

        while len(yet_to_visit_list) > 0:
            #print("new A star iteration")

            # Every time any node is referred from yet_to_visit list, counter of limit operation incremented
            outer_iterations += 1

            # Get the current node
            current_node = yet_to_visit_list[0]
            current_index = 0
            for index, item in enumerate(yet_to_visit_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # if we hit this point return the path such as it may be no solution or
            # computation cost is too high
            if outer_iterations > max_iterations:
                print("Giving up on A* - too many iterations")
                return self.return_path(current_node, maze)

            # Pop current node out off yet_to_visit list, add to visited list
            yet_to_visit_list.pop(current_index)
            visited_list.append(current_node)

            # test if goal is reached or not, if yes then return the path
            if current_node == end_node:
                #print("A star has reached goal")
                return self.return_path(current_node, maze)

            # Generate children from all adjacent squares
            children = []


            # move options dubins/ reedsshepp:
            if self.kinematic=="dubins" or self.kinematic=="reedsshepp":
                move = []

                # Checking whether the current position is on a row edge or at a vertex
                cur_index = self.row_nrs.index(current_node.position[1])
                booledge = False
                if self.row_edges[cur_index][1] == current_node.position[0]:
                    boolright = 1
                    booledge = True
                elif self.row_edges[cur_index][0] == current_node.position[0]:
                    boolright = 0
                    booledge = True
                boolvertex = False
                for vertex in self.field_vertex:
                    if current_node.position[0] == vertex[0] and current_node.position[1] == vertex[1]:
                        boolvertex = True
                        boolright = vertex[3]
                        break

                # if the current position is an edge or vertex and it's not at the same side of a vertex as the start/end, the start/end location is not a possible move
                if booledge or boolvertex:
                    # for index in range(len(self.row_nrs) - 1):  # all edges on same side
                    #     move.append([self.row_edges[index][boolright], self.row_nrs[index]])
                    includestart = True
                    for vertex in self.field_vertex:
                        if vertex[3] == boolright and not (
                                vertex[0] == current_node.position[0] and vertex[1] == current_node.position[
                            1]):  # at the same side as the edge and not at the same location
                            if not ((cur_index <= vertex[2][0] and start_index <= vertex[2][0]) or
                                    (cur_index >= vertex[2][1] and start_index >= vertex[2][1])):
                                move.append([vertex[0], vertex[1]])
                                includestart = False
                                break
                    # includestart=True # for now
                    if includestart:
                        move.append([self.row_edges[start_index][boolright], start_node.position[1]])
                    includeend = True
                    for vertex in self.field_vertex:
                        if vertex[3] == boolright and not (
                                vertex[0] == current_node.position[0] and vertex[1] == current_node.position[
                            1]):  # at the same side as the edge and not at the same location
                            if not ((cur_index <= vertex[2][0] and end_index <= vertex[2][0]) or (
                                    cur_index >= vertex[2][1] and end_index >= vertex[2][1])):
                                move.append([vertex[0], vertex[1]])
                                includeend = False
                                break
                    # includeend=True # for now
                    if includeend:
                        move.append([self.row_edges[end_index][boolright], end_node.position[1]])
                    if not boolvertex or (boolvertex and current_node.position[1] in self.row_nrs): # vertex but also a row
                        move.append([self.row_edges[cur_index][abs(boolright - 1)],
                                     self.row_nrs[cur_index]])  # edge on other side of current row
                # if the current position is within a row, any position within the row is a possible move
                if not boolvertex and booledge or (boolvertex and current_node.position[1] in self.row_nrs):
                    for xpos in range(self.row_edges[cur_index][0],
                                      self.row_edges[cur_index][1]):  # any point within the row
                        move.append([xpos, self.row_nrs[cur_index]])
                # if the current position is not an edge or vertex, the only move possible is the end of the row (left or right)
                if not boolvertex and not booledge:
                    move.append([self.row_edges[cur_index][0],
                                 self.row_nrs[cur_index]])
                    move.append([self.row_edges[cur_index][1],
                                 self.row_nrs[cur_index]])
            # move options reedsshepp reverse:
            elif self.kinematic == "reedsshepprev" or self.kinematic=="none" or self.kinematic=="limit" or self.kinematic=="ranger":
                move=[]
                cur_index = self.row_nrs.index(current_node.position[1])
                booledge=False
                if self.row_edges[cur_index][1] == current_node.position[0]:
                    boolright = 1
                    booledge = True
                elif self.row_edges[cur_index][0] == current_node.position[0]:
                    boolright=0
                    booledge = True
                boolvertex = False
                for vertex in self.field_vertex:
                    if current_node.position[0] == vertex[0] and current_node.position[1] == vertex[1]:
                        boolvertex = True
                        boolright=vertex[3]
                        break
                if booledge or boolvertex:
                    # for index in range(len(self.row_nrs) - 1):  # all edges on same side
                    #     move.append([self.row_edges[index][boolright], self.row_nrs[index]])
                    includestart=True
                    for vertex in self.field_vertex:
                        if vertex[3]==boolright and not (vertex[0]==current_node.position[0] and vertex[1]==current_node.position[1]): # at the same side as the edge and not at the same location
                            if not ((cur_index<=vertex[2][0] and start_index<=vertex[2][0]) or
                                    (cur_index>=vertex[2][1] and start_index>=vertex[2][1])):
                                move.append([vertex[0],vertex[1]])
                                includestart=False
                                break
                    #includestart=True # for now
                    if includestart:
                        move.append([self.row_edges[start_index][boolright],start_node.position[1]])
                    includeend = True
                    for vertex in self.field_vertex:
                        if vertex[3]==boolright and not (vertex[0]==current_node.position[0] and vertex[1]==current_node.position[1]): # at the same side as the edge and not at the same location
                            if not ((cur_index <= vertex[2][0] and end_index <= vertex[2][0]) or (
                                    cur_index >= vertex[2][1] and end_index >= vertex[2][1])):
                                move.append([vertex[0], vertex[1]])
                                includeend = False
                                break
                    #includeend=True # for now
                    if includeend:
                        move.append([self.row_edges[end_index][boolright],end_node.position[1]])
                    if not boolvertex or (boolvertex and current_node.position[1] in self.row_nrs):
                        move.append([self.row_edges[cur_index][abs(boolright - 1)], self.row_nrs[cur_index]]) # edge on other side of current row


                if not booledge and not boolvertex:
                    # move.append([current_node.position[0] - 1, current_node.position[1]])  # 1 step left
                    # move.append([current_node.position[0] + 1, current_node.position[1]])  # 1 step right
                    move.append([self.row_edges[cur_index][0], self.row_nrs[cur_index]]) # edge on either side of current row
                    move.append([self.row_edges[cur_index][1], self.row_nrs[cur_index]])  # edge on either side of current row
                if not boolvertex or (boolvertex and current_node.position[1] in self.row_nrs):
                    for xpos in range(self.row_edges[cur_index][0],self.row_edges[cur_index][1]): #any point within the row
                        move.append([xpos,self.row_nrs[cur_index]])
                # print(current_node.position)
                # print(move)
                # print(booledge,boolvertex)
            for new_position in move:
                #cost = new_position[2]

                #cost = 1
                #cost = np.sqrt((current_node.position[0]-new_position[0])**2+(current_node.position[1]-new_position[1])**2)
                #cost = ((current_node.position[0]-new_position[0])**2+(current_node.position[1]-new_position[1])**2)

                # Get node position
                #if self.kinematic=="dubins" or self.kinematic=="reedsshepp" or self.kinematic=="reedsshepprev":
                node_position = (new_position[0],new_position[1])
                cost = abs(current_node.position[0] - new_position[0]) + 1
                #else:
                #    node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
                #    cost= new_position[2]

                # Make sure within range (check if within maze boundary)
                if (node_position[0] > (no_rows - 1) or
                        node_position[0] < 0 or
                        node_position[1] > (no_columns - 1) or
                        node_position[1] < 0):
                    continue

                # Make sure walkable terrain
                # note: if .. continue means: if the statement returns true, we skip the rest of the loop
                # two conditions: if the position is not within the rows/edges or if the position is changing in y position while it's not on "edge" terrain
                # if maze[node_position[1]][node_position[0]] != 0 or (new_position[1]!=0 and (edgemaze[node_position[1]][node_position[0]]!=0 or edgemaze[current_node.position[1]][current_node.position[0]]!=0)): #or (node_position[0]==end[0] and node_position[1]==end[1])
                #     continue

                # Create new node
                new_node = NodeA(current_node, node_position)
                new_node.g = current_node.g + cost
                # Append
                children.append(new_node)

            # Loop through children
            for child in children:

                # Child is on the visited list (search entire visited list)
                if len([visited_child for visited_child in visited_list if visited_child == child]) > 0:
                    continue

                # Create the f, g, and h values
                #child.g = current_node.g + cost
                ## Heuristic costs calculated here, this is using eucledian distance
                # child.h = (((child.position[0] - end_node.position[0]) ** 2) +
                #           ((child.position[1] - end_node.position[1]) ** 2)) # manhattan: overestimates

                # child.h = ((abs(child.position[0] - end_node.position[0])) +
                #            (abs(child.position[1] - end_node.position[1])))**2 # euclidean
                #child.h=0 # dijkstra --> no heuristic
                #child.h = (abs(child.position[1] - end_node.position[1])) # only y position

                child.h = (abs(child.position[0] - end_node.position[0])) # only x position
                #child.h  = (child.position[0]**2+(abs(child.position[1] - end_node.position[1]))**2)
                child.f = child.g + child.h

                # Child is already in the yet_to_visit list and g cost is already lower
                if len([i for i in yet_to_visit_list if child == i and child.g > i.g]) > 0:
                    continue

                # Add the child to the yet_to_visit list
                yet_to_visit_list.append(child)

        print("Reached end of A* without solution, maze(start)="+str(self.maze[start[1],start[0]])+" maze(end)="+str(self.maze[end[1],end[0]]))
        print("Start location: ("+str(start[0])+","+str(start[1])+")")
        print("End location: ("+str(end[0])+","+str(end[1])+")")
        return [],np.inf
    def FindCostInfoA(self, node_end_x, node_end_y, node_start_x, node_start_y, node, totalpath=True, costOnly=False):
        # node_end = the goal or new node
        # node_start = the (potential) parent
        # currentcost = cost of the parent (node_start)
        # currentinfopath = the infopath of the parent (node_start)
        # costOnly = shortcut in case we only want the distance and not the info

        cost = self.costmatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x]
        #node_direction = self.directionmatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x]

        info = self.infomatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x]
        infopath = self.infopathmatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x]
        #if not cost or (not info):
        if not cost or (not info and not costOnly):
            #print("cost: "+str(cost)+" costonly: "+str(costOnly))
            # a star
            #maze=self.maze
            start=[node_start_x,node_start_y]
            end = [node_end_x,node_end_y]
            #end = [node_start_x+20,node_start_y]
            #print(start +end)

            [infopath,cost] = self.search(start, end) # A star steps
            [infopath,cost,info] = self.getInfopath(node_end_x, node_end_y, node_start_x, node_start_y, node,infopath) # actual cost and infopath
                    # if boolvertex:
                    #     prevnode=True
            infopathnew = []
            # for location in infopath:
            #     infopathnew.append(location)
            #print("starting A star")
            #print(infopath)
            #print(infopathnew)
            #print(cost)

            # fig, ax = plt.subplots()
            # colormap = cm.Blues
            # colormap.set_bad(color='black')
            # im = ax.imshow(self.uncertaintymatrix, colormap, vmin=0, vmax=3, origin='lower')
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # plt.colorbar(im, cax=cax)
            # # ax.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
            # ax.plot([x for x, _ in infopath], [y for _, y in infopath], '-r')
            #
            # ax.set_title("Spatial distribution of uncertainty and final path")
            # fig.tight_layout()
            # plt.show()

            #if not costOnly:


            # for index, infopoint in enumerate(infopath):
            #     if infopoint[1]==node_start_y:
            #         self.costmatrix[infopoint[1]*100+infopoint[0], node_end_y*100 + node_end_x] = cost-index
            #         self.costmatrix[node_end_y * 100 + node_end_x,infopoint[1]*100+infopoint[0]] = cost-index
            #
            #     else:
            #         break;
            # for index, infopoint in enumerate(infopath[::-1]):
            #     if infopoint[1]==node_end_y:
            #         self.costmatrix[infopoint[1]*100+infopoint[0], node_start_y*100 + node_start_x] = cost-index
            #         self.costmatrix[node_start_y * 100 + node_start_x,infopoint[1]*100+infopoint[0]] = cost-index
            #
            #     else:
            #         break;
        # else:
        #     print("reusing known costs")
        if costOnly:
            return cost

        #finding the actual infopath here
        infopath = infopath
        infonode = 0
        currentinfopath = []
        curnode = node
        while curnode.parent:
            # print("index 1 = " + str(curnode.parent.y*100+curnode.parent.x) + " index 2 = " + str(
            #    curnode.y*100+curnode.x))
            if self.kinematic!="dubins" and self.kinematic!="reedsshepp":
                currentinfopath.extend(
                    self.infopathmatrix[curnode.parent.y * 100 + curnode.parent.x, curnode.y * 100 + curnode.x])
            else:
                start = [curnode.parent.x, curnode.parent.y]
                end = [curnode.x, curnode.y]
                [infoastar, costastar] = self.search(start, end)  # A star steps
                currentinfopath.extend(self.getInfopath(curnode.x,curnode.y,curnode.parent.x,curnode.parent.y,curnode.parent,infoastar)[0])
            curnode = curnode.parent
        if not any(element in currentinfopath for element in infopath):  # the whole infopath is new
            infonode += info
        else:  # if some infopoints overlap
            for element in infopath:
                if not element in currentinfopath and not np.isnan(self.uncertaintymatrix[element[1], element[0]]):
                    infonode += self.uncertaintymatrix[element[1], element[0]]
        return cost, infonode
        # return [cost,infopath,info]
    def getInfopath(self, node_end_x, node_end_y, node_start_x, node_start_y, node,infopathastar):
        info = 0
        if self.kinematic == "none" or self.kinematic == "ranger" or self.kinematic == "limit":
            cost = 0
            #infopathastar = infopath
            infopath = []
            for i in range(len(infopathastar) - 1):
                dcost = np.sqrt((infopathastar[i + 1][0] - infopathastar[i][0]) ** 2 + (
                            infopathastar[i + 1][1] - infopathastar[i][1]) ** 2)
                cost += dcost
                # Ranger kinematic constraints: taking time to turn
                if self.kinematic == "ranger":
                    curangle = math.atan2(infopathastar[i + 1][1] - infopathastar[i][1],
                                          infopathastar[i + 1][0] - infopathastar[i][0])
                    if i > 0:
                        prevangle = math.atan2(infopathastar[i][1] - infopathastar[i - 1][1],
                                               infopathastar[i][0] - infopathastar[i - 1][0])
                    elif i==0 and node.parent!=None:
                        prevangle = self.anglematrix[node.parent.y * 100 + node.parent.x, node.y * 100 + node.x]
                    else:
                        prevangle = curangle
                    dangle = (curangle - prevangle) ** 2
                    cost += np.sqrt(dangle) * self.angularcost
                # Limited angle kinematic constraint
                if self.kinematic == "limit":
                    anglelimit = 2 * math.pi / 4  # 90 degrees
                    curangle = math.atan2(infopathastar[i + 1][1] - infopathastar[i][1],
                                          infopathastar[i + 1][0] - infopathastar[i][0])
                    if i > 0:
                        prevangle = math.atan2(infopathastar[i][1] - infopathastar[i - 1][1],
                                               infopathastar[i][0] - infopathastar[i - 1][0])
                    elif i == 0 and node.parent != None:
                        prevangle = self.anglematrix[node.parent.y * 100 + node.parent.x, node.y * 100 + node.x]
                    else:
                        prevangle = curangle
                    dangle = (curangle - prevangle) ** 2
                    if (dangle > (anglelimit ** 2)):
                        cost += np.inf  # if the angle exceeds the limit, inf is added
                    else:
                        cost += np.sqrt(dangle) * self.angularcost

                infopath.extend(self.FindInfo(infopathastar[i + 1][0], infopathastar[i + 1][1], infopathastar[i][0],
                                              infopathastar[i][1], None, dcost, False)[0])
            if self.kinematic!="none":
                self.anglematrix[node_start_y * 100 + node_start_x,node_end_y * 100 + node_end_x] = curangle # last angle of the piece
        if self.kinematic == "dubins" or self.kinematic == "reedsshepp" or self.kinematic == "reedsshepprev":
            cost = 0
            #infopathastar = infopath
            infopath = []
            if node.parent != None:
                prevnode = [node.parent.x, node.parent.y]
            else:
                prevnode = None
            for i in range(len(infopathastar) - 1):
                boolvertex = False
                for vertex in self.field_vertex:
                    if infopathastar[i + 1][0] == vertex[0] and infopathastar[i + 1][1] == vertex[1]:
                        boolvertex = True
                        break
                if boolvertex:
                    [dcost, dubins_x, dubins_y, infopathpart] = self.getDubins(infopathastar[i], infopathastar[i + 1],
                                                                               prevnode,
                                                                               True)  # twist end point 180 degrees around for more logical path
                else:
                    [dcost, dubins_x, dubins_y, infopathpart] = self.getDubins(infopathastar[i], infopathastar[i + 1],
                                                                               prevnode)
                cost += dcost
                infopath.extend(infopathpart)
                prevnode = infopathastar[i]

        infopathcopy = deepcopy(infopath)
        infopath = []
        for i, infopoint in enumerate(infopathcopy):
            # if np.isnan(self.uncertaintymatrix[infopoint[1], infopoint[0]]) and self.maze[infopoint[1],infopoint[0]]!=0: #not within allowed width around edges or within a row
            if self.maze[infopoint[1], infopoint[0]] != 0:  # not within allowed width around edges or within a row
                if infopathcopy[i - 1][1] != infopoint[1]:
                    cost = np.inf
                    break;
            if not np.isnan(self.uncertaintymatrix[infopoint[1], infopoint[0]]) and not [infopoint[0],
                                                                                         infopoint[1]] in infopath:
                info += self.uncertaintymatrix[infopoint[1], infopoint[0]]
                infopath.append(infopoint)
            if self.inforadius > 0:
                for rowdist in range(-self.inforadius, self.inforadius + 1):
                    for coldist in range(-self.inforadius, self.inforadius + 1):
                        if (coldist ** 2 + rowdist ** 2) <= self.inforadius ** 2:  # radius
                            xpoint_ = infopoint[0] + coldist
                            ypoint_ = infopoint[1] + rowdist
                            if not [xpoint_, ypoint_] in infopath and not np.isnan(
                                    self.uncertaintymatrix[ypoint_, xpoint_]):
                                info += self.uncertaintymatrix[ypoint_, xpoint_]
                                infopath.append([xpoint_, ypoint_])

        if self.kinematic != "dubins" and self.kinematic != "reedsshepp":
            self.infomatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x] = info
            self.infomatrix[
                node_end_y * 100 + node_end_x, node_start_y * 100 + node_start_x] = info  # mirror the matrix

            self.costmatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x] = cost
            self.costmatrix[
                node_end_y * 100 + node_end_x, node_start_y * 100 + node_start_x] = cost  # mirror the matrix

            self.infopathmatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x] = infopath
            self.infopathmatrix[node_end_y * 100 + node_end_x, node_start_y * 100 + node_start_x] = infopath[::-1]
        return infopath,cost,info

    def getDubins(self,node_start,node_end,node_prev,turnend=False):
        #print("getDubinsfunction")
        #print(last)
        syaw=None
        if node_prev!=None and np.any(self.anglematrix[node_prev[1] * 100 + node_prev[0],node_start[1] * 100 + node_start[0]]): #not empty
            syaw = self.anglematrix[node_prev[1] * 100 + node_prev[0],node_start[1] * 100 + node_start[0]]
        elif node_prev!=None and not np.any(self.anglematrix[node_prev[1] * 100 + node_prev[0],node_start[1] * 100 + node_start[0]]):
            #print("missing the angle, getting it from the path")
            [infopath, cost] = self.search(node_prev, node_start)
            if len(infopath)>=2:
                syaw = self.anglematrix[infopath[-2][1] * 100 + infopath[-2][0],infopath[-1][1] * 100 + infopath[-1][0]]
        if syaw==None: #still need to find syaw after the two things tried above
            if node_start[1]==self.x_start.y and node_start[0]==self.x_start.x:
                if node_end[0]>node_start[0]:
                    syaw=0
                else:
                    syaw=math.pi
            elif node_start[1]==self.x_start.y and node_start[0]>self.x_start.x:
                syaw=math.pi
                #print("no syaw, scenario 1, x = "+str(node_start[0]))
            elif node_start[1]==self.x_start.y and node_start[0]<self.x_start.x:
                syaw=0
                #print("no syaw, scenario 2")
            elif node_start[1]!=self.x_start.y and node_start[0]>=self.x_start.x:
                syaw=0
                #print("no syaw, scenario 3")
            elif node_start[1]!=self.x_start.y and node_start[0]<self.x_start.x:
                syaw=math.pi
                #print("no syaw, scenario 4")

        if node_end[1]==node_start[1]: # crossing a row
            if node_start[0]<node_end[0]: # towards the right
                #syaw=0
                gyaw=0
            else:
                #syaw=math.pi
                gyaw=math.pi
        else:
            endindex=self.row_nrs.index(node_end[1])
            if abs(self.row_edges[endindex][0]-node_end[0])<abs(self.row_edges[endindex][1]-node_end[0]): #closer to left than right
                #syaw=0
                #print("left end: "+str(node_end[0]))
                gyaw=0
                if turnend:
                    gyaw=math.pi
                # elif turnend and self.kinematic=="dubins":
                #     syaw=0
            else:
                #print("right end: "+str(node_end[0]))
                #syaw=math.pi
                gyaw=math.pi
                if turnend:
                    gyaw=0
                # elif turnend and self.kinematic=="dubins":
                #     syaw=math.pi
            #node_prev=None
            # if node_prev:
            #     if node_prev==True: # just for now
            #         node_prev=False
        # if node_prev:
        #     if node_prev==True or (node_prev[0] == self.row_edges[self.row_nrs.index(node_prev[1])][0] and node_start[0] ==
        #         self.row_edges[self.row_nrs.index(node_start[1])][0]) or (
        #             node_prev[0] == self.row_edges[self.row_nrs.index(node_prev[1])][1] and node_start[0] ==
        #             self.row_edges[self.row_nrs.index(node_start[1])][1]):
        #         #print("Starting angle is other way around")
        #         # we went from edge to edge so we are facing inwards to the row, not outwards to the edge
        #         if syaw==0:
        #             syaw=math.pi
        #         elif syaw==math.pi:
        #             syaw=0


        # if node_prev==True:
        #     node_prev=None
        # if node_prev:
        #     if node_prev==True or node_prev[1] == node_start[1] or (node_prev[0] == self.row_edges[self.row_nrs.index(node_prev[1])][0] and node_start[0] ==
        #         self.row_edges[self.row_nrs.index(node_start[1])][1]) or (
        #             node_prev[0] == self.row_edges[self.row_nrs.index(node_prev[1])][1] and node_start[0] ==
        #             self.row_edges[self.row_nrs.index(node_start[1])][0]): # crossing the row or opposite side edges
        #         #print("Starting angle is other way around")
        #         # we went from edge to edge so we are facing inwards to the row, not outwards to the edge
        #         if syaw==0:
        #             syaw=math.pi
        #         elif syaw==math.pi:
        #             syaw=0

        self.anglematrix[node_start[1] * 100 + node_start[0],node_end[1] * 100 + node_end[0]] = gyaw
        #print("nodestart="+str(node_start)+" nodeend="+str(node_end)+" syaw="+str(syaw)+" gyaw="+str(gyaw))
        maxc = 1

        sx = node_start[0]
        sy = node_start[1]

        gx = node_end[0]
        gy = node_end[1]

        if (gx - sx, gy - sy, syaw, gyaw) not in self.dubinsmat:  # relative position + end angle
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
            self.dubinsmat[(gx - sx, gy - sy, syaw, gyaw)]=[dubinspath.L, dubinspath.x, dubinspath.y, infopathrel]
        else:
            #print("reusing the dubinsmat")
            #index = self.dubinsmat[0].tolist().index([gx - sx, gy - sy, gyaw])
            #index = self.dubinsmat.index([gx - sx, gy - sy, gyaw])
            # index = np.where(self.dubinsmat[0] == [gx - sx, gy - sy, gyaw])

            [cost, dubins_rel_x, dubins_rel_y, infopathrel] = self.dubinsmat[(gx - sx, gy - sy, syaw, gyaw)]

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
            # if self.inforadius > 0:
            #     for rowdist in range(-self.inforadius, self.inforadius + 1):
            #         for coldist in range(-self.inforadius, self.inforadius + 1):
            #             if (coldist ** 2 + rowdist ** 2) <= self.inforadius ** 2:  # radius
            #                 xpoint_ = cell[0] + sx + coldist
            #                 ypoint_ = cell[1] + sy + rowdist
            #                 if not [xpoint_, ypoint_] in infopath and not np.isnan(
            #                         self.uncertaintymatrix[ypoint_, xpoint_]):
            #                     # info += self.uncertaintymatrix[ypoint_, xpoint_]
            #                     infopath.append([xpoint_, ypoint_])
        return cost,dubins_x,dubins_y,infopath

    def Rewiring_afterv2(self, best_node, doubleround): #rewiring afterwards
        # goal: gain more info while remaining within the budget
        print("Start rewiring after v2")
        print(" Info: " + str(best_node.info) + " Tot. info: " + str(
            best_node.totalinfo) + " Cost: " + str(best_node.totalcost))
        bestpath=[]
        infosteps=[]
        node=best_node
        prev_copynode = None
        costfirstround=best_node.prevroundcost
        while node != self.x_start:
            # copynode = deepcopy(node) # just now
            copynode = Node((node.x, node.y))
            if prev_copynode:
                prev_copynode.parent = copynode
            # copynode.parent = copyparent
            copynode.info = node.info
            copynode.cost = node.cost
            copynode.totalinfo = node.totalinfo
            copynode.totalcost = node.totalcost
            if node == best_node:
                copybest_node = copynode
            self.V.append(copynode)
            self.X_soln.append(copynode)
            # if len(bestpath)>0:
            #     bestpath[-1].parent=copynode # just now
            bestpath.append(copynode) #
            #bestpath.append(node)
            #infosteps.append(node.info-node.parent.info)
            infosteps.append((node.info-node.parent.info)/(node.cost-node.parent.cost)) # density of increase of info
            # the infosteps contain the added info for the parent to the node (of that node)
            prev_copynode=copynode
            node=node.parent
        # bestpath[-1].parent=self.x_start # just now
        prev_copynode.parent=self.x_start # for the last one
        best_node=copybest_node

        if costfirstround==0 or not doubleround:
            #print("Rewiring: updated first round cost")
            costfirstround=best_node.totalcost

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

            #print("Best index: "+str(bestindex))
            #print(node.parent.parent.x, node.parent.parent.y)

            checked_locations = []
            for x_near in self.Near(self.V, node, self.search_radius,False):
                # for x_near in self.Near(self.V, node, 10):
                if not ([x_near.x,x_near.y]==[node.x,node.y]) and not ([x_near.x,x_near.y] in checked_locations) and not ([node.parent.x,node.parent.x]==[self.x_goal.x,self.x_goal.y]):
                    checked_locations.append([x_near.x, x_near.y])
                    x_temp = Node((node.x, node.y))
                    x_temp.parent = node.parent
                    x_temp.info = node.info
                    x_temp.cost = node.cost
                    x_temp.totalinfo = node.totalinfo
                    x_temp.totalcost = node.totalcost

                    #if node.parent != self.x_start and node != self.x_start:  # because otherwise there's no "old" path to go back to
                    c_old = node.cost

                    # c_new = node.parent.parent.cost + self.Line(node.parent.parent, x_near) + self.Line(node,
                    #                                                                                        x_near)
                    [cost, info] = self.FindCostInfoA(node.parent.parent.x, node.parent.parent.y, x_near.x, x_near.y, node.parent.parent, True)
                    cost2 = self.FindCostInfoA(x_near.x, x_near.y, node.x, node.y, node.parent.parent, True,True)
                    c_new = node.parent.parent.cost + cost + cost2

                    # if x_new.parent.x==x_near.x and x_new.parent.y==x_near.y:
                    #     return # if the parent of x_new = x_near, we don't want to make the parent of x_near = x_new (because then we create a loose segment
                    costsecondround = best_node.totalcost - costfirstround
                    if (node.round == 1 and (c_new - c_old) < (self.budget - costfirstround)) or (node.round == 2 and (c_new - c_old) < (self.budget - costsecondround)):  # still within budget
                        newnode = Node((x_near.x, x_near.y))
                        newnode.parent = node.parent.parent
                        newnode.info = node.parent.parent.info + info
                        newnode.cost = node.parent.parent.cost + cost
                        self.LastPath(newnode)
                        info = newnode.info + self.FindCostInfoA(newnode.x, newnode.y, node.x, node.y, newnode, True)[1]


                        info_old = node.info

                        info_new = info
                        if info_new >= info_old:  # note: this is different than the condition in pruning
                            self.V.append(newnode)
                            if (newnode.round==1 and newnode.totalcost<=self.budget) or (newnode.round==2 and (newnode.totalcost-newnode.prevroundcost)<=self.budget):
                                self.X_soln.append(newnode)

                            # rewiring:

                            node.parent = newnode
                            node.info = info_new
                            node.cost = c_new
                            self.LastPath(node)  # also recalculate the last info part


                            # Scenario 1:
                            # else:
                            self.Recalculate(node)  # recalculates the cost and info for nodes further down the path
                            if totalinfo>=best_node.totalinfo:
                                # reverse rewiring
                                node.parent = x_temp.parent
                                node.info = x_temp.info
                                node.cost = x_temp.cost
                                node.totalinfo = x_temp.totalinfo
                                node.totalcost = x_temp.totalcost
                                self.Recalculate(node)
                            else:
                                bestpath[bestindex + 1] = newnode
                                # tworoundstrategy2:
                                if doubleround:
                                    if node.round == 1:
                                        costfirstround += (c_new - c_old)
                                    else:  # round 2:
                                        costsecondround += (c_new - c_old)
                                else:
                                    costfirstround += (c_new - c_old)

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
        best_node=copybest_node
        print("End rewiring after v2")
        print(" Info: " + str(best_node.info) + " Tot. info: " + str(
            best_node.totalinfo) + " Cost: " + str(best_node.totalcost))
        return best_node


    def RoundTwoAdd(self, x_new):
        for node in self.X_soln:
            if [node.x,node.y]==[x_new.x,x_new.y]: #all nodes at the new sampled location
                # tworoundstrategy2:
                if node.round == 1:
                    Round2StartNode = Node((self.x_goal.x, self.x_goal.y))
                    Round2StartNode.cost = node.totalcost
                    Round2StartNode.totalcost = node.totalcost
                    Round2StartNode.info = node.info
                    Round2StartNode.totalinfo = node.totalinfo
                    Round2StartNode.parent = node
                    Round2StartNode.round = 2
                    Round2StartNode.prevroundcost= node.totalcost
                    self.V.append(Round2StartNode)
                    #self.X_soln.add(Round2StartNode)
                    self.X_soln.append(Round2StartNode)
    def Recalculate(self,parent):
        for node in self.V:  # to recalculate the cost and info for nodes further down the line
            if node.parent == parent:
                if node==self.x_best:
                    previnfo=node.info
                    prevtotalinfo=node.totalinfo

                [dist,info] = self.FindCostInfoA(parent.x, parent.y, node.x, node.y, parent, True)

                node.info = parent.info + info

                node.cost = parent.cost + dist

                if node.round==2:
                    if [node.x,node.y]==[self.x_start.x,self.x_start.y]: # start of round 2
                        node.prevroundcost=node.totalcost
                    else:
                        node.prevroundcost=node.parent.prevroundcost

                self.LastPath(node)
                self.Recalculate(node)
    def Rewiring(self, x_near, x_new):
        if x_new.parent.x == x_near.x and x_new.parent.y == x_near.y:
            return  # if the parent of x_new = x_near, we don't want to make the parent of x_near = x_new (because then we create a loose segment

        #cost_heuristic = abs(x_new.x-x_near.x)+abs(x_new.y-x_near.y) #manhattan distance
        #info_heuristic = cost_heuristic*(cost_heuristic/x_new.cost)*x_new.info #weighted average of x_new info
        cost_heuristic_x = (x_new.x-x_near.x)**2
        cost_heuristic_y = (x_new.y-x_near.y)**2
        c_near = x_near.cost

        if cost_heuristic_x < ((c_near-x_new.cost)**2) and cost_heuristic_y < ((c_near-x_new.cost)**2): #first do heuristic and then the actual calculation
        #if True:

            [cost,info] = self.FindCostInfoA(x_near.x,x_near.y,x_new.x,x_new.y,x_new,True)

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

    def Recalculate_old(self,parent):
        for node in self.V:  # to recalculate the cost and info for nodes further down the line
            if node.parent == parent:
                [dist,info] = self.FindCostInfoA(node.x,node.y,node.parent.x,node.parent.y,parent,True)

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
                if (node2.cost<=node1.cost and node2.info>node1.info) or (node2.cost<node1.cost and node2.info==node1.info) or (node1.parent==node2.parent and index!=index2): #prune lesser paths or doubles
                #if (node2.cost<=node1.cost and node2.info>node1.info) or (node1.parent==node2.parent and index!=index2): #prune lesser paths or doubles
                    # if (node1.cost==node2.cost and node1.info==node2.info and index!=index2):
                    #     print("Double detected, now pruned")
                    # else:
                    #     print("Alternative pruned")
                    # prune the node from all nodes
                    nochildren = True
                    for allnode in self.V[::-1]:
                        if allnode.parent == node1:  # in this case we can't prune the node because other nodes depend on it
                            nochildren = False
                            print("!!!Node not pruned because it has children")
                            break
                    if nochildren:
                        if node1 in self.V:
                            self.V.remove(node1)
                        if node1 in self.X_soln:
                            self.X_soln.remove(node1)  # also from the solutions


    def SteerAstar(self, x_nearest, x_rand):
        dist = self.FindCostInfoA(x_rand.x,x_rand.y,x_nearest.x,x_nearest.y,x_nearest,False,True)

        if dist <= self.step_len:
            # print("x_nearest=(" + str(x_start.x) + "," + str(x_start.y) + ")")
            # print("x_rand close enough, x_rand = x_new --> x_rand=("+str(x_goal.x)+","+str(x_goal.y)+")")
            print("nearest=(" + str(x_nearest.x) + "," + str(x_nearest.y) + ") - x_rand=(" + str(x_rand.x) + "," + str(
                x_rand.y) + ") - dist = " + str(dist) + " - x_new=(" + str(x_rand.x) + "," + str(
                x_rand.y) + ")")
            return Node((int(x_rand.x), int(x_rand.y)))

        # if dist > self.steplen: use the astar path to find node closer by
        # if no constraints/ ranger/ limit/ reedsshepprev: this can also be within a row
        start = [x_nearest.x,x_nearest.y]
        end = [x_rand.x, x_rand.y]
        [infopath, cost] = self.search(start, end)

        for i in range(len(infopath)):
            # TODO decide whether it's allowed to sample a vertex
            # boolvertex = False
            # for vertex in self.field_vertex:
            #     if infopath[i + 1][0] == vertex[0] and infopath[i + 1][1] == vertex[1]:
            #         boolvertex = True
            #         break
            # if not boolvertex:

            xpoint = infopath[-(i+1)][0]
            ypoint = infopath[-(i+1)][1]
            # sampling within the row whenever the kinematic constraint allows for it:
            if i>0 and ypoint==infopath[-i][1] and self.kinematic!="dubins" and self.kinematic!="reedsshepp": # in same row
                rightbool=-1
                if xpoint>infopath[-i][0]:
                    rightbool=1
                for xpoint in range(infopath[-i][0],infopath[-(i+1)][0],rightbool):
                    dist = self.FindCostInfoA(xpoint, ypoint, x_nearest.x, x_nearest.y, x_nearest, False, True)
                    if dist <= self.step_len:
                        print("nearest=(" + str(x_nearest.x) + "," + str(x_nearest.y) + ") - x_rand=(" + str(
                            x_rand.x) + "," + str(
                            x_rand.y) + ") - dist = " + str(dist) + " - x_new=(" + str(xpoint) + "," + str(
                            ypoint) + ")")

                        return Node((int(xpoint), int(ypoint)))
            # sampling the Astar points:
            dist = self.FindCostInfoA(xpoint,ypoint,x_nearest.x,x_nearest.y,x_nearest,False,True)
            if dist <= self.step_len:
                print("nearest=(" + str(x_nearest.x) + "," + str(x_nearest.y) + ") - x_rand=(" + str(x_rand.x) + "," + str(
                    x_rand.y) + ") - dist = " + str(dist) + " - x_new=(" + str(xpoint) + "," + str(
                    ypoint) + ")")

                return Node((int(xpoint), int(ypoint)))
        print("no point close enough found, current dist = "+str(dist))
        print("nearest=(" + str(x_nearest.x) + "," + str(x_nearest.y) + ") - x_rand=(" + str(x_rand.x) + "," + str(
            x_rand.y) + ") - dist = " + str(dist) + " - x_new=(" + str(x_rand.x) + "," + str(
            x_rand.y) + ")")
        return Node((int(x_rand.x),int(x_rand.y)))


    def Steer_section(self, x_start, x_goal): # with sectioning
        #print("Steer start and end: nearest=(" + str(x_start.x) + "," + str(x_start.y) + ") - x_rand=(" + str(x_goal.x) + "," + str(
        #    x_goal.y)+")")
        xpoint=x_start.x
        ypoint = x_start.y # just for debugging now

        #dist, theta = self.get_distance_and_angle(x_start, x_goal)
        dist = self.FindCostInfoA(x_goal.x,x_goal.y,x_start.x,x_start.y,x_start,False,True)

        # closer=False
        # if dist>self.step_len:
        #     #just for now for debugging purposes
        #     closer=True
        if dist<=self.step_len:
            #print("x_nearest=(" + str(x_start.x) + "," + str(x_start.y) + ")")
            #print("x_rand close enough, x_rand = x_new --> x_rand=("+str(x_goal.x)+","+str(x_goal.y)+")")
            print("nearest=(" + str(x_start.x) + "," + str(x_start.y) + ") - x_rand=(" + str(x_goal.x) + "," + str(
                x_goal.y) + ") - dist = " + str(dist) + " - x_new=(" + str(x_goal.x) + "," + str(
                x_goal.y) + ")")
            return Node((int(x_goal.x), int(x_goal.y)))
        distleft = self.FindCostInfoA(self.row_edges[self.row_nrs.index(x_goal.y)][0], self.row_nrs[self.row_nrs.index(x_goal.y)], x_start.x, x_start.y,
                                     x_start, False, True)
        distright = self.FindCostInfoA(self.row_edges[self.row_nrs.index(x_goal.y)][1], self.row_nrs[self.row_nrs.index(x_goal.y)], x_start.x, x_start.y,
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
        dist = self.FindCostInfoA(self.row_edges[self.row_nrs.index(x_start.y)][boolright], self.row_nrs[self.row_nrs.index(x_start.y)], x_start.x, x_start.y,
                                     x_start, False, True)
        #print("step 1 dist = "+str(dist))
        #print("x position: "+str(self.row_edges[self.row_nrs.index(x_start.y)][boolright]))

        if dist>=self.step_len:
            #print("step 1 (within start row)")
            # find point within start row
            xpoint=self.row_edges[self.row_nrs.index(x_start.y)][boolright]
            ypoint=self.row_nrs[self.row_nrs.index(x_start.y)]
            step=0
            while dist>self.step_len:
                xpoint = self.row_edges[self.row_nrs.index(x_start.y)][boolright]-direction*step #note the - sign because we're going backwards on the path
                ypoint = x_start.y
                #print(xpoint,ypoint)
                dist = self.FindCostInfoA(xpoint, ypoint, x_start.x, x_start.y,
                                     x_start, False, True)
                step+=1
            #print("x_rand=(" + str(x_goal.x) + "," + str(x_goal.y) + ") - dist = " + str(dist) + " - x_new=(" + str(
            #    xpoint) + "," + str(ypoint) + ")")
            print("nearest=(" + str(x_start.x) + "," + str(x_start.y) + ") - x_rand=(" + str(x_goal.x) + "," + str(
                x_goal.y) + ") - dist = " + str(dist) + " - x_new=(" + str(xpoint) + "," + str(
                ypoint) + ")")

            return Node((int(xpoint),int(ypoint)))
        # step 2:

        dist = self.FindCostInfoA(self.row_edges[self.row_nrs.index(x_goal.y)][boolright], x_goal.y, x_start.x, x_start.y,
                                     x_start, False, True)
        #print("step 2 dist = "+str(dist))
        #print("x position: "+str(self.row_edges[self.row_nrs.index(x_goal.y)][boolright]))
        if dist>=self.step_len:
           xpoint = self.row_edges[self.row_nrs.index(x_goal.y)][boolright]
           ypoint = self.row_nrs[self.row_nrs.index(x_goal.y)]
           #print("step 2 (on edge)")
           # find point on edge between start and end edge
           updown=-1
           if x_goal.y>x_start.y:
               updown=1
           step=0
           while dist > self.step_len:
               xpoint = self.row_edges[self.row_nrs.index(x_goal.y)-updown*step][
                     boolright]
               ypoint = self.row_nrs[self.row_nrs.index(x_goal.y)-updown*step]
               dist = self.FindCostInfoA(xpoint,ypoint,x_start.x,x_start.y,x_start,False,True)
               step+=1
           #print("x_rand=(" + str(x_goal.x) + "," + str(x_goal.y) + ") - dist = " + str(dist) + " - x_new=(" + str(
           #    xpoint) + "," + str(ypoint) + ")")
           print("nearest=(" + str(x_start.x) + "," + str(x_start.y) + ") - x_rand=(" + str(x_goal.x) + "," + str(
               x_goal.y) + ") - dist = " + str(dist) + " - x_new=(" + str(xpoint) + "," + str(
               ypoint) + ")")
           return Node((int(xpoint),int(ypoint)))


        # step 3:
        # find point on end row
        #print("step 3 (in end row)")
        dist = self.FindCostInfoA(x_goal.x,x_goal.y,x_start.x,x_start.y,x_start,False,True)
        xpoint = x_goal.x
        ypoint = x_goal.y
        step = 0
        while dist > self.step_len:
            xpoint = x_goal.x + direction * step  # note the - sign because we're going backwards on the path
            ypoint = x_goal.y
            dist = self.FindCostInfoA(xpoint, ypoint, x_start.x, x_start.y, x_start, False, True)
            #print("new dist = "+str(dist))
            step+=1

        if xpoint==x_start.x and ypoint==x_start.y:
            print("Something went wrong, no new location found")
        #print("x_rand=("+str(x_goal.x)+","+str(x_goal.y)+") - dist = "+str(dist)+" - x_new=("+str(xpoint)+","+str(ypoint)+")")
        print("nearest=(" + str(x_start.x) + "," + str(x_start.y) + ") - x_rand=(" + str(x_goal.x) + "," + str(
            x_goal.y) + ") - dist = " + str(dist) + " - x_new=(" + str(xpoint) + "," + str(
            ypoint) + ")")
        return Node((int(xpoint),int(ypoint)))


    def Near(self, nodelist, node,max_dist=0,reduction=True):
        if max_dist==0:
            max_dist = self.step_len
        timestart=time.time()
        dist_table = [self.FindCostInfoA(node.x, node.y, nd.x, nd.y, nd, False, True) for nd in nodelist]
        X_near = [nodelist[ind] for ind in range(len(dist_table)) if
                  (dist_table[ind] <= max_dist and dist_table[ind] > 0.0)]
        # print("number of near nodes: "+str(len(X_near)))
        timeend = time.time()
        self.time[3] += (timeend - timestart)

        if len(X_near) > 500 and max_dist >= 5 and reduction:
            X_near_reducted = self.Near(nodelist, node, max_dist - 1)
            if len(X_near_reducted) > 0:
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
        location = random.choice(self.allpoints)
        return Node((location[0],location[1]))

    def ExtractPath(self, node):
        print("Final cost: "+str(node.totalcost)+" Cost without final part: "+str(node.cost))
        print("Final info value: "+str(node.totalinfo))
        # to visualize radius of infopath
        curnode = node
        currentinfopath = []
        if self.kinematic!="dubins" and self.kinematic!="reedsshepp":
            currentinfopath.extend((self.infopathmatrix[node.y * 100 + node.x, self.x_goal.y * 100 + self.x_goal.x])[::-1])
        else:
            start = [node.x, node.y]
            end = [self.x_goal.x, self.x_goal.y]
            [infoastar, costastar] = self.search(start, end)  # A star steps
            currentinfopath.extend(
                self.getInfopath(self.x_goal.x, self.x_goal.y, node.x, node.y, node,infoastar)[0])

        while curnode.parent:
            # print("index 1 = " + str(curnode.parent.y*100+curnode.parent.x) + " index 2 = " + str(
            #    curnode.y*100+curnode.x))
            if self.kinematic!="dubins" and self.kinematic!="reedsshepp":
                currentinfopath.extend((
                    self.infopathmatrix[curnode.parent.y * 100 + curnode.parent.x, curnode.y * 100 + curnode.x])[::-1])
            else:
                start = [curnode.parent.x, curnode.parent.y]
                end = [curnode.x, curnode.y]
                [infoastar, costastar] = self.search(start, end)  # A star steps
                currentinfopath.extend((
                    self.getInfopath(curnode.x, curnode.y, curnode.parent.x, curnode.parent.y, curnode.parent,infoastar)[0])[::-1])
            curnode = curnode.parent

        if self.kinematic=="dubins" or self.kinematic=="reedsshepp" or self.kinematic=="reedsshepprev":
            path = []
            infopath=[]
            start = [node.x, node.y]
            end = [self.x_goal.x, self.x_goal.y]
            [infopathastar, cost] = self.search(start, end)
            if node.parent!=None:
                prevnode=[node.parent.x,node.parent.y]
            else:
                prevnode=None
            for i in range(len(infopathastar) - 1):
                boolvertex=False
                for vertex in self.field_vertex:
                    if infopathastar[i+1][0] == vertex[0] and infopathastar[i+1][1] == vertex[1]:
                        boolvertex = True
                        break
                if boolvertex:
                    print("Twist path!")
                    [dcost, dubins_x, dubins_y, infopathpart] = self.getDubins(infopathastar[i], infopathastar[i + 1],prevnode,True) # twist end point 180 degrees around for more logical path
                else:
                    [dcost, dubins_x, dubins_y, infopathpart] = self.getDubins(infopathastar[i], infopathastar[i+1],prevnode)
                for j in range(len(dubins_x)):
                    infopath.append([dubins_x[j],dubins_y[j]])
                    prevnode=infopathastar[i]
                    # if boolvertex:
                    #     prevnode=True
            path.extend(infopath[::-1])
            while node.parent:
                #print("new piece")
                infopath=[]
                start = [node.parent.x, node.parent.y]
                end = [node.x, node.y]
                # end = [node_start_x+20,node_start_y]
                # print(start +end)
                [infopathastar, cost] = self.search(start, end)
                if node.parent.parent!=None:
                    prevnode=[node.parent.parent.x,node.parent.parent.y]
                else:
                    prevnode=None
                for i in range(len(infopathastar) - 1):
                    boolvertex=False
                    for vertex in self.field_vertex:
                        if infopathastar[i + 1][0] == vertex[0] and infopathastar[i + 1][1] == vertex[1]:
                            boolvertex = True
                            break
                    if boolvertex:
                        print("Twist path!")
                        [dcost, dubins_x, dubins_y, infopathpart] = self.getDubins(infopathastar[i],infopathastar[i + 1],prevnode,True)  # twist end point 180 degrees around for more logical path
                    else:
                        [dcost, dubins_x, dubins_y, infopathpart] = self.getDubins(infopathastar[i], infopathastar[i+1],prevnode)
                    for j in range(len(dubins_x)):
                        infopath.append([dubins_x[j], dubins_y[j]])
                    prevnode=infopathastar[i]
                    # if boolvertex:
                    #     prevnode=True
                path.extend(infopath[::-1])
                node = node.parent

        if self.kinematic=="none" or self.kinematic=="ranger" or self.kinematic=="limit":
            path=[]
            start  = [node.x,node.y]
            end= [self.x_goal.x,self.x_goal.y]
            [infopath, cost] = self.search(start, end)
            path.extend(infopath[::-1])
            while node.parent:
                start = [node.parent.x, node.parent.y]
                end = [node.x, node.y]
                # end = [node_start_x+20,node_start_y]
                # print(start +end)

                [infopath, cost] = self.search(start, end)
                path.extend(infopath[::-1])
                node=node.parent
        # to extract the path
        # path= []
        # path.extend((self.infopathmatrix[node.y*100+node.x, self.x_goal.y*100+self.x_goal.x])[::-1])
        # while node.parent:
        #     #path.append([node.x, node.y])
        #     path.extend((self.infopathmatrix[node.parent.y*100+node.parent.x, node.y*100+node.x])[::-1])
        #     node = node.parent

        #path.append([self.x_start.x, self.x_start.y])
        #print(path)
        #return path
        return path,currentinfopath



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
    def Nearest(self,nodelist, node):
        nearestnode =nodelist[int(np.argmin(
                [self.FindCostInfoA(node.x, node.y, nd.x, nd.y, nd, False, True) for nd in nodelist]))]
        return nearestnode
        # if len(nodelist_new)==0:
        #     return nodelist[int(np.argmin(
        #         [self.FindCostInfoA(node.x, node.y, nd.x, nd.y, nd, False, True) for nd in nodelist]))]
        # return nodelist_new[int(np.argmin([self.FindCostInfoA(node.x, node.y, nd.x, nd.y, nd, False, True) for nd in nodelist_new]))]

    def LastPath(self, node):
        # new with kinematics stuff
        [cost,info] = self.FindCostInfoA(self.x_goal.x,
                                          self.x_goal.y, node.x, node.y,
                                          node, False, False)

        #
        # #left:
        # distleft=0
        # infoleft=0
        # boolright=0
        # [dist, info] = self.FindCostInfoA(self.row_edges[self.row_nrs.index(node.y)][boolright],
        #                                   self.row_nrs[self.row_nrs.index(node.y)], node.x, node.y,
        #                                   node, False, False)
        # distleft+=dist
        # infoleft+=info
        # [dist, info] = self.FindCostInfoA(self.row_edges[self.row_nrs.index(node.y)][boolright],
        #                                   self.row_nrs[self.row_nrs.index(node.y)], self.row_edges[self.row_nrs.index(self.x_goal.y)][boolright],
        #                                   self.row_nrs[self.row_nrs.index(self.x_goal.y)],
        #                                   node, False, False)
        # distleft += dist
        # infoleft += info
        # [dist,info] = self.FindCostInfoA(self.row_edges[self.row_nrs.index(self.x_goal.y)][boolright],
        #                           self.row_nrs[self.row_nrs.index(self.x_goal.y)], self.x_goal.x, self.x_goal.y,
        #                           node, False, False)
        # distleft += dist
        # infoleft += info
        #
        # # right:
        # distright = 0
        # inforight = 0
        # boolright = 1
        # [dist, info] = self.FindCostInfoA(self.row_edges[self.row_nrs.index(node.y)][boolright],
        #                                   self.row_nrs[self.row_nrs.index(node.y)], node.x, node.y,
        #                                   node, False, False)
        # distright += dist
        # inforight += info
        # [dist, info] = self.FindCostInfoA(self.row_edges[self.row_nrs.index(node.y)][boolright],
        #                                   self.row_nrs[self.row_nrs.index(node.y)],
        #                                   self.row_edges[self.row_nrs.index(self.x_goal.y)][boolright],
        #                                   self.row_nrs[self.row_nrs.index(self.x_goal.y)],
        #                                   node, False, False)
        # distright += dist
        # inforight += info
        # [dist, info] = self.FindCostInfoA(self.row_edges[self.row_nrs.index(self.x_goal.y)][boolright],
        #                                   self.row_nrs[self.row_nrs.index(self.x_goal.y)], self.x_goal.x, self.x_goal.y,
        #                                   node, False, False)
        # distright += dist
        # inforight += info
        #
        #
        # if distleft<distright:
        #     cost = distleft
        #     info = infoleft
        # else:
        #     cost = distright
        #     info = inforight

        #[cost,info] = self.FindCostInfoA(self.x_goal.x,self.x_goal.y,node.x,node.y,node,False)
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

        cost = self.FindCostInfoA(node.x, node.y, node.parent.x, node.parent.y, node.parent, False, True)


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
        [cost,info] = self.FindCostInfoA(node.x,node.y,node.parent.x,node.parent.y,node.parent,True)
        #node.infopath = infopath
        info+=node.parent.info

        return info

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def animation(self,k=None,x_new=None):

        plt.cla()
        if k:
            self.plot_grid("RIG, k = " + str(k)+", new node = ("+str(x_new.x)+","+str(x_new.y)+")")
        elif not k:
            self.plot_grid("RIG, k_max = " + str(self.iter_max))
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])


        for node in self.V:
            if node.parent:
                reachedparent=False
                prevpoint=[node.x,node.y]
                infopath = self.infopathmatrix[node.parent.y*100+node.parent.x, node.y*100+node.x]
                for point in infopath[::-1]:
                    reachedparent= (point[0]==node.parent.x and point[1]==node.parent.y)

                    plt.plot([point[0], prevpoint[0]], [point[1], prevpoint[1]], "-g")
                    prevpoint=point
                    if reachedparent:
                       break


        if x_new:
            plt.plot(x_new.x, x_new.y, "bs", linewidth=3)

        plt.xlim([0, 100])
        plt.ylim([0, 100])
        self.ax.set_xlim([0, 100])
        self.ax.set_ylim([0, 100])

        self.fig.tight_layout()
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
                # if self.maze[row,col]==0:
                #     self.ax.add_patch(patches.Rectangle((col-0.5,row-0.5),1,1,
                #                                         edgecolor='black',
                #                                         facecolor='0.5',
                #                                         fill=True
                #                                         ))



        colormap = cm.Blues
        colormap.set_bad(color='black')
        im = self.ax.imshow(self.uncertaintymatrix, cmap=colormap, vmin=0, vmax=1, origin='lower',
                            extent=[0, 100, 0, 100])

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



def main(uncertaintymatrix,row_nrs,row_edges,field_vertex,scenario=None,matrices=None,samplelocations=None):
    x_start = (50, 48)  # Starting node
    #x_goal = (37, 18)  # Goal node
    x_goal = (50,48)
    # scenario = [rowsbool, budget, informed, rewiring, step_len, search_radius, stopsetting, horizonplanning]
    if scenario:
        rrt_star = IRrtStar(x_start, x_goal, scenario[4], 0.0, scenario[5], 2000, uncertaintymatrix, row_nrs,row_edges,field_vertex,scenario, matrices,
                            samplelocations)
    else:
        rrt_star = IRrtStar(x_start, x_goal, 100, 0.0, 15, 2000,uncertaintymatrix,row_nrs,row_edges,field_vertex,scenario,matrices,samplelocations)
    [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration,matrices,samplelocations]=rrt_star.planning()

    return finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration, matrices,samplelocations


if __name__ == '__main__':
    main()
