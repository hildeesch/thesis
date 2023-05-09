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
import time
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

        self.x_best = self.x_start

        self.reductioncount = 0 # to count the shortening of range in Near()
        self.reduction = 0

        self.budget=250
        self.inforadius=1
        self.scenario = 1
        self.kinematic = "none"  # kinematic constraint
        # choices: "none", "dubins", "dubinsrev", "reedsshepp", "ranger", "limit"

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

        self.time = np.zeros(8) # for debugging
        # 0 = sample, 1 = nearest, 2 = steer, 3 = near, 4 = rewiring, 5 = lastpath, 6 = pruning, 7 = total time

        self.maze = np.ones((100,100) ) # for A star
        self.edgemaze = np.ones((100,100)) # for A star (to know where we can go up/down)


    def init(self):
        #cMin, theta = self.get_distance_and_angle(self.x_start, self.x_goal)
        #C = self.RotationToWorldFrame(self.x_start, self.x_goal, cMin)
        #xCenter = np.array([[(self.x_start.x + self.x_goal.x) / 2.0],
        #                    [(self.x_start.y + self.x_goal.y) / 2.0], [0.0]])
        x_best = self.x_start

        #return theta, cMin, xCenter, C, x_best

        self.EdgeCostInfo()
        for row in range(100):
            for col in range(100):
                if [col,row] in self.allpoints:
                    self.maze[row,col]=0
        # fig, ax = plt.subplots()
        # colormap = cm.Greys
        # colormap.set_bad(color='black')
        # im = ax.imshow(self.maze, cmap=colormap, vmin=0, vmax=1, origin='lower')
        #
        # ax.set_title("A star maze")
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
        return x_best

    def planning(self):
        show=True
        #theta, dist, x_center, C, x_best = self.init()
        #theta, dist, x_center, x_best = self.init()
        x_best = self.init()
        #c_best = np.inf
        count_down=3
        i_best = 0
        startlen=0 # for checking node increase
        totalstarttime=time.time()
        for k in range(self.iter_max):
            #time.sleep(0.1)
            if k>50-3: #only evaluate from when we might want it to stop
                #print("Start countdown")
                cost = {node: node.totalcost for node in self.X_soln}
                info = {node: node.totalinfo for node in self.X_soln}
                #x_best = min(cost, key=cost.get)
                x_best = max(info, key=info.get)
                #c_best = cost[x_best]
                i_last_best = i_best
                i_best = info[x_best]
                #if i_last_best>0: # to prevent division by zero
                if ((i_best-i_last_best)/i_last_best)<0.01: #smaller than 1% improvement
                    count_down-=1
                else:
                    count_down=10 #reset
                    print("Reset countdown")
            if k==51: # to test up to certain iteration
                count_down=0
            if count_down<=0 and k>50:
                print("Reached stopping criterion at iteration "+str(k))
                break # we stop iterating if the best score is not improving much anymore and we already passed at least ... cycles

            if k%50==0:
                print("ATTENTION!!! ATTENTION!!! ATTENTION!!! AGAIN FIFTY CYCLES FURTHER, CURRENT CYCLE ="+str(k)) # to know how far we are
            endlen=len(self.V)
            print("Nr of nodes added: "+str(endlen-startlen))
            startlen = len(self.V)
            #x_rand = self.Sample(c_best, dist, x_center, C)
            timestart = time.time()
            x_rand = self.Sample()
            timeend= time.time()
            self.time[0]+=(timeend-timestart)
            timestart=time.time()
            x_nearest = self.Nearest(self.V, x_rand)
            timeend=time.time()
            self.time[1] += (timeend - timestart)
            timestart=time.time()
            x_new = self.Steer_section(x_nearest, x_rand) #so that we only generate one new node, not multiple
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
                    break
            #if x_nearest.cost + self.Line(x_nearest, x_new) + self.Line(x_new, self.x_goal) < self.budget and not double:  # budget check for nearest parent (to make it more efficient)
            if not double:  # budget check for nearest parent (to make it more efficient)
                # print(x_nearest.cost + self.Line(x_nearest, x_new) + self.Line(x_new, self.x_goal))
                for x_near in self.Near(self.V,x_new):

                    if x_new and not self.utils.is_collision(x_near, x_new):
                        c_min = x_near.cost+self.FindCostInfoA(x_new.x,x_new.y,x_near.x,x_near.y,x_near,False,True) #cost from near node to new node

                        if c_min+self.FindCostInfoA(self.x_goal.x,self.x_goal.y,x_new.x,x_new.y,x_near,False,True) <=self.budget: #extra check for budget for actual parent (cmin+ cost to goal node)

                            node_new = Node((x_new.x,x_new.y))
                            node_new.parent = x_near #added
                            node_new.cost = self.Cost(node_new) #+self.Line(x_new, self.x_goal)
                            node_new.info = self.Info_cont(node_new)
                            self.V.append(node_new) #generate a "node"/trajectory to each near point


                            if self.InGoalRegion(node_new):
                                if not self.utils.is_collision(node_new, self.x_goal):
                                    self.X_soln.add(node_new)
                                    timestart=time.time()
                                    self.LastPath(node_new)
                                    timeend = time.time()
                                    self.time[5] += (timeend - timestart)

                timestart=time.time()
                self.Pruning(x_new)
                timeend = time.time()
                self.time[6] += (timeend - timestart)

            if k % 50 == 0 and show:
                self.animation()
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

        #self.path = self.ExtractPath(x_best)
        [self.path,infopathradius] = self.ExtractPath(x_best)

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
            node = x_best
            while node.parent:
                # reachedparent=False
                prevpoint = [node.x, node.y]
                infopath = self.infopathmatrix[node.parent.y * 100 + node.parent.x, node.y * 100 + node.x]
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
            #ax.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
            for cell in infopathradius:
                ax.plot(cell[0],cell[1],marker="o",markersize=1,color="blue")
            ax.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')


            #ax.plot([x for x, _ in x_best.infopath], [y for _, y in x_best.infopath], '-r')
            #ax.plot([x for x, _ in x_best.lastinfopath], [y for _, y in x_best.lastinfopath], '-r')
            ax.set_title("Spatial distribution of uncertainty and final path")
            fig.tight_layout()
            plt.show()
            #plt.close()

        return self.path, infopathradius, x_best.totalcost, x_best.totalinfo, self.budget, self.step_len, self.search_radius, k

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
            self.edgemaze[self.row_nrs[index],self.row_edges[index][0]]=0
            self.edgemaze[self.row_nrs[index],self.row_edges[index][1]]=0

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
            for gridpoint in infopath:
                self.maze[gridpoint[1],gridpoint[0]]=0
                self.edgemaze[gridpoint[1],gridpoint[0]]=0
                width_path=1
                for i in range(width_path):
                    self.maze[gridpoint[1], gridpoint[0]+(i+1)] = 0
                    self.edgemaze[gridpoint[1], gridpoint[0]+(i+1)] = 0

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
            for gridpoint in infopath:
                self.maze[gridpoint[1],gridpoint[0]]=0
                self.edgemaze[gridpoint[1],gridpoint[0]]=0

                width_path=1
                for i in range(width_path):
                    self.maze[gridpoint[1], gridpoint[0]-(i+1)] = 0
                    self.edgemaze[gridpoint[1], gridpoint[0]-(i+1)] = 0

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

    def search(self,maze, edgemaze, start, end):
        #TODO: why do we pass maze/ edgemaze as arguments (while they're in the self)
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
                print("giving up on pathfinding too many iterations")
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

            for new_position in move:
                cost = new_position[2]


                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # Make sure within range (check if within maze boundary)
                if (node_position[0] > (no_rows - 1) or
                        node_position[0] < 0 or
                        node_position[1] > (no_columns - 1) or
                        node_position[1] < 0):
                    continue

                # Make sure walkable terrain
                # note: if .. continue means: if the statement returns true, we skip the rest of the loop
                # two conditions: if the position is not within the rows/edges or if the position is changing in y position while it's not on "edge" terrain
                if maze[node_position[1]][node_position[0]] != 0 or (new_position[1]!=0 and (edgemaze[node_position[1]][node_position[0]]!=0 or edgemaze[current_node.position[1]][current_node.position[0]]!=0)): #or (node_position[0]==end[0] and node_position[1]==end[1])
                    continue

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
                child.h = (abs(child.position[1] - end_node.position[1]))**2
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
            info=0
            # a star
            #maze=self.maze
            start=[node_start_x,node_start_y]
            end = [node_end_x,node_end_y]
            #end = [node_start_x+20,node_start_y]
            #print(start +end)

            [infopath,cost] = self.search(self.maze, self.edgemaze, start, end)

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
            infopathcopy=deepcopy(infopath)
            for infopoint in infopathcopy:
                if np.isnan(self.uncertaintymatrix[infopoint[1], infopoint[0]]):
                    cost=np.inf
                    break;
                info += self.uncertaintymatrix[infopoint[1], infopoint[0]]
                if self.inforadius>0:
                    xpoint_ = infopoint[0]
                    for rowdist in range(-self.inforadius,self.inforadius+1):
                        ypoint_=infopoint[1]+rowdist
                        if not [xpoint_, ypoint_] in infopath and not np.isnan(self.uncertaintymatrix[ypoint_, xpoint_]):
                            info += self.uncertaintymatrix[ypoint_, xpoint_]
                            infopath.append([xpoint_, ypoint_])

            self.infomatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x] = info
            self.infomatrix[
                node_end_y * 100 + node_end_x, node_start_y * 100 + node_start_x] = info  # mirror the matrix

            self.costmatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x] = cost
            self.costmatrix[node_end_y * 100 + node_end_x, node_start_y * 100 + node_start_x] = cost #mirror the matrix

            self.infopathmatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x] = infopath
            self.infopathmatrix[node_end_y * 100 + node_end_x, node_start_y * 100 + node_start_x] = infopath[::-1]

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
            currentinfopath.extend(
                self.infopathmatrix[curnode.parent.y * 100 + curnode.parent.x, curnode.y * 100 + curnode.x])
            curnode = curnode.parent
        if not any(element in currentinfopath for element in infopath):  # the whole infopath is new
            infonode += info
        else:  # if some infopoints overlap
            for element in infopath:
                if not element in currentinfopath:
                    infonode += self.uncertaintymatrix[element[1], element[0]]
        return cost, infonode
        # return [cost,infopath,info]

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
                    if np.isnan(self.uncertaintymatrix[ypoint, xpoint]):
                        self.costmatrix[node_start_y * 100 + node_start_x, node_end_y * 100 + node_end_x] = np.inf
                        self.costmatrix[node_end_y * 100 + node_end_x, node_start_y * 100 + node_start_x] = np.inf # mirror the matrix
                    else:
                        info += self.uncertaintymatrix[ypoint, xpoint]
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
    def Rewiring_afterv2(self, best_node): #rewiring afterwards
        # goal: gain more info while remaining within the budget
        print("Start rewiring after v2")
        print(" Info: " + str(best_node.info) + " Tot. info: " + str(
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
            print(node.parent.parent.x, node.parent.parent.y)

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
                if self.kinematic == "dubins" or self.kinematic=="reedsshepp" or self.kinematic=="dubinsrev":
                    #TODO
                    #[cost, info] = self.dubins(node, x_near, False)
                    c_new = node.parent.parent.cost + self.dubinsnomatrix(node.parent.parent,x_near,True)[0] + self.dubinsnomatrix(node,x_near,True)[0]

                else:
                    # c_new = node.parent.parent.cost + self.Line(node.parent.parent, x_near) + self.Line(node,
                    #                                                                                        x_near)
                    [cost, info] = self.FindCostInfoA(node.parent.parent.x, node.parent.parent.y, x_near.x, x_near.y, node.parent.parent, True)
                    cost2 = self.FindCostInfoA(x_near.x, x_near.y, node.x, node.y, node.parent.parent, True,True)
                    c_new = node.parent.parent.cost + cost + cost2

                # if x_new.parent.x==x_near.x and x_new.parent.y==x_near.y:
                #     return # if the parent of x_new = x_near, we don't want to make the parent of x_near = x_new (because then we create a loose segment
                if (c_new-c_old) < (self.budget-best_node.totalcost): # still within budget
                    if not self.kinematic == "dubins" and not self.kinematic=="reedsshepp" and not self.kinematic=="dubinsrev":
                        newnode = Node((x_near.x, x_near.y))
                        newnode.parent = node.parent.parent
                        addnew = True
                        # for node in self.V[::-1]: # to prevent adding doubles
                        #     if node.parent==newnode.parent and node.x==newnode.x and node.y==newnode.y:
                        #         newnode=node
                        #         addnew = False
                        #         break
                        if addnew:
                            newnode.info = node.parent.parent.info + info
                            newnode.cost = node.parent.parent.cost + cost
                            self.LastPath(newnode)
                        info = newnode.info + self.FindCostInfoA(newnode.x, newnode.y, node.x, node.y, newnode, True)[1]
                    if self.kinematic=="dubins" or self.kinematic=="reedsshepp" or self.kinematic=="dubinsrev":
                        #TODO (dubins part etc)
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
                    if self.kinematic=="dubins" or self.kinematic=="reedsshepp" or self.kinematic=="dubinsrev":
                        #[dist,info] = self.dubins(parent,node)
                        [dist,info] = self.dubinsnomatrix(parent,node,True)
                        node.info = parent.info + info
                    else:
                        #dist = self.Line(parent, node)
                        [dist,info] = self.FindCostInfoA(parent.x, parent.y, node.x, node.y, parent, True)

                        node.info = parent.info + info

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
                            break;
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
        dist = self.FindCostInfoA(x_goal.x,x_goal.y,x_start.x,x_start.y,x_start,False,True)

        # closer=False
        # if dist>self.step_len:
        #     #just for now for debugging purposes
        #     closer=True
        maxdist = self.step_len-self.reduction
        if dist<=maxdist:
            #print("x_rand close enough, x_rand = x_new --> x_rand=("+str(x_goal.x)+","+str(x_goal.y)+")")
            print("nearest=(" + str(x_start.x) + "," + str(x_start.y) + ") - x_rand=(" + str(x_goal.x) + "," + str(
                x_goal.y) + ") - dist = " + str(dist) + " - x_new=(" + str(x_goal.x) + "," + str(
                x_goal.y) + ")")

            #return x_goal
            return Node((int(x_goal.x), int(x_goal.y)))
        distleft = self.FindCostInfoA(self.row_edges[self.row_nrs.index(x_goal.y)][0], self.row_nrs[self.row_nrs.index(x_goal.y)], x_start.x, x_start.y,
                                     x_start, False, True)
        distright = self.FindCostInfoA(self.row_edges[self.row_nrs.index(x_goal.y)][1], self.row_nrs[self.row_nrs.index(x_goal.y)], x_start.x, x_start.y,
                                      x_start, False, True)
        boolright = 0
        if distright<distleft:
            boolright=1 #bool right means right is shorter if boolright=1
        #sampling within the end row
        withinbounds=True
        distance=5
        while withinbounds and dist>maxdist:
            if boolright==1:
                xpoint = x_goal.x+distance
                ypoint = self.row_nrs[self.row_nrs.index(x_goal.y)]
                if xpoint>=self.row_edges[self.row_nrs.index(x_goal.y)][boolright]:
                    withinbounds=False
                else:
                    dist = self.FindCostInfoA(xpoint,ypoint, x_start.x, x_start.y,
                                         x_start, False, True)

            else:
                xpoint = x_goal.x-distance
                ypoint = self.row_nrs[self.row_nrs.index(x_goal.y)]
                if xpoint<=self.row_edges[self.row_nrs.index(x_goal.y)][boolright]:
                    withinbounds=False
                else:
                    dist = self.FindCostInfoA(xpoint, ypoint, x_start.x, x_start.y,
                                             x_start, False, True)
            distance+=5

        index = self.row_nrs.index(x_goal.y)
        while dist>maxdist:
            xpoint = self.row_edges[index][boolright]
            ypoint = self.row_nrs[index]
            dist= self.FindCostInfoA(xpoint,ypoint, x_start.x, x_start.y,
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
        if dist>maxdist:
            xpoint=random.choice([x_start.x-maxdist,x_start.x+maxdist])
            ypoint=x_start.y
            dist=maxdist
            print("last resort sampling: in own row")
        #node_new.parent = x_start
        #print("WE NEED TO SAMPLE CLOSER TO THE NEAREST: ("+str(x_start.x)+","+str(x_start.y)+")")
        #print("x_nearest=("+str(x_start.x)+","+str(x_start.y)+")")
        #print("x_rand=("+str(x_goal.x)+","+str(x_goal.y)+") - dist = "+str(dist)+" - x_new=("+str(xpoint)+","+str(ypoint)+")")
        print("nearest=("+str(x_start.x)+","+str(x_start.y)+") - x_rand=("+str(x_goal.x)+","+str(x_goal.y)+") - dist = "+str(dist)+" - x_new=("+str(xpoint)+","+str(ypoint)+")")

        return Node((int(xpoint),int(ypoint)))

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
            #return x_goal
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


    def Near(self, nodelist, node,max_dist=0):
        timestart=time.time()
        if max_dist==0:
            max_dist = self.step_len
        if max_dist==self.step_len or max_dist==self.search_radius:
            max_dist-=self.reduction
        #max_dist-=self.reduction
        #heuristic:
        nodelist_new = nodelist[:]
        for nd in nodelist: #TODO check if this actually speeds things up
            if (nd.x-node.x)**2>max_dist**2 and (nd.y-node.y)**2>max_dist**2:
                nodelist_new.remove(nd)
        #actual calculation
        dist_table = [self.FindCostInfoA(nd.x, nd.y, node.x, node.y, node, False, True) for nd in nodelist]
        X_near = [nodelist[ind] for ind in range(len(dist_table)) if (dist_table[ind] <= max_dist and dist_table[ind] > 0.0
                                                      and not self.utils.is_collision(nodelist[ind], node)==True)]
        #print("number of near nodes: "+str(len(X_near)))
        timeend = time.time()
        self.time[3] += (timeend - timestart)
        if len(X_near)>500 and max_dist>=5:
            self.reductioncount+=1
            #print("Shortening the range for Near: "+str(max_dist-1))
            if self.reductioncount==1000:
                self.reductioncount=0
                self.reduction+=1
                print("Range is reducted by "+str(self.reduction))
            return self.Near(nodelist,node,max_dist-1)
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
        print("Final cost: "+str(node.totalcost)+" Cost without final part: "+str(node.cost))
        print("Final info value: "+str(node.totalinfo))
        # to visualize radius of infopath
        curnode = node
        currentinfopath = []
        currentinfopath.extend((self.infopathmatrix[self.x_goal.y * 100 + self.x_goal.x, node.y * 100 + node.x])[::-1])

        while curnode.parent:
            # print("index 1 = " + str(curnode.parent.y*100+curnode.parent.x) + " index 2 = " + str(
            #    curnode.y*100+curnode.x))
            currentinfopath.extend((
                self.infopathmatrix[curnode.parent.y * 100 + curnode.parent.x, curnode.y * 100 + curnode.x])[::-1])
            curnode = curnode.parent

        path=[]
        start  = [node.x,node.y]
        end= [self.x_goal.x,self.x_goal.y]
        [infopath, cost] = self.search(self.maze, self.edgemaze, start, end)
        path.extend(infopath[::-1])
        while node.parent:
            start = [node.parent.x, node.parent.y]
            end = [node.x, node.y]
            # end = [node_start_x+20,node_start_y]
            # print(start +end)

            [infopath, cost] = self.search(self.maze, self.edgemaze, start, end)
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

    def InGoalRegion(self, node):
        #if self.Line(node, self.x_goal) < self.step_len:
        if node.cost+self.FindCostInfoA(self.x_goal.x, self.x_goal.y, node.x, node.y, node, False, True) <= self.budget:
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
        return nodelist[int(np.argmin([self.FindCostInfoA(nd.x, nd.y, n.x, n.y, n, False, True) for nd in nodelist]))]

    def LastPath(self, node):

        #left:
        distleft=0
        infoleft=0
        boolright=0
        [dist, info] = self.FindCostInfoA(self.row_edges[self.row_nrs.index(node.y)][boolright],
                                          self.row_nrs[self.row_nrs.index(node.y)], node.x, node.y,
                                          node, False, False)
        distleft+=dist
        infoleft+=info
        [dist, info] = self.FindCostInfoA(self.row_edges[self.row_nrs.index(node.y)][boolright],
                                          self.row_nrs[self.row_nrs.index(node.y)], self.row_edges[self.row_nrs.index(self.x_goal.y)][boolright],
                                          self.row_nrs[self.row_nrs.index(self.x_goal.y)],
                                          node, False, False)
        distleft += dist
        infoleft += info
        [dist,info] = self.FindCostInfoA(self.row_edges[self.row_nrs.index(self.x_goal.y)][boolright],
                                  self.row_nrs[self.row_nrs.index(self.x_goal.y)], self.x_goal.x, self.x_goal.y,
                                  node, False, False)
        distleft += dist
        infoleft += info

        # right:
        distright = 0
        inforight = 0
        boolright = 1
        [dist, info] = self.FindCostInfoA(self.row_edges[self.row_nrs.index(node.y)][boolright],
                                          self.row_nrs[self.row_nrs.index(node.y)], node.x, node.y,
                                          node, False, False)
        distright += dist
        inforight += info
        [dist, info] = self.FindCostInfoA(self.row_edges[self.row_nrs.index(node.y)][boolright],
                                          self.row_nrs[self.row_nrs.index(node.y)],
                                          self.row_edges[self.row_nrs.index(self.x_goal.y)][boolright],
                                          self.row_nrs[self.row_nrs.index(self.x_goal.y)],
                                          node, False, False)
        distright += dist
        inforight += info
        [dist, info] = self.FindCostInfoA(self.row_edges[self.row_nrs.index(self.x_goal.y)][boolright],
                                          self.row_nrs[self.row_nrs.index(self.x_goal.y)], self.x_goal.x, self.x_goal.y,
                                          node, False, False)
        distright += dist
        inforight += info


        if distleft<distright:
            cost = distleft
            info = infoleft
        else:
            cost = distright
            info = inforight

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

    rrt_star = IRrtStar(x_start, x_goal, 15, 0.0, 15, 2000,uncertaintymatrix,row_nrs,row_edges)
    [finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration]=rrt_star.planning()

    return finalpath, infopath, finalcost, finalinfo, budget, steplength, searchradius, iteration


if __name__ == '__main__':
    main()
