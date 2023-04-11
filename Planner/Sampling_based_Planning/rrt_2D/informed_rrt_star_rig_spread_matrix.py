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
                 goal_sample_rate, search_radius, iter_max,uncertaintymatrix):
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

        self.reductioncount = 0 # to count the shortening of range in Near()
        self.reduction = 0

        self.budget=200
        self.kinematic = "none" # kinematic constraint
        # choices: "none", "dubins", "ranger", "limit"
        self.dubinsmatrix = np.empty((100, 100), dtype=object)

        self.uncertaintymatrix = uncertaintymatrix

        self.x_best = self.x_start # just for now, remove later


        if self.kinematic=="dubins":
            print("dubins")
            self.costmatrix = np.empty((100 * 100 * 8, 100 * 100))
            self.anglematrix = np.empty((100 * 100 * 8, 100 * 100))
            self.infopathmatrix = np.empty((100 * 100 * 8, 100 * 100), dtype=object)
            self.infomatrix = np.empty((100 * 100 * 8, 100 * 100))
        else:
            self.costmatrix = np.empty((100 * 100, 100 * 100))
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
        show = True
        #theta, dist, x_center, C, x_best = self.init()
        self.x_best = self.init()
        x_best = self.init()
        c_best = np.inf
        count_down=20
        i_best = 0.001
        totalstarttime=time.time()
        startlen=0 # for checking node increase
        for k in range(self.iter_max):
            #time.sleep(0.1)
            if k>=50-3: #only evaluate from when we might want it to stop #TODO make 400 a variable
                cost = {node: node.totalcost for node in self.X_soln}
                info = {node: node.totalinfo for node in self.X_soln}
                #x_best = min(cost, key=cost.get)
                self.x_best = max(info, key=info.get)
                x_best = max(info, key=info.get)
                #c_best = cost[x_best]
                i_last_best = i_best
                i_best = info[x_best]
                #print("i_best: "+str(i_best)+" i_last_best: "+str(i_last_best)+" Criterion value: "+str(((i_best-i_last_best)*100/i_last_best)))
                if ((i_best-i_last_best)/i_last_best)<0.001: #smaller than 1% improvement
                    count_down-=1
                else:
                    count_down=20 #reset
                    print("reset countdown")
            # if k==101: # to test up to certain iteration
            #     count_down=0
            if count_down<=0 and k>1000:
                print("Reached stopping criterion at iteration "+str(k))
                break # we stop iterating if the best score is not improving much anymore and we already passed at least ... cycles

            if k%50==0:
                print("ATTENTION!!! ATTENTION!!! ATTENTION!!! AGAIN FIFTY CYCLES FURTHER, CURRENT CYCLE ="+str(k)) # to know how far we are
            endlen=len(self.V)
            print("Nr of nodes added: "+str(endlen-startlen))
            # if (endlen-startlen)==0:
            #     print("Len X_Near was: "+str(len(self.Near(self.V,x_new))))
            startlen = len(self.V)
            #x_rand = self.Sample(c_best, dist, x_center, C)
            #x_rand = self.Sample(c_best, dist, x_center)
            timestart=time.time()
            x_rand = self.Sample()
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
                    break
            #if x_nearest.cost + self.Line(x_nearest, x_new) + self.Line(x_new, self.x_goal) < self.budget and not double:  # budget check for nearest parent (to make it more efficient)
            if not double:  # budget check for nearest parent (to make it more efficient)
                # print(x_nearest.cost + self.Line(x_nearest, x_new) + self.Line(x_new, self.x_goal))
                node_new=[]
                for x_near in self.Near(self.V,x_new):
                    #if x_new and not self.utils.is_collision(x_near, x_new):

                    if not self.utils.is_collision(x_near, x_new):
                        if self.kinematic=="dubins":
                            [dubinscost,dubinsinfo] = self.dubins(x_near,x_new)
                            c_min = x_near.cost + dubinscost
                            endcost = self.dubins(x_new,self.x_goal,True)
                        else:
                            c_min = x_near.cost + self.Line(x_near, x_new)
                            endcost = self.Line(x_new, self.x_goal)

                        # if c_min+self.Line(x_new, self.x_goal) > self.budget:
                        #     print("past budget (step 2): "+str(c_min+self.Line(x_new, self.x_goal)))

                        if c_min+endcost <=self.budget: #extra check for budget for actual parent
                            node_new = Node((x_new.x,x_new.y))
                            node_new.cost = c_min #+self.Line(x_new, self.x_goal)
                            node_new.parent = x_near #added
                            #node_new.info = self.Info(node_new)
                            if self.kinematic=="dubins":
                                node_new.info = dubinsinfo
                            else:
                                node_new.info = self.Info_cont(node_new)
                            self.V.append(node_new) #generate a "node"/trajectory to each near point

                            # rewire
                            for x_near in self.Near(self.V,x_new,self.search_radius):
                                timestart=time.time()
                                self.Rewiring(x_near,node_new)
                                timeend = time.time()
                                self.time[4] += (timeend - timestart)


                            #if self.InGoalRegion(node_new): # skip because we already check this earlier
                            if not self.utils.is_collision(node_new, self.x_goal):
                                self.X_soln.add(node_new)
                                timestart=time.time()
                                self.LastPath(node_new)
                                timeend = time.time()
                                self.time[5] += (timeend - timestart)

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
                print("It.: " + str(k) + " Time: " + str(self.time[7]) + " Info: " + str(x_best.info) + " Tot. info: "+str(x_best.totalinfo) + " Cost: " + str(x_best.totalcost) + " Nodes: "+str(len(self.V)))

        self.path = self.ExtractPath(x_best)
        #for point in reversed(x_best.infopath):
        #    print("infopoint (x,y)=("+str(point[0])+","+str(point[1])+")")
        #print("length of infopath: "+str(len(x_best.infopath)+len(x_best.lastinfopath)))

        node = x_best
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
            ax.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
            ax.set_title("Spatial distribution of uncertainty and final path")
            fig.tight_layout()
            plt.show()

        return self.path, x_best.totalcost, x_best.totalinfo, self.budget, self.step_len, self.search_radius, k

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
                xpoint = math.floor(node_start_x + t * xline)
                ypoint = math.floor(node_start_y + t * yline)
                if not [xpoint,ypoint] in infopath:
                    info += self.uncertaintymatrix[ypoint, xpoint]
                    infopath.append([xpoint, ypoint])
                    if np.isnan(self.uncertaintymatrix[ypoint, xpoint]):  # to prevent going through edges and/or obstacles
                        self.costmatrix[node_start_y * 100 + node_start_x,node_end_y * 100 + node_end_x] = np.inf
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
    def Rewiring(self, x_near,x_new):
        c_near = x_near.cost
        if self.kinematic=="dubins":
            [cost,info] = self.dubins(x_new, x_near, False)
            c_new = x_new.cost + cost
        else:
            c_new = x_new.cost + self.Line(x_new, x_near)

        if x_new.parent.x==x_near.x and x_new.parent.y==x_near.y:
            return # if the parent of x_new = x_near, we don't want to make the parent of x_near = x_new (because then we create a loose segment
        if c_new < c_near:
            if not self.kinematic=="dubins":
                info = self.FindInfo(x_near.x,x_near.y,x_new.x,x_new.y,x_new,self.search_radius,True)

            info += x_new.info

            info_near = x_near.info

            info_new = info
            if info_new>=info_near: #note: this is different than the condition in pruning
                if x_near==self.x_best:
                    previnfo=x_near.info
                    prevtotalinfo=x_near.totalinfo
                # saving temporary node to compare the total info values (TODO: check if deepcopy is faster than assigning all the node parameters)
                x_temp = Node((x_near.x,x_near.y))
                x_temp.parent = x_near.parent
                x_temp.info = x_near.info
                x_temp.cost = x_near.cost
                x_temp.totalinfo = x_near.totalinfo
                x_temp.totalcost = x_near.totalcost

                # rewiring:
                x_near.parent = x_new
                x_near.info = info_new
                x_near.cost = c_new
                self.LastPath(x_near) # also recalculate the last info part
                if x_near==self.x_best and prevtotalinfo>x_near.totalinfo:
                    print("[REWIRING] Best node is removed, prev totinfo: "+str(prevtotalinfo)+" Previnfo: "+str(previnfo)+" New totinfo: "+str(x_near.totalinfo)+" Newinfo: "+str(x_near.info))

                oldparent=None
                # if totalcost not increasing, also add the temp (the old one)
                if x_temp.totalinfo>x_near.totalinfo:
                    # Scenario 1:
                    # x_near = x_temp
                    # Other scenarios
                    self.V.append(x_temp)
                    oldparent=x_temp
                    #print("Totalinfo rewiring not increased, copy of old node added")

                # Scenario 1:
                # else:
                self.Recalculate(x_near,oldparent) # recalculates the cost and info for nodes further down the path


                # # bit of debugging:
                # thisnode=x_new
                # while thisnode.parent:
                #     thisnode=thisnode.parent
                # if not (thisnode.x==self.x_start.x and thisnode.y==self.x_start.y):
                #     print("WARNING WARNING WARNING LOOSE END LOOSE END LOOSE END LOOSE END LOOSE END LOOSE END AT X_NEAR = ("+str(x_near.x)+","+str(x_near.y)+")")

                #x_near.infopath = infopath
                #print("Rewiring took place!!")
    def Recalculate(self,parent, prevparent=None):
        for node in self.V:  # to recalculate the cost and info for nodes further down the line
            if node.parent == parent:
                # Scenario 3:
                # if not prevparent==None:
                #     # saving temporary node to compare the total info values (TODO: check if deepcopy is faster than assigning all the node parameters)
                #     x_temp = Node((node.x, node.y))
                #     x_temp.parent = node.parent
                #     x_temp.info = node.info
                #     x_temp.cost = node.cost
                #     x_temp.totalinfo = node.totalinfo
                #     x_temp.totalcost = node.totalcost

                # Scenario 2:
                # if the old one had a higher total value, we change the parent to the old (the copy) #TODO check how valid this is
                if not prevparent==None:
                    node.parent=prevparent
                if prevparent==None:
                    if node==self.x_best:
                        previnfo=node.info
                        prevtotalinfo=node.totalinfo
                    if self.kinematic=="dubins":
                        [dist,info] = self.dubins(parent,node)
                        node.info = parent.info + info
                    else:
                        dist = self.Line(parent, node)
                        node.info = parent.info + self.FindInfo(node.x, node.y, parent.x, parent.y, parent,
                                                                dist, True)

                    node.cost = parent.cost + dist

                    self.LastPath(node)

                    if node == self.x_best and node.totalinfo<prevtotalinfo:
                        print("[RECALCULATE] Best node is removed, prev totinfo: " + str(
                                    prevtotalinfo) + " Previnfo: " + str(previnfo) + " New totinfo: " + str(
                                    node.totalinfo) + " Newinfo: " + str(node.info))
                # oldparent=None
                # if not prevparent==None:
                #     if x_temp.totalinfo>node.totalinfo:
                #         self.V.append(x_temp)
                #         oldparent=x_temp
                #         print("Totalinfo recalculating not increased, copy of old node added")
                #     else:
                #         print("Totalinfo recalculating increased, copy of old node not added")
                #only continue if the rewiring increased the totalinfo (prevparent==None)
                self.Recalculate(node,oldparent)
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
                    if (node1.cost==node2.cost and node1.info==node2.info and index!=index2 and node1.parent==node2.parent):
                        print("Double detected, now pruned")
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
                        if node1 in self.V: #still have to figure out why this is needed (TODO)
                            if node1 == self.x_best:
                                print("[PRUNING] Best node is removed, info: " + str(
                                    self.x_best.totalinfo))
                            self.V.remove(node1)
                        #nodelist.pop(index)
                        #costlist.pop(index)
                        #infolist.pop(index)
                        #costlist[index]=np.nan
                        #infolist[index]=np.nan
                        # TODO still: how to "pop" or "remove" from the list we iterate over? to speed up pruningz
                        #if node1 in self.X_soln:
                        #    self.X_soln.remove(node1)  # also from the solutions
                        self.X_soln.discard(node1)

    def Steer(self, x_start, x_goal):
        dist, theta = self.get_distance_and_angle(x_start, x_goal)
        dist = min(self.step_len-self.reduction, dist)
        notfinished=True
        while notfinished and dist>=0:
            node_new = Node((math.floor(x_start.x + dist * math.cos(theta)),
                             math.floor(x_start.y + dist * math.sin(theta))))
            if not np.isnan(self.uncertaintymatrix[node_new.y,node_new.x]) and self.Line(x_start,node_new)<=(self.step_len-self.reduction):
                dist = self.Line(x_start,node_new)
                notfinished=False # to prevent sampling in obstacle and sampling too far due to rounding
            dist-=1
        #node_new.parent = x_start

        if self.kinematic=="dubins":
            dist = self.dubins(x_start,node_new,True)

        print("nearest=("+str(x_start.x)+","+str(x_start.y)+") - x_rand=("+str(x_goal.x)+","+str(x_goal.y)+") - dist = "+str(dist+1)+" - x_new=("+str(node_new.x)+","+str(node_new.y)+")")
        return node_new

    def Near(self, nodelist, node, max_dist=0):
        timestart=time.time()
        if max_dist==0:
            max_dist = self.step_len
        if max_dist==self.step_len or max_dist==self.search_radius:
            max_dist-=self.reduction
        dist_table = [self.get_distance_and_angle(nd,node)[0] for nd in nodelist]
        X_near = [nodelist[ind] for ind in range(len(dist_table)) if (dist_table[ind] <= max_dist and dist_table[ind] > 0.0
                                                                    and not self.utils.is_collision(nodelist[ind], node))]
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

#    def Sample(self, c_max, c_min, x_center, C):
    #def Sample(self, c_max, c_min, x_center): #TODO can we leave out this function?
    def Sample(self):
        x_rand = self.SampleFreeSpace()

        return x_rand

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
        print("Final cost: "+str(node.totalcost))
        print("Final info value: "+str(node.totalinfo))
        #path = [[self.x_goal.x, self.x_goal.y]]
        # while node.parent:
        #     path.append([node.x, node.y])
        #     node = node.parent
        # path.append([node.x,node.y]) # this should be the start



        path = []
        maxc = 2


        sx = self.x_goal.x
        sy = self.x_goal.y
        syaw = self.anglematrix[node.y * 100 + node.x, self.x_goal.y * 100 + self.x_goal.x]+math.pi
        #[syaw,int] = self.roundAngle(syaw)
        gx = node.x
        gy = node.y
        gyaw = self.anglematrix[node.parent.y*100 + node.parent.x,node.y * 100 + node.x]+math.pi
        #[gyaw,int] = self.roundAngle(gyaw)
        [dubinspath,self.dubinsmatrix] = dubins.calc_dubins_path(sx, sy, syaw, gx, gy, gyaw, maxc, self.dubinsmatrix)
        for i in range((len(dubinspath.x))):
            pathpart = []
            pathpart.append([dubinspath.x[i], dubinspath.y[i]])
            path.extend(pathpart[::-1])

        # gx = node.parent.x
        # gy = node.parent.y
        # gyaw = self.anglematrix[(node.parent).parent.y * 100 + (node.parent).parent.x, node.parent.y * 100 + node.parent.x] + math.pi
        node=node.parent
        last= False
        while node.parent or last:
            sx = gx
            sy = gy
            syaw=gyaw
            gx = node.x
            gy= node.y
            if not last:
                gyaw = self.anglematrix[node.parent.y*100 + node.parent.x,node.y * 100 + node.x] + math.pi
                #[gyaw, int] = self.roundAngle(gyaw)
            else:
                gyaw = syaw
            #print(gyaw)

            [dubinspath,self.dubinsmatrix] = dubins.calc_dubins_path(sx, sy, syaw, gx, gy, gyaw, maxc, self.dubinsmatrix)
            for i in range((len(dubinspath.x))):
                pathpart=[]
                pathpart.append([dubinspath.x[i],dubinspath.y[i]])
                path.extend(pathpart[::-1])

            node = node.parent
            if last:
                break
            elif not node.parent:
                last = True

            #node = Node((0,0))


        return path

    def InGoalRegion(self, node):
        #if self.Line(node, self.x_goal) < self.step_len:
        if node.cost+self.Line(node, self.x_goal) <= self.budget:
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
        return nodelist[int(np.argmin([self.get_distance_and_angle(nd, n)[0] for nd in nodelist]))]

        # return nodelist[int(np.argmin([(nd.x - n.x) ** 2 + (nd.y - n.y) ** 2
        #                                for nd in nodelist]))]

    #@staticmethod
    def Line(self,x_start, x_goal):
        dist,angle = self.get_distance_and_angle(x_start,x_goal)
        return dist

    def LastPath(self,node):
        #node.totalinfo=node.info+self.FindInfo(node.x,node.y,self.x_start.x,self.x_start.y,node,self.step_len,True)
        if self.kinematic=="dubins":
            [cost,info] = self.dubins(node,self.x_goal)
            node.totalcost=node.cost+cost
            node.totalinfo=node.info+info
        else:
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
        if self.kinematic=="dubins":
            return self.dubins(node_start,node_end,False)
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
        # if self.kinematic=="ranger":
        #   if node_start.parent:
        #         dangle = (self.anglematrix[node_start.parent.y*100 + node_start.parent.x,node_start.y*100 + node_start.x]-angle)**2# squared difference in angle between the line segments
        #       distance+=dangle

        # Limited angle kinematic constraint
        if self.kinematic=="limit":
            anglelimit = 2*math.pi/4 #90 degrees
            if node_start.parent:
                dangle = (self.anglematrix[node_start.parent.y*100 + node_start.parent.x,node_start.y*100 + node_start.x]-angle)**2# squared difference in angle between the line segments
                distance+=dangle
                if (dangle>(anglelimit**2)):
                    distance+=np.inf # if the angle exceeds the limit, inf is added

        # Working with angular velocity (over distance)
        # Note: difficult because it also depends on the next direction so we know what the desired direction should be at the goal position
        # if node_start.parent:
        #     totaldistance = distance+self.anglematrix[node_start.parent.y*100 + node_start.parent.x,node_start.y*100 + node_start.x]
        #     dangle = (self.anglematrix[node_start.parent.y*100 + node_start.parent.x,node_start.y*100 + node_start.x]-angle)**2# squared difference in angle between the line segments
        #     vangle = dangle/totaldistance # angular velocity (Note: dangle still squared)
        #     distance+=vangle*10

        return distance, angle
    def dubinsnomatrix(self, node_start, node_end, costOnly=False): # to get info and cost for dubins kinematic constraints (=smooth trajectory)
        # Note: probably all the syaw and gyaw should be replaced by node.angle

        path = []
        maxc = 2


        sx = node_end.x
        sy = self.x_goal.y
        syaw = self.anglematrix[node_start.y * 100 + node_start.x, node_end.y * 100 + node_end.x]+math.pi
        #[syaw,int] = self.roundAngle(syaw)
        gx = node_start.x
        gy = node_start.y
        gyaw = self.anglematrix[node_start.parent.y*100 + node_start.parent.x,node_start.y * 100 + node_start.x]+math.pi
        #[gyaw,int] = self.roundAngle(gyaw)
        [dubinspath,self.dubinsmatrix] = dubins.calc_dubins_path(sx, sy, syaw, gx, gy, gyaw, maxc, self.dubinsmatrix)
        for i in range((len(dubinspath.x))):
            pathpart = []
            pathpart.append([dubinspath.x[i], dubinspath.y[i]])
            path.extend(pathpart[::-1])

        # gx = node.parent.x
        # gy = node.parent.y
        # gyaw = self.anglematrix[(node.parent).parent.y * 100 + (node.parent).parent.x, node.parent.y * 100 + node.parent.x] + math.pi
        node=node_start.parent
        last= False
        while node.parent or last:
            sx = gx
            sy = gy
            syaw=gyaw
            gx = node.x
            gy= node.y
            if not last:
                gyaw = self.anglematrix[node.parent.y*100 + node.parent.x,node.y * 100 + node.x] + math.pi
                #[gyaw, int] = self.roundAngle(gyaw)
            else:
                gyaw = syaw
            #print(gyaw)

            [dubinspath,self.dubinsmatrix] = dubins.calc_dubins_path(sx, sy, syaw, gx, gy, gyaw, maxc, self.dubinsmatrix)
            for i in range((len(dubinspath.x))):
                pathpart=[]
                pathpart.append([dubinspath.x[i],dubinspath.y[i]])
                path.extend(pathpart[::-1])

            node = node.parent
            if last:
                break
            elif not node.parent:
                last = True

            #node = Node((0,0))
            return cost,info

    def dubins(self, node_start, node_end, costOnly=False): # to get info and cost for dubins kinematic constraints (=smooth trajectory)
        # the matrices probably take too much memory
        cost = self.costmatrix[(node_start.y*100+node_start.x)*8 + node_start.angle,(node_end.y*100+node_end.x)]
        angle = self.anglematrix[(node_start.y*100+node_start.x)* 8 + node_start.angle,(node_end.y*100+node_end.x)]
        info = self.infomatrix[(node_start.y * 100 + node_start.x) * 8 + node_start.angle, (
                       node_end.y * 100 + node_end.x)]
        infopath = self.infopathmatrix[(node_start.y * 100 + node_start.x) * 8 + node_start.angle, (
                       node_end.y * 100 + node_end.x) ]
        if node_end.angle==-1 or (infopath==None and not costOnly):
        #if node_end.angle==-1 or (not costOnly):
            print("starting dubins cost")
            dx = node_end.x - node_start.x
            dy = node_end.y - node_start.y
            [syaw, int] = self.roundAngle(math.atan2(dy, dx) + math.pi)
            node_end.angle=int
            self.anglematrix[(node_start.y * 100 + node_start.x) * 8 + node_start.angle, (
                        node_end.y * 100 + node_end.x) ] = syaw
            # self.anglematrix[(node_start.y * 100 + node_start.x) * 8 + node_start.angle, (
            #     node_end.y * 100 + node_end.x)] = syaw #mirror

            path = []
            maxc = 2

            sx = node_end.x
            sy = node_end.y
            #syaw = self.anglematrix[node_start.y * 100 + node_start.x, node_end.y * 100 + node_end.x] + math.pi
            gx = node_start.x
            gy = node_start.y
            if node_start.parent:
                 gyaw = self.anglematrix[(node_start.parent.y * 100 + node_start.parent.x) * 8 + node_start.parent.angle, (
                         node_start.y * 100 + node_start.x) ]
                #gyaw = node_start.parent.angle
                #gyaw = self.roundAngle(gyaw)
            else:
                gyaw= syaw
            [dubinspath,self.dubinsmatrix] = dubins.calc_dubins_path(sx, sy, syaw, gx, gy, gyaw, maxc, self.dubinsmatrix)

                #path.extend(pathpart[::-1])

            cost = dubinspath.L
            self.costmatrix[(node_start.y * 100 + node_start.x) * 8 + node_start.angle, (
                node_end.y * 100 + node_end.x)] = cost
        if costOnly:
            return cost


        if infopath==None:
        #if not costOnly: # not necessary condition, just put here temporarily
            print("start dubins info")
            infopath = []
            info=0
            for i in range((len(dubinspath.x))):
                if [math.floor(dubinspath.x[i]), math.floor(dubinspath.y[i])] not in infopath:
                    infopath.append([math.floor(dubinspath.x[i]), math.floor(dubinspath.y[i])])
                    info+=self.uncertaintymatrix[math.floor(dubinspath.x[i]), math.floor(dubinspath.y[i])]
                    if np.isnan(self.uncertaintymatrix[math.floor(dubinspath.x[i]), math.floor(dubinspath.y[i])]):  # to prevent going through edges and/or obstacles
                        self.costmatrix[(node_start.y * 100 + node_start.x) * 8 + node_start.angle, (
                    node_end.y * 100 + node_end.x) * 8 + node_end.angle] = np.inf
                        cost = np.inf

            self.infomatrix[(node_start.y * 100 + node_start.x) * 8 + node_start.angle, (
                    node_end.y * 100 + node_end.x) ] = info
            # self.infomatrix[(node_start.y * 100 + node_start.x) * 8 + node_start.angle, (
            #         node_end.y * 100 + node_end.x) ] = info  # mirror the matrix
            self.infopathmatrix[(node_start.y * 100 + node_start.x) * 8 + node_start.angle, (
                    node_end.y * 100 + node_end.x) ] = infopath
            # self.infopathmatrix[(node_start.y * 100 + node_start.x) * 8 + node_start.angle, (
            #         node_end.y * 100 + node_end.x) ] = infopath[::-1]  # mirror the matrix

        currentinfopath=[]
        if node_start.parent:
            node = node_start.parent
            while node.parent:
                currentinfopath.extend(self.infopathmatrix[(node.parent.y * 100 + node.parent.x) * 8 + node.parent.angle, (node.y * 100 + node.x)])
                node = node.parent
        infonode=0
        if not any(element in currentinfopath for element in infopath): # the whole infopath is new
            infonode+=info
        else: #if some infopoints overlap
            for element in infopath:
                if not element in currentinfopath:
                    infonode+=self.uncertaintymatrix[element[1],element[0]]



        return cost, infonode
    @staticmethod
    def roundAngle(angle):
        while angle<0:
            angle+=2*math.pi
        while angle>2*math.pi:
            angle-=2*math.pi
        int = round(angle*10/(2*math.pi))
        rounded = int*2*math.pi/10


        return rounded, int
    def animation(self):
        plt.cla()
        self.plot_grid("Informed rrt*, N = " + str(self.iter_max))
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        for node in self.V:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")
            elif not node.parent and not (node.x==self.x_start.x and node.y==self.x_start.y):
                plt.plot(node.x, node.y, "bs", linewidth=3)

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



def main(uncertaintymatrix):
    x_start = (50, 50)  # Starting node
    #x_goal = (37, 18)  # Goal node
    x_goal = (50,50)

    rrt_star = IRrtStar(x_start, x_goal, 15, 0.0, 15, 2000,uncertaintymatrix)
    [finalpath, finalcost, finalinfo, budget, steplength, searchradius, iteration]=rrt_star.planning()

    return finalpath, finalcost, finalinfo, budget, steplength, searchradius, iteration


if __name__ == '__main__':
    main()
