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
#from Sampling_based_Planning.rrt_2D import env, plotting, utils
from Planner.Sampling_based_Planning.rrt_2D import env, plotting, utils

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.cost = 0
        self.info = 0
        self.infopath = []
        self.direction = []


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

        self.budget=500
        self.uncertaintymatrix = uncertaintymatrix
        self.rowdist=4

    def init(self):
        cMin, theta = self.get_distance_and_angle(self.x_start, self.x_goal)
        C = self.RotationToWorldFrame(self.x_start, self.x_goal, cMin)
        xCenter = np.array([[(self.x_start.x + self.x_goal.x) / 2.0],
                            [(self.x_start.y + self.x_goal.y) / 2.0], [0.0]])
        x_best = self.x_start

        return theta, cMin, xCenter, C, x_best

    def planning(self):
        theta, dist, x_center, C, x_best = self.init()
        c_best = np.inf
        count_down=3
        i_best = 0
        for k in range(self.iter_max):
            #time.sleep(0.1)
            if self.X_soln:
                cost = {node: node.cost for node in self.X_soln}
                info = {node: node.info for node in self.X_soln}
                #x_best = min(cost, key=cost.get)
                x_best = max(info, key=info.get)
                #c_best = cost[x_best]
                i_last_best = i_best
                i_best = info[x_best]
                if i_last_best>0: # to prevent division by zero
                    if ((i_best-i_last_best)/i_last_best)<0.01: #smaller than 1% improvement
                        count_down-=1
                    else:
                        count_down=3 #reset
            if count_down<=0 and k>500:
                print("Reached stopping criterion at iteration "+str(k))
                break # we stop iterating if the best score is not improving much anymore and we already passed at least ... cycles

            if k%50==0:
                print("ATTENTION!!! ATTENTION!!! ATTENTION!!! AGAIN FIFTY CYCLES FURTHER, CURRENT CYCLE ="+str(k)) # to know how far we are

            x_rand = self.Sample(c_best, dist, x_center, C)
            x_nearest = self.Nearest(self.V, x_rand)
            x_new = self.Steer(x_nearest, x_rand) #so that we only generate one new node, not multiple
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
                        #c_min = x_near.cost + self.Line(x_near, x_new) #TODO makes no sense with rows
                        c_min = x_near.cost+self.get_distance_rows(x_near,x_new)
                        # if c_min+self.Line(x_new, self.x_goal) > self.budget:
                        #     print("past budget (step 2): "+str(c_min+self.Line(x_new, self.x_goal)))
                        if c_min+self.get_distance_rows(x_new,self.x_goal) <=self.budget: #extra check for budget for actual parent
                            node_new = Node((x_new.x,x_new.y))
                            node_new.parent = x_near #added
                            node_new.cost = self.Cost(node_new) #+self.Line(x_new, self.x_goal)
                            #node_new.info = self.Info(node_new)
                            node_new.info = self.Info_cont(node_new)
                            self.V.append(node_new) #generate a "node"/trajectory to each near point
                            # choose parent
                            # for x_near in X_near:
                            #     c_new = self.Cost(x_near) + self.Line(x_near, x_new)
                            #     if c_new < c_min:
                            #         x_new.parent = x_near
                            #         c_min = c_new

                            #print("append new node x: "+str(x_new.x)+" parent x:"+str(x_new.parent.x))

                            #rewire
                            # for x_near in X_near:
                            #     c_near = self.Cost(x_near)
                            #     c_new = self.Cost(x_new) + self.Line(x_new, x_near)
                            #     if c_new < c_near:
                            #         x_near.parent = x_new

                            if self.InGoalRegion(node_new):
                                if not self.utils.is_collision(node_new, self.x_goal):
                                    self.X_soln.add(node_new)
                                    # new_cost = self.Cost(x_new) + self.Line(x_new, self.x_goal)
                                    # if new_cost < c_best:
                                    #     c_best = new_cost
                                    #     x_best = x_new
                self.Pruning(x_new)

            if k % 20 == 0:
                self.animation(x_center=x_center, c_best=c_best, dist=dist, theta=theta)

        self.path = self.ExtractPath(x_best)
        self.animation(x_center=x_center, c_best=c_best, dist=dist, theta=theta)
        #plt.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
        plt.plot([x for x, _ in x_best.infopath], [y for _, y in x_best.infopath], '-r')
        #plt.plot([x for x, _ in self.path[:2]],[y for _, y in self.path[:2]], '-b') # to see whether the path actually ends at the goal
        plt.pause(0.01)
        plt.show()

        fig, ax = plt.subplots()
        colormap = cm.Blues
        colormap.set_bad(color='black')
        im = ax.imshow(self.uncertaintymatrix, colormap, vmin=0, vmax=1, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        #ax.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
        ax.plot([x for x, _ in x_best.infopath], [y for _, y in x_best.infopath], '-r')
        ax.set_title("Spatial distribution of uncertainty and final path")
        fig.tight_layout()
        plt.show()

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
                if (node2.cost<=node1.cost and node2.info>node1.info) or (node1.cost==node2.cost and node1.info==node2.info and index!=index2): #prune lesser paths or doubles
                    # if (node1.cost==node2.cost and node1.info==node2.info and index!=index2):
                    #     print("Double detected, now pruned")
                    # else:
                    #     print("Alternative pruned")
                    # prune the node from all nodes

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
        dist, theta = self.get_distance_and_angle(x_start, x_goal)
        dist = self.get_distance_rows(x_start,x_goal)
        closer=False
        if dist>self.step_len:
            #just for now for debugging purposes
            closer=True
        else:
            print("x_rand close enough, x_rand = x_new")
            #return x_goal
            return Node((int(x_goal.x), int(x_goal.y)))
        dist = min(self.step_len, dist)
        withinreach = False
        while not withinreach:
            node_new = Node((math.floor(x_start.x + dist * math.cos(theta)),
                         int((x_start.y + dist * math.sin(theta)//self.rowdist)*self.rowdist)))
            if self.get_distance_rows(x_start,node_new)>self.step_len:
                dist-=1
            else:
                withinreach=True

            if dist<=0: #to prevent going into negative numbers, we will just pick a node in the row of the nearest node
                print("dist: "+str(dist))
                # see if we can go up
                yvalue= min(99,x_start.y+self.rowdist) # make sure we don't go above 99
                if x_start.x<50:
                    xvalue=0
                else:
                    xvalue=99
                #xvalue= np.random.choice([0,99])
                node_new = Node((int(xvalue),int(yvalue)))
                if self.get_distance_rows(x_start,node_new)>self.step_len: # if going up is too far
                    print("can't go up")
                    yvalue = x_start.y
                    xvalue = np.random.choice([max(0,x_start.x-self.step_len),min(99,x_start.x+self.step_len)])
                    node_new = Node((int(xvalue), int(yvalue)))

                withinreach=True
        #node_new.parent = x_start
        print("WE NEED TO SAMPLE CLOSER TO THE NEAREST: ("+str(x_start.x)+","+str(x_start.y)+")")
        print("x_rand=("+str(x_goal.x)+","+str(x_goal.y)+") - dist = "+str(dist)+" - x_new=("+str(node_new.x)+","+str(node_new.y)+")")
        return node_new

    def Near(self, nodelist, node):
        max_dist = self.step_len
        dist_table = [self.get_distance_rows(node,nd) for nd in nodelist]
        X_near = [nodelist[ind] for ind in range(len(dist_table)) if (dist_table[ind] <= max_dist and dist_table[ind] > 0.0
                                                      and not self.utils.is_collision(nodelist[ind], node)==True)]
        print("number of near nodes: "+str(len(X_near)))
        return X_near

    def Sample(self, c_max, c_min, x_center, C):
        c_max=np.inf
        if c_max < np.inf:
            print("not random sampling")
            r = [c_max / 2.0,
                 math.sqrt(c_max ** 2 - c_min ** 2) / 2.0,
                 math.sqrt(c_max ** 2 - c_min ** 2) / 2.0]
            L = np.diag(r)

            while True:
                x_ball = self.SampleUnitBall()
                x_rand = np.dot(np.dot(C, L), x_ball) + x_center
                if self.x_range[0] + self.delta <= x_rand[0] <= self.x_range[1] - self.delta and \
                        self.y_range[0] + self.delta <= x_rand[1] <= self.y_range[1] - self.delta:
                    break
            x_rand = Node((x_rand[(0, 0)], x_rand[(1, 0)]))
        else:
            x_rand = self.SampleFreeSpace()

        return x_rand

    @staticmethod
    def SampleUnitBall():
        while True:
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if x ** 2 + y ** 2 < 1:
                return np.array([[x], [y], [0.0]])

    def SampleFreeSpace(self):
        delta = self.delta

        if np.random.random() > self.goal_sample_rate: # TODO maybe take this out
            #return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
            #             np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))
            return Node((np.random.random_integers(int(self.x_range[0]), int(self.x_range[1])),
                         (min(99,((np.random.random_integers(int(self.y_range[0]),int(self.y_range[1])))//self.rowdist)*self.rowdist))))
        return self.x_goal

    def ExtractPath(self, node):
        print("Final cost: "+str(node.cost))
        print("Final info value: "+str(node.info))
        path = [[self.x_goal.x, self.x_goal.y]]

        while node.parent:
            path.append([node.x, node.y])
            node = node.parent

        path.append([self.x_start.x, self.x_start.y])

        return path

    def InGoalRegion(self, node):
        #if self.Line(node, self.x_goal) < self.step_len:
        if node.cost+self.get_distance_rows(node,self.x_goal) < self.budget:
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
        return nodelist[int(np.argmin([self.get_distance_rows(nd,n) for nd in nodelist]))]

    @staticmethod
    def Line(x_start, x_goal):
        return math.hypot(x_goal.x - x_start.x, x_goal.y - x_start.y)

    def Cost(self, node):
        if node == self.x_start:
            return 0.0

        if node.parent is None:
            return np.inf

        cost = self.get_distance_rows(node,node.parent)

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
        dt = 1/(node.cost-node.parent.cost)
        t=0
        points=[] # TODO clean up such that this is not a necessary variable anymore
        points_new=[] #TODO fix this so that the order is right immediately
        info=0
        node.infopath=[]
        for point in node.parent.infopath:
            node.infopath.append(point)
            points.append(point)

        done=False
        while done==False:
            if node.direction==["left","none"]:
                for xpoint in range(node.parent.x,node.x-1,-1):
                    ypoint = node.y
                    if [xpoint, ypoint] not in points:  # only info value when the point is not already monitored before
                        info += self.uncertaintymatrix[ypoint, xpoint]
                    points_new.append([xpoint, ypoint])
                done=True
            if node.direction==["right","none"]:
                for xpoint in range(node.parent.x,node.x+1):
                    ypoint = node.y
                    if [xpoint, ypoint] not in points:  # only info value when the point is not already monitored before
                        info += self.uncertaintymatrix[ypoint, xpoint]
                    points_new.append([xpoint, ypoint])
                done=True
            if node.direction==["left","up"]:
                for xpoint in range(node.parent.x,-1,-1):
                    ypoint=node.parent.y
                    if [xpoint, ypoint] not in points:  # only info value when the point is not already monitored before
                        info += self.uncertaintymatrix[ypoint, xpoint]
                    points_new.append([xpoint, ypoint])
                for ypoint in range(node.parent.y, node.y + 1):
                    xpoint = 0
                    if [xpoint, ypoint] not in points:  # only info value when the point is not already monitored before
                        info += self.uncertaintymatrix[ypoint, xpoint]
                    points_new.append([xpoint, ypoint])
                for xpoint in range(0,node.x+1):
                    ypoint=node.y
                    if [xpoint, ypoint] not in points:  # only info value when the point is not already monitored before
                        info += self.uncertaintymatrix[ypoint, xpoint]
                    points_new.append([xpoint, ypoint])
                done=True
            if node.direction==["left","down"]:
                for xpoint in range(node.parent.x,-1,-1):
                    ypoint=node.parent.y
                    if [xpoint, ypoint] not in points:  # only info value when the point is not already monitored before
                        info += self.uncertaintymatrix[ypoint, xpoint]
                    points_new.append([xpoint, ypoint])
                for ypoint in range(node.parent.y,node.y-1,-1):
                    xpoint =0
                    if [xpoint, ypoint] not in points:  # only info value when the point is not already monitored before
                        info += self.uncertaintymatrix[ypoint, xpoint]
                    points_new.append([xpoint, ypoint])
                for xpoint in range(0,node.x+1):
                    ypoint=node.y
                    if [xpoint, ypoint] not in points:  # only info value when the point is not already monitored before
                        info += self.uncertaintymatrix[ypoint, xpoint]
                    points_new.append([xpoint, ypoint])
                done=True
            if node.direction==["right","up"]:
                for xpoint in range(node.parent.x,100):
                    ypoint=node.parent.y
                    if [xpoint, ypoint] not in points:  # only info value when the point is not already monitored before
                        info += self.uncertaintymatrix[ypoint, xpoint]
                    points_new.append([xpoint, ypoint])
                for ypoint in range(node.parent.y,node.y+1):
                    xpoint =99
                    if [xpoint, ypoint] not in points:  # only info value when the point is not already monitored before
                        info += self.uncertaintymatrix[ypoint, xpoint]
                    points_new.append([xpoint, ypoint])
                for xpoint in range(99,node.x-1,-1):
                    ypoint=node.y
                    if [xpoint, ypoint] not in points:  # only info value when the point is not already monitored before
                        info += self.uncertaintymatrix[ypoint, xpoint]
                    points_new.append([xpoint, ypoint])
                done=True
            if node.direction==["right","down"]:
                for xpoint in range(node.parent.x,100):
                    ypoint=node.parent.y
                    if [xpoint, ypoint] not in points:  # only info value when the point is not already monitored before
                        info += self.uncertaintymatrix[ypoint, xpoint]
                    points_new.append([xpoint, ypoint])
                for ypoint in range(node.parent.y,node.y-1,-1):
                    xpoint =99
                    if [xpoint, ypoint] not in points:  # only info value when the point is not already monitored before
                        info += self.uncertaintymatrix[ypoint, xpoint]
                    points_new.append([xpoint, ypoint])
                for xpoint in range(99,node.x-1,-1):
                    ypoint=node.y
                    if [xpoint, ypoint] not in points:  # only info value when the point is not already monitored before
                        info += self.uncertaintymatrix[ypoint, xpoint]
                    points_new.append([xpoint, ypoint])
                done=True
            # if node.direction[1]=="up" and done==False:
            #     for ypoint in range(node.y,node.parent.y-1,-1):
            #         #xpoint is already given in the if-statements above
            #         if [xpoint, ypoint] not in points:  # only info value when the point is not already monitored before
            #             info += self.uncertaintymatrix[ypoint, xpoint]
            #         points.append([xpoint, ypoint])
            #     done=True
            # if node.direction[1]=="down" and done==False:
            #     for ypoint in range(node.y,node.parent.y+1):
            #         #xpoint is already given in the if-statements above
            #         if [xpoint, ypoint] not in points:  # only info value when the point is not already monitored before
            #             info += self.uncertaintymatrix[ypoint, xpoint]
            #         points.append([xpoint, ypoint])
            #     done=True
            if not done:
                print("something went wrong in the info path")

        for point in points_new:
            node.infopath.append(point)
        #node.infopath=points
        info+=node.parent.info

        return info

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)
    @staticmethod
    def get_distance_rows(node_start,node_end):
        #node_start = node
        #node_end=parent
        if node_start.y == node_end.y: #in the same row
            dxright=abs(node_start.x-node_end.x)
            if node_start.x<node_end.x:
                direction="left"
                dx=dxright
            else:
                direction="right"
                dx=dxright #makes no difference for left or right
            dy=0
            if node_start.parent:
                node_start.direction=[direction,"none"]
        else:  # not in same row
            dxleft=node_start.x+node_end.x
            dxright=99-node_start.x+99-node_end.x
            if dxright<dxleft:
                direction="right"
                #print("direction RIGHT")
                dx=dxright
            else:
                direction="left"
                #print("direction LEFT")
                dx=dxleft
            dy=abs(node_end.y-node_start.y)
            if node_start.y>node_end.y:
                updown="up"
            else:
                updown="down"
            if node_start.parent:
                node_start.direction=[direction,updown]
        return dx+dy

    def animation(self, x_center=None, c_best=None, dist=None, theta=None):
        plt.cla()
        self.plot_grid("Informed rrt*, N = " + str(self.iter_max))
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        for node in self.V:
            if node.parent:
                #reachedparent=False
                prevpoint=[node.x,node.y]
                for point in node.infopath[::-1]:
                    reachedparent= (point[0]==node.parent.x and point[1]==node.parent.y)

                    plt.plot([point[0], prevpoint[0]], [point[1], prevpoint[1]], "-g")
                    prevpoint=point
                    if reachedparent:
                        break

        if c_best != np.inf:
            self.draw_ellipse(x_center, c_best, dist, theta)

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
    x_start = (18, 8)  # Starting node
    #x_goal = (37, 18)  # Goal node
    x_goal = (19,8)

    rrt_star = IRrtStar(x_start, x_goal, 20, 0.0, 15, 2000,uncertaintymatrix)
    rrt_star.planning()

if __name__ == '__main__':
    main()
