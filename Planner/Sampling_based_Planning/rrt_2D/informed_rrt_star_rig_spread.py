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
        self.cost = 0 # cost upto this node
        self.info = 0
        self.infopath = [] # all grid points that are monitored
        self.totalcost = 0 # including the distance to the goal point
        self.totalinfo = 0 # including the last part to the goal point
        self.lastinfopath = [] # the monitored grid elements from the node to the goal


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

        self.budget=250
        self.uncertaintymatrix = uncertaintymatrix

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
                cost = {node: node.totalcost for node in self.X_soln}
                info = {node: node.totalinfo for node in self.X_soln}
                #x_best = min(cost, key=cost.get)
                x_best = max(info, key=info.get)
                #c_best = cost[x_best]
                i_last_best = i_best
                i_best = info[x_best]
                if ((i_best-i_last_best)/i_last_best)<0.01: #smaller than 1% improvement
                    count_down-=1
                else:
                    count_down=3 #reset
            if count_down<=0 and k>400:
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
                    # print("double")
                    break
            #if x_nearest.cost + self.Line(x_nearest, x_new) + self.Line(x_new, self.x_goal) < self.budget and not double:  # budget check for nearest parent (to make it more efficient)
            if not double:  # budget check for nearest parent (to make it more efficient)
                # print(x_nearest.cost + self.Line(x_nearest, x_new) + self.Line(x_new, self.x_goal))
                for x_near in self.Near(self.V,x_new):

                    if x_new and not self.utils.is_collision(x_near, x_new):
                        c_min = x_near.cost + self.Line(x_near, x_new)

                        # if c_min+self.Line(x_new, self.x_goal) > self.budget:
                        #     print("past budget (step 2): "+str(c_min+self.Line(x_new, self.x_goal)))
                        if c_min+self.Line(x_new, self.x_goal) <self.budget: #extra check for budget for actual parent
                            node_new = Node((x_new.x,x_new.y))
                            node_new.cost = c_min #+self.Line(x_new, self.x_goal)
                            node_new.parent = x_near #added
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
                                    self.LastPath(node_new)
                                    # new_cost = self.Cost(x_new) + self.Line(x_new, self.x_goal)
                                    # if new_cost < c_best:
                                    #     c_best = new_cost
                                    #     x_best = x_new
                self.Pruning(node_new)

            if k % 20 == 0:
                self.animation(x_center=x_center, c_best=c_best, dist=dist, theta=theta)

        self.path = self.ExtractPath(x_best)
        #for point in reversed(x_best.infopath):
        #    print("infopoint (x,y)=("+str(point[0])+","+str(point[1])+")")
        print("length of infopath: "+str(len(x_best.infopath)+len(x_best.lastinfopath)))
        self.animation(x_center=x_center, c_best=c_best, dist=dist, theta=theta)
        plt.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
        #plt.plot([x for x, _ in x_best.infopath], [y for _, y in x_best.infopath], '-b')
        #plt.plot([x for x, _ in x_best.lastinfopath], [y for _, y in x_best.lastinfopath], '-c')
        #plt.plot([x for x, _ in self.path[:2]],[y for _, y in self.path[:2]], '-k') # to see whether the path actually ends at the goal
        plt.pause(0.01)
        plt.show()

        fig, ax = plt.subplots()
        colormap = cm.Blues
        colormap.set_bad(color='black')
        im= ax.imshow(self.uncertaintymatrix, colormap, vmin=0, vmax=1, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
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
                    #if node1 in self.X_soln:
                    #    self.X_soln.remove(node1)  # also from the solutions
                    self.X_soln.discard(node1)

    def Steer(self, x_start, x_goal):
        dist, theta = self.get_distance_and_angle(x_start, x_goal)
        dist = min(self.step_len, dist)
        node_new = Node((math.floor(x_start.x + dist * math.cos(theta)),
                         math.floor(x_start.y + dist * math.sin(theta))))
        #node_new.parent = x_start
        print("x_rand=("+str(x_goal.x)+","+str(x_goal.y)+") - dist = "+str(dist)+" - x_new=("+str(node_new.x)+","+str(node_new.y)+")")
        return node_new

    def Near(self, nodelist, node):
        max_dist = self.step_len
        dist_table = [self.get_distance_and_angle(nd, node)[0] for nd in nodelist]
        X_near = [nodelist[ind] for ind in range(len(dist_table)) if (dist_table[ind] <= max_dist and dist_table[ind] > 0.0
                                                                    and not self.utils.is_collision(nodelist[ind], node))]
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
                         np.random.random_integers(int(self.y_range[0]),int(self.y_range[1]))))
        return self.x_goal

    def ExtractPath(self, node):
        print("Final cost: "+str(node.totalcost))
        print("Final info value: "+str(node.totalinfo))
        path = [[self.x_goal.x, self.x_goal.y]]

        while node.parent:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x,node.y]) # this should be the start
        #path.append([self.x_start.x, self.x_start.y]) #in principle this shouldn't be necessary

        return path

    def InGoalRegion(self, node):
        #if self.Line(node, self.x_goal) < self.step_len:
        if node.cost+self.Line(node, self.x_goal) < self.budget:
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

    @staticmethod
    def Nearest(nodelist, n):
        return nodelist[int(np.argmin([(nd.x - n.x) ** 2 + (nd.y - n.y) ** 2
                                       for nd in nodelist]))]

    @staticmethod
    def Line(x_start, x_goal):
        return math.hypot(x_goal.x - x_start.x, x_goal.y - x_start.y)

    def LastPath(self,node):
        node.totalcost=node.cost+math.hypot(node.x - self.x_goal.x, node.y - self.x_goal.y)
        dt = 1/(2*(node.totalcost-node.cost))
        t=0
        node.lastinfopath=[]
        info=0

        while t<1.0:
            xline = node.x-self.x_goal.x
            yline = node.y-self.x_goal.y
            xpoint = math.floor(self.x_goal.x+t*xline)
            ypoint = math.floor(self.x_goal.y+t*yline)
            if [xpoint,ypoint] not in node.infopath and [xpoint,ypoint] not in node.lastinfopath: # only info value when the point is not already monitored before
                info+=self.uncertaintymatrix[ypoint,xpoint]
                node.lastinfopath.append([xpoint,ypoint])
            t+=dt

        info+=node.parent.info


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
        dt = 1/(2*self.step_len)
        t=0
        #points=node.parent.infopath
        points=[]
        info=0
        node.infopath=[]
        for point in node.parent.infopath:
            node.infopath.append(point)
        #node.infopath=node.parent.infopath
        #print("node coordinates: ("+str(node.x)+","+str(node.y)+")")
        #print("parent node coordinates: ("+str(node.parent.x)+","+str(node.parent.y)+")")

        while t<1.0:
            xline = node.x-node.parent.x
            yline = node.y-node.parent.y
            xpoint = math.floor(node.parent.x+t*xline)
            ypoint = math.floor(node.parent.y+t*yline)
            if [xpoint,ypoint] not in node.infopath: # only info value when the point is not already monitored before
                info+=self.uncertaintymatrix[ypoint,xpoint]
                #points.append([xpoint,ypoint])
                node.infopath.append([xpoint,ypoint])
                #print("new info point coordinates: (" + str(xpoint) + "," + str(ypoint) + ")")
            t+=dt
        #node.infopath=points
        # while node.parent:
        #     info += node.parent.info
        #     node = node.parent
        info+=node.parent.info
        return info

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def animation(self, x_center=None, c_best=None, dist=None, theta=None):
        plt.cla()
        self.plot_grid("Informed rrt*, N = " + str(self.iter_max))
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        for node in self.V:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")

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

    rrt_star = IRrtStar(x_start, x_goal, 10, 0.0, 15, 2000,uncertaintymatrix)
    rrt_star.planning()

if __name__ == '__main__':
    main()
