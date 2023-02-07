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

#from Sampling_based_Planning.rrt_2D import env, plotting, utils
from PathPlanning.Sampling_based_Planning.rrt_2D import env, plotting, utils

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.cost = 0
        self.info = 0


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
                cost = {node: node.cost for node in self.X_soln}
                info = {node: node.info for node in self.X_soln}
                #x_best = min(cost, key=cost.get)
                x_best = max(info, key=info.get)
                #c_best = cost[x_best]
                i_last_best = i_best
                i_best = info[x_best]
                if ((i_best-i_last_best)/i_last_best)<0.01: #smaller than 1% improvement
                    count_down-=1
                else:
                    count_down=3 #reset
            if count_down==0 and k>1000:
                print("Reached stopping criterion at iteration "+str(k))
                break # we stop iterating if the best score is not improving much anymore and we already passed at least 1000 cycles

            if k%100==0:
                print("AGAIN ONE HUNDRED CYCLES FURTHER, CURRENT CYCLE ="+str(k))

            x_rand = self.Sample(c_best, dist, x_center, C)
            x_nearest = self.Nearest(self.V, x_rand)
            x_new = self.Steer(x_nearest, x_rand) #so that we only generate one new node, not multiple
            #if self.Cost(x_nearest) + self.Line(x_nearest, x_rand) + self.Line(x_rand, self.x_goal) > self.budget:
                #just for debugging purposes (for now)
                #print("Past the budget")
            double=False
            for node in self.V:
                if node.x == x_new.x and node.y == x_new.y:  # co-located nodes
                    double=True #there is already a node at this located, so we skip it
                    # print("double")
                    break
            #if x_nearest.cost + self.Line(x_nearest, x_new) + self.Line(x_new, self.x_goal) < self.budget and not double:  # budget check for nearest parent (to make it more efficient)
            if not double:  # budget check for nearest parent (to make it more efficient)
                # print(x_nearest.cost + self.Line(x_nearest, x_new) + self.Line(x_new, self.x_goal))
                for x_near in self.Near(self.V,x_new):

                    if x_new and not self.utils.is_collision(x_near, x_new):
                        #X_near = self.Near(self.V, x_new)
                        c_min = x_near.cost + self.Line(x_near, x_new)
                        #print(c_min)
                        #budget = 70
                        # if c_min+self.Line(x_new, self.x_goal) > self.budget:
                        #     print("past budget (step 2): "+str(c_min+self.Line(x_new, self.x_goal)))
                        if c_min+self.Line(x_new, self.x_goal) <self.budget: #extra check for budget for actual parent
                            #if True:
                            node_new = Node((x_new.x,x_new.y))
                            node_new.cost = c_min #+self.Line(x_new, self.x_goal)
                            node_new.parent = x_near #added
                            node_new.info = self.Info(node_new)
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
                self.Pruning(node_new)

            if k % 20 == 0:
                self.animation(x_center=x_center, c_best=c_best, dist=dist, theta=theta)

        self.path = self.ExtractPath(x_best)
        self.animation(x_center=x_center, c_best=c_best, dist=dist, theta=theta)
        plt.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
        plt.pause(0.01)
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
                    # prune the node from all nodes and from our lists
                    # self.V.remove(node1)
                    # nodelist.remove(node1)
                    # costlist.remove(node1.cost)
                    # infolist.remove(node1.cost)
                    # if node1 in self.X_soln:
                    #     self.X_soln.remove(node1) #also from the solutions
                    if node1 in self.V: #still have to figure out why this is needed (to do)
                        self.V.remove(node1)
                    #nodelist.pop(index)
                    #costlist.pop(index)
                    #infolist.pop(index)
                    #costlist[index]=np.nan
                    #infolist[index]=np.nan
                    # to do still: how to "pop" or "remove" from the list we iterate over? to speed up pruningz
                    if node1 in self.X_soln:
                        self.X_soln.remove(node1)  # also from the solutions

    def Steer(self, x_start, x_goal):
        dist, theta = self.get_distance_and_angle(x_start, x_goal)
        dist = min(self.step_len, dist)
        node_new = Node((int(x_start.x + dist * math.cos(theta)),
                         int(x_start.y + dist * math.sin(theta))))
        #node_new.parent = x_start
        print("x_rand=("+str(x_goal.x)+","+str(x_goal.y)+") - dist = "+str(dist)+" - x_new=("+str(node_new.x)+","+str(node_new.y)+")")
        return node_new

    def Near(self, nodelist, node):
        n = len(nodelist) + 1
        r = 50 * math.sqrt((math.log(n) / n))
        max_dist = self.step_len
        #dist_table = [(nd.x - node.x) ** 2 + (nd.y - node.y) ** 2 for nd in nodelist]
        dist_table = [self.get_distance_and_angle(nd, node)[0] for nd in nodelist]
        # X_near = [nodelist[ind] for ind in range(len(dist_table)) if dist_table[ind] <= r ** 2 and dist_table[ind] > 0.0 and
        #           not self.utils.is_collision(nodelist[ind], node)]
        # X_near = [nodelist[ind] for ind in range(len(dist_table)) if dist_table[ind] <= max_dist and dist_table[ind] > 0.0 and
        #           not self.utils.is_collision(nodelist[ind], node)]
        X_near = [nodelist[ind] for ind in range(len(dist_table)) if (dist_table[ind] <= max_dist and dist_table[ind] > 0.0
                                                                    and not self.utils.is_collision(nodelist[ind], node))]
        #print("len xnear:"+str(len(X_near)))
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

        if np.random.random() > self.goal_sample_rate:
            #return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
            #             np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))
            return Node((np.random.random_integers(int(self.x_range[0]), int(self.x_range[1])),
                         np.random.random_integers(int(self.y_range[0]),int(self.y_range[1]))))
        return self.x_goal

    def ExtractPath(self, node):
        print("Final cost: "+str(self.Cost(node)))
        print("Final info value: "+str(self.Info(node)))
        path = [[self.x_goal.x, self.x_goal.y]]

        while node.parent:
            path.append([node.x, node.y])
            node = node.parent

        path.append([self.x_start.x, self.x_start.y])

        return path

    def InGoalRegion(self, node):
        #if self.Line(node, self.x_goal) < self.step_len:
        if self.Cost(node)+self.Line(node, self.x_goal) < self.budget:
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
        if node.parent is None:
            return 0.0
        #info = -self.Line(node, self.x_goal)
        info = self.uncertaintymatrix[int(node.y),int(node.x)]
        # if node.x > 20 and node.y>10:
        #     info+=10
        # else:
        #     info +=1
        while node.parent:
            #info += math.hypot(node.x - node.parent.x, node.y - node.parent.y)
            info += node.parent.info
            node = node.parent
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

        for (ox, oy, w, h) in self.obs_boundary:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

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

        #added:
        for row in range(100):
            for col in range(100):
                if np.isnan(self.uncertaintymatrix[row,col]):
                    self.ax.add_patch(patches.Rectangle((col,row),1,1,
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


# def main():
#     x_start = (18, 8)  # Starting node
#     #x_goal = (37, 18)  # Goal node
#     x_goal = (19,8)
#
#     rrt_star = IRrtStar(x_start, x_goal, 2, 0.1, 6, 1000)
#     rrt_star.planning()

def main(uncertaintymatrix):
    x_start = (18, 8)  # Starting node
    #x_goal = (37, 18)  # Goal node
    x_goal = (19,8)

    rrt_star = IRrtStar(x_start, x_goal, 10, 0.1, 15, 2000,uncertaintymatrix)
    rrt_star.planning()

if __name__ == '__main__':
    main()
