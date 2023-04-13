import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt

from copy import deepcopy
from shapely.geometry import Point, Polygon

#fake_img = np.random.random((10, 10))
# fake_img=np.ones((10,10))
#
# plt.imshow(fake_img, interpolation='none')
# ax = plt.gca()
# fig = plt.gcf()

#fig, ax = plt.subplots()

fig,ax=plt.subplots()
#ax.plot((np.random.rand(10)),(np.random.rand(10)))
plt.ylim((0,10))
plt.xlim((0,10))

linepoints = np.array([])
def onclick(event):
    #print("enter onclick")
    #print(event.button)
    #if event.button==3 or event.button==MouseButton.LEFT:
    if True:
        #print("enter button")
        global linepoints
        #print(linepoints)

        x=event.xdata
        y=event.ydata
        linepoints=np.append(linepoints,round(x))
        linepoints=np.append(linepoints,round(y))

        if np.size(linepoints)==4:
            #print("draw line")
            print(linepoints)
            print(linepoints[0])
            print(linepoints[1])
            print(linepoints[2])
            print(linepoints[3])
            ax.plot([linepoints[0],linepoints[2]],[linepoints[1],linepoints[3]],'-b')
            ax.plot(linepoints[0],linepoints[1], "-b", marker="o",markersize=5)
            ax.plot(linepoints[2],linepoints[3], "-b", marker="o",markersize=5)

            linepoints=np.array([])
            plt.show()


def interactivePlot():

    cid=fig.canvas.mpl_connect('button_press_event',onclick)
    plt.show()

def plotLines():
    nodes = np.empty(10,dtype=object)
    nodes[0] = (1,1)
    nodes[1] = (3,6)
    nodes[2] = (9,1)

    links = np.empty(10,dtype=object)
    links[0] = (0,1)
    links[1] = (0,2)

    fig, ax = plt.subplots()
    plt.grid()
    for link in links:
        if link!=None:
            ax.plot([nodes[link[0]][0],nodes[link[1]][0]], [nodes[link[0]][1],nodes[link[1]][1]], '-r')
    for index,node in enumerate(nodes):
        if node!=None:
            ax.plot(node[0],node[1], "-b", marker="o",markersize=5)
            plt.annotate(chr(index+65),(node[0],node[1]),textcoords='offset points',xytext=(3,3))
    ax.set_title("Nodes and links")
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    #plotLines()
    interactivePlot()