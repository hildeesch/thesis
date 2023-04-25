import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import math

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
plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
plt.yticks([0,1,2,3,4,5,6,7,8,9,10])
plt.xlim((0,10))

linepoints = np.array([])
color="-b"
draw="line" #whether we want to draw an angle or a line



def onkey(event):
    global color
    global draw
    print("You pressed: "+str(event.key))
    if event.key=="r":
        color=str(color[0]+"r")
    elif event.key=="k":
        color=str(color[0]+"k")
    elif event.key=="b":
        color=str(color[0]+"b")
    elif event.key=="v":
        color=str(color[0]+"g")
    elif event.key=="d":
        color=str("--"+color[1])
    elif event.key=="z":
        color="-b"
        draw="line"
    elif event.key=="a":
        draw="angle"
        print(draw)
    elif event.key=="i":
        draw="line"
        print("line")
    elif event.key=="j":
        draw="line_nonode"
    elif event.key=="g": # happens automatically
        print("Changing grid style")
    elif event.key=="q": # happens automatically
        print("Quit plot")
    else:
        print("No recognized option. Options: 'b' for blue, 'k' for black, 'v' for green, 'r' for red, 'd' for dotted, 'z' for reset, \n'a' for angle, 'i' for line, 'j' for line without nodes, or 'g' to change gridstyle")



def drawAngle(center, point1, point2):
    global linepoints
    #print(center)
    #print(point1)
    #print(point2)
    #width = np.sqrt((point1[0] - center[0]) ** 2 + (point1[1] - center[1]) ** 2)
    width=2
    #print(width)
    theta1 = math.degrees(math.atan2(point1[1] - center[1], point1[0] - center[0]))
    theta2 = math.degrees(math.atan2(point2[1] - center[1], point2[0] - center[0]))
    #angle1= min(theta1,theta2)
    #angle2= max(theta1,theta2)
    angle1=theta1
    angle2=theta2
    angle = abs(theta2 - theta1)
    ax.add_patch(mpl.patches.Arc(center, width, width, 0, angle1, angle2))
    linepoints = np.array([])

    plt.show()


def onclick_angle(event):
    global color
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

        if np.size(linepoints)==6:
            #print("draw line")
            drawAngle([linepoints[0],linepoints[1]],[linepoints[2],linepoints[3]],[linepoints[4],linepoints[5]])
            #linepoints=np.array([])

def onclick(event):
    global draw
    if draw=="angle":
        onclick_angle(event)
    #global color
    #print("enter onclick")
    #print(event.button)
    #if event.button==3 or event.button==MouseButton.LEFT:
    else:
    #if True:
        #print("enter button")
        global linepoints
        #print(linepoints)

        #print(linepoints)

        x=event.xdata
        y=event.ydata
        linepoints=np.append(linepoints,round(x))
        linepoints=np.append(linepoints,round(y))

        if np.size(linepoints)==4:
            #print("draw line")
            #print(linepoints)
            #print(linepoints[0])
            #print(linepoints[1])
            #print(linepoints[2])
            #print(linepoints[3])
            ax.plot([linepoints[0],linepoints[2]],[linepoints[1],linepoints[3]],color)
            if draw=="line":
                ax.plot(linepoints[0],linepoints[1], "-k", marker="o",markersize=5)
                ax.plot(linepoints[2],linepoints[3], "-k", marker="o",markersize=5)

            linepoints=np.array([])
            plt.show()


def interactivePlot():

    cid=fig.canvas.mpl_connect('button_press_event',onclick)

    cid=fig.canvas.mpl_connect('key_press_event',onkey)
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