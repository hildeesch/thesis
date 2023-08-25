import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import math

from Planner.Sampling_based_Planning.rrt_2D import dubins_path as dubins


from copy import deepcopy
from shapely.geometry import Point, Polygon



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
        print("Color red")
    elif event.key=="n":
        color=str(color[0]+"k")
        print("Color black")
    elif event.key=="b":
        color=str(color[0]+"b")
        print("Color blue")
    elif event.key=="v":
        color=str(color[0]+"g")
        print("Color green")
    elif event.key=="w":
        color=str(color[0]+"w")
        print("Color white (eraser)")
    elif event.key=="d":
        print("Dotted line")
        color=str("--"+color[-1])
    elif event.key=="z":
        print("Reset")
        color="-b"
        draw="line"
    elif event.key=="a":
        draw="angle"
        print("Angle")
    elif event.key=="c":
        draw="curve"
        print("Curve")
    elif event.key=="i":
        draw="line"
        print("Line")
    elif event.key=="j":
        draw="line_nonode"
        print("Line without nodes")
    elif event.key=="g": # happens automatically
        print("Changing grid style")
    elif event.key=="q": # happens automatically
        print("Quit plot")
    else:
        print("No recognized option. Options: 'b' for blue, 'n' for black, 'w' for white, 'v' for green, 'r' for red, 'd' for dotted, 'z' for reset, \n'a' for angle, 'c' for curve, 'i' for line, 'j' for line without nodes, or 'g' to change gridstyle")



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

def onclick_curve(event):
    global linepoints
    # print(linepoints)

    # print(linepoints)

    x = event.xdata
    y = event.ydata
    linepoints = np.append(linepoints, round(x))
    linepoints = np.append(linepoints, round(y))

    if np.size(linepoints) == 4:
        sx = linepoints[0]
        sy = linepoints[1]
        gx = linepoints[2]
        gy = linepoints[3]
        maxc=1
        syaw=np.random.choice([0,math.pi/2,math.pi,3*math.pi/2])
        gyaw=np.random.choice([0,math.pi/2,math.pi,3*math.pi/2])

        dubinsmat=[]

        [dubinspath, dubinsmat, infopathrel] = dubins.calc_dubins_path(sx, sy, syaw, gx, gy, gyaw, maxc,
                                                                            dubinsmat)
        dubins_rel_x = dubinspath.x
        dubins_rel_y = dubinspath.y
        dubins_x = [x + sx for x in dubins_rel_x]
        dubins_y = [y + sy for y in dubins_rel_y]
        if color[-1] == "w":
            linewidth = 3
        else:
            linewidth = 1.5

        for index in range(1,len(dubins_x)):
            #print(len(color))
            ax.plot([dubins_x[index-1], dubins_x[index]], [dubins_y[index-1], dubins_y[index]], color, linewidth=linewidth)

        ax.plot(linepoints[0], linepoints[1], "-k", marker="o", markersize=5)
        ax.plot(linepoints[2], linepoints[3], "-k", marker="o", markersize=5)

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
    elif draw=="curve":
        onclick_curve(event)
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
            if color[-1]=="w":
                linewidth=3
            else:
                linewidth=1.5
            if len(color)==2 and color[-1]!="w" and draw!="line_nonode":
                print("arrow")
                ax.arrow(linepoints[0],linepoints[1],(linepoints[2]-linepoints[0]),(linepoints[3]-linepoints[1]),color=color[-1],linewidth=linewidth,head_width=0.22,length_includes_head=True)
            else:
                print(len(color))
                ax.plot([linepoints[0],linepoints[2]],[linepoints[1],linepoints[3]],color,linewidth=linewidth)
            if draw=="line":
                ax.plot(linepoints[0],linepoints[1], "-k", marker="o",markersize=5)
                ax.plot(linepoints[2],linepoints[3], "-k", marker="o",markersize=5)

            linepoints=np.array([])
            plt.show()


def interactivePlot():
    print("Options: 'b' for blue, 'n' for black, 'w' for white, 'v' for green, 'r' for red, 'd' for dotted, 'z' for reset, \n'a' for angle, 'c' for curve, 'i' for line, 'j' for line without nodes, or 'g' to change gridstyle")

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