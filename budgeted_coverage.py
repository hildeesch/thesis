import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import math
import itertools

def visualize_back_and_forth(budget,x0=0,y0=0,incl_travel=False,startpos=[],show=False):
    """
    Visualizes a back-and-forth coverage planner over a discretized grid.
    Inputs:
    - budget: max path length
    - x0,y0: start of coverage (left corner)
    - incl_travel: including the path from and to the start pos in the budget
    Parameters:
    - w: Width of the grid (number of columns).
    - l: Length of the grid (number of rows).
    """
    # Width and Length calculation with given budget
    l = math.floor(math.sqrt(budget+1))
    w=l
    # Path length calculation
    path_length = l * (w-1) + (l-1)  # Simple total length, row-by-row
    
    if show:
        # Initialize figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_aspect('equal')
        ax.set_facecolor("white")  # White background
        ax.set_xlim(-0.5 + x0, w + 0.5 + x0)  # Add padding around the grid
        ax.set_ylim(-0.5 + y0, l + 0.5 + y0)  # Add padding around the grid
        ax.grid(visible=True, color="grey", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_xticks(range(0 + x0, w+1 + x0))
        ax.set_yticks(range(0 + y0, l+1 + y0))
        ax.tick_params(colors="black")  # Ensure ticks are visible on white
    within_budget=False
    while within_budget==False:
        # Draw the coverage path
        x, y = [], []
        direction = 1  # 1 for left-to-right, -1 for right-to-left
        for row in range(l):
            if direction == 1:
                x.extend([i + 0.5 + x0 for i in range(w)])  # Centered Left to right
            else:
                x.extend([i + 0.5 + x0 for i in range(w - 1, -1, -1)])  # Centered Right to left
            y.extend([row + 0.5 + y0] * w)  # Centered row coordinate
            direction *= -1  # Switch direction
        if incl_travel:
            travel_start = math.sqrt((x[0]-(startpos[0]+0.5))**2+(y[0]-(startpos[1]+0.5))**2)
            travel_end = math.sqrt((x[-1]-(startpos[0]+0.5))**2+(y[-1]-(startpos[1]+0.5))**2)
            # if including the travel surpasses the budget, shorten the coverage
            if (path_length + travel_start + travel_end)>budget:
                w=w-1
                l=l-1
                path_length = l * (w-1) + (l-1)
                if w==0: # coverage at this position is not within budget
                    return False, False
            else:
                path_length += (travel_start + travel_end)
                within_budget = True
                x = [startpos[0]+0.5]+x+[startpos[0]+0.5] # include start and end in viz path
                y = [startpos[1]+0.5]+y+[startpos[1]+0.5]
        else: 
            within_budget = True
    if show:
        # Plot the field of view (glow effect)
        ax.fill_betweenx(
            [min(y)-0.5,max(y)+0.5], min(x)-0.5, max(x)+0.5, color="lightblue", alpha=0.3, label="Field of View"
        )
        # ax.fill_betweenx(
        #     [min(y),max(y)+1], min(x), max(x)+1, color="lightblue", alpha=0.3, label="Field of View"
        # )
        # Plot the path
        #Plot each node
        #ax.plot(x, y, "-o", color="black", linewidth=1.5, markersize=5, alpha=0.8, label="Coverage Path")
        # Plot the path
        ax.plot(x, y,  color="black", linewidth=1.5, alpha=0.8, label="Coverage Path")
        
        # Annotate the start and end points
        # ax.scatter(x[0], y[0], color="green", s=80, label="Start Point", edgecolor="black")
        # ax.scatter(x[-1], y[-1], color="red", s=80, label="End Point", edgecolor="black")

        # Add width and length labels on the figure
        ax.text(w / 2 + x0, l + 0.2 + y0, f"$w$ = {w}", fontsize=10, color="black", ha="center")
        ax.text(w + 0.3 + x0, l / 2 + y0, f"$l$ = {l}", fontsize=10, color="black", rotation=90, va="center")

        # Emphasize path length
        ax.text(
            0.5 * (w) + x0, -1.1 + y0, f"$Total$ $Path$ $Length$ $=$ ${path_length}$ ", 
            fontsize=12, color="grey", ha="center", fontweight="bold"
        )

        # Add labels, legend, and title
        # ax.set_title(f"Back-and-Forth Coverage Planner (w={w}, l={l})", color="black")
        ax.set_title(f"Back-and-Forth Coverage Planner ", color="black")
        ax.set_xlabel("Width ", color="black")
        ax.set_ylabel("Length ", color="black")
        ax.legend(loc="upper right", facecolor="white", edgecolor="black")

        # Show the visualization
        plt.tight_layout()
        plt.show()

    # Make a list of the coords to return 
    coords=[]
    for i in range(len(x)):
        coords.append([int(x[i]-0.5),int(y[i]-0.5)])
    return coords, path_length


def max_coverage(map,budget,startpos,multirobot=1,show=True):
    # Iterate over possible translations of the base coverage path
    multi_best_coords = []
    multi_info = []
    multi_costs = []
    for robot in range(multirobot):
        best_coords = []
        best_travel_coords=[]
        best_info = 0
        best_costs = 0
        for x in range(map.shape[1]):
            for y in range(map.shape[0]):
                translated_coords,costs = visualize_back_and_forth(budget,x,y,True,startpos,False)
                if translated_coords:
                    if np.max(translated_coords)>map.shape[1]: # if it goes outside of the map, skip it
                        continue 
                    info=0
                    for [x_coord,y_coord] in translated_coords:
                        if x_coord in range(map.shape[0]) and y_coord in range(map.shape[1]):
                            info+= map[y_coord][x_coord]
                    if [x,y]!=startpos:
                        info,travel_coords = addTravelInfo(map,translated_coords,info)
                    if info>best_info:
                        best_info=info
                        best_coords=translated_coords
                        best_travel_coords=travel_coords
                        best_costs = costs
        for coords in best_travel_coords+best_coords:
            map[coords[1],coords[0]]=0 # update the map to avoid redundancy in paths
        multi_best_coords.append(best_coords)
        multi_info.append(best_info)
        multi_costs.append(best_costs)
        # Visualize the results if requested
        if show:
            animation_max_coverage(map, best_coords, best_info)
    sum_best_coords = list(itertools.chain(*multi_best_coords))
    sum_info = sum(multi_info)
    sum_costs = sum(multi_costs)
    #return best_coords,best_info,best_costs
    #return multi_best_coords, multi_info, multi_costs
    return sum_best_coords, sum_info, sum_costs

def addTravelInfo(map,coords,info):
    distance_start = math.dist(coords[0],coords[1])
    dt = 1 / (2 * distance_start)
    t = 0
    infopath = []
    while t < 1.0:
        xline = coords[1][0] - coords[0][0]
        yline = coords[1][1] - coords[0][1]
        xpoint = round(coords[0][0] + t * xline)
        ypoint = round(coords[0][1] + t * yline)
        if not [xpoint,ypoint] in infopath and not [xpoint,ypoint] in coords:
            infopath.append([xpoint,ypoint])
            info+= map[ypoint][xpoint]
        t += dt

    distance_end = math.dist(coords[-1],coords[-2])
    dt = 1 / (2 * distance_end)
    t = 0
    while t < 1.0:
        xline = coords[-2][0] - coords[-1][0]
        yline = coords[-2][1] - coords[-1][1]
        xpoint = round(coords[-1][0] + t * xline)
        ypoint = round(coords[-1][1] + t * yline)
        if not [xpoint,ypoint] in infopath and not [xpoint,ypoint] in coords:
            infopath.append([xpoint,ypoint])
            info+= map[ypoint][xpoint]
        t += dt
    return info, infopath


def animation_max_coverage(info_map, best_path, best_info, title="Max Coverage with Information Map",show=True,pathname=None,rounds=None,size=(100,100)):
    """
    Visualizes the max coverage path over the given information map.
    
    Parameters:
    - info_map: 2D numpy array representing the information field.
    - coverage_path: List of (x, y) coordinates representing the coverage path.
    - best_info: Total information gathered along the path.
    - title: Title for the plot (optional).
    """
    startpos=[50,0]
    if size!=(100,100):
        startpos=[50,50-size[1]/2]
    # Translate the points to fit nicely on grid
    coverage_path=[]
    for [x,y] in best_path:
        coverage_path.append([x-0.5,y-0.5])


    # Define grid size
    l, w = info_map.shape
    
    # Calculate the field of view radius
    radius = 1  # Default radius = 1 (adjustable)
    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))
    #ax.set_aspect('equal')
    #ax.set_facecolor("white")  # White background
    
    # Display the information map as a heatmap
    colormap = cm.Blues
    colormap.set_bad(color="black")
    heatmap = ax.imshow(
        info_map, 
        cmap=colormap, 
        vmin=0, 
        vmax=1, 
        origin="lower", 
        extent=[-0.5, w - 0.5, -0.5, l - 0.5], 
        alpha=0.8
    )


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(heatmap, cax=cax, label="Information Value",ticks=[])
    
    # Extract x and y coordinates from the path
    if coverage_path==[]:
        x, y = [],[]
    else:
        x, y = zip(*coverage_path)
        x = [xi + 0.5 for xi in x]  # Center path within grid cells
        y = [yi + 0.5 for yi in y]
    
    # # Plot the field of view (glow effect)
    # for (xi, yi) in coverage_path:
    #     rect = plt.Rectangle(
    #         (xi - radius + 0.5, yi - radius + 0.5), 
    #         2 * radius, 2 * radius, 
    #         color="lightblue", 
    #         alpha=0.2
    #     )
    #     ax.add_patch(rect)
    
    # Draw the coverage path
    colors = mpl.colormaps['Dark2'].colors
    color_list = []
    c=0
    linewidth = 0.5/(size[0]/100)
    if not rounds:
        for i in range(len(x)-1):
            color_list.append(colors[c])
            if i>0 and ([x[i],y[i]] == startpos and not [x[i-1],y[i-1]]==startpos):
                c+=1
                ax.plot(
                    [x[i],x[i+1]], [y[i],y[i+1]], color=colors[c], linewidth=linewidth, alpha=0.8, label="Robot "+str((c+1))
                )
            elif i==0:
                ax.plot(
                    [x[i],x[i+1]], [y[i],y[i+1]], color=colors[c], linewidth=linewidth, alpha=0.8, label="Robot "+str((c+1))
                )
            else:
                ax.plot(
                    [x[i],x[i+1]], [y[i],y[i+1]], color=colors[c], linewidth=linewidth, alpha=0.8
                )
    else:
        for i in range(len(x)-1):
            color_list.append(colors[c])
            if i>0 and (rounds[i-1]<rounds[i]):
                c+=1
                ax.plot(
                    [x[i],x[i+1]], [y[i],y[i+1]], color=colors[c], linewidth=linewidth, alpha=0.8, label="Robot "+str((c+1))
                )
            elif i==0:
                ax.plot(
                    [x[i],x[i+1]], [y[i],y[i+1]], color=colors[c], linewidth=linewidth, alpha=0.8, label="Robot "+str((c+1))
                )
            else:
                ax.plot(
                    [x[i],x[i+1]], [y[i],y[i+1]], color=colors[c], linewidth=linewidth, alpha=0.8
                )

    ax.plot(
        [x[-1],startpos[0]], [y[-1],startpos[1]], color=colors[c], linewidth=linewidth, alpha=0.8
    )
    #ax.set_prop_cycle(color=color_list)         
    # ax.plot(
    #     x, y, linewidth=0.5, alpha=0.8, label="Coverage Path"
    # )



    
    
    # Add grid and formatting
    ax.set_xlim(-0.5+(50-size[0]/2), w - 0.5 - (50-size[0]/2))
    ax.set_ylim(-0.5+(50-size[1]/2), l - 0.5 - (50-size[1]/2))
    #ax.grid(visible=True, color="grey", linestyle="--", linewidth=0.5, alpha=0.6)
    #ax.set_xticks(range(0, w))
    #ax.set_yticks(range(0, l))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(colors="black")
    
    # Add width and length labels
    # ax.text(w / 2, l, f"$w$ = {w}", fontsize=10, color="black", ha="center")
    # ax.text(w, l / 2, f"$l$ = {l}", fontsize=10, color="black", rotation=90, va="center")
    
    # Emphasize total information collected
    if size==(100,100):
        ax.text(
            w / 2, -3.5, f"$Total$ $Information$ $Collected$ = ${best_info:.2f}$", 
            fontsize=12, color="grey", ha="center", fontweight="bold"
        )
    else:
        ax.text(
            w / 2, -1.5 + (50-size[1]/2), f"$Total$ $Information$ $Collected$ = ${best_info:.2f}$", 
            fontsize=12, color="grey", ha="center", fontweight="bold"
        )

    # Add title and legend
    ax.set_title(title, fontsize=14, color="black")
    if c==0 and not pathname:
        ax.legend(["Coverage Path"],loc="upper right", facecolor="white", edgecolor="black")
    elif c==0 and pathname: # 1-robot tests
        ax.legend(["Robot 1"],loc="upper right", facecolor="white", edgecolor="black")
    else:
        ax.legend(loc="upper right", facecolor="white", edgecolor="black")
    
    plt.tight_layout()
    if pathname:
        plt.savefig(pathname + "animation")

    if show:
        # Display the visualization
        plt.show()
    else:
        plt.close()






if __name__ == '__main__':
    # Example Usage
    visualize_back_and_forth(budget=9,x0=2,y0=4)