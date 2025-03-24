
import matplotlib.pyplot as plt
import numpy as np
import math

def visualize_back_and_forth(budget,x0=0,y0=0,show=False):
    """
    Visualizes a back-and-forth coverage planner over a discretized grid.
    
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
    if show:
        # Plot the field of view (glow effect)
        ax.fill_betweenx(
            [min(y)-0.5,max(y)+0.5], x[0]-0.5, max(x)+0.5, color="lightblue", alpha=0.3, label="Field of View"
        )
        # ax.fill_betweenx(
        #     [min(y),max(y)+1], x[0], max(x)+1, color="lightblue", alpha=0.3, label="Field of View (Radius=1)"
        # )
        # Plot the path
        ax.plot(x, y, "-o", color="black", linewidth=1.5, markersize=5, alpha=0.8, label="Coverage Path")
        
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
    return coords


def max_coverage(map,budget,show=True):
    base_coords = visualize_back_and_forth(budget,0,0)
    best_coords = []
    best_info = 0
    # Iterate over possible translations of the base coverage path
    for x in range(map.shape[1] - max(coord[0] for coord in base_coords)):
        for y in range(map.shape[0] - max(coord[1] for coord in base_coords)):
            translated_coords=[]
            info=0
            for [x_base,y_base] in base_coords:
                translated_coords.append([x_base+x,y_base+y])
                info+= map[y_base+y][x_base+x]
            if info>best_info:
                best_info=info
                best_coords=translated_coords

    # Visualize the results if requested
    if show:
        animation_max_coverage(map, best_coords, best_info)
    return best_coords,info

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

def animation_max_coverage(info_map, best_path, best_info, title="Max Coverage with Information Map"):
    """
    Visualizes the max coverage path over the given information map.
    
    Parameters:
    - info_map: 2D numpy array representing the information field.
    - coverage_path: List of (x, y) coordinates representing the coverage path.
    - best_info: Total information gathered along the path.
    - title: Title for the plot (optional).
    """
    # Translate the points to fit nicely on grid
    coverage_path=[]
    for [x,y] in best_path:
        coverage_path.append([x+0.5,y+0.5])


    # Define grid size
    l, w = info_map.shape
    
    # Calculate the field of view radius
    radius = 1  # Default radius = 1 (adjustable)
    
    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.set_facecolor("white")  # White background
    
    # Display the information map as a heatmap
    colormap = cm.Blues
    colormap.set_bad(color="black")
    heatmap = ax.imshow(
        info_map, 
        cmap=colormap, 
        vmin=np.min(info_map), 
        vmax=np.max(info_map), 
        origin="lower", 
        extent=[-0.5, w - 0.5, -0.5, l - 0.5], 
        alpha=0.8
    )
    plt.colorbar(heatmap, ax=ax, label="Information Value")
    
    # Extract x and y coordinates from the path
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
    ax.plot(
        x, y, color="blue", linewidth=0.5, alpha=0.8, label="Coverage Path"
    )
    
    
    # Add grid and formatting
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, l - 0.5)
    ax.grid(visible=True, color="grey", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xticks(range(0, w))
    ax.set_yticks(range(0, l))
    ax.tick_params(colors="black")
    
    # Add width and length labels
    ax.text(w / 2, l, f"$w$ = {w}", fontsize=10, color="black", ha="center")
    ax.text(w, l / 2, f"$l$ = {l}", fontsize=10, color="black", rotation=90, va="center")
    
    # Emphasize total information collected
    ax.text(
        w / 2, -1, f"$Total$ $Information$ $Collected$ = ${best_info:.2f}$", 
        fontsize=12, color="grey", ha="center", fontweight="bold"
    )
    
    # Add title and legend
    ax.set_title(title, fontsize=14, color="black")
    ax.legend(loc="upper right", facecolor="white", edgecolor="black")
    
    # Display the visualization
    plt.tight_layout()
    plt.show()





if __name__ == '__main__':
    # Example Usage
    visualize_back_and_forth(budget=9,x0=2,y0=4)