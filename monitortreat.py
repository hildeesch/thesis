import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
import os
def showpath(uncertaintymatrix, path, cost, info,budget, steplength, searchradius, iteration,show,save):
    fig, ax = plt.subplots()
    colormap = cm.Blues
    colormap.set_bad(color='black')
    im = ax.imshow(uncertaintymatrix, colormap, vmin=0, vmax=3, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    # ax.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
    ax.plot([x for x, _ in path], [y for _, y in path], '-r')

    # ax.plot([x for x, _ in x_best.infopath], [y for _, y in x_best.infopath], '-r')
    # ax.plot([x for x, _ in x_best.lastinfopath], [y for _, y in x_best.lastinfopath], '-r')
    ax.set_title("Final path in entropy heatmap")
    #plt.title("Final path in entropy heatmap")
    plt.suptitle("Cost: "+str(cost)+" Info: "+str(info)+" Iteration: "+str(iteration))
    fig.text(0.25,0.01,"Budget ="+str(budget)+" Step length ="+str(steplength)+" Search radius = "+str(searchradius))
    #fig.tight_layout()
    if save:
        now = datetime.now()
        now_string = now.strftime("%m_%d_%H_%M")
        path = "../../Documents/Thesis_project/Figures/New_sim_figures/"
        dirname = os.path.dirname(path)
        filename= "/Final_path_"+now_string+".png"
        #plt.savefig(os.path.join(dirname,filename))
        plt.savefig(path+filename)
    if show:
        plt.show()
