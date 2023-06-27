import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
import os
from copy import deepcopy
from spreading import pathogenupdate
from spreading import weedsupdate


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
def showpathlong(day,total_days,fig,ax,uncertaintymatrix, path, cost, info,budget, steplength, searchradius, iteration,show,save):
    if day == 1:
        # 4 days
        # i_max = round(np.ceil(total_days/2))
        # j_max = 2
        # 20 days
        i_max=4
        j_max=5
        # 10 days
        i_max=3
        j_max=4
        fig, ax = plt.subplots(i_max,j_max)
        #fig.subplots_adjust(wspace=0.075)
    # ax index
    # four days:
    # i = (day-1)//2
    # j = (day-1)%2
    # 20 days
    # i = (day - 1) // 5
    # j = (day - 1) % 5
    # 10 days:
    i = (day - 1) // 4
    j = (day - 1) % 4
    # figure
    colormap = cm.Blues
    colormap.set_bad(color='black')
    im = ax[i,j].imshow(uncertaintymatrix, colormap, vmin=0, vmax=1, origin='lower')
    #divider = make_axes_locatable(ax[i,j])
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #plt.colorbar(im, cax=cax)
    ax[i,j].plot([x for x, _ in path], [y for _, y in path], '-r')

    # ax.plot([x for x, _ in x_best.infopath], [y for _, y in x_best.infopath], '-r')
    # ax.plot([x for x, _ in x_best.lastinfopath], [y for _, y in x_best.lastinfopath], '-r')
    ax[i,j].set_title("Path on day "+str(day)+"\nCost: "+str(round(cost))+" Info: "+str(round(info))+" \nSum entropy: "+str(round(np.nansum(uncertaintymatrix))),fontsize=4)
    #plt.title("Final path in entropy heatmap")
    #plt.suptitle("Cost: "+str(cost)+" Info: "+str(info)+" Sum entropy: "+str(np.nansum(uncertaintymatrix)))
    if day==total_days:
        fig.text(0.25,0.01,"Budget ="+str(budget)+" Step length ="+str(steplength)+" Search radius = "+str(searchradius))
        fig.tight_layout()
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
    return fig,ax
def updatematrix(disease,plantmatrix,spreadmatrix,worldmodel,uncertaintymatrix,infopath, sensoruncertainty=0,dailyuncertainty=0, reproductionrate=None,show=False):
    # disease = pathogen or weed
    saturation=disease.saturation
    # spreadmatrix = reality (ground truth on the spread of the pathogens/weeds) (= just for simulation purposes)
    # worldmodel = spreadmatrix that we know of
    # sensoruncertainty = uncertainty in monitoring, value between 0 (= no uncertainty) and 1 (monitoring gives no info at all)
    #   this uncertainty represents the standard deviation (STD) relative to the actual value
    #dailyuncertainty=0.01 # how much the uncertainty rises (across the entire matrix) each day

    uncertaintymatrixnew = deepcopy(uncertaintymatrix)
    worldmodelnew= deepcopy(worldmodel)
    for cell in infopath:
        worldmodelnew[cell[1],cell[0]]=np.random.normal(spreadmatrix[cell[1],cell[0]], spreadmatrix[cell[1],cell[0]]*sensoruncertainty) # updating the worldmodel with the monitored data

    if disease.type=="pathogen":
        [spreadnew,worldmodelnew,uncertaintymatrixupdate]=pathogenupdate(disease,plantmatrix,spreadmatrix,worldmodelnew,reproductionrate)
    else:
        [spreadnew,worldmodelnew,uncertaintymatrixupdate]=weedsupdate(disease,plantmatrix,spreadmatrix,worldmodelnew,reproductionrate)

    for row in range(100):
        for col in range(100):
            if not uncertaintymatrixupdate[row,col]==0:
                uncertaintymatrixnew[row,col]=uncertaintymatrixupdate[row,col] # we only take over those that have a value
    for cell in infopath:
        uncertaintymatrixnew[cell[1],cell[0]]=sensoruncertainty*uncertaintymatrix[cell[1],cell[0]] # decrease the uncertainty at the monitored cells

    uncertaintymatrixnew+=(np.ones_like(uncertaintymatrixnew)*dailyuncertainty) # add the daily uncertainty to each cell
    findDifferenceModel(spreadnew,worldmodelnew,False)
    if show:
        fig, ax = plt.subplots(3,2)
        colormap = cm.Oranges
        colormap.set_bad(color='black')
        im5 = ax[0,0].imshow(spreadmatrix, colormap, vmin=0, vmax=saturation, origin='lower')
        divider = make_axes_locatable(ax[0,0])
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im5, cax=cax)
        ax[0,0].set_title("Spread model")

        colormap = cm.Oranges
        colormap.set_bad(color='black')
        im6 = ax[0,1].imshow(spreadnew, colormap, vmin=0, vmax=saturation, origin='lower')
        divider = make_axes_locatable(ax[0,1])
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im6, cax=cax)
        ax[0,1].set_title("Spread model - new")


        colormap = cm.Oranges
        colormap.set_bad(color='black')
        im1 = ax[1,0].imshow(worldmodel, colormap, vmin=0, vmax=saturation, origin='lower')
        divider = make_axes_locatable(ax[1,0])
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im1, cax=cax)
        ax[1,0].set_title("World model")

        colormap = cm.Oranges
        colormap.set_bad(color='black')
        im2 = ax[1,1].imshow(worldmodelnew, colormap, vmin=0, vmax=saturation, origin='lower')
        divider = make_axes_locatable(ax[1,1])
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im2, cax=cax)
        ax[1,1].set_title("World model - new")


        colormap = cm.Blues
        colormap.set_bad(color='black')
        im3 = ax[2,0].imshow(uncertaintymatrix, colormap, vmin=0, vmax=1, origin='lower')
        divider = make_axes_locatable(ax[1,0])
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im3, cax=cax)
        ax[2,0].set_title("Spatial distribution of uncertainty")

        colormap = cm.Blues
        colormap.set_bad(color='black')
        im4 = ax[2,1].imshow(uncertaintymatrixnew, colormap, vmin=0, vmax=1, origin='lower')
        divider = make_axes_locatable(ax[2,1])
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im4, cax=cax)
        ax[2,1].set_title("Spatial distribution of uncertainty - new")
        fig.tight_layout()
        plt.show()
    return spreadnew, worldmodelnew, uncertaintymatrixnew

def findDifferenceModel(model1,model2,show=False):
    differencematrix = np.zeros((100,100))
    for row in range(100):
        for col in range(100):
            differencematrix[row,col]=abs(model1[row,col]-model2[row,col])
    if show:
        fig, ax = plt.subplots()
        colormap = cm.Oranges
        colormap.set_bad(color='black')
        im5 = ax.imshow(differencematrix, colormap, vmin=0, vmax=1, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im5, cax=cax)
        ax.set_title("Difference in models")
