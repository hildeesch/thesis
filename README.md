# Informative Path Planning for the Monitoring of Pathogens and Weeds (2023)
_Author: Hilde van Esch_

_Thesis Project at Technical University of Eindhoven, part of Synergia project_
# Introduction and References
This repository contains all the code for the thesis project by Hilde van Esch for the master of Systems & Control at the Technical University of Eindhoven. 

The planning algorithm (in the Planner folder) is based on the code of Huiming Zhou (https://github.com/zhm-real/PathPlanning/). The final algorithm is a Rapidly-Exploring Information Gathering sampling-based algorithm, as introduced by Hollinger and Sukhatme in their paper "Sampling-based robotic information gathering algorithms" (2014).

Below, the files and functions are outlined. Note that this does not include all files of the Planner folder, but only those that have been adapted for this project. Furthermore, module descriptions entails pseudo-code for functions for improved understanding.
For deeper understanding, it is recommended to analyze the code and comments, starting in the main.py file.

# Files and Functions:
**main.py** : controls the sequence between the modules
* default(): used for running the algorithm during development
* prepandtest(): used for creating testing scenarios and retrieving testing results for validation
**heatmap.py** : contains different functions to create the matrix defining the field shape and generate the plants in the matrix structure and create images
* show_map(matrix): creates an image of the field structure
  * matrix = the structure matrix defining the field (from the main)
  * No output (only for visualization)
* polygon(shape)
  * shape = desired field shape (e.g. "rectangle","hexagon_convex",etc.)
  * Output: matrix = matrix where 0 is within the field and NaN is outside, coords = corner points
* withrows(matrix,density,rowdist,show=True)
  * matrix = the structure matrix defining the field (from the main)
  * plantdist = distance between the plants (1= every meter, 2 is every 2 meters, etc.)
  * rowdist = distance between the rows (1= a row every meter, 2 is every 2 meters, etc.)
  * show = whether the figure is generated (standard True)
  * Output: heatmatrix= matrix containing the plant locations
* norows(matrix,density,show=True)
  * Same as for withrows(), but then without the rows (just uniformly plants)
**spreading.py** : spread model and uncertainty
* weedsspread(matrixstructure,matrixplants,weed,show=True): simplified model for weed spread
  * matrixstructure = the structure matrix defining the field (from main)
  * matrixplants = the matrix containing the plant locations (from heatmap)
  * weed = an instance of the Weed class (see weed.py). Attributes used from the class in this function: patchnr, patchsize, plantattach (MAY CHANGE STILL??)
  * show = whether the figure is generated 
  * Output: weedmatrix= matrix containing the weed locations
* uncertaintyweeds(matrixstructure,matrixweeds,matrixplants,weed,show=True): generates the uncertainty (entropy)
  * matrixstructure = the structure matrix defining the field (from main)
  * matrixweeds = the matrix containing the weed locations (from weedspread)
  * matrixplants = the matrix containing the plant locations (from heatmap)
  * weed = an instance of the Weed class. Attributes used in this function: spreadrange, plantattach
  * show = whether the figure is generated 
  * Output: uncertaintymatrix= the uncertainty about the growth of the weed
* weedsupdate(weed,matrixplants,weedmatrix,weedmatrix_est,reproductionrate=None): updates the spreading and uncertainty matrix for the next timestep
  * Output: weedmatrix, uncertaintymatrix
* pathogenspread(matrixstructure,matrixplants,pathogen,show=True): simplified model for pathogen spread
  * matrixstructure = the structure matrix defining the field (from main)
  * matrixplants = the matrix containing the plant locations (from heatmap)
  * pathogen = an instance of the Pathogen class (see pathogen.py). Attributes used from the class in this function: patchnr, infectionduration, spreadrange, spreadspeed, reproductionrate, saturation (MAY CHANGE STILL??)
  * show = whether the figure is generated 
  * Output: pathogenmatrix= matrix containing the pathogen locations
* uncertaintypathogen(matrixstructure,matrixpathogen,matrixplants,pathogen,show=True): generates the uncertainty (entropy)
  * matrixstructure = the structure matrix defining the field (from main)
  * matrixpathogen = the matrix containing the pathogen locations (from pathogenspread)
  * matrixplants = the matrix containing the plant locations (from heatmap)
  * pathogen = an instance of the Pathogen class (see pathogen.py). Attributes used from the class in this function: spreadrange, spreadspeed, reproductionrate, saturation (MAY CHANGE STILL??)
  * show = whether the figure is generated 
  * Output: uncertaintymatrix= the uncertainty about the growth of the pathogen
* pathogenupdate(pathogen,matrixplants,pathogenmatrix, pathogenmatrix_est, reproductionrate=None): updates the spreading and uncertainty matrix for the next timestep
  * Output: pathogenmatrix, uncertaintymatrix

**monitortreat.py** : relates to the modules of monitoring and treating, contains function to visualize the path
* showpath(matrix, path): visualize a path overlaid on a matrix
  * matrix = a matrix (preferably the spreading matrix)
  * path = the generated path from the planner
  * No output, simply generates a figure to visualize the path
* showpathlong(day,total_days,fig,ax,matrix, path): visualize the paths of a multi-day simulation
  * day = the current testing day
  * total_days = amount of days in the multi-day simulation
  * fig, ax = the earlier created figure handle
  * uncertaintymatrix = the matrix to overlay the path on
  * path = the generated path from the planner
  * Output: fig, ax
* updatematrix(disease,plantmatrix,spreadmatrix,worldmodel,uncertaintymatrix,infopath): update the matrices based on the monitored path
  * disease = weed type or pathogen type
  * infopath = the path generated by the planner in terms of grid cells
  * Output: spreadnew, worldmodelnew, uncertaintymatrixnew (updated spread, worldmodel and uncertainty matrices)
* findDifferenceModel(model1,model2): find the difference matrix between 2 matrices and visualize
  * model1 = e.g. worldmodel
  * model2 = e.g. spreadmodel
**visualizations.py**: not required for the algorithm, independent file used for drawing visualizations
**weed.py**: class to create a weed type based on type-specific characteristics
**pathogen.py**: class to create a pathogen type based on type-specific characteristics

# Module Descriptions
Below, pseudo-code is given that explains the basics of the workings for the most important functions.

```bash
def weedsspread(matrixstructure,matrixplants,weed,show=True):
  for each patch:
    choose random center point of patch
    random choose patch size of the patch, which is at max the given patchsize
    for every grid point within the patch size distance of the patch center:
      value grid point = 1 or 0.5 (depending on plantattach)
    uncertaintyweeds(matrixstructure,matrixplants,weedmatrix,weed)

  if(show):
    plot weedmatrix
  return weedmatrix
  
def uncertaintyweeds(matrixstructure,matrixplants,matrixweeds,weed,show=True):
  for each grid point in the map:
        find max value of all grid points around the grid point within the spreadrange (that is still within the map) 
        uncertainty of grid point = 1 or 0.5 (depending on plantattach)
  if(show):
    plot uncertaintymatrix
  return uncertaintymatrix      

def pathogenspread(matrixstructure,matrixplants,pathogen,show=True):
  for each patch:
    choose random center point of patch
    random choose current infection duration of the patch, which is at max the given infectionduration
    for each day until current infection duration:
      for each grid point in the map:
        find max value of all grid points around the grid point within the spreadrange (that is still within the map)
        increase = max value of grid points in reach, modified by the reproduction rate and spreadspeed, and whether the grid point contains a crop
        value of grid point = value of grid point + increase (until it reaches the saturation value)
        
  uncertaintypathogen(matrixstructure,matrixplants,pathogenmatrix,pathogen)
  
  if(show):
    plot pathogenmatrix
  return pathogenmatrix
      
def uncertaintypathogen(matrixstructure,matrixplants,matrixpathogen,pathogen,show=True):
  for each grid point in the map:
        find max value of all grid points around the grid point within the spreadrange (that is still within the map)
        increase = max value of grid points in reach, modified by the reproduction rate and spreadspeed, and whether the grid point contains a crop
        uncertainty of grid point = uncertainty of grid point + increase (until it reaches the saturation value)
  if(show):
    plot uncertaintymatrix
  return uncertaintymatrix
```