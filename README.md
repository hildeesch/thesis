# Thesis Hilde van Esch
# Synergia project
# Topic: ...

# Files and functions:
**main.py** : contains the matrices (different field structures) and controls the sequence between the modules

**heatmap.py** : contains different functions to generate the plants in the matrix structure and create images
* show_map(matrix): creates an image of the field structure
  * matrix = the structure matrix defining the field (from the main)
  * No output (only for visualization)
* withrows(matrix,density,rowdist,show=True)
  * matrix = the structure matrix defining the field (from the main)
  * plantdist = distance between the plants (1= every meter, 2 is every 2 meters, etc.)
  * rowdist = distance between the rows (1= a row every meter, 2 is every 2 meters, etc.)
  * show = whether the figure is generated (standard True)
  * Output: heatmatrix= matrix containing the plant locations
* norows(matrix,density,show=True)
  * Same as for withrows(), but then without the rows (just uniformly plants)


**planner.py** : empty 

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
  
**monitortreat.py** : relates to the modules of monitoring and treating, contains function to visualize the path
* showpath(matrix, path)
  * matrix = a matrix (preferably the spreading matrix)
  * path = the generated path from the planner
  * No output, simply generates a figure to visualize the path

# Design choices and module descriptions
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