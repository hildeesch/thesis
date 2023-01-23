# Thesis Hilde van Esch
# Synergia project
# Topic: ...

# Files and functions:
**main.py** : contains the matrices (different field structures) and controls the sequence between the modules

**heatmap.py** : contains different functions to generate the plants in the matrix structure and create images
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
* weedsspread(matrixstructure,matrixplants,show=True): simplified model for weed spread
  * matrixstructure = the structure matrix defining the field (from main)
  * matrixplants = the matrix containing the plant locations (from heatmap)
  * show = whether the figure is generated
  * patchnr = the number of patches of weeds
  * patchsize = the radius of the patch
  * spreadrange = the range which the weed travels when invading new territory from the existing patch
  * spreadspeed = the speed in which the weed invades (1 = each day the full spreadrange, 2 = the full spreadrange in 2 days, etc.)
  * plantattach = boolean, whether the weed is more likely to grow on the places where crops grow (True) or on empty places
  * Output: weedmatrix= matrix containing the weed locations
* uncertainty(matrixstructure,matrixweeds,matrixplants,spreadrange,plantattach,show=True): generates the uncertainty (entropy)
  * matrixstructure = the structure matrix defining the field (from main)
  * matrixweeds = the matrix containing the weed locations (from weedspread)
  * matrixplants = the matrix containing the plant locations (from heatmap)
  * spreadrange = the range which the weed travels when invading new territory from the existing patch
  * plantattach = boolean, whether the weed is more likely to grow on the places where crops grow (True) or on empty places
  * show = whether the figure is generated
  * Output: uncertaintymatrix= the uncertainty about the growth of the weed
  
**monitortreat.py** : relates to the modules of monitoring and treating, contains function to visualize the path
* showpath(matrix, path)
  * matrix = a matrix (preferably the spreading matrix)
  * path = the generated path from the planner
  * No output, simply generates a figure to visualize the path

