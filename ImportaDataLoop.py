
# import packages
import numpy as np
import matplotlib.pyplot as plt
import os, fnmatch
import pandas as pd
import glob

# this need to be fixed. I should imoprt all data (the 2 columns) and then merge along with the common column
data_file = np.loadtxt('01-AC-100-1D-RX-AB1.dpt',delimiter = '\t')
w = data_file[:,0]
a = data_file[:,1]

## Import the data

# glob uses: I use the wildcard '?'. In this example I ask glob to find
# files that end with WG then any character ('?') and then .dpt.
# Therefore I can extract all the wings measurements regardiless the numbers

fileList=glob.glob("*WG?.dpt") # read the files in the current directory
dfList=[] # create a empty list
for filename in fileList: # intirate on each file in fileList
    print(filename) # print the name of the current file
    # read the file and extrac the second column
    df=pd.read_csv(filename,header=None,delimiter='\t',usecols=[1])
    dfList.append(df) # append the data read into the dfList
# after the loop it takes all the files in dflist and join them them into a new file. Vertically (axis = 1)
concatanedDf = pd.concat(dfList,axis=1)

Wing_spectra = plt.plot(w, concatanedDf)
plt.setp(Wing_spectra, 'color', 'r', 'linewidth', 1.0)
plt.xlabel('Wavenumbers (cm-1)')
plt.ylabel("Absorbance (a.u)")
plt.gca().invert_xaxis()
plt.xlim((4000, 600))
plt.title('Wing spectra in Rx Mode')
plt.show() # important if you want to see your plot
