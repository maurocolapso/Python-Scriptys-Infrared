
# import packages
import numpy as np
import matplotlib.pyplot as plt
import os, fnmatch
import pandas as pd
import glob

# read files of the directory
#listOfFiles = os.listdir('.') #List of files on directory ('.')
#pattern = "*.dpt" # the extension for the filter

# This one works good
path = ('.')
text_files = [f for f in os.listdir(path) if f.endswith('.dpt')]
for text_file in text_files:
    print(text_file)


# import dataset
data_file = np.loadtxt('01-AC-100-1D-RX-AB1.dpt',delimiter = '\t')
w = data_file[:,0]
a = data_file[:,1]

# making a plot
#fig, ax = plt.subplots()
#ax.plot(w, a)
#plt.gca().invert_xaxis()
#ax.set(xlabel='Wavenumbers (cm-1)', ylabel='Absorbance (a.u)',
#       title='Spectra of Abdomen in Reflection Mode')
#ax.grid()
#plt.show()

graph_spectra = plt.plot(w, a,)
plt.xlabel('Wavenumbers (cm-1)')
plt.ylabel("Absorbance (a.u)")
plt.title('Spectra of Abdomen in Reflection Mode')
#MATLAB style string value pairs
plt.setp(graph_spectra, 'color', 'r', 'linewidth', 1.0)
plt.gca().invert_xaxis()

plt.show()




# glob uses: I use the wildcard '?'. In this example I ask glob to find
# files that end with WG then any character ('?') and then .dpt.
# Therefore I can extract all the wings measurements regardiless the numbers

## Import the data

fileList=glob.glob("*WG?.dpt") # read the files in the current directory
dfList=[] # create a empty list
for filename in fileList: # intirate on each file in fileList
    print(filename) # print the name of the current file
    # read the file and extrac the second column
    df=pd.read_csv(filename,header=None,delimiter='\t',usecols=[1])
    dfList.append(df) # append the data read into the dfList
# after the loop it takes all the files in dflist and concantaned them into a new file. Vertically (axis = 1)
concatanedDf = pd.concat(dfList,axis=1)

concatanedDf
filename

Wing_spectra = plt.plot(w, concatanedDf)
plt.setp(Wing_spectra, 'color', 'r', 'linewidth', 1.0)
plt.xlabel('Wavenumbers (cm-1)')
plt.ylabel("Absorbance (a.u)")
plt.gca().invert_xaxis()
plt.xlim((4000, 600))
plt.title('Wing spectra in Rx Mode')

# the next problem is how I extract with just the age or maybe I just need to
# extract everything than make a proper dataframe will all the data
# and add species, age...ect
