
# import packages
import numpy as np
import matplotlib.pyplot as plt
import os, fnmatch
import pandas as pd
import glob

# this need to be fixed. I should imoprt all data (the 2 columns) and then merge along with the common column
data_file = np.loadtxt('01-AC-100-1D-RX-AB1.dpt',delimiter = '\t')
w = data_file[:,0] # wavenumbers
a = data_file[:,1] # Absorbance

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

#Extract the wavenumbers for indexing
wavenumbers = w.tolist()

# Extract the species and age from the name's files
speciesList=[]  # create a empty list where the species are going to be store
ageList=[] # create a empty list where the age are going to be store

for filename in fileList:
    x = filename[3:5]
    y = filename[6:9]
    speciesList.append(x)
    ageList.append(y)

#------------------------------------------------------------------#
# Make the data frame

# round up the values of the wavenumbers and use them as index of the dataframe

w_round = np.ceil(w).astype(int) # round wavenumbers
index_df = concatanedDf.set_index(w_round) # index the rows with the wavenumebers
trans_df = index_df.T # Transpose the concatanedDf
trans_df

# Add columns of species and age
trans_df['species'] = speciesList # add a new column to a DataFrame
trans_df['Age'] = ageList

trans_df

# Exporting data frame to work in orange

export_csv = trans_df.to_csv (r'C:\Users\2166611p\Desktop\PhD\Python Scripts\dataframesp.csv',index = False)





#------------------------------------------------------------------#
#Ploting
Wing_spectra = plt.plot(w, concatanedDf)
plt.setp(Wing_spectra, 'color', 'r', 'linewidth', 1.0)
plt.xlabel('Wavenumbers (cm-1)')
plt.ylabel("Absorbance (a.u)")
plt.gca().invert_xaxis()
plt.xlim((4000, 600))
plt.title('Wing spectra in Rx Mode')
plt.show() # important if you want to see your plot
