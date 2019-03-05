# Export data frame to work in Orange
# Date: 05-03-2019


# import packages
import numpy as np
import matplotlib.pyplot as plt
import os, fnmatch
import pandas as pd
import glob
from os import path
## Import the data

# Define the path 
pattern = 'C:/Users/2166611p/Desktop/PhD/Data/FTIR data/Mosquito Samples/RAW_FTIR_Experiment_2/*.dpt'

# Create a list of all the files from the path 
fileList = [os.path.basename(x) for x in glob.glob(pattern)] # path.basename returns only the name of the file not all the path
filepath = glob.glob(pattern)
# read the first element of the list to extract wavenumbers

data_file = np.loadtxt(filepath[1],delimiter = '\t')
w = data_file[:,0] # wavenumbers

dfList=[] # create a empty list

for filename in filepath: # intirate on each file in fileList
    print(filename) # print the name of the current file
    # read the file and extrac the second column
    df=pd.read_csv(filename,header=None,delimiter='\t',usecols=[1])
    dfList.append(df) # append the data read into the dfList

# after the loop it takes all the files in dflist and join them them into a new file. Vertically (axis = 1)
concatanedDf = pd.concat(dfList,axis=1)
concatanedDf


# Extract the species and age from the name's files
speciesList=[]  # create a empty list where the species are going to be store
ageList=[] # create a empty list where the age is going to be store
idList=[]   # create a empty list where the ID is going to be store
mozziepart=[] # create a empty list where the part(head, thorax, etc) is going to be store
mozziespec=[] # # create a empty list where the specific part is going to be store
for filename in fileList:
    sp = filename[4:6]
    ag = filename[7:10]
    id = filename[0:3]
    mp = filename[11:13]
    ms = filename[14:16]
    speciesList.append(sp)
    ageList.append(ag)
    idList.append(id)
    mozziepart.append(mp)
    mozziespec.append(ms)
speciesList
ageList
idList
mozziepart
mozziespec

#------------------------------------------------------------------#
# build the data frame to work in Orange or Machine learning use

# round up the values of the wavenumbers and use them as index of the dataframe

w_round = np.ceil(w).astype(int) # round wavenumbers
index_df = concatanedDf.set_index(w_round) # index the rows with the wavenumebers
trans_df = index_df.T # Transpose the concatanedDf
trans_df

# Add columns of species and age
trans_df['Species'] = speciesList # add a new column to a DataFrame
trans_df['Age'] = ageList
trans_df['ID'] = idList
trans_df['Part'] = mozziepart
trans_df['Sp Part'] = mozziespec

trans_df

# Export data frame to work in Orange
export_csv = trans_df.to_csv (r'C:\Users\2166611p\Desktop\PhD\Python Scripts\dataframeLEGS.csv',index = False) #Don't forget to add '.csv' at the end of the path
