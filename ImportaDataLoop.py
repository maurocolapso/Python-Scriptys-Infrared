# import packages
import numpy as np
import matplotlib.pyplot as plt
import os, fnmatch
import pandas as pd
import glob

# maybe the person input the directory folder
# program reads the files take one and import it and make the next lines

data_file = np.loadtxt('01-AC-100-1D-RX-AB1.dpt',delimiter = '\t')
w = data_file[:,0] # wavenumbers
a = data_file[:,1] # Absorbance

## Import the data

# glob uses: I use the wildcard '?'. In this example I ask glob to find
# files that end with WG then any character ('?') and then .dpt.
# Therefore I can extract all the wings measurements regardiless the numbers

fileList=glob.glob("*.dpt") # read the files in the current directory
dfList=[] # create a empty list

for filename in fileList: # intirate on each file in fileList
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
    sp = filename[3:5]
    ag = filename[6:8]
    ij = filename[0:2]
    mp = filename[9:11]
    ms = filename[12:14]
    speciesList.append(sp)
    ageList.append(ag)
    idList.append(ij)
    mozziepart.append(mp)
    mozziespec.append(ms)

#------------------------------------------------------------------#
# build the data frame

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
export_csv = trans_df.to_csv (r'C:\Users\2166611p\Desktop\PhD\Python Scripts\dataframesp.csv',index = False) #Don't forget to add '.csv' at the end of the path


#--------------------------------------------------------------------------#
# Plotting each general part against age (whole spectra)
pr = input('Which general part of the mosquito do you want? ')
age_one = input('Which age are you looking for? ')
age_two = input('Which other age are you looking for? ')
criteria_part = trans_df['Part'] == pr
criteria_age = trans_df['Age'] == age_one
criteria_age_two = trans_df['Age'] == age_two
criteria_all_one = criteria_age & criteria_part
criteria_all_two = criteria_age_two & criteria_part
oneday = trans_df[criteria_all_one]
twoday = trans_df[criteria_all_two]

X_one = oneday.values[0:,0:1763].astype('float32')
X_two = twoday.values[0:,0:1763].astype('float32')
whole_spectra = plt.plot(w, X.T,'r',w,X_two.T,'b')
plt.setp(whole_spectra, 'linewidth', 1.0)
plt.xlabel('Wavenumbers (cm-1)')
plt.ylabel("Absorbance (a.u)")
plt.gca().invert_xaxis()
plt.xlim((4000, 600))
# legends are not right yet
plt.legend(('1 day old', '3 days old'),
           loc='upper right')
plt.show()

#Ploting
Wing_spectra = plt.plot(w, concatanedDf)
plt.setp(Wing_spectra, 'color', 'r', 'linewidth', 1.0)
plt.xlabel('Wavenumbers (cm-1)')
plt.ylabel("Absorbance (a.u)")
plt.gca().invert_xaxis()
plt.xlim((4000, 600))
plt.title('Wing spectra in Rx Mode')
plt.show() # important if you want to see your plot

https://www.idtools.com.au/two-scatter-correction-techniques-nir-spectroscopy-python/

