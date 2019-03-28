#Date: 3/20/2019
#Author: Salome Perez
#If you find this code useful, please include a citation for it. Thanks!

import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
import time

#s=time.time()

measurementsDF = pd.read_csv("C:/Users/2166611p/Desktop/PhD/Scripts/Python Scripts/dfLEGS.csv", skiprows = 0)
dList=['Species','Age', 'ID', 'Part', 'Sp Part']
descriptorsDF = measurementsDF[dList]


measurementsDF.drop(dList, axis=1,inplace=True)
print(measurementsDF.info() , descriptorsDF.info())
#measurements['idColumn'].describe()


wnLabels= measurementsDF.columns.values.tolist() #wavenumbers labels
waveNums = [int(x) for x in wnLabels] #wavenumbers numbers (for plotting)

#waveNums = list(map(int,waveNums))
idAllL=list(map(str, descriptorsDF['ID']))


### Figure 2 Age
data3D = measurementsDF.loc[descriptorsDF['Age'] == '03D']
data10D = measurementsDF.loc[descriptorsDF['Age'] == '10D']
#id3D=list(map(str, descriptorsDF['ID']['Age'=='03D']))
#id10D=list(map(str, descriptorsDF['ID']['Age'=='10D']))


fig, ax = plt.subplots(figsize=(15,10))
for i,aRow in data3D.iterrows():
   p1= ax.plot(waveNums, aRow,'r-')

for i,aRow in data10D.iterrows():
   p2= ax.plot(waveNums, aRow,'k-')

ax.legend((p1[0],p2[0]),('3 days old','10 days old'), loc=0, fontsize=10)
ax.set_title("MIR spectra by diffuse reflection",fontsize=16)
ax.set_xlabel("Wavenumber (1/cm)",fontsize=12)
ax.set_ylabel("Absorbance",fontsize=12)
ax.set_ylim((0,3))
ax.set_xlim((600,4000))
ax.invert_xaxis()
plt.show()

### FIGURE 3 Age mean 
plt.figure(figsize=(12,7))
plt.plot(waveNums,data3D.mean(axis=0),'r-',label ='3 days old')
plt.plot(waveNums,data10D.mean(axis=0),'k-',label = '10 days old')
plt.legend(loc=0,fontsize=9)
plt.title("Mean Absorbance",fontsize=16)
plt.xlabel("Wavenumber (1/cm)",fontsize=12)
plt.ylabel("Absorbance",fontsize=12)
plt.ylim((0,2))
plt.xlim((600,4000))
plt.gca().invert_xaxis()
plt.show()

#print(time.time()-s)

######################    EOF    ######################

### -- Legend Position --
#'best'=0,'upper right'=1,'upper left'=2,'lower left'=3,'lower right'=4,'right'=5
#'center left'=6,'center right'=7,'lower center'=8,'upper center'=9,'center'=10

### -- Graph Formatting Reference --
#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html

### -- Prototyping code --
# labels=['IR1','IR2','IR3','IR4','IR5','IR6','IR7','IR8','IR9','IR10']
# #pd.DataFrame(data).plot()
# #plt.legend(labels, loc=1)
# #plt.show()
