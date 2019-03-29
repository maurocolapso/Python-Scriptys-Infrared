import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
import time




`measurementsDF = pd.read_csv("C:/Users/2166611p/Desktop/PhD/Scripts/Python Scripts/dfExperiment1.csv", skiprows = 0)
dList=['Species','Age', 'ID', 'Part', 'Sp Part']
descriptorsDF = measurementsDF[dList]

measurementsDF.drop(dList, axis=1,inplace=True)
print(measurementsDF.info() , descriptorsDF.info())
#measurements['idColumn'].describe()


wnLabels= measurementsDF.columns.values.tolist() #wavenumbers labels
waveNums = [int(x) for x in wnLabels] #wavenumbers numbers (for plotting)

#waveNums = list(map(int,waveNums))
idAllL=list(map(str, descriptorsDF['ID']))

# Plot abdomen with diferent ages
dataabodmen = measurementsDF.loc[descriptorsDF['Part'] == 'AB']

data1D = dataabodmen.loc[descriptorsDF['Age'] == '01D']
data3D = dataabodmen.loc[descriptorsDF['Age'] == '03D']
data12D = dataabodmen.loc[descriptorsDF['Age'] == '12D']
#id3D=list(map(str, descriptorsDF['ID']['Age'=='03D']))
#id10D=list(map(str, descriptorsDF['ID']['Age'=='10D']))


fig, ax = plt.subplots(figsize=(12,7))
for i,aRow in data1D.iterrows():
   p1= ax.plot(waveNums, aRow,'r-')

for i,aRow in data3D.iterrows():
   p2= ax.plot(waveNums, aRow,'k-')

for i, aRow in data12D.iterrows():
    p3 = ax.plot(waveNums, aRow,'b-')

ax.legend((p1[0],p2[0],p3[0]),('1 days old','3 days old','10 days old'), loc=0, fontsize=10)
ax.set_title("MIR spectra by diffuse reflection",fontsize=16)
ax.set_xlabel("Wavenumber (1/cm)",fontsize=12)
ax.set_ylabel("Absorbance",fontsize=12)
ax.set_ylim((1,5))
ax.set_xlim((600,4000))
ax.invert_xaxis()
plt.show()




### FIGURE 3 AGE MEAN SPECTRA
plt.figure(figsize=(12,7))
plt.plot(waveNums,data1D.mean(axis=0),'r-',label ='3 days old')
plt.plot(waveNums,data3D.mean(axis=0),'k-',label = '10 days old')
plt.plot(waveNums,data12D.mean(axis=0),'b-',label= '12 days old')
plt.legend(loc=0,fontsize=9)
plt.title("Mean Absorbance",fontsize=16)
plt.xlabel("Wavenumber (1/cm)",fontsize=12)
plt.ylabel("Absorbance",fontsize=12)
plt.ylim((1,3.5))
plt.xlim((1000,1800))
plt.gca().invert_xaxis()
plt.savefig('C:/Users/2166611p/Desktop/PhD/Scripts/Python Scripts/MeanAbodomen_ageEx1.png')


#------------- Principal components analysis for Age-----------------------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter


# Seperating the features

raw = pd.read_csv("C:/Users/2166611p/Desktop/PhD/Scripts/Python Scripts/dfExperiment1.csv")
dataabodmen =  raw.loc[raw['Part'] == 'AB']
dataabodmen = dataabodmen.reset_index()
X = dataabodmen.loc[:,"1850":"951"] # fingerprint region
y = dataabodmen.loc[:,"Age"] # Seperate the labels
y = y.reset_index(drop = True)
X = X.reset_index(drop = True)

# Savitzky-Golay savgol_filter
dX = savgol_filter(X, 25, polyorder = 5, deriv=1)
dX_dataframe = pd.DataFrame(dX)




scaled_data = preprocessing.scale(dX) # center and scale the data
pca = PCA()
pca.fit(scaled_data) #PCA math thing
pca_data = pca.transform(scaled_data) # generate coordinates for a PCA graph
per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1) # percentage of variation that each PC accounts for
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)] # create PC labels

# plot of how much variance have each PC
plt.figure(figsize=(15,10))
plt.bar(x=range(1,len(per_var)+1),height=per_var,tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

pca_df = pd.DataFrame(pca_data ,columns=labels) # put the coordinates into a data frame

finalDF = pd.concat([pca_df,dataabodmen[['Age']]],axis = 1)

# PCA Plot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('PC1 -{0}%'.format(per_var[0]), fontsize = 15)
ax.set_ylabel('PC2 -{0}%'.format(per_var[1]), fontsize = 15)
ax.set_title('PCA', fontsize = 20)
targets = ['01D', '03D','12D']
colors = ['k','r','b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDF['Age'] == target
    ax.scatter(finalDF.loc[indicesToKeep, 'PC1']
               , finalDF.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 50)
ax.legend(targets)
plt.savefig('C:/Users/2166611p/Desktop/PhD/Scripts/Python Scripts/PCA_smooth_ageEx1.png')


# LDA as a classifier --------------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from itertools import product

# Split the data intro training data and test data
X_train, X_test, y_train, y_test = train_test_split(dX, y, test_size=0.25, random_state=2)

#LDA on the train data set
lda = LDA()
lda.fit(X_train,y_train) # we are not looking into generate a new data set with reduced featrues, we want to train the data

# Predction of the LDA model to the test data set
y_pred = lda.predict(X_test)
print(lda.score(X_test,y_test))


# Checking the perfomance of the classifer with crossvalidation
scores = cross_val_score(LDA(), dX, y, cv=4)
predicted = cross_val_predict(LDA(), dX, y, cv=4)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

# Using PCA to improve LDA
pca = PCA(n_components=20) # define number of components of PCA
Xpc = pca.fit_transform(dX)# Apply PCA to data
scores = cross_val_score(LDA(), Xpc, y, cv=4) #Training with cross validation and reduced dimension data set
predicted = cross_val_predict(LDA(), Xpc, y, cv=4)

print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))


# Confusion matrix
cm = confusion_matrix(y, predicted)
cm


def plot_confusion_matrix(cm, classes,
                          normalise=True,
                          text=False,
                          title='Confusion matrix',
                          xrotation=0,
                          yrotation=0,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalisation can be applied by setting 'normalise=True'.
    """

    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "{0} (normalised)".format(title)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=xrotation)
    plt.yticks(tick_marks, classes, rotation=yrotation)

    if text:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]),
                                      range(cm.shape[1])):
            plt.text(j, i, "{0:.2f}".format(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class_names = np.unique(np.sort(y))

# Plot non-normalized confusion matrix
plt.figure(figsize=(12,7))
plot_confusion_matrix(cm, classes=class_names, normalise = True, title='Confusion matrix')
plt.savefig('C:/Users/2166611p/Desktop/PhD/Scripts/Python Scripts/ConfusionAGE_EX1_PCA-LDA.png')
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, precision_recall_fscore_support, mean_squared_error, r2_score

cr = classification_report(y, predicted)
print(cr)
