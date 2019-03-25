# Preprocessing infrared data
# Date: 05-03-2019

# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter


# Preprocessing algorithms

# Standard normal variate correction
def snv(input_data):

    # Define a new array and populate it with the corrected data
    data_snv = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):

        # Apply correction
        data_snv[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])

    return data_snv


# multiplicative scatter correction
def msc(input_data, reference=None):
    ''' Perform Multiplicative scatter correction'''

    # mean centre correction
    for i in range(input_data.shape[0]):
        input_data[i,:] -= input_data[i,:].mean()

    # Get the reference spectrum. If not given, estimate it from the mean
    if reference is None:
        # Calculate mean
        ref = np.mean(input_data, axis=0)
    else:
        ref = reference

    # Define a new array and populate it with the corrected data
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Run regression
        fit = np.polyfit(ref, input_data[i,:], 1, full=True)
        # Apply correction
        data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0]

    return (data_msc, ref)


# Import wavelenght range from one files (needs to improve :( )
data_file = np.loadtxt('C:/Users/2166611p/Desktop/PhD/Data/FTIR data/Mosquito Samples/RAW_FTIR_Experiment_2/031-AC-03D-LG-LG.dpt',delimiter = '\t')
w = data_file[:,0] #wavenumbers

raw = pd.read_csv("C:/Users/2166611p/Desktop/PhD/Scripts/Python Scripts/dfLEGS.csv") # Import whole data set


#---------------Aqui empiezan los problemas-------------------------------#

#Plot all samples

X = raw.values[:,0:1763].astype('float32')
plt.plot(w,X.T)
plt.xlabel('Wavenumbers (cm-1)')
plt.ylabel("Absorbance (a.u)")
plt.gca().invert_xaxis()
plt.show()


# Plot only the fingerprint region (1789 to 972 cm-1)

# Select finger print region
Xf = raw.values[:,1146:1570].astype('float32') #Aqui uso el numero de columnas mas no el numero de longitud de onda, que seria lo correcto
wf = w[1146:1570] #longitud de onda del fingerprint region

# Plot
plt.figure(figsize=(15,10))
plt.plot(wf,Xf.T)
plt.xlabel('Wavenumbers (cm-1)')
plt.ylabel("Absorbance (a.u)")
plt.title('Spectra (1789 - 972)')
plt.gca().invert_xaxis()
plt.show()

# Plot the mean of the spectra

meanspecies = raw.groupby(['Species','Age']).mean() #Meanspectra by species
species_mean = meanspecies.iloc[:,1146:1570]
species_mean.T

list(species_mean.T)

# Ploting the means
plt.figure(figsize=(15,10))
plt.plot(wf,species_mean.T)
plt.xlabel('Wavenumbers (cm-1)')
plt.ylabel("Absorbance (a.u)")
plt.title('Mean spectra')
plt.gca().invert_xaxis()
plt.legend(loc='upper right')
plt.show()



# --------------Preprocessing--------------------------------------------#
# SNV
Xsnv = snv(Xf)
Xsnv

# MSC
Xmsc = msc(Xf)[0]

# Savitzky-Golay
Xf2 = savgol_filter(Xf, 9, polyorder = 2,deriv=2)

# plot Savitzky-golay
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(wf, Xf2.T)
ax.set_title('Savitzky-Golay Smoothing')
ax.set_xlabel('wavenumbers(cm-1)')
ax.set_ylabel('second derivative')
plt.gca().invert_xaxis()

# plot mean Savitzky-golay

raw_speciesIndex = raw.set_index('Species')
raw_speciesIndex
Speciesp = raw.loc[:,"Species"]
Speciesp
Xf2.set_index('Speciesp')


#Plot SNV
plt.figure(figsize=(15,10))
plt.plot(wf, Xsnv.T)
plt.title('SNV')
plt.xlabel('Wavenumbers (cm-1)')
plt.ylabel("Absorbance (a.u)")
plt.gca().invert_xaxis()


#Plot MSC
plt.plot(wf, Xmsc.T)
plt.xlabel('Wavenumbers (cm-1)')
plt.ylabel("Absorbance (a.u)")
plt.title('MSC')
plt.gca().invert_xaxis()




# Principal components analysis for Species

from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA as sk_pca
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn import preprocessing


# Seperating the features


X = raw.loc[:,"1850":"951"] # fingerprint region
X.T.index
# Seperate the labels
y = raw.loc[:, "Species"].values
y = raw.loc[:,"Species"]
y
# Savitzky-Golay savgol_filter

dX = savgol_filter(X, 25, polyorder = 5, deriv=1)

scaled_data = preprocessing.scale(dX) # center and scale the data
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data) # generate coordinates for a PCA graph

per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1) # percentage of variation that each PC accounts for
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)] # create PC labels

plt.figure(figsize=(15,10))
plt.bar(x=range(1,len(per_var)+1),height=per_var,tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

pca_df = pd.DataFrame(pca_data, index = y ,columns=labels)
pca_df

plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA')
plt.xlabel('PC1 -{0}%'.format(per_var[0]))
plt.ylabel('PC2 -{0}%'.format(per_var[1]))

for sample in pca_df.index:
    plt.annotate(sample,(pca_df.PC1.loc[sample],pca_df.PC2.loc[sample]))
plt.show()

wt = ['wt' + str(i) for i in range(1,6)]
ko = ['ko' + str(i) for i in range(1,6)]

index = [*wt,*ko]
index

yL = y.tolist
pca_df.index

loading_scores = pd.Series(pca.components_[0],index = X.T.index )
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

top_10_w = sorted_loading_scores[0:10].index.values

print(loading_scores[top_10_w])



#---------------Linear Discriminat Analysis on fingerprint region without preprocessing---------------------------#

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from itertools import product

# Defining the number of components of the linear discrminant analysis
lda = LDA(n_components=2)
Xlda = lda.fit_transform(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

lda = LDA()
lda.fit(X_train,y_train)

y_pred = lda.predict(X_test)
print(lda.score(X_test,y_test))

scores = cross_val_score(LDA(), X, y, cv=4)
predicted = cross_val_predict(LDA(), X, y, cv=4)

print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

# Using PCA to improve LDA
pca = PCA(n_components=20)
Xpc = pca.fit_transform(X)
scores = cross_val_score(LDA(), Xpc, y, cv=4)
predicted = cross_val_predict(LDA(), Xpc, y, cv=4)

print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))


# Confusion matrix
cm = confusion_matrix(y, predicted)

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
class_names
cm
# Plot non-normalized confusion matrix
plot_confusion_matrix(cm, classes=class_names, normalise = True, title='Confusion matrix, without normalization')


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, precision_recall_fscore_support, mean_squared_error, r2_score

cr = classification_report(y, predicted)
print(cr)

from itertools import *
itertools.product(range(cm.shape[0]))
