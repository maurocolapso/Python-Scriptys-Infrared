#---------------Linear Discriminat Analysis on fingerprint region---------------------------#

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from itertools import product
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, precision_recall_fscore_support, mean_squared_error, r2_score

# Defining the number of components of the linear discrminant analysis
lda = LDA(n_components=2)

# change labels into numbers for better management

from sklearn.preprocessing import LabelEncoder
y = raw['Species'].values
enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1
label_dict = {1: 'AC', 2: 'AK'}

# Using dX (savitsky-golay smoothing)
Xlda = lda.fit_transform(dX,y)
Xlda


#-------------------------- LDA as a classifier --------------------------------------

# Split the data intro training data and test data
X_train, X_test, y_train, y_test = train_test_split(dX, y, test_size=0.25, random_state=2)

#LDA on the train data set
lda = LDA()
lda.fit(X_train,y_train) # we are not looking into generate a new data set with reduced featrues, we want to train the data

# Predction of the LDA model to the test data set
y_pred = lda.predict(X_test)
print(lda.score(X_test,y_test))


# Training and testing the model using crossvalidation
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

# report
cr = classification_report(y, predicted)
print(cr)
