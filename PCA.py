#------------- Principal components analysis for Species-----------------------

# Import modules
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Seperate data into data and labels
X = raw.loc[:,"1850":"951"] # fingerprint region
y = raw.loc[:,"Species"] # Seperate the labels

# Savitzky-Golay savgol_filter
dX = savgol_filter(X, 25, polyorder = 5, deriv=1)

# PCA
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

pca_df = pd.DataFrame(pca_data, index = y ,columns=labels) # put the coordinates into a data frame 
colormap = {
    1 : '#ff0000',  # Red
    2 : '#0000ff',  # Blue
}
colorlist = [colormap[c] for c in y]

# PCA plot for PC1 and PC2
plt.scatter(pca_df.PC1, pca_df.PC2,c=colorlist)
plt.title('My PCA')
plt.xlabel('PC1 -{0}%'.format(per_var[0]))
plt.ylabel('PC2 -{0}%'.format(per_var[1]))

