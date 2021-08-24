import numpy as np
from sklearn.decomposition import PCA 
from sklearn.mixture import GaussianMixture
from numpy import  where
from sklearn.metrics import f1_score
from numpy import genfromtxt


X = genfromtxt(r'c:\data\kdd_anomaly.txt', delimiter=' ')
Y=X[:,-1]
X = np.delete(X, -1, axis=1)

#PCA
pca = PCA(n_components= 'mle', whiten = False )
X = pca.fit_transform(X)
a,b=X.shape


X = X + np.absolute(X).max()
X = np.sqrt(X)


#GaussianMixture
gm = GaussianMixture(n_components = b).fit(X)

Z = gm.predict_proba(X)
cluster = gm.predict(X)

#p(x) 
Z = 1 - Z
array = np.cumprod(Z, axis = 1)[:,-1]
array = 1-array

#bestEpsilon
bestEpsilon = 0
F1=0
bestF1=0

for epsilon in np.arange(0, 1, 0.01):
    not_normal = where(array <= epsilon)
    cluster[not_normal] = 1

    normal = where(array > epsilon)
    cluster[normal] = 0
    F1 = f1_score(y_true=Y, y_pred= cluster, average='binary')
    if F1 > bestF1:
        bestEpsilon = epsilon
        bestF1 = F1
        
print("Best F1", bestF1)
print("Best epsilon", bestEpsilon)

#elective epsilon
# epsilon = ???
# not_normal = where(array <= epsilon)
# cluster[not_normal] = 1

# normal = where(array > epsilon)
# cluster[normal] = 0

# print("F1 score: ", f1_score(y_true=Y, y_pred= cluster, average='binary'))