import numpy as np
from sklearn.decomposition import PCA 
from sklearn.mixture import GaussianMixture
from numpy import where
from sklearn.metrics import f1_score
from numpy import genfromtxt
import pickle

X= genfromtxt("kdd_anomaly.txt",delimiter="")
Y =X[:,-1]
X = np.delete(X, -1, axis=1)

try:
    pca=pickle.load(open("pca_data","rb"))
except:
    pca = PCA(n_components= 'mle', whiten = False )
    pca = pca.fit(X)
    pickle.dump(pca,open("pca_data","wb"))

X=pca.transform(X)
b=X.shape[1]
X = X + np.absolute(X).max()
X = np.sqrt(X)

try:
    gm=pickle.load(open("gm_data.txt","rb"))
except:
    gm = GaussianMixture(n_components = b)
    gm = gm.fit(X)
    pickle.dump(gm,open("gm_data.txt","wb"))

Z = gm.predict_proba(X)

# Calculate p(x) 
Z = 1 - Z
array = np.cumprod(Z, axis = 1)[:,-1]
array = 1 - array



