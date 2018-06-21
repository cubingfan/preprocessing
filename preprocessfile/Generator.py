
# coding: utf-8

# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')

from collections import defaultdict
import scipy
import traceback
import os
import numpy as np
import scipy
import scipy.stats
import scipy.io
import matplotlib.pyplot as plt
import sys
import seaborn as sns; sns.set()  # for plot styling

import sklearn.cluster
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


#LOAD DATA
# input_path = 'filtered_gene_bc_matrices/zv10_gtf89_cloche_gfp/'

# if os.path.isfile(input_path + '/matrix.npz'):
#     E = scipy.sparse.load_npz(input_path + '/matrix.npz')
# else:
#     E = scipy.io.mmread(input_path + '/matrix.mtx').T.tocsc()
#     scipy.sparse.save_npz(input_path + '/matrix.npz', E, compressed=True)

# print(E.shape)

# # Convert to numpy array and standardize
# X = E.toarray()
# X = StandardScaler().fit_transform(X)


# In[3]:


def genPCA(E,n,k):
    X = E.toarray()
    X = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=n)
    pca.fit(X)
    X_pca = pca.transform(X)
    print("original shape:   ", X.shape)
    print("transformed shape:", X_pca.shape)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_pca)
    y_kmeans = kmeans.predict(X_pca)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='.',alpha=0.8,c=y_kmeans, cmap='viridis')
    name = 'PCA'+str(n)+'Kmeans'+str(k)
    print(name)
    plt.title(name)
    plt.savefig('GeneratedPCA/'+name)
    plt.show()
    plt.close()
    return [X_pca,pca.components_]


# In[5]:


def contributor(pca_comp, comp, cont):
    p1 = np.fabs(pca_comp[comp])
    cont = cont*-1
    ind = np.argpartition(p1,cont)[cont:]
    return ind


# In[28]:


# genPCA(20,5)
# genPCA(30,5)
# genPCA(40,5)
# genPCA(50,5)
# genPCA(60,5)


# In[16]:


#TSNE
# n = 20
# k = 5
# pca = PCA(n_components=n)
# pca.fit(X)
# X_pca = pca.transform(X)
# print("original shape:   ", X.shape)
# print("transformed shape:", X_pca.shape)
# kmeans = KMeans(n_clusters=k)
# kmeans.fit(X_pca)
# y_kmeans = kmeans.predict(X_pca)


# In[2]:


def genTSNE(X,k,p):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    tsne = TSNE(n_components=2, verbose=1, perplexity=p, n_iter=300)
    tsne_results = tsne.fit_transform(X)
    plt.scatter(tsne_results[:,0], tsne_results[:,1], marker='.',alpha=0.6,c=y_kmeans, cmap='viridis')
    name = 'PCA'+str(X.shape)+'Kmeans'+str(k)+'TSNE'+str(p)
    print(name)
    plt.title(name)
    plt.show()
    #plt.savefig('GeneratedTSNE/'+name)
    plt.close()
    return tsne_results


# In[22]:


#genTSNE(5)


# In[27]:


# for i in range(75,100):
#     genTSNE(i)

