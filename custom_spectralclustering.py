from scipy.sparse import csgraph
from sklearn.cluster import KMeans
import numpy as np

def custom_spectral_custering(A,k):

    #apply laplacian
    L = csgraph.laplacian(A, normed=True)

    #compute eigenvectors and eigenvalues
    eigvals, eigvects = np.linalg.eig(L)

    #take the k lowest eigen values
    eigvals_sorted = sorted(eigvals)[:k]

    #get indexes of these values in the original list
    idx = [eigvals.tolist().index(el) for el in eigvals_sorted]

    #get k eigen vectors corresponding to the k eigen values
    k_eigvects = [eigvects[i] for i in idx]

    #build matrix of k eigenvectors
    mat_k_eigvects = np.transpose(np.asmatrix(k_eigvects))

    R = L.dot(mat_k_eigvects)

    #apply k means
    kmeans = KMeans(n_clusters=k, random_state=0).fit(R)
    kmeans_result = kmeans.fit_predict(R)

    #matrix of clusters
    clusters = [[] for i in range(k)]

    #feed result with A rows to build clusters
    for i in range(len(A)):
        clusters[kmeans_result[i]].append(A[i])
    
    #merge clusters to create a matrix
    result = [item for sublist in clusters for item in sublist]

    return result