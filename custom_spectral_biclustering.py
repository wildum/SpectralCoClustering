from scipy.sparse import csgraph
from sklearn.cluster import KMeans
import numpy as np
import scipy as sp
import math
from scipy.sparse.linalg import svds

def rearrange(labels, A):
	#matrix of clusters
	clusters = [[] for i in range(k)]

	#feed result with A rows to build clusters
	for i in range(len(A)):
		clusters[labels[i]].append(A[i])

	#merge clusters to create a matrix
	A_rearranged = [item for sublist in clusters for item in sublist]

	return A_rearranged


def custom_spectral_biclustering(A,k):

	# #Compute diagonal degree matrix
	D1 = np.diag(A.sum(axis=0))
	D2 = np.diag(A.sum(axis=1))

	#normalized data matrix --> pb si matrice rect
	D1_sqt_inv = np.absolute(sp.linalg.sqrtm(np.linalg.inv(D1)))
	D2_sqt_inv = np.absolute(sp.linalg.sqrtm(np.linalg.inv(D2)))
	A_norm = D1_sqt_inv.dot(A).dot(D2_sqt_inv)

	num_of_pcs = 1 + int(np.ceil(np.log2(k)))

	#Compute SVD of A_norm
	u, s, v = svds(A_norm, num_of_pcs)
	# s = singular values of A_norm (order desc)
	# u = eigenvectors (column) & v_transpose = eigenvectors(row)

	#pair of eigenvectors corresponding to the k-largest eigenvalue

	z = np.vstack((D1_sqt_inv[:] * u[:, 1:], D2_sqt_inv[:] * v[:, 1:]))
	# z1 = np.transpose(D1_sqt_inv.dot(np.transpose(u[:][1:num_of_pcs+1])))
	# z2 = np.transpose(D2_sqt_inv.dot(np.transpose(v[:][1:num_of_pcs+1])))

	#build matrix of k eigenvectors
	# mat_k_eigvects = np.transpose(np.matrix([z1[0], z2[0]]))

	#apply k means
	labels = KMeans(n_clusters=k, random_state=0).fit(z).labels_

	row_labels_ = labels[:len(A)]
	column_labels_ = labels[len(A):]

	A_rearranged = rearrange(row_labels_, A)
	A_rearranged = np.transpose(A_rearranged)
	A_rearranged = rearrange(column_labels_, A_rearranged)

	return A_rearranged
	