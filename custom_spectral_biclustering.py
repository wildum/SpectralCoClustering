from scipy.sparse import csgraph
from sklearn.cluster import KMeans
import numpy as np
import scipy as sp
import math
from scipy.sparse.linalg import svds
from scipy.sparse import dia_matrix

def rearrange(labels, A, k):
	#matrix of clusters
	clusters = [[] for i in range(k)]

	#feed result with A rows to build clusters
	for i in range(len(A)):
		clusters[labels[i]].append(A[i])

	#merge clusters to create a matrix
	A_rearranged = [item for sublist in clusters for item in sublist]

	return A_rearranged


def custom_spectral_biclustering(A,k):

	n_rows = len(A)
	n_cols = len(A[0])

	row_diag = np.asarray(1.0 / np.sqrt(A.sum(axis=1))).squeeze()
	col_diag = np.asarray(1.0 / np.sqrt(A.sum(axis=0))).squeeze()

	#check if there is no nan value because of the division of the previoius operation
	row_diag = np.where(np.isnan(row_diag), 0, row_diag)
	col_diag = np.where(np.isnan(col_diag), 0, col_diag)

	# #Compute diagonal degree matrix
	D1 = dia_matrix((row_diag, [0]), shape=(n_rows, n_rows))
	D2 = dia_matrix((col_diag, [0]), shape=(n_cols, n_cols))

	A_norm = D1 * A * D2

	num_of_pcs = 1 + int(np.ceil(np.log2(k)))

	#Compute SVD of A_norm
	u, s, v = svds(A_norm, k=num_of_pcs)

	z = np.vstack((row_diag[:, np.newaxis] * u[:, 1:], col_diag[:, np.newaxis] * v.T[:, 1:]))

	#apply k means
	labels = KMeans(n_clusters=k, random_state=0).fit(z).labels_

	row_labels_ = labels[:n_rows]
	column_labels_ = labels[n_rows:]

	#compute biclusters to calculate consensus score
	biclusters_rows = np.vstack(row_labels_ == c for c in range(k))
	biclusters_cols = np.vstack(column_labels_ == c for c in range(k))

	#rearrange rows and cols or data matrix according to the clusters
	A_rearranged = rearrange(row_labels_, A, k)
	A_rearranged = np.transpose(A_rearranged)
	A_rearranged = rearrange(column_labels_, A_rearranged, k)

	return np.transpose(A_rearranged), biclusters_rows, biclusters_cols
	