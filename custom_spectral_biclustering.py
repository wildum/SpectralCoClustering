from scipy.sparse import csgraph
from sklearn.cluster import KMeans
import numpy as np
import scipy as sp

def custom_spectral_biclustering(A,k):
	#Compute diagonal degree matrix
		# initialize list to hold values of degree
	degree_col = np.zeros(len(A)) 
	degree_row = np.zeros(len(A))
	# calculate the sums along rows and sum along columns
	colsum = A.sum(axis=0)
	rowsum = A.sum(axis=1)

	# loop through matrix and add up all degree connections
	for j in range(0, len(A)):
	    degree_col[j] = colsum[0,j]
	    degree_row[j] = rowsum[j,0]
	# construct matrix (column & row) 
	C = np.diag(degree_col)
	R = np.diag(degree_row)

	#normalized data matrix --> pb si matrice rect
	R_sqt_inv = sp.linalg.sqrtm(np.linalg.inv(R))
	C_sqt_inv = sp.linalg.sqrtm(np.linalg.inv(C))
	A_norm = R_sqt_inv.dot(A).dot(C_sqt_inv)

	#Compute SVD of A_norm
	u, s, v = np.linalg.svd(A_norm)
	# s = singular values of A_norm (order desc)
	# u = eigenvectors (column) & v_transpose = eigenvectors(row)
	
	#pair of eigenvectors corresponding to the k-largest eigenvalue
	