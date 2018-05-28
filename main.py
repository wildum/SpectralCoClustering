import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_biclusters
from sklearn.datasets import samples_generator as sg
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.metrics import consensus_score

import custom_spectral_biclustering as bs


NB_CLUSTERS = 5
SIZE = 300
NOIZE = 5

#build data
data_init, rows, columns = make_biclusters(shape=(SIZE, SIZE), n_clusters=NB_CLUSTERS, noise=NOIZE,shuffle=False, random_state=0)

# we dont want negative data
data_init = np.absolute(data_init)

#shuffle rows and columns!
data, row_idx, col_idx = sg._shuffle(data_init, random_state=0)

######### sklearn algorithm #########
model = SpectralCoclustering(n_clusters=NB_CLUSTERS, random_state=0)
model.fit(data)
score = consensus_score(model.biclusters_,(rows[:, row_idx], columns[:, col_idx]))
fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]
#################################

######### our algorithm #########
custom_spectralcoclustering_result, r, c = bs.custom_spectral_biclustering(data, NB_CLUSTERS)
custom_score = consensus_score((r, c),(rows[:, row_idx], columns[:, col_idx]))
#################################


######### plot results part #########
print("consensus score sklearn: {:.3f}".format(score))
print("consensus score custom: {:.3f}".format(custom_score))

plt.matshow(data_init, cmap=plt.cm.Blues)
plt.title("Original dataset")

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Shuffled dataset")

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("Sklearn spectral coclustering")

plt.matshow(custom_spectralcoclustering_result, cmap=plt.cm.Blues)
plt.title("Our spectral coclustering")

plt.show()
#####################################