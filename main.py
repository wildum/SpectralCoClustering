import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_biclusters
from sklearn.datasets import samples_generator as sg
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.metrics import consensus_score

import custom_spectralclustering as cs


NB_CLUSTERS = 2
SIZE = 30

#build data
data_init, rows, columns = make_biclusters(shape=(SIZE, SIZE), n_clusters=NB_CLUSTERS, noise=5,shuffle=False, random_state=0)

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
custom_spectralclustering_result = cs.custom_spectral_custering(data, NB_CLUSTERS)
custom_spectralcoclustering_result = cs.custom_spectral_custering(np.transpose(custom_spectralclustering_result), NB_CLUSTERS)
#################################


######### plot results part #########
print("consensus score: {:.3f}".format(score))

plt.matshow(data_init, cmap=plt.cm.Blues)
plt.title("Original dataset")

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Shuffled dataset")

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")

plt.matshow(custom_spectralclustering_result, cmap=plt.cm.Blues)
plt.title("Custom spectral clustering")

plt.matshow(custom_spectralcoclustering_result, cmap=plt.cm.Blues)
plt.title("Custom spectral coclustering")

plt.show()
#####################################