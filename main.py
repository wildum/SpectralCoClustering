import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import make_biclusters
from sklearn.datasets import samples_generator as sg
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.metrics import consensus_score

import custom_spectralclustering as cs

data, rows, columns = make_biclusters(shape=(30, 30), n_clusters=5, noise=5,shuffle=False, random_state=0)

data = np.absolute(data)

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")

data, row_idx, col_idx = sg._shuffle(data, random_state=0)
plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Shuffled dataset")

model = SpectralCoclustering(n_clusters=5, random_state=0)
model.fit(data)
score = consensus_score(model.biclusters_,
                        (rows[:, row_idx], columns[:, col_idx]))

print("consensus score: {:.3f}".format(score))

custom_spectralclustering_result = cs.custom_spectral_custering(data, 5)
custom_spectralcoclustering_result = cs.custom_spectral_custering(np.transpose(custom_spectralclustering_result), 5)

fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")

plt.matshow(custom_spectralclustering_result, cmap=plt.cm.Blues)
plt.title("Custom spectral clustering")

plt.matshow(custom_spectralcoclustering_result, cmap=plt.cm.Blues)
plt.title("Custom spectral coclustering")

plt.show()