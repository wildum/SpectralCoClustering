import numpy as np
import random
import math

#adapted version of https://gist.github.com/iandanforth/5862470 implementation

class Point(object):

    def __init__(self, coords):

        self.coords = coords
        self.n = len(coords)

    def __repr__(self):
        return str(self.coords)

class Cluster(object):

    def __init__(self, points):

        self.points = points
        self.n = points[0].n
        self.centroid = self.calculateCentroid()

    def __repr__(self):
        return str(self.points)

    def update(self, points):
        old_centroid = self.centroid
        self.points = points

        if len(self.points) == 0:
            return 0

        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid)
        return shift

    def calculateCentroid(self):

        numPoints = len(self.points)
        coords = [p.coords for p in self.points]
        unzipped = zip(*coords)
        centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]

        return Point(centroid_coords)

    def getTotalDistance(self):

        sumOfDistances = 0.0
        for p in self.points:
            sumOfDistances += getDistance(p, self.centroid)

        return sumOfDistances


def getDistance(a, b):

    accumulatedDifference = 0.0
    for i in range(a.n):
        squareDifference = pow((a.coords[i]-b.coords[i]), 2)
        accumulatedDifference += squareDifference

    return accumulatedDifference

def calculateError(clusters):

    accumulatedDistances = 0
    num_points = 0
    for cluster in clusters:
        num_points += len(cluster.points)
        accumulatedDistances += cluster.getTotalDistance()

    error = accumulatedDistances / num_points
    return error


def custom_k_means(data, k):

    points = [Point(d) for d in data]

    initial_centroids = random.sample(points, k)
    
    clusters = [Cluster([p]) for p in initial_centroids]

    labels = np.zeros(len(data), dtype=int)

    cutoff = 0.1

    # Loop through the dataset until the clusters stabilize
    loopCounter = 0

    error = 1.0

    #security
    max_iteration = 300

    iterations = 0

    while error != 0.0 and iterations < max_iteration:

        # Create a list of lists to hold the points in each cluster
        lists = [[] for _ in clusters]
        clusterCount = len(clusters)

        for it in range(len(points)):

            p = points[it]

            # Get the distance between that point and the centroid of the first cluster.
            smallest_distance = getDistance(p, clusters[0].centroid)

            # Set the cluster this point belongs to
            clusterIndex = 0

            for i in range(1, clusterCount):
                # calculate the distance of that point to each other cluster's centroid.
                distance = getDistance(p, clusters[i].centroid)
                # If it's closer to that cluster's centroid update what we think the smallest distance is
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i

            # After finding the cluster the smallest distance away, set the point to belong to that cluster
            lists[clusterIndex].append(p)

            #update labels list
            labels[it] = clusterIndex

        # Set our error to zero for this iteration
        error = 0.0

        # For each cluster ...
        for i in range(clusterCount):
            # Calculate how far the centroid moved in this iteration
            shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
            error = max(error, shift)

        # Remove empty clusters
        clusters = [c for c in clusters if len(c.points) != 0]

        iterations+=1

    return labels, clusters


def custom_iterative_kmeans(data, k, iteration_count):

    print("Running K-means",iteration_count,"times to find best clusters ...")

    candidate_labels = []
    errors = []

    for _ in range(iteration_count):
        labels, clusters = custom_k_means(data, k)
        error = calculateError(clusters)
        candidate_labels.append(labels)
        errors.append(error)

    highest_error = max(errors)
    lowest_error = min(errors)

    print("Lowest error found:",lowest_error)
    print("Highest error found:",highest_error)

    ind_of_lowest_error = errors.index(lowest_error)
    best_clusters_labels = candidate_labels[ind_of_lowest_error]

    print("Best K-means candidate: ",(ind_of_lowest_error+1))

    return best_clusters_labels