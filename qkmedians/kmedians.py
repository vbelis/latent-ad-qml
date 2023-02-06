""" Modification of KMedians from github account @hounslow  """

import numpy as np
import util as u


class Kmedians:
    def __init__(self, k, tolerance):
        self.k = k
        self.centroids = []
        self.loss = []
        self.tolerance = tolerance

    def fit(self, data):
        # init centroids
        indexes = np.random.randint(data.shape[0], size=self.k)
        centroids = data[indexes]
        centroids = np.array(centroids, dtype=np.float)

        iteration = 0
        iter_additional = 0
        while True:
            dist = []
            for i in range(data.shape[0]):  # through all training samples
                d = []
                for j in range(
                    self.k
                ):  # distance of each training example to each centroid
                    d.append(u.euclidean_dist(data[i, :], centroids[j, :]))
                dist.append(d)
            dist = np.array(dist)
            dist[np.isnan(dist)] = np.inf
            cluster_labels = np.argmin(dist, axis=1)

            # Update centroids
            new_centroids = []
            for kk in range(self.k):
                if data[cluster_labels == kk].shape[0] > 0:
                    new_centroids.append(
                        np.array(
                            np.median(data[cluster_labels == kk], axis=0),
                            dtype=np.float,
                        )
                    )
                else:
                    new_centroids[kk] = medians[kk]
            new_centroids = np.array(new_centroids)

            self.loss.append(np.linalg.norm(np.subtract(centroids, new_centroids)))

            if np.linalg.norm(np.subtract(centroids, new_centroids)) < self.tolerance:
                centroids = new_centroids.copy()
                iter_additional += 1
                if iter_additional == 5:  # additional 5 epoch to be sure
                    print(f"KMedians converged after {iteration+1} iterations.")
                    break

            centroids = new_centroids
            iteration += 1

        self.centroids = centroids

    def predict(self, data):
        centroids = self.centroids
        dist = u.euclidean_dist(data, centroids)
        dist[np.isnan(dist)] = np.inf
        return np.argmin(dist, axis=1), dist
