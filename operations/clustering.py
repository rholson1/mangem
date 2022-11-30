from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np


def cluster_gmm(A, B, num_clusters):
    """Gaussian Mixture clustering"""

    estimator = GaussianMixture(n_components=num_clusters,
                                covariance_type='tied',  # could be 'spherical', 'diag', 'tied', 'full'
                                max_iter=50,  # number of EM iterations to perform
                                random_state=1  # random seed
                                )

    gmm = estimator.fit(np.hstack((A[:, :3], B[:, :3])))
    labels = gmm.predict(np.hstack((A[:, :3], B[:, :3]))) + 1
    return labels


def cluster_kmeans(A, B, num_clusters):
    """K-Means clustering"""
    data = np.hstack((A[:, :3], B[:, :3]))
    labels = KMeans(init='k-means++', n_clusters=num_clusters, n_init=4, random_state=0).fit_predict(data) + 1
    return labels


def cluster_hierarchical(A, B, num_clusters):
    """Ward hierarchical clustering"""
    data = np.hstack((A[:, :3], B[:, :3]))
    labels = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward').fit_predict(data) + 1
    return labels

