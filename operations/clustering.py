from sklearn.mixture import GaussianMixture
import numpy as np


def cluster_gmm(A, B, num_clusters):

    estimator = GaussianMixture(n_components=num_clusters,
                                covariance_type='tied',  # could be 'spherical', 'diag', 'tied', 'full'
                                max_iter=50,  # number of EM iterations to perform
                                random_state=1  # random seed
                                )

    gmm = estimator.fit(np.hstack((A[:, :3], B[:, :3])))
    labels = gmm.predict(np.hstack((A[:, :3], B[:, :3])))
    return labels
