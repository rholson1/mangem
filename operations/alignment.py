# adapted from https://github.com/Oafish1/WR2MD/blob/main/mmd_wrapper/algs/source/maninetcluster/alignment.py
# Align data sets

import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigs, eigsh
from .maninetcluster.util import pairwise_error, block_antidiag
from .maninetcluster.neighborhood import neighbor_graph, laplacian
from .maninetcluster.distance import SquaredL2
from .maninetcluster.correspondence import Correspondence
from .maninetcluster.util import Timer

from sklearn.decomposition import PCA

from .mmd_ma_helper import mmd_ma_helper
from unioncom import UnionCom

from functools import reduce


class UnexpectedAlignmentMethodException(Exception):
    pass


def alignment(method, X, Y, num_dims, neighbors=2):
    """ Interface to alignment methods
    'lma': 'Linear Manifold Alignment',
    'nlma': 'Nonlinear Manifold Alignment (NLMA)',
    'unioncom': 'UnionCom',
    'cca': 'CCA',
    'mmdma': 'MMD-MA'

    """
    # Use PCA to reduce dimension
    USE_PCA = True
    if USE_PCA:
        pca = PCA(.90)  # choose number of principal components to retain 95% of the variance
        #pca = PCA(min(50, X.shape[1]))
        X = pca.fit_transform(X)
        print(pca.n_components_, sum(pca.explained_variance_ratio_))
        #pca = PCA(min(50, Y.shape[1]))
        Y = pca.fit_transform(Y)
        print(pca.n_components_, sum(pca.explained_variance_ratio_))


    if method in ('lma', 'nlma'):
        Wx = neighbor_graph(X, k=neighbors)
        Wy = neighbor_graph(Y, k=neighbors)
        eig_method = 'eigs'
        eig_count = num_dims + 2
        if method == 'lma':
            corr = Correspondence(matrix=np.eye(len(X)))
            proj = ManifoldLinear(X, Y, corr, num_dims, Wx, Wy).project(X, Y, num_dims)
        elif method == 'nlma':
            proj = manifold_nonlinear(X, Y, num_dims, Wx, Wy, eig_method=eig_method, eig_count=eig_count)
    elif method == 'unioncom':
        pass
        uc = UnionCom.UnionCom(output_dim=num_dims, epoch_pd=20, epoch_DNN=10)
        proj = uc.fit_transform(dataset=[X, Y])
    elif method == 'cca':
        corr = Correspondence(matrix=np.eye(len(X)))
        proj = CCA(X, Y, corr, num_dims).project(X, Y, num_dims)
    elif method == 'mmdma':
        pass




        # Perform MMD-MA alignment
        X = X / np.linalg.norm(X, axis=1).reshape(-1, 1)
        Y = Y / np.linalg.norm(Y, axis=1).reshape(-1, 1)
        X = np.matmul(X, X.T)
        Y = np.matmul(Y, Y.T)
        proj, history, ab = mmd_ma_helper(X, Y, p=num_dims, max_iterations=100, training_rate=.0001)  # p=32, training_rate=.00005
    else:
        raise UnexpectedAlignmentMethodException()
    return proj


def unioncom_alignment(X, Y, num_dims=32):

    uc = UnionCom.UnionCom(output_dim=num_dims)
    proj = uc.fit_transform(dataset=[X, Y])  # ??? output_dim
    return proj, {'pairwise_error': pairwise_error(*proj, metric=SquaredL2)}  # ??? correct?


def mmd_ma_alignment(X, Y, num_dims=32):
    # Perform MMD-MA alignment
    X = X / np.linalg.norm(X, axis=1).reshape(-1, 1)
    Y = Y / np.linalg.norm(Y, axis=1).reshape(-1, 1)
    X = np.matmul(X, X.T)
    Y = np.matmul(Y, Y.T)
    proj, history, ab = mmd_ma_helper(X, Y, p=32)
    return proj, {'pairwise_error': pairwise_error(*proj, metric=SquaredL2)}


def cca_alignment(X, Y, num_dims):
    """ CCA Alignment"""
    corr = Correspondence(matrix=np.eye(len(X)))
    proj = CCA(X, Y, corr, num_dims).project(X, Y, num_dims)
    return proj, {'pairwise_error': pairwise_error(*proj, metric=SquaredL2)}


def lma_alignment(X, Y, num_dims):
    """ LMA: Linear Manifold Alignment"""
    corr = Correspondence(matrix=np.eye(len(X)))
    proj = ManifoldLinear(X, Y, corr, num_dims).project(X, Y, num_dims)
    return proj, {'pairwise_error': pairwise_error(*proj, metric=SquaredL2)}


def _linear_decompose(X, Y, L, num_dims, eps):
    Z = sp.linalg.block_diag(X.T, Y.T)
    u, s, _ = np.linalg.svd(np.dot(Z, Z.T))
    Fplus = np.linalg.pinv(np.dot(u, np.diag(np.sqrt(s))))
    T = reduce(np.dot, (Fplus, Z, L, Z.T, Fplus.T))
    L = 0.5 * (T + T.T)
    d1, d2 = X.shape[1], Y.shape[1]
    return _manifold_decompose(L, d1, d2, num_dims, eps, lambda v: np.dot(Fplus.T, v))


class LinearAlignment(object):
    def project(self, X, Y, num_dims=None):
        if num_dims is None:
            return np.dot(X, self.pX), np.dot(Y, self.pY)
        return np.dot(X, self.pX[:, :num_dims]), np.dot(Y, self.pY[:, :num_dims])

    def apply_transform(self, other):
        self.pX = np.dot(self.pX, other.pX)
        self.pY = np.dot(self.pY, other.pY)


class ManifoldLinear(LinearAlignment):
    def __init__(self, X, Y, corr, num_dims, Wx, Wy, mu=0.9, eps=1e-8):
        L = _manifold_setup(Wx, Wy, corr.matrix(), mu)
        self.pX, self.pY = _linear_decompose(X, Y, L, num_dims, eps)


class CCA(LinearAlignment):
    def __init__(self, X, Y, corr, num_dims, eps=1e-8):
        Wxy = corr.matrix()
        L = laplacian(block_antidiag(Wxy, Wxy.T))
        self.pX, self.pY = _linear_decompose(X, Y, L, num_dims, eps)


def nonlinear_manifold_alignment(X, Y, num_dims=2, neighbors=2): #, eig_method='eigs', eig_count=5):
    # e.g.
    # day_ortho = np.genfromtxt("data/maninetcluster/dayOrthoExpr.csv", delimiter=',')[1:, 1:]
    # night_ortho = np.genfromtxt("data/maninetcluster/nightOrthoExpr.csv", delimiter=',')[1:, 1:]
    #
    # X = day_ortho
    # Y = night_ortho

    Wx = neighbor_graph(X, k=neighbors)  # was k=5
    Wy = neighbor_graph(Y, k=neighbors)  # was k=5

    eig_method = 'eigs'
    eig_count = num_dims + 2  # compute a couple of extra eigenvalues in case of 0-valued eigenvalues

    proj = manifold_nonlinear(X, Y, num_dims, Wx, Wy, eig_method=eig_method, eig_count=eig_count)
    return proj, {'pairwise_error': pairwise_error(*proj, metric=SquaredL2)}


def _manifold_setup(Wx, Wy, Wxy, mu):
    Wxy = mu * (Wx.sum() + Wy.sum()) / (2 * Wxy.sum()) * Wxy
    W = np.asarray(np.bmat(((Wx, Wxy), (Wxy.T, Wy))))
    return laplacian(W)


def _manifold_decompose(L, d1, d2, num_dims, eps, vec_func=None, eig_method='eig', eig_count=0):
    if eig_method == 'eig' or vec_func is not None:
        vals, vecs = np.linalg.eig(L)
    elif eig_method == 'eigs':
        vals, vecs = eigs(L, k=eig_count or num_dims + 2, sigma=0, which='LR')
    elif eig_method == 'eigsh':
        vals, vecs = eigsh(L, k=eig_count or num_dims + 2, sigma=0, which='LM')
    else:
        raise Exception('no eigenvalue method specified')

    idx = np.argsort(vals)

    # skip over eigenvalues < eps
    for i in range(len(idx)):
        if vals[idx[i]] >= eps:
            break

    vecs = vecs.real[:, idx[i:]]
    if vec_func:
        vecs = vec_func(vecs)

    # normalize eigenvectors to unit length.  But np.linalg.eig returns eigenvectors of unit length already!
    for i in range(vecs.shape[1]):
        vecs[:, i] /= np.linalg.norm(vecs[:, i])

    map1 = vecs[:d1, :num_dims]
    map2 = vecs[d1:d1 + d2, :num_dims]
    return map1, map2


def manifold_nonlinear(X, Y, num_dims, Wx, Wy, mu=0.9, eps=1e-8, eig_method=None, eig_count=0):
    corr = np.eye(len(X))  # X and Y are perfectly correlated
    L = _manifold_setup(Wx, Wy, corr, mu)
    return _manifold_decompose(L, X.shape[0], Y.shape[0], num_dims, eps, eig_method=eig_method, eig_count=eig_count)


#
# From https://github.com/rsinghlab/SCOT/blob/master/src/evals.py
# Author: Ritambhara Singh, Pinar Demetci, Rebecca Santorella
# 19 February 2020
#
def calc_frac_idx(x1_mat, x2_mat):
    """
    Returns fraction closer than true match for each sample (as an array)
    """
    fracs = []
    x = []
    nsamp = x1_mat.shape[0]
    rank = 0
    for row_idx in range(nsamp):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(x1_mat[row_idx, :], x2_mat)), axis=1))
        true_nbr = euc_dist[row_idx]
        sort_euc_dist = sorted(euc_dist)
        rank = sort_euc_dist.index(true_nbr)
        frac = float(rank) / (nsamp - 1)

        fracs.append(frac)
        x.append(row_idx + 1)

    return fracs, x


def calc_domainAveraged_FOSCTTM(x1_mat, x2_mat):
    """
    Outputs average FOSCTTM measure (averaged over both domains)
    Get the fraction matched for all data points in both directions
    Averages the fractions in both directions for each data point
    """
    fracs1, xs = calc_frac_idx(x1_mat, x2_mat)
    fracs2, xs = calc_frac_idx(x2_mat, x1_mat)
    fracs = []
    for i in range(len(fracs1)):
        fracs.append((fracs1[i] + fracs2[i]) / 2)
    return fracs
