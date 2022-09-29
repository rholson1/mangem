# Align data sets
import numpy as np
from scipy.sparse.linalg import eigs
from .maninetcluster.util import pairwise_error
from .maninetcluster.neighborhood import neighbor_graph, laplacian
from .maninetcluster.distance import SquaredL2
from .maninetcluster.util import Timer


def nonlinear_manifold_alignment(X, Y, num_dims=2, eig_method='eigs', eig_count=5):
    # e.g.
    # day_ortho = np.genfromtxt("data/maninetcluster/dayOrthoExpr.csv", delimiter=',')[1:, 1:]
    # night_ortho = np.genfromtxt("data/maninetcluster/nightOrthoExpr.csv", delimiter=',')[1:, 1:]
    #
    # X = day_ortho
    # Y = night_ortho

    Wx = neighbor_graph(X, k=2)  # was k=5
    Wy = neighbor_graph(Y, k=2)  # was k=5
    # num_dims = 2

    proj = manifold_nonlinear(X, Y, num_dims, Wx, Wy, eig_method=eig_method, eig_count=eig_count)
    return proj, {'pairwise_error': pairwise_error(*proj, metric=SquaredL2)}


def _manifold_setup(Wx, Wy, Wxy, mu):
    Wxy = mu * (Wx.sum() + Wy.sum()) / (2 * Wxy.sum()) * Wxy
    W = np.asarray(np.bmat(((Wx, Wxy), (Wxy.T, Wy))))
    return laplacian(W)


def _manifold_decompose(L, d1, d2, num_dims, eps, vec_func=None, eig_method=None, eig_count=0):
    if eig_method == 'eig':
        vals, vecs = np.linalg.eig(L)
    elif eig_method == 'eigs':
        vals, vecs = eigs(L, k=eig_count, sigma=0, which='LR')  # assume no more than 2 0-valued eigenvalues


    # with Timer('eig'):
    #     vals, vecs = np.linalg.eig(L)
    # with Timer('eigs'):
    #     vals2, vecs2 = eigs(L, k=num_dims + 2)  # assume no more than 2 0-valued eigenvalues

    idx = np.argsort(vals)

    # skip over eigenvalues < eps
    for i in range(len(idx)):
        if vals[idx[i]] >= eps:
            break
    vecs = vecs.real[:, idx[i:]]
    if vec_func:
        vecs = vec_func(vecs)

    # normalize eigenvectors to unit length.  But np.linagl.eig returns eigenvectors of unit length already!
    for i in range(vecs.shape[1]):
        vecs[:, i] /= np.linalg.norm(vecs[:, i])

    map1 = vecs[:d1, :num_dims]
    map2 = vecs[d1:d1 + d2, :num_dims]
    return map1, map2


def manifold_nonlinear(X, Y, num_dims, Wx, Wy, mu=0.9, eps=1e-8, eig_method=None, eig_count=0):
    corr = np.eye(len(X))  # X and Y are perfectly correlated
    L = _manifold_setup(Wx, Wy, corr, mu)
    return _manifold_decompose(L, X.shape[0], Y.shape[0], num_dims, eps, eig_method=eig_method, eig_count=eig_count)
