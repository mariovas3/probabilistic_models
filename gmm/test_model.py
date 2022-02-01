import numpy as np
from model import GMM

num_mixtures = 2
gmm = GMM(num_mixtures, 42)
X = np.arange(1, 16).reshape(5, 3)
gmm._init_params(X.shape[1])
R = gmm._compute_responsibilities(X)


def test_init_params():
    assert gmm.prior_weights.sum().round(5) == 1
    assert gmm.means.shape == (num_mixtures, X.shape[1])
    eigvals = np.concatenate([np.linalg.eigvals(sigma) for sigma in gmm.covariances])
    assert all(eigvals > 0)
    assert all(np.array(gmm.determinants) > 0)


def test_compute_responsibilities():
    assert all(R.sum(0).round(5) == 1)
    assert R.sum() == X.shape[0]
    assert R.shape == (num_mixtures, X.shape[0])
