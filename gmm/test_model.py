import numpy as np
from model import GMM

num_mixtures = 2
N, d = 5, 3
gmm = GMM(num_mixtures, 42)
X = np.random.normal(2, 4, size=(N, d))
gmm._init_params(d)
R = gmm._compute_many_gaussians(X)


def test_init_params():
    assert gmm.prior_weights.sum().round(5) == 1
    assert gmm.means.shape == (num_mixtures, d)
    eigvals = np.concatenate([np.linalg.eigvals(sigma) for sigma in gmm.covariances])
    assert all(eigvals > 0)
    assert all(np.array(gmm.determinants) > 0)


def test_compute_many_gaussians():
    assert (R > 0).sum() == R.size
    assert all(gmm.posterior_weights_matrix.sum(0).round(5) == 1)
    assert gmm.posterior_weights_matrix.sum() == N
    assert gmm.posterior_weights_matrix.shape == (num_mixtures, N)


def test_compute_log_likelihood():
    l = 0
    for i in range(N):
        temp = 0
        for j in range(num_mixtures):
            temp += R[j, i]
        l += np.log(temp)
    my_l = gmm._compute_log_likelihood(R)
    assert l.round(5) == my_l.round(5)


def test_update_prior_weights():
    p = np.empty(num_mixtures)
    for j in range(num_mixtures):
        p[j] = gmm.posterior_weights_matrix[j, :].sum() / N
    gmm._update_prior_weights()
    assert (p.round(5) == gmm.prior_weights.round(5)).all()


def test_update_covariances():
    covs = [np.zeros((d, d)) for j in range(num_mixtures)]
    for j in range(num_mixtures):
        rj_sum = gmm.posterior_weights_matrix[j, :].sum()
        for i in range(N):
            u = (X[i, :] - gmm.means[j, :]).reshape((-1, 1))
            covs[j] += gmm.posterior_weights_matrix[j, i] * u @ u.T / rj_sum
    # update my covariances stored in the gmm object;
    gmm._update_covariances(X)
    result = [(covs[i].round(5) == gmm.covariances[i].round(5)).sum()
                for i in range(num_mixtures)]
    assert np.sum(result) == num_mixtures * d * d


def test_update_means():
    mus = np.zeros((num_mixtures, d))
    for j in range(num_mixtures):
        rj_sum = gmm.posterior_weights_matrix[j, :].sum()
        for i in range(N):
            mus[j] += gmm.posterior_weights_matrix[j, i] * X[i, :] / rj_sum
    # update my means stored in the gmm object;
    gmm._update_means(X)
    result = [(mus[j, :].round(5) == gmm.means[j, :].round(5)).sum()
                for j in range(num_mixtures)]
    assert np.sum(result) == num_mixtures * d

