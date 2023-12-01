from pytest import fixture
import numpy as np
from math import sqrt

np.random.seed(0)


@fixture(scope="session")
def gmm_data():
    means = np.array([[5.0, 5.0], [-5.0, 5.0], [0.0, -5.0]])

    eigvals = np.array([[2.5, 0], [0, 1.0]])

    eig2 = np.array([[-1, 1], [1, 1]]) / sqrt(2)

    eig3 = np.array([[1, -1], [1, 1]]) / sqrt(2)

    Sigmas = np.array(
        [eigvals, eig2 @ eigvals @ eig2.T, eig3 @ eigvals @ eig3.T],
    )

    data = np.concatenate(
        [
            np.random.multivariate_normal(mean=m, cov=S, size=(100,))
            for m, S in zip(means, Sigmas)
        ]
    )
    return data


@fixture(scope="module")
def warm_params():
    priors = np.ones((3, 1)) / 3
    means = np.array([[4.0, 6.0], [-6.0, 4.0], [-1.0, -6.0]])
    Covs = [np.eye(2) for _ in range(3)]
    return dict(priors=priors, means=means, Covs=Covs)


@fixture(scope="module")
def warm_params_bad_example():
    priors = np.ones((3, 1)) / 3
    return dict(priors=priors, means=None, Covs=None)


@fixture
def gauss_pdf_setup():
    Sigma = np.random.uniform(low=0.5, high=1.5, size=(7, 1))
    Sigma = Sigma @ Sigma.T + 0.3 * np.eye(7)
    mean = np.random.uniform(low=-30, high=30, size=(7,))
    X = np.random.multivariate_normal(mean=mean, cov=Sigma, size=(100,))
    return mean, np.linalg.inv(Sigma), X
