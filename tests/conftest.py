from pytest import fixture
import numpy as np

np.random.seed(0)


@fixture
def gmm_data():
    means = np.array([[5.0, 5.0], [-5.0, 5.0], [0.0, -5.0]])

    Sigma = np.eye(2)
    data = np.concatenate(
        [
            np.random.multivariate_normal(mean=m, cov=Sigma, size=(100,))
            for m in means
        ]
    )
    return data


@fixture
def gauss_pdf_setup():
    Sigma = np.random.uniform(low=0.5, high=1.5, size=(7, 1))
    Sigma = Sigma @ Sigma.T + 0.3 * np.eye(7)
    mean = np.random.uniform(low=-30, high=30, size=(7,))
    X = np.random.multivariate_normal(mean=mean, cov=Sigma, size=(100,))
    return mean, np.linalg.inv(Sigma), X
