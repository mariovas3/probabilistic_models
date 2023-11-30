import numpy as np
from math import log, pi


# x.T @ A @ x = sum(x_i * x_j * A_{ij}) = Tr(A @ x @ x.T)
# X.T @ X = sum(x_n @ x_n.T)
# log-kernel is Tr(A @ X.T @ X) takes O(D^3 + N * D^2) time;
# using Hadamard (A * (X.T @ X)).sum() takes O((N + 2) * D^2) time;
def log_gauss_ker(mean, Precision, X):
    X = X - mean.reshape(1, -1)
    assert len(Precision) == X.shape[-1]
    return -(Precision * (X.T @ X)).sum() * 0.5


def log_gauss_pdf(mean, Precision, X):
    N, D = X.shape
    return (
        -log(2 * pi) * (N * D / 2)
        + log(np.linalg.det(Precision)) * (N / 2)
        + log_gauss_ker(mean, Precision, X)
    )
