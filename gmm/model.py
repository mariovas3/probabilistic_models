import numpy as np


class GMM:
    def __init__(self, num_mixtures: int, random_seed):
        assert num_mixtures > 0
        np.random.seed(random_seed)
        self.num_mixtures = num_mixtures
        self.means = None
        self.covariances = None
        self.prior_weights = None
        self.determinants = None
        self.inverses = None

    def _init_params(self, dimensions: int):
        # uninformative prior;
        self.prior_weights = np.repeat(1 / self.num_mixtures, self.num_mixtures)
        self.means = np.random.standard_normal((self.num_mixtures, dimensions))
        # all positive eigenvalues since covariance matrices are positive definite;
        eigen_vals = np.random.uniform(low=0.1, high=3, size=(self.num_mixtures, dimensions))
        # get orthogonal matrix by generating Householder reflection matrices;
        v = np.random.standard_normal(dimensions).reshape((-1, 1))
        H = np.eye(dimensions) - (2 / (v * v).sum()) * (v @ v.T)  # note the outer product on v;
        # generate the covariance matrices;
        self.covariances = np.array([eigen_vals[i, :] * H @ H.T for i in range(self.num_mixtures)])
        # precomute determinants for gaussian pdfs later to have O(1) time operations with determinants;
        self.determinants = [np.linalg.det(sigma) for sigma in self.covariances]
        # compute inverses once so that when computing gaussian pdf
        # you take O(d^2) time for the inverted_sigma_j @ (x_i - mu_j) where d is the dimensionality of one 
        # observation x_i;
        self.inverses = [np.linalg.inv(sigma) for sigma in self.covariances]

    def _compute_gaussian(self, x: np.ndarray, idx: int) -> float:
        d = self.means.shape[1]
        u = (x - self.means[idx, :])
        assert u.shape == (d,)
        assert type(u @ self.inverses[idx] @ u.T) == np.float64
        # idx is the index of the mixture;
        return self.prior_weights[idx] * ((2 * np.pi) ** (-d/2)) * (self.determinants[idx] ** (-0.5)) * \
                np.exp(-0.5 * u @ self.inverses[idx] @ u)  # precomputed inverse so matrix vector multiply takes O(d^2) time;

    def _compute_responsibilities(self, X: np.ndarray) -> np.ndarray:
        # X should be (N x d)
        # responsibilities matrix should be (num_mixtures x N)
        N, d = X.shape
        R = np.empty((self.num_mixtures, N))
        for i in range(self.num_mixtures):
             # compute gaussians on each row/observation;
             R[i, :] = np.apply_along_axis(self._compute_gaussian, axis=1, idx=i, arr=X)
        R = R / R.sum(0)
        return R

    def fit(self, X: np.ndarray):
        raise NotImplementedError
        _, dimensions = ndarray.shape
        self._init_params(dimensions)
        R = self._compute_responsibilities(X)

    def predict(self, samples):
        raise NotImplementedError

