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
        self.R_res = None

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
        self.covariances = [eigen_vals[j, :] * H @ H.T for j in range(self.num_mixtures)]
        # precomute determinants for gaussian pdfs later to have O(1) time operations with determinants;
        self.determinants = [np.linalg.det(sigma) for sigma in self.covariances]
        # compute inverses once so that when computing gaussian pdf
        # you take O(d^2) time for the inverted_sigma_j @ (x_i - mu_j) where d is the dimensionality of one 
        # observation x_i;
        self.inverses = [np.linalg.inv(sigma) for sigma in self.covariances]

    def _compute_gaussian(self, x: np.ndarray, idx: int) -> np.float64:
        d = self.means.shape[1]
        u = (x - self.means[idx, :])
        assert u.shape == (d,)
        # idx is the index of the mixture;
        return self.prior_weights[idx] * ((2 * np.pi) ** (-d/2)) * (self.determinants[idx] ** (-0.5)) * \
                np.exp(-0.5 * u.T @ self.inverses[idx] @ u)  # precomputed inverse so matrix vector multiply takes O(d^2) time;

    def _compute_many_gaussians(self, X: np.ndarray) -> np.ndarray:
        # X should be (N x d)
        # the matrix of gaussians should be (num_mixtures x N)
        # so that it can be used to compute the log-likelihood
        # easily; and I can obtain the responsibilities by
        # dividing by R.sum(0);
        # Note: R[j, i] = p_j * N(x_i | mu_j, sigma_j)
        N, d = X.shape
        R = np.empty((self.num_mixtures, N))
        for j in range(self.num_mixtures):
             # compute Gaussian pdfs multiplied by prior weights
             # on each row/observation;
             R[j, :] = np.apply_along_axis(self._compute_gaussian, axis=1, idx=j, arr=X)
        self.R_res = R / R.sum(0)
        return R

    def _compute_log_likelihood(self, R):
        # recall R[j, i] = p_j * N(x_i | mu_j, sigma_j);
        return np.log(R.sum(0)).sum()

    def _update_prior_weights(self):
        assert self.R_res is not None
        self.prior_weights = self.R_res.sum(1) / self.R_res.shape[1]

    def _update_covariances(self, X):
        assert self.R_res is not None
        for j in range(self.num_mixtures):
            rj = self.R_res[j, :].reshape((-1, 1))
            X_tilde = rj * (X - self.means[j, :])
            # here this is a neat trick since in the Deisenroth book
            # the update of an individual covariance matrix is given as a
            # weighted sum of outer products sum_i(rji * (xi - mu_j) @ (xi - mu_j)^T)
            # imagine rji * (xi - mu_j) is the i^th column of X_tilde 
            # and (xi-mu_j)^T is just the i^th row of X - mu_j
            # then the sum of outerproducts is just X_tilde @ (X - mu_j)
            self.covariances[j] = (X_tilde.T @ (X - self.means[j, :])) / rj.sum()

    def _update_means(self, X):
        assert self.R_res is not None
        # since mu_j^T = (X.T @ rj) ^ T / rj.sum()
        # here R_res is (num_mixtures, N);
        self.means = (self.R_res @ X) / self.R_res.sum(1).reshape((-1, 1))

    def fit(self, X: np.ndarray):
        raise NotImplementedError
        _, dimensions = ndarray.shape
        self._init_params(dimensions)
        R = self._compute_responsibilities(X)

    def predict(self, samples):
        raise NotImplementedError

