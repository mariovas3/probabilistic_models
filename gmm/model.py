import numpy as np


class GMM:
    def __init__(self, num_mixtures: int, random_seed: int, max_iter: int = 20, min_change: float=1e-3, cov_reg: float=1e-5):
        assert num_mixtures > 0
        np.random.seed(random_seed)
        self.num_mixtures = num_mixtures
        self.max_iter = max_iter
        self.min_change = min_change
        self.cov_reg = cov_reg
        self.means = None
        self.covariances = None
        self.prior_weights = None
        self.determinants = None
        self.inverses = None
        self.posterior_weights_matrix = None

    def _init_params(self, dimensions: int):
        # uninformative prior;
        self.prior_weights = np.repeat(1 / self.num_mixtures, self.num_mixtures)
        self.means = np.random.standard_normal((self.num_mixtures, dimensions))
        # all positive eigenvalues since covariance matrices are positive definite;
        eigen_vals = np.random.uniform(low=1, high=2.5, size=(self.num_mixtures, dimensions))
        # get orthogonal matrix by generating a Householder reflection matrix;
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
        self.posterior_weights_matrix = R / R.sum(0)
        return R

    def _compute_log_likelihood(self, R: np.ndarray) -> np.float64:
        # recall R[j, i] = p_j * N(x_i | mu_j, sigma_j);
        return np.log(R.sum(0)).sum()

    def _update_prior_weights(self):
        assert self.posterior_weights_matrix is not None
        self.prior_weights = self.posterior_weights_matrix.sum(1) / self.posterior_weights_matrix.shape[1]

    def _update_covariances(self, X: np.ndarray):
        assert self.posterior_weights_matrix is not None
        N, d = X.shape
        a = np.arange(d).reshape((-1, 1))
        mask = a == a.T
        for j in range(self.num_mixtures):
            rj = self.posterior_weights_matrix[j, :].reshape((-1, 1))
            X_tilde = rj * (X - self.means[j, :])
            # here this is a neat trick since in the Deisenroth book
            # the update of an individual covariance matrix is given as a
            # weighted sum of outer products sum_i(rji * (xi - mu_j) @ (xi - mu_j)^T)
            # imagine rji * (xi - mu_j) is the i^th column of X_tilde 
            # and (xi-mu_j)^T is just the i^th row of X - mu_j
            # then the sum of outerproducts is just X_tilde @ (X - mu_j)
            self.covariances[j] = (X_tilde.T @ (X - self.means[j, :])) / rj.sum()
            # a little bit of black magic so that your covariance matrices have positive diagonals;
            self.covariances[j][mask] = self.covariances[j][mask] + self.cov_reg


    def _update_means(self, X: np.ndarray):
        assert self.posterior_weights_matrix is not None
        # since mu_j^T = (X.T @ rj) ^ T / rj.sum()
        # here posterior_weights_matrix is (num_mixtures, N);
        self.means = (self.posterior_weights_matrix @ X) / self.posterior_weights_matrix.sum(1).reshape((-1, 1))

    def fit(self, X: np.ndarray) -> int:
        X_std = (X - X.mean(0)) / (X.std(0) + 1e-6)
        N, d = X_std.shape
        self._init_params(d)
        R = self._compute_many_gaussians(X_std)
        l1 = self._compute_log_likelihood(R)
        l0 = None
        for t in range(self.max_iter):
            try:
                self._update_prior_weights()
                self._update_covariances(X_std)
                self._update_means(X_std)
                self.determinants = [np.linalg.det(sigma) for sigma in self.covariances]
                self.inverses = [np.linalg.inv(sigma) for sigma in self.covariances]
                R = self._compute_many_gaussians(X_std)
                l0 = l1
                l1 = self._compute_log_likelihood(R)
                if abs(l1 - l0) < self.min_change:
                    print(f"fitting converged on pass {t}")
                    return 0
            except np.linalg.LinAlgError:
                print(f"singular covariance matrix detected at pass {t}")
                return 1
        print("max_iter reached")
        return 2

    def predict(self, samples):
        raise NotImplementedError

