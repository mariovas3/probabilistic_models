import numpy as np
from typing import Iterable


class GMM:
    """
    Gaussian Mixture Models class.

    Attributes:
        num_mixtures (int): positive number of mixtures in the model;
        max_iter (int): maximum number of iterations in the fit() method (greater than 1);
        min_change (float): non-negative float; stop the fitting funciton if the change in the
            log-likelihood from the previous iteration and the current iteration is less than min_change;
        cov_reg (float): regularisation positive term to be added to the diagonals of covariance matrices
            for maintaining positive definite matrices (this one I looked up in the sklearn documentation
            for sklearn.mixture.GaussianMixture; for more info check the documentation for reg_covar here:
            https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
        actual_means (np.ndarray): the means of the mixtures for the non-standardised data;
        actual_covariances (Iterable[np.ndarray]): the covariances for the non-standardised data;
    """
    def __init__(self, num_mixtures: int, random_seed: int, max_iter: int = 20, min_change: float=1e-3, cov_reg: float=1e-5):
        """
        Instantiate object of GMM type;

        Args:
            num_mixtures (int)
            random_seed (int): random seed to be passed to the numpy random number generator;
            max_iter (int, optional)
            min_change (float, optional)
            cov_reg (float, optional)
        """
        assert num_mixtures > 0 and max_iter > 1 and min_change >= 0 and cov_reg >= 0
        np.random.seed(random_seed)
        self.num_mixtures = num_mixtures
        self.max_iter = max_iter
        self.min_change = min_change
        self.cov_reg = cov_reg
        self._means = None
        self._covariances = None
        self._prior_weights = None
        self._determinants = None
        self._inverses = None
        self._posterior_weights_matrix = None
        self.actual_means = None
        self.actual_covariances = None

    def _init_params(self, dimensions: int):
        """
        Initialise parameters of mixtures;

        Args:
            dimensions (int): dimension of a single data point x_i;

        Note:
            Priors are initialised to be non-negative, equal and sum to 1;
            Means are sampled from a standard normal distribution
                and are stored as a (num_mixtures, dimensions) np.ndarray.
            Covariance matrices must be symmetric positive definite. I have generated
                positive eigenvalues from U(1, 2.5) distribution and stored them as a
                (num_mixtures, dimensions) np.ndarray. As an orthogonal matrix I have used
                a Householder reflection matrix given by H = I - (2 * <v, v>) * v @ v.T
                where the values in v are sampled from a standard normal distribution;
                Finally each covariance matrix is given by Sigma = H @ eigen_values @ H.T;
        """
        # uninformative prior;
        self._prior_weights = np.repeat(1 / self.num_mixtures, self.num_mixtures)
        self._means = np.random.standard_normal((self.num_mixtures, dimensions))
        # all positive eigenvalues since covariance matrices are positive definite;
        eigen_vals = np.random.uniform(low=1, high=2.5, size=(self.num_mixtures, dimensions))
        # get orthogonal matrix by generating a Householder reflection matrix;
        v = np.random.standard_normal(dimensions).reshape((-1, 1))
        H = np.eye(dimensions) - (2 / (v * v).sum()) * (v @ v.T)  # note the outer product on v;
        # generate the covariance matrices;
        self._covariances = [eigen_vals[j, :] * H @ H.T for j in range(self.num_mixtures)]
        # precomute determinants for gaussian pdfs later to have O(1) time operations with determinants;
        self._determinants = [np.linalg.det(sigma) for sigma in self._covariances]
        # compute inverses once so that when computing gaussian pdf
        # you take O(d^2) time for the inverted_sigma_j @ (x_i - mu_j) where d is the dimensionality of one 
        # observation x_i;
        self._inverses = [np.linalg.inv(sigma) for sigma in self._covariances]

    def _compute_gaussian(self, x: np.ndarray, idx: int) -> np.float64:
        """
        Multiply prior_weight by Gaussian density evaluated at vector x;

        Args:
            x (np.ndarray): single observation (d-dimensional vector);
            idx (int): index of mixture model;

        Returns:
            np.float64
            output = p_idx * N(x | mu_idx, sigma_idx)
        """
        d = self._means.shape[1]
        u = (x - self._means[idx, :])
        assert u.shape == (d,)
        # idx is the index of the mixture;
        return self._prior_weights[idx] * ((2 * np.pi) ** (-d/2)) * (self._determinants[idx] ** (-0.5)) * \
                np.exp(-0.5 * u.T @ self._inverses[idx] @ u)  # precomputed inverse so matrix vector multiply takes O(d^2) time;

    def _compute_many_gaussians(self, X: np.ndarray) -> np.ndarray:
        """
        Compute a matrix with prior-weights-Gaussian-density products;

        Args:
            X (np.ndarray): a matrix;

        Returns:
            np.ndarray
            A matrix output with output[j, i] = p_j * N(x_i | mu_j, sigma_j)
        """
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
        self._posterior_weights_matrix = R / R.sum(0)
        return R

    def _compute_posteriors(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posteriors/responsibilities;

        Args:
            X (np.ndarray): dataset as a matrix;

        Returns:
            np.ndarray
            A (num_mixtures x number_samples) matrix with
            output[j, i] = p_j * N(x_i | mu_j, sigma_j) / sum_k(p_k * N(x_i | mu_k, sigma_k))
        """
        N, d = X.shape
        R = np.empty((self.num_mixtures, N))
        for j in range(self.num_mixtures):
            # the applied _compute_gaussian function actually computes p_j * N(x_i|mu_j, sigma_j)
            # rather than just N(x_i|mu_j, sigma_j);
            R[j, :] = np.apply_along_axis(self._compute_gaussian, axis=1, idx=j, arr=X)
        return R / R.sum(0)

    def _compute_log_likelihood(self, R: np.ndarray) -> np.float64:
        """
        Compute log-likelihood.

        Args:
            R (np.ndarray): matrix with R[j, i] = p_j * N(x_i | mu_j, sigma_j)

        Returns:
            np.float64
        """
        # recall R[j, i] = p_j * N(x_i | mu_j, sigma_j);
        return np.log(R.sum(0)).sum()

    def _update_prior_weights(self):
        """
        Update prior_weights;

        Note:
            Follows the procedure p_j = sum_i(posterior[j, i]) / N
        """
        assert self._posterior_weights_matrix is not None
        self._prior_weights = self._posterior_weights_matrix.sum(1) / self._posterior_weights_matrix.shape[1]

    def _update_covariances(self, X: np.ndarray):
        """
        Update covariances;

        Args:
            X (np.ndarray): dataset as a matrix;

        Note:
            Instead of summing outer products, I create a matrix X_tilde
            with X_tilde^T[:, i] = posterior[j, i] * (x_i - mu_j)
            so X_tilde^T @ (X - mu_j) = sum_i(posterior[j, i] * (x_i - mu_j) @ (x_i - mu_j)^T)
            then sigma_j = X_tilde^T @ (X-mu_j) / sum_i(posterior[j, i])
            This is basically done to avoid slow for loops over N, otherwise the time
            complexity for a single covariance matrix update is O(N * d^2) for both approaches.

            At the end I add a positive constant to the diagonals of covariance matrices
            as in https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html;
        """
        assert self._posterior_weights_matrix is not None
        N, d = X.shape
        a = np.arange(d).reshape((-1, 1))
        mask = a == a.T
        for j in range(self.num_mixtures):
            rj = self._posterior_weights_matrix[j, :].reshape((-1, 1))
            X_tilde = rj * (X - self._means[j, :])
            # here this is a neat trick since in the Deisenroth book
            # the update of an individual covariance matrix is given as a
            # weighted sum of outer products sum_i(rji * (xi - mu_j) @ (xi - mu_j)^T) / sum_i(rji)
            # imagine rji * (xi - mu_j) is the i^th column of X_tilde^T
            # and (xi-mu_j)^T is just the i^th row of X - mu_j
            # then the sum of outerproducts is just X_tilde^T @ (X - mu_j)
            self._covariances[j] = (X_tilde.T @ (X - self._means[j, :])) / rj.sum()
            # a little bit of magic so that your covariance matrices have positive diagonals;
            self._covariances[j][mask] = self._covariances[j][mask] + self.cov_reg

    def _update_means(self, X: np.ndarray):
        """
        Update means;

        Args:
            X (np.ndarray): dataset in the form of a matrix;
        """
        assert self._posterior_weights_matrix is not None
        # since mu_j^T = (X.T @ rj) ^ T / rj.sum()
        # here posterior_weights_matrix is (num_mixtures, N);
        self._means = (self._posterior_weights_matrix @ X) / self._posterior_weights_matrix.sum(1).reshape((-1, 1))

    def _compute_actual_means_and_covariances(self, overall_means: np.ndarray, overall_stds: np.ndarray, overall_stds_mat: np.ndarray):
        """
        Calculate back the means and covariances for the non-standardised datasets;

        Args:
            overall_means (np.ndarray): d-dimensional vector with means of non-standardised data;
            overall_stds (np.ndarray): d-dimensional vector with standard deviations of non-standardised data;
            overall_stds_mat (np.ndarray): outer product of overall_stds;

        Note:
            The fit method estimates means and covariances after the input is standardised by
            subtracting overall_means and dividing by overall_stds using broadcasting.

            To "go" to the "non-standardised" means I need to multiply _means by overall_stds and
            then add overall_means;

            For the covariances, constants don't matter so I only need to multiply each covariance
            by the outer product of overall_stds elementwise;
        """
        self.actual_means = self._means * overall_stds + overall_means
        self.actual_covariances = [sigma * overall_stds_mat for sigma in self._covariances]

    def get_means(self) -> np.ndarray:
        """
        Get means for non-standardised data;

        Returns:
            np.ndarray
            matrix of size (num_mixtures, dimensions)
        """
        return self.actual_means

    def get_covariances(self) -> Iterable[np.ndarray]:
        """
        Get covariance matrices for non-standardised data;

        Returns:
            Iterable[np.ndarray]
            List of size (dimension, dimension) matrices;
        """
        return self.actual_covariances

    def get_priors(self) -> np.ndarray:
        """
        Get priors calculated by the fit method;

        Returns:
            np.ndarray
            Vector of length num_mixtures;
        """
        return self._prior_weights

    def fit(self, X: np.ndarray) -> int:
        """
        Estimate priors, means and covariances for the num_mixtures mixtures;

        Args:
            X (np.ndarray): raw dataset as matrix;

        Returns:
            int

        Note:
            The exit status can be
            {
                0: "algorithm converged because the absolute difference in log-likelihood was less than min_change",
                1: "singular covariance matrix encountered during fitting gmm",
                2: "max_iter reached"
            }
            Exit status 1 hasn't occurred after introducing standardising and adding cov_reg as in
            https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
            to the fitting procedure; Both these "tricks" have contributed for numerical stability
            of determinants and inverses.
        """
        # first standardise for safer computations, otherwise
        # determinants might be unstable, depending on your data;
        overall_means = X.mean(0)
        overall_stds = X.std(0)
        # saving means and stds for computing relevant means and covariances
        # for non-standardised data;
        overall_stds_mat = overall_stds.reshape((-1, 1)) * overall_stds  # outer product;
        X_std = (X - overall_means) / (overall_stds + 1e-6)
        N, d = X_std.shape
        self._init_params(d)
        R = self._compute_many_gaussians(X_std)
        l1 = self._compute_log_likelihood(R)
        l0 = None
        for t in range(self.max_iter):
            try:
                self._update_prior_weights()
                self._update_means(X_std)  # update means before covariances;
                self._update_covariances(X_std)
                self._determinants = [np.linalg.det(sigma) for sigma in self._covariances]
                self._inverses = [np.linalg.inv(sigma) for sigma in self._covariances]
                R = self._compute_many_gaussians(X_std)
                l0 = l1
                l1 = self._compute_log_likelihood(R)
                if abs(l1 - l0) < self.min_change:
                    self._compute_actual_means_and_covariances(overall_means, overall_stds, overall_stds_mat)
                    print(f"fitting converged on pass {t}")
                    return 0
            except np.linalg.LinAlgError:
                print(f"singular covariance matrix detected at pass {t}")
                return 1
        self._compute_actual_means_and_covariances(overall_means, overall_stds, overall_stds_mat)
        print("max_iter reached")
        return 2

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute posteriors/responsibilities of each mixture to each sample;

        Args:
            samples (np.ndarray): matrix with same number of columns as the matrix passed to the fit method;

        Returns:
            np.ndarray
            Matrix of posterior probabilities of size (num_mixtures, sample_size);
        """
        samples = (samples - samples.mean(0)) / (samples.std(0) + 1e-6)
        posteriors = self._compute_posteriors(samples)
        return posteriors

