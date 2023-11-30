import numpy as np
from scipy.special import softmax
from scipy.stats import multivariate_normal
from base.model_abcs import MixtureModel


class GMM(MixtureModel):
    def __init__(self, n_components, tol=1e-8, maxiter=500, random_state=0):
        self._n_components = n_components
        self._tol, self._maxiter = tol, maxiter
        self._random_state = random_state
        self._converged = False
        self._priors, self._means, self._Covs = None, None, None
        self._last_iter = None
        np.random.seed(random_state)

    @property
    def converged(self):
        return self._converged

    @property
    def last_iter(self):
        return self._last_iter

    @property
    def maxiter(self):
        return self._maxiter

    @property
    def tol(self):
        return self._tol

    @property
    def priors(self):
        return self._priors

    @property
    def means(self):
        return self._means

    @property
    def Covs(self):
        return self._Covs

    @property
    def n_components(self):
        return self._n_components

    def __len__(self):
        return self.n_components

    def __getitem__(self, idx):
        return self.priors[idx], self.means[idx], self.Covs[idx]

    def _Estep(self, X):
        # work in log joint space and then call softmax;
        # KxN - softmax of log pdfs along axis=0;
        return softmax(
            np.concatenate(
                [
                    # gets log-gauss-pdf + log prior;
                    multivariate_normal.logpdf(
                        x=X, mean=self._means[i], cov=self._Covs[i]
                    )[None, :]
                    for i in range(self.n_components)
                ]
            )
            + np.log(self.priors),
            0,
        )

    def _Mstep(self, X, R):
        # X is NxD and R is KxN
        self._priors = R.mean(-1, keepdims=True)
        R = R / R.sum(-1, keepdims=True)
        # is KxD array;
        assert R.shape[-1] == len(X)
        self._means = R @ X
        X = X[None, :, :] - self._means[:, None, :]
        # list of DxD arrays;
        self._Covs = [X[i].T @ (X[i] * R[i][:, None]) for i in range(len(R))]

    def log_lik(self, X):
        marginals = (
            self.priors
            * np.concatenate(
                [
                    multivariate_normal.pdf(
                        x=X, mean=self.means[i], cov=self.Covs[i]
                    )[None, :]
                    for i in range(len(self.priors))
                ]
            )
        ).sum(0)
        return np.log(marginals).mean()

    def random_init(self, X):
        N = len(X)
        # R is of shape KxN
        R = softmax(
            np.random.uniform(low=5, high=10, size=(self.n_components, N)), 0
        )
        self._Mstep(X, R)
        l = self.log_lik(X)
        return l

    def fit(self, X):
        log_lik1, log_lik2 = None, self.random_init(X)
        for it in range(self.maxiter):
            R = self._Estep(X)
            self._Mstep(X, R)
            log_lik1, log_lik2 = log_lik2, self.log_lik(X)
            assert log_lik1 <= log_lik2
            if abs(log_lik1 - log_lik2) < self.tol:
                print(f"fitting reached tol at iter: {it + 1}.")
                self._converged = True
                self._last_iter = it + 1
                break

    def predict(self, X):
        return self.predict_proba(X).argmax(0)

    def predict_proba(self, X):
        if X.ndim == 1:
            X = X[None, :]
        return softmax(
            np.concatenate(
                [
                    multivariate_normal.logpdf(
                        x=X, mean=self.means[i], cov=self.Covs[i]
                    )[None, :]
                    for i in range(self.n_components)
                ]
            )
            + np.log(self.priors),
            0,
        )
