from gmm.utils import log_gauss_pdf
import numpy as np
from scipy.stats import multivariate_normal


def test_log_gauss_pdf(gauss_pdf_setup):
    mean, Precision, X = gauss_pdf_setup
    res1 = log_gauss_pdf(mean, Precision, X)
    res2 = multivariate_normal.logpdf(
        x=X, mean=mean, cov=np.linalg.inv(Precision)
    ).sum()
    assert np.allclose(res1, res2)
