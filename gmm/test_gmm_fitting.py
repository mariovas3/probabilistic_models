import numpy as np
from model import GMM


X = np.arange(1, 51).reshape((10, 5))
num_mixtures, random_seed = 3, 42
gmm = GMM(num_mixtures, random_seed)


def test_fit():
    val = gmm.fit(X)
    print(val)

