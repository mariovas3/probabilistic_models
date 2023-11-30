from gmm.model import GMM
import matplotlib.pyplot as plt


def test_GMM(gmm_data):
    model = GMM(n_components=3)
    model.fit(gmm_data)

    # check __getitem__
    prior_k, mean_k, Cov_k = model[1]
    assert isinstance(prior_k, float)
    assert mean_k.ndim == 1 and len(mean_k) == 2
    assert Cov_k.shape == (2, 2)

    # save plot of predictions;
    classes = model.predict(gmm_data)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(gmm_data[:, 0], gmm_data[:, 1], c=classes)
    ax.set_title(
        f"converged: {model.converged}; last_iter: {model.last_iter}"
    )
    fig.tight_layout()
    plt.savefig("bob.png")
    plt.close()
