from gmm.gmm import GMM
import matplotlib.pyplot as plt


def test_GMM(gmm_data):
    model = GMM(n_components=3)
    model.fit(gmm_data)
    classes = model.predict(gmm_data)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(gmm_data[:, 0], gmm_data[:, 1], c=classes)
    ax.set_title(
        f"converged: {model.converged}; last_iter: {model.last_iter}"
    )
    fig.tight_layout()
    plt.savefig("bob.png")
    plt.close()
