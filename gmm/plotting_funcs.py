import numpy as np
from math import acos, pi, sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def get_ellipses(gmm_model, ax, n_stds=1, facecolor="none", **kwargs):
    ellipses = []
    for _, m, S in gmm_model:
        eig_pairs = np.linalg.eigh(S)
        angle = (
            acos(eig_pairs.eigenvectors[0, 0])
            / pi
            * 180
            * np.sign(eig_pairs.eigenvectors[0, 1])
        )
        ellipses.append(
            Ellipse(
                (m[0], m[1]),
                # eigvals of Precision are inverse eigvals of Cov;
                width=sqrt(eig_pairs.eigenvalues[0] * n_stds),
                height=sqrt(eig_pairs.eigenvalues[1] * n_stds),
                angle=angle,
                facecolor=facecolor,
                **kwargs,
            )
        )
    return ellipses


def plot_ci(
    png_name, X, predictions, gmm_model, stds, facecolor, edgecolors, **kwargs
):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X[:, 0], X[:, 1], c=predictions, alpha=0.5)
    for n_stds, edgecolor in zip(stds, edgecolors):
        ellipses = get_ellipses(
            gmm_model,
            ax,
            n_stds,
            facecolor=facecolor,
            edgecolor=edgecolor,
            **kwargs,
        )
        for e in ellipses:
            ax.add_patch(e)
    ax.grid()
    ax.set_title(
        f"converged: {gmm_model.converged}; last_iter: {gmm_model.last_iter}"
    )
    fig.tight_layout()
    plt.savefig(png_name)
    plt.close()
