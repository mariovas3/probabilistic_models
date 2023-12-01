import matplotlib.pyplot as plt

plt.style.use("dark_background")
import pytest

from gmm.model import GMM
from gmm.plotting_funcs import plot_ci


@pytest.mark.e2e
def test_GMM(gmm_data):
    model = GMM(n_components=3)
    model.fit(gmm_data)

    # check __getitem__
    prior_k, mean_k, Cov_k = model[1]
    assert isinstance(prior_k, float)
    assert mean_k.ndim == 1 and len(mean_k) == 2
    assert Cov_k.shape == (2, 2)


def test_plotting(gmm_data):
    model = GMM(n_components=3)
    model.fit(gmm_data)

    classes = model.predict(gmm_data)
    plot_ci(
        "ci_plots.png",
        X=gmm_data,
        predictions=classes,
        gmm_model=model,
        stds=(1, 2, 3),
        facecolor="none",
        edgecolors=("firebrick",) * 3,
    )


def test_warm_start(gmm_data, warm_params):
    model = GMM(3, **warm_params)
    model.fit(gmm_data)

    classes = model.predict(gmm_data)
    plot_ci(
        "ci_plots_warm_start.png",
        X=gmm_data,
        predictions=classes,
        gmm_model=model,
        stds=(1, 2, 3),
        facecolor="none",
        edgecolors=("firebrick",) * 3,
    )


def test_warm_start_bad_example(warm_params_bad_example):
    try:
        model = GMM(3, **warm_params_bad_example)
        raise AssertionError("shouldn't reach this")
    except ValueError as e:
        print(e)
