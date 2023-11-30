from gmm.model import GMM
import matplotlib as mpl
from gmm.plotting_funcs import plot_ci

mpl.style.use("dark_background")


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
