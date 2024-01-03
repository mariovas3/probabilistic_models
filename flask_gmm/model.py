import datetime
import sys
from collections import namedtuple
from math import sqrt
from pathlib import Path
from typing import Dict

MODELS_PATH = Path(__file__).absolute().parent
now = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
sys.path.append(str(MODELS_PATH.parent))

import joblib
import numpy as np

np.random.seed(0)

from gmm.model import GMM
from gmm.plotting_funcs import plot_ci


def gmm_data():
    means = np.array([[5.0, 5.0], [-5.0, 5.0], [0.0, -5.0]])

    eigvals = np.array([[2.5, 0], [0, 1.0]])

    eig2 = np.array([[-1, 1], [1, 1]]) / sqrt(2)

    eig3 = np.array([[1, -1], [1, 1]]) / sqrt(2)

    Sigmas = np.array(
        [eigvals, eig2 @ eigvals @ eig2.T, eig3 @ eigvals @ eig3.T],
    )

    data = np.concatenate(
        [
            np.random.multivariate_normal(mean=m, cov=S, size=(100,))
            for m, S in zip(means, Sigmas)
        ]
    )
    return data


def fit_model(verbose=False):
    # get data;
    data = gmm_data()

    # fit model;
    model = GMM(n_components=3)
    model.fit(data)

    # save plot of gmm fit;
    if verbose:
        classes = model.predict(data)
        plot_ci(
            MODELS_PATH / f"ci_plots-{now}.png",
            X=data,
            predictions=classes,
            gmm_model=model,
            stds=(1, 2, 3),
            facecolor="none",
            edgecolors=("firebrick",) * 3,
        )

    # persist model;
    joblib.dump(model, MODELS_PATH / f"gmm-{now}.joblib")


def get_predictions(model_path: Path, data: namedtuple) -> Dict[str, float]:
    if not model_path.exists():
        raise FileNotFoundError

    model = joblib.load(model_path)

    predictions = model.predict(data.coords)
    return {ds: str(group) for ds, group in zip(data.dates, predictions)}
