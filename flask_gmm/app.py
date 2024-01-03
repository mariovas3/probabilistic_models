import datetime
from collections import namedtuple

data = namedtuple("data", "dates,coords")

import numpy as np
from flask import Flask, render_template, request
from markupsafe import escape

from .model import MODELS_PATH, get_predictions

app = Flask(__name__)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict_form.html")
    coords = np.array(
        str(escape(request.form["coords"])).split(",")[:-1], dtype=float
    )
    return get_predictions(
        model_path=MODELS_PATH / "gmm-2024-01-03-10-06-14.joblib",
        data=data(
            dates=[datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")],
            coords=coords,
        ),
    )
