import datetime
from collections import namedtuple

data = namedtuple("data", "dates,coords")

import numpy as np
from flask import Flask, render_template, request
from markupsafe import escape

from .model import MODELS_PATH, get_predictions
from .utils import RecencyStore

app = Flask(__name__)

rs = RecencyStore(max_len=3)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict_form.html")
    str_coords = str(escape(request.form["coords"])).split(",")[:-1]
    coords = np.array(str_coords, dtype=float)
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    inputs = data(dates=[now], coords=coords)

    out = get_predictions(
        model_path=MODELS_PATH / "gmm.joblib",
        data=inputs,
    )

    rs.add_item((now, str_coords, out[now]))
    return out


@app.route("/show_predictions")
def show_predictions():
    return render_template("show_predictions.html", db=rs)
