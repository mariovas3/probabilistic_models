import flask_gmm.model as model


def test_fit_predict(gmm_json_data):
    # fit and serialise;
    model.fit_model()

    # predict on data;
    out = model.get_predictions(
        model_path=model.MODELS_PATH / f"gmm-{model.now}.joblib",
        data=gmm_json_data,
    )
    assert isinstance(out, dict)
