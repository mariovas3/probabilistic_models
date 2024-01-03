import flask_gmm.model as model


def test_fit_predict(gmm_json_data):
    # fit and serialise;
    model.fit_model()

    # predict on data;
    out = model.get_predictions(gmm_json_data)
    assert isinstance(out, dict)
