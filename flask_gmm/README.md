# Flask app for Gaussian Mixture Model classification:

## Endpoints:
### The `/predict` endpoint:
The `/predict` endpoint gives a form to be filled. The form is validated on the client side by requiring a regex match. It requires two comma-separated floats up to 3 decimal places precision. This is because in the demo, the GMM works with 2d inputs.
```html
<input type="text" name="coords" id="coords" value="4.,3.123," required pattern="^(-?[0-9]\.([0-9]){0,3},){2}$"/>
```

Once the submit button is clicked, or alternatively `curl` is executed with appropriate flags:
```bash
curl -X POST -F coords=4.,3.123, localhost:5000/predict
```
a json is returned of the type:
```
{
  "2024-01-04-12-02-44": "1"
}
```
denoting when the form data was read and `"1"` stands for the index of the group in which the inputs were classified by the model. So the json is `{"date": "predicted_class"}`.

### The `/show_predictions` endpoint:
The `/show_predictions` endpoint gives a table of recent predictions together with their inputs in reverse chronological order. An example table is shown below:
<img src="./flask_gmm_show_predictions.png" alt="Table of date, input coordinates and predicted class columns."/>

If no POST requests have been given yet, an empty table will be displayed.

The records in the table are managed by the `RecencyStore` class in `flask_gmm/utils.py`. It takes an `int` as its `max_len` parameter in its constructor and once `max_len` is reached, whenever a new record is added, the oldest one is evicted.

The links to the `/predict` and `/show_predictions` endpoints are given as a navigation list on each page.

### Running a demo:
To run a demo on a flask dev server, I suggest you go to the `probabilistic_models` root directory and build the docker image from the `Dockerfile-gmm` using:
```bash
DOCKER_BUILDKIT=1 docker build -t mixture-models-flask -f Dockerfile-gmm .
```

and then again from that directory run the container to test the tests in `probabilistic_models/tests` which will create a `gmm.joblib` file in your `probabilistic_models/flask_gmm` directory which will be read in order to make predictions.
```bash
docker container run --rm -v $PWD:/model/probabilistic_models mixture-models-flask
```

Then you can go to `probabilistic_models/flask_gmm` and run the flask development server:
```bash
flask --app app.py --debug run
```
and you should be able to see the prediction form if you enter
```
localhost:5000/predict
```
in your browser.