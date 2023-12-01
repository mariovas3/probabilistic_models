## A repo for probabilistic models;

## How to use this:
### Mixture models:
* It is best to build an image using `Dockerfile-gmm` like so from the root of `probabilistic_models`
    ```bash
    DOCKER_BUILDKIT=1 docker build -t mixture-models -f Dockerfile-gmm .
    ```
* Irrelevant files are excluded from the context with `.dockerignore`.
* Once built, to run all tests with `pytest` just type:
    ```bash
    docker container run --rm -v $PWD:/model/probabilistic_models  mixture-models
    ```
    * You can skip end-to-end tests by adding `python3 -m pytest -m 'not e2e'` to the above command.

### Currently has:
* [Gaussian Mixture Model](https://github.com/mariovas3/probabilistic_models/tree/master/gmm)
* [Bayesian Linear Regression](https://github.com/mariovas3/probabilistic_models/tree/master/bayesian_linear_reg)

