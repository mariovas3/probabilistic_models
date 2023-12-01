## Implementation of Gaussian Mixture Model;
* Currently initialise with random responsibilities, $$p(s_n=k|x_n, \theta),$$ and then immediately call `_Mstep` to get initial values of parameters - done using:
	```Python
	R = softmax(
			np.random.uniform(
					low=5, 
					high=10, 
					size=(self.n_components, N)
			), 
			0
		)
	```
* From tag `v1` you can also pass parameters as initial point to warm start the model.

### Example of fitted GMM:

![alt text](https://github.com/mariovas3/probabilistic_models/blob/master/ci_plots.png)

