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

### For the plotting funcs:
I am illustrating the boundaries, such that all points within lie $ \sqrt{k}$ standard deviations away from the mean. 
For example 1, 2 and 3 standard deviations correspond to $ k=1, 4, 9$ respectively.
Since the covariance matrix is positive definite for the Gaussian density to be difined, we use the spectral theorem for real symmetric matrices and get real, positive eigenvalues and orthogonal unit vectors.
Let the quadratic form be defined as

$$
\begin{equation}
	\Delta(x):=(x-\mu)^T \Sigma^{-1}(x-\mu).
\end{equation}
$$

Then for any eigenvector, $v$, with eigenvalue, $\lambda$, of $\Sigma$ we have:

$$
\begin{align}
	v^T \Sigma^{-1} v &= \frac{1}{\lambda}\notag\\
	&\iff x^T \Sigma^{-1} x = 1  && \text{(for $x=\sqrt{\lambda}v$)}.
\end{align}
$$

The equations of the type given above, define an ellipse such that all vectors, $y$ within it are sure to have $\Delta(y+\mu)\le 1$. This ellipse corresponds to the 1 standard deviation boundary. If we want $\sqrt{k}$ standard deviations, we will scale the eigenvectors by $\sqrt{\lambda k}$, such that $\Delta(\sqrt{\lambda k}v + \mu) = k$ for all eigenvectors $v$ with corresponding eigenvalues $\lambda$.

### Example of fitted GMM showing contours for 1, 2, 3 standard deviations:

![alt text](https://github.com/mariovas3/probabilistic_models/blob/master/ci_plots.png)

