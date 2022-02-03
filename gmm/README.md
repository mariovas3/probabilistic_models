## Implementation of Gaussian Mixture Model;

* Initial motivation from [MML book](https://mml-book.github.io/);

### Example of fitted GMM:

![alt text](https://github.com/mariovas3/probabilistic_models/blob/master/gmm/gmm_on_iris.png)


## The fitting procedure:

The objective is to maximise the log-likelihood with respect to the parameters of the $K$ mixtures, $\theta$. However, this is actually a constrained optimisation problem since our prior weights $\{\pi_j\}_{j=1}^{K}$ need to obey the following "rules":

$\forall j:  1\le j\le K$ we need $\pi_j\ge 0$ and $\sum_{j=1}^{K}\pi_j=1$.

Hence, it is really a lagrangian that we have to optimise.

$\mathcal{L}(\theta) = \sum_{i=1}^{N} log\left(\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)\right) + \lambda \left(\sum_{j=1}^{K}\pi_j -1\right)$

where $\theta = \left\{(\pi_j, \mu_j, \Sigma_j)\right\}_{j=1}^{K}$ and $\mathcal{N}(x_i | \mu_j, \Sigma_j)$ is 
the Gaussian density, parameterised by $\mu_j$ and $\Sigma_j$, and evaluated at the $d$-dimensional datapoint $x_i$.

The first order necessary conditions for the parameters:

* $\frac{\partial \mathcal{L}(\theta)}{\pi_j} = 0$

* $\frac{\partial \mathcal{L}(\theta)}{\Sigma_j} = 0$

* $\frac{\partial \mathcal{L}(\theta)}{\mu_j} = 0$

* $\frac{\partial \mathcal{L}(\theta)}{\lambda} = 0$

do not yield closed form solutions.
In fact, the above lead to:

* $\pi_j = \frac{\sum_{i=1}^{N}r_{i,j}}{N}$

* $\Sigma_j=\frac{\sum_{i=1}^{N}r_{i, j}(x_i-\mu_j)(x_i - \mu_j)^T}{\sum_{i=1}^{N}r_{i, j}}$

* $\mu_j^T = \frac{\sum_{i=1}^{N}r_{i, j}x_i^T}{\sum_{i=1}^{N}r_{i,j}}$

* $\lambda=-N$

where 

$r_{i, j} = \frac{\pi_j\mathcal{N}(x_i|\mu_j, \Sigma_j)}{\sum_{j=1}^{K}\pi_j\mathcal{N}(x_i|\mu_j, \Sigma_j)} \ge0$

and

$\sum_{j=1}^{K}r_{i,j}=1$.

The derivation of the above is skipped, but many good texts show how to do it. The reader might wish to consult with Chapter 11 of [MML book](https://mml-book.github.io/) for more details.

These $r$'s are known as the "responsibilities", or the posterior probability of the $i^{th}$ observation "coming" from the $j^{th}$ mixture from a latent variable model's perspective. Since the $r$'s depend on all parameters $\theta$ we cannot obtain a closed form solution for this optimisation problem in contrast to e.g. Least squares linear regression.

The maximisation problem is attempted to be solved, therefore, by an iterative procedure:

1. Initialise parameters $\theta = \left\{(\pi_j, \mu_j, \Sigma_j)\right\}_{j=1}^{K}$
    * In my implementation of the algorithm, I sampled eigenvalues, $\Lambda_j$ for each covariance matrix from a $\mathcal{U}(1, 2.5)$ distribution and then multiplied those on both "sides" by a Householder reflection matrix which has the general form 
    
        $H=I_n - 2\frac{vv^T}{<v, v>}$ for any $n$-dimensional vector $v$. This is an orthogonal matrix which allows me to make use of the eigen decomposition of real symmetric matrices and construct 
        
        $\Sigma_j = H\Lambda_j H^T$.
2. Compute $r_{i,j}$
3. Compute $\pi_j$, $\Sigma_j$, $\mu_j$ using the equations above for all $K$ mixtures with the $r$'s from step 2 (don't update the "responsibilities" unless you are at step 2).
4. Repeat steps 2 and 3 until the change in the log-likelihood becomes "small enough".

This maximisation procedure is an example of the Expectation Maximisation algorithm (popular with other latent variable models).
