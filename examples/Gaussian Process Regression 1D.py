from src.main import GradientDescent as GD
import numpy as np
import jax
import jax.numpy as jnp
import optax
from scipy import stats
from time import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def objective_fn(theta, data):
    d, y = data[0], data[1]
    K = kernel_fn(d, theta)
    L = jnp.linalg.cholesky(K, upper=False)
    alpha = jnp.linalg.inv(L.T).dot(jnp.linalg.inv(L).dot(y))
    # alpha = jax.scipy.linalg.solve(L.dot(L.T), y)
    loglike = - 0.5 * y.T.dot(alpha) - jnp.sum(jnp.log(jnp.diag(L))) # - constant
    return -loglike

def initilization_fn(rng):
    return jax.random.uniform(rng, minval=0, maxval=2, shape=(1,))

def kernel_fn(d, ls):
    return jnp.exp(-d**2/(2*ls)) + jnp.eye(d.shape[0]) * 1e-6


if __name__ == "__main__":

    n = 100
    n_dim = 1
    mean = 0
    sigma = 1
    sigma_noise = 0
    ls_true = np.ones(n_dim)

    np.random.seed(42)
    x = np.random.uniform(size=(n, n_dim))
    np.random.seed(32)
    d = np.abs(x - x.T)
    K_true = np.asarray(kernel_fn(d, ls_true))
    np.random.seed(33)
    y = mean + np.sqrt(sigma**2+sigma_noise**2) * np.linalg.cholesky(K_true).dot(np.random.randn(n))
    data = (jnp.asarray(d), jnp.asarray(y))
    true_loglike = -objective_fn(ls_true, data)

    scheduler = optax.exponential_decay(init_value=1e-3, transition_steps=1000, decay_rate=0.99)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
        optax.scale_by_adam(),  # Use the updates from adam.
        optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
        optax.scale(-1.0)  # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
    )

    rng = jax.random.PRNGKey(42)
    rng_fit, rng_boot = jax.random.split(rng, 2)
    gd = GD(objective_fn, initilization_fn, data, optimizer, obj_threshold=1e-3, grad_threshold=1e-3, max_epochs=5_000)
    start_time = time()
    res, metrics = gd.fit(rng_fit, n_inits=1)
    end_time = time()
    print(end_time - start_time)


    ls_grid = jnp.linspace(0.01, 5, 1_000).reshape(-1, 1)
    obj, grad = jax.jit(jax.vmap(jax.value_and_grad(objective_fn, argnums=0, has_aux=False), in_axes=(0, None)))(ls_grid, data)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(ls_grid, -obj, c="b")
    ax2.plot(ls_grid, grad, c="r")
    ax.axvline(ls_true, color="k", label="True")
    ax.axvline(res.theta, color="g", label="MLE")
    ax.legend(fontsize=10)
    ax.set_xlabel("Lengthscale", fontsize=12)
    ax.set_xlim(0, np.maximum(res.theta, ls_grid.max()))
    ax2.set_xlim(0, np.maximum(res.theta, ls_grid.max()))
    ax.set_ylabel("LML", fontsize=12)
    ax2.set_ylabel(r"$\nabla$LML", fontsize=12)
    ax.yaxis.label.set_color("b")
    ax2.yaxis.label.set_color("r")

