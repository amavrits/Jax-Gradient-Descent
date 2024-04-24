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

if __name__ == "__main__":

    np.random.seed(42)
    x = np.random.normal(size=100)
    np.random.seed(32)
    y = 3 + 2 * x + np.random.normal(size=100)
    data = (np.c_[np.ones_like(x), x], y)

    res_analytical = stats.linregress(x, y)

    def objective_fn(theta, data):
        yhat = data[0].dot(theta)
        y = data[1]
        loss = jnp.sum((y - yhat)**2) / y.size
        return loss

    def initilization_fn(rng):
        return jax.random.normal(rng, shape=(2, ))

    scheduler = optax.exponential_decay(init_value=1e-3, transition_steps=1000, decay_rate=0.99)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
        optax.scale_by_adam(),  # Use the updates from adam.
        optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
        optax.scale(-1.0) # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
    )

    rng = jax.random.PRNGKey(42)
    rng_fit, rng_boot = jax.random.split(rng, 2)
    gd = GD(objective_fn, initilization_fn, data, optimizer, obj_threshold=1e-3, grad_threshold=1e-3, max_epochs=2_000)
    start_time = time()
    res, metrics = gd.fit(rng_fit, n_inits=100)
    res_boot, metrics_boot = gd._bootstrap(rng_boot, n_inits=1, n_boot=1_000)
    end_time = time()
    print(end_time - start_time)


    fig, axs = plt.subplots(1, 3, figsize=(8, 5))

    axs[0].scatter(res_boot.theta[:, 0].mean(axis=0), ["GD"], color="b", label="Mean")
    axs[0].plot(np.quantile(res_boot.theta[:, 0], q=[0.05, 0.95]), ["GD", "GD"], color="b", label="90% CI")
    axs[0].scatter(res.theta[0], ["GD"], color="r", marker="x", label="Optimal fit")
    axs[0].scatter([res_analytical.intercept], ["OLS"], color="b")
    axs[0].plot(res_analytical.intercept+stats.norm.ppf([0.05, 0.95])*res_analytical.intercept_stderr,
                ["OLS", "OLS"], color="b")
    axs[0].scatter([res_analytical.intercept], ["OLS"], color="r", marker="x")
    axs[0].set_xlabel("Intercept", fontsize=12)
    axs[0].set_ylabel("Method", fontsize=12)
    axs[0].legend(fontsize=10)

    axs[1].scatter(res_boot.theta[:, 1].mean(axis=0), ["GD"], color="b", label="GD Bootstrap mean")
    axs[1].plot(np.quantile(res_boot.theta[:, 1], q=[0.05, 0.95]), ["GD", "GD"], color="b", label="GD 90% CI")
    axs[1].scatter(res.theta[1], ["GD"], color="r", marker="x", label="GD fit")
    axs[1].scatter([res_analytical.slope], ["OLS"], color="b", label="Analytical mean")
    axs[1].plot(res_analytical.slope+stats.norm.ppf([0.05, 0.95])*res_analytical.stderr, ["OLS", "OLS"], color="b", label="Analytical 90% CI")
    axs[1].scatter([res_analytical.slope], ["OLS"], color="r", marker="x", label="Analytical fit")
    axs[1].set_xlabel("Slope", fontsize=12)
    axs[1].set_ylabel("Method", fontsize=12)

    axs[2].plot(metrics["epoch"]+1, metrics["objective_value"], color="r", label="Fit")
    axs[2].plot(metrics["epoch"]+1, metrics_boot["objective_value"].mean(axis=0), color="b", label="Bootstrap mean")
    axs[2].fill_between(metrics["epoch"]+1,
                        np.quantile(metrics_boot["objective_value"], 0.05, axis=0),
                        np.quantile(metrics_boot["objective_value"], 0.95, axis=0),
                        color="b", alpha=0.3, label="Bootstrap 90% CI")
    axs[2].set_xlabel("Epoch", fontsize=12)
    axs[2].set_ylabel("Objective value", fontsize=12)
    axs[2].legend(fontsize=10)

    plt.tight_layout()

