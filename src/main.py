import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import optax
from jax_tqdm import scan_tqdm, loop_tqdm
from flax.training.train_state import TrainState
from dataclasses import dataclass
from functools import partial

class TrainState(TrainState):
    old_obj_value: jnp.array
    old_grad: jnp.array
    max_epochs: jnp.array

@dataclass
class GD_Output:
    theta: jnp.array
    converged: bool
    step: int

class GradientDescent:

    def __init__(self, objective_fn, initilization_fn, data, optimizer):
        self.objective_fn = objective_fn
        self.initilization_fn = initilization_fn
        self.optimizer = optimizer
        self.data = data

    @partial(jax.jit, static_argnums=(0, ))
    def _initialize_theta(self, rng):
        return self.initilization_fn(rng)

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, args):

        training, data, converged = args

        obj_old = training.old_obj_value
        grad_old = training.old_grad

        obj_value, grad = jax.jit(jax.value_and_grad(self.objective_fn, argnums=0, has_aux=False))(training.params, data)

        converged = jnp.logical_or(
            jnp.logical_or(
                jnp.less_equal(jnp.linalg.norm(obj_value - obj_old), self.obj_threshold),
                jnp.less_equal(jnp.linalg.norm(grad - grad_old), self.grad_threshold)
            ),
            jnp.greater_equal(training.step, training.max_epochs)
        )

        training = training.apply_gradients(grads=grad, old_obj_value=obj_value.squeeze(), old_grad=grad)

        return (training, data, converged)

    @staticmethod
    @partial(jax.jit)
    def cond_fun(args):
        return jnp.logical_not(args[-1])

    @partial(jax.jit, static_argnums=(0, ))
    def _run(self, rng, max_epochs=20_000, obj_threshold=1e-12, grad_threshold=1e-12):

        self.obj_threshold = obj_threshold
        self.grad_threshold = grad_threshold

        theta_init = self._initialize_theta(rng)

        training = TrainState.create(
            apply_fn=self.objective_fn,
            params=theta_init,
            old_obj_value=jnp.array(9999, dtype=jnp.float32),
            old_grad=jnp.ones_like(theta_init, dtype=jnp.float32) * 9999,
            max_epochs=max_epochs,
            tx=self.optimizer
        )

        runner = (training, self.data, jnp.array(False, dtype=jnp.bool_))
        runner = lax.while_loop(self.cond_fun, self._step, runner)

        # res = GD_Output(runner[0].params, runner[-1])
        # res = GD_Output(runner[0].params, runner[-1], runner[0].step)
        res = runner

        return res


if __name__ == "__main__":

    from scipy import stats

    np.random.seed(42)
    x = np.random.normal(size=100)
    np.random.seed(32)
    y = 3 + 2 * x + np.random.normal(size=100)
    data = (np.c_[np.ones_like(x), x], y)

    ols_slope, ols_intercept, _, _, _ = stats.linregress(x, y)

    def objective_fn(theta, data):
        yhat = data[0].dot(theta)
        y = data[1]
        loss = jnp.sum((y - yhat)**2) / y.size
        return loss

    def initilization_fn(rng):
        return jnp.ones(2)

    # optimizer = optax.adam(learning_rate=1e-3, eps=1e-5)
    scheduler = optax.exponential_decay(init_value=1e-3, transition_steps=1000, decay_rate=0.99)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
        optax.scale_by_adam(),  # Use the updates from adam.
        optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
        # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
        optax.scale(-1.0)
    )

    rng = jax.random.PRNGKey(42)
    gd = GradientDescent(objective_fn, initilization_fn, data, optimizer)
    runner = gd._run(rng, obj_threshold=1e-10, grad_threshold=1e-10, max_epochs=10_000)

