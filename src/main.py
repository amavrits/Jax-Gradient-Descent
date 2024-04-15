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
    obj_keeper: jnp.float32
    grads_keeper: jnp.array
    max_epochs: jnp.int32
    converged: jnp.bool_

@dataclass
class GD_Output:
    theta: np.array
    converged: bool
    step: int
    objective_value: float
    grads: np.array

class GradientDescent:

    def __init__(self, objective_fn, initilization_fn, data, optimizer):
        self.objective_fn = objective_fn
        self.initilization_fn = initilization_fn
        self.optimizer = optimizer
        self.data = data

    @staticmethod
    def _collect_output(res):
        return GD_Output(
            theta=np.asarray(res.params),
            converged=res.converged.item(),
            step=res.step.item(),
            objective_value=res.obj_keeper.item(),
            grads=np.asarray(res.grads_keeper)
        )

    @staticmethod
    def _clean_results(res):
        idx_fittest = jnp.argmin(res.obj_keeper)
        res = res.replace(
            params=res.params[idx_fittest],
            converged=res.converged[idx_fittest],
            step=res.step[idx_fittest],
            obj_keeper=res.obj_keeper[idx_fittest],
            grads_keeper=res.grads_keeper[idx_fittest]
        )
        return res

    @partial(jax.jit, static_argnums=(0, ))
    def _initialize_theta(self, rng):
        return self.initilization_fn(rng)

    @partial(jax.jit, static_argnums=(0,))
    def _check_convergence(self, training, obj_value, grads):

        obj_old, grads_old = training.obj_keeper, training.grads_keeper

        obj_converged = jnp.less_equal(jnp.linalg.norm(obj_value - obj_old), self.obj_threshold)
        grads_converged = jnp.less_equal(jnp.linalg.norm(grads - grads_old), self.grad_threshold)

        return jnp.logical_or(obj_converged, grads_converged)

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, args):

        training, data = args

        obj_value, grads = jax.jit(jax.value_and_grad(self.objective_fn, argnums=0, has_aux=False))(training.params, data)

        converged = self._check_convergence(training, obj_value, grads)

        training = training.apply_gradients(grads=grads)
        training = training.replace(obj_keeper=obj_value.squeeze(), grads_keeper=grads, converged=converged)

        return (training, data)

    @staticmethod
    @partial(jax.jit)
    def cond_fun(args):
        steps_exceeded = jnp.greater_equal(args[0].step, args[0].max_epochs)
        return jnp.logical_or(jnp.logical_not(args[0].converged), steps_exceeded)

    @partial(jax.jit, static_argnums=(0, ))
    def _run_single(self, rng, max_epochs=20_000):

        theta_init = self._initialize_theta(rng)

        training = TrainState.create(
            apply_fn=self.objective_fn,
            params=theta_init,
            converged=jnp.array(False, dtype=jnp.bool_),
            obj_keeper=jnp.array(9999, dtype=jnp.float32),
            grads_keeper=jnp.ones_like(theta_init, dtype=jnp.float32) * 9999,
            max_epochs=max_epochs,
            tx=self.optimizer
        )

        runner = (training, self.data)
        runner = lax.while_loop(self.cond_fun, self._step, runner)


        training, data = runner

        return training

    def _run(self, rng, n_inits=1, max_epochs=100_000, obj_threshold=1e-6, grad_threshold=1e-6):

        self.obj_threshold = obj_threshold
        self.grad_threshold = grad_threshold

        if n_inits > 1:
            rngs = jax.random.split(rng, n_inits)
            res = jax.jit(jax.vmap(self._run_single, in_axes=(0, None)))(rngs, max_epochs)
            res = self._clean_results(res)
            return self._collect_output(res)
        else:
            res = jax.jit(self._run_single)(rng, max_epochs)
            return self._collect_output(res)


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
        # return jax.random.normal(rng, shape=(rng.shape[0], ))
        return jax.random.normal(rng, shape=(2, ))

    # optimizer = optax.adam(learning_rate=1e-3, eps=1e-5)
    scheduler = optax.exponential_decay(init_value=1e-3, transition_steps=1000, decay_rate=0.99)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
        optax.scale_by_adam(),  # Use the updates from adam.
        optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
        optax.scale(-1.0) # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
    )

    rng = jax.random.PRNGKey(42)
    gd = GradientDescent(objective_fn, initilization_fn, data, optimizer)
    res = gd._run(rng, n_inits=5, obj_threshold=1e-10, grad_threshold=1e-10, max_epochs=10_000)

