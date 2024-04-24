import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax_tqdm import scan_tqdm, loop_tqdm
from flax.training.train_state import TrainState
from dataclasses import dataclass
from functools import partial

class TrainState(TrainState):
    obj_keeper: jnp.float32
    grads_keeper: jnp.array
    converged: jnp.bool_
    convergence_epoch: jnp.int32

@dataclass
class FitResults:
    theta: np.array
    converged: bool
    convergence_epoch: int
    objective_value: float
    grads: np.array

class GradientDescent:

    def __init__(self,
                 objective_fn,
                 initilization_fn,
                 data,
                 optimizer,
                 obj_threshold=1e-6,
                 grad_threshold=1e-6,
                 max_epochs=10_000):
        self.objective_fn = objective_fn
        self.initilization_fn = initilization_fn
        self.optimizer = optimizer
        self.data = tuple([jnp.asarray(d) for d in data])
        self.n_datapoints = self.data[0].shape[0]
        self.obj_threshold = obj_threshold
        self.grad_threshold = grad_threshold
        self.max_epochs = max_epochs

    def __repr__(self):
        pass

    @staticmethod
    def _collect_output(res):
        bootstraped = res.converged.size > 1
        return FitResults(
            theta=np.asarray(res.params),
            converged=res.converged.item() if not bootstraped else np.asarray(res.converged),
            convergence_epoch=res.convergence_epoch.item() if not bootstraped else np.asarray(res.convergence_epoch),
            objective_value=res.obj_keeper.item() if not bootstraped else np.asarray(res.obj_keeper),
            grads=np.asarray(res.grads_keeper)
        )

    @staticmethod
    def _clean_results(res, metrics):

        bootstraped = res.converged.ndim > 1

        if not bootstraped:

            idx_fittest = jnp.argmin(res.obj_keeper)

            res = res.replace(
                params=res.params[idx_fittest],
                converged=res.converged[idx_fittest],
                convergence_epoch=res.convergence_epoch[idx_fittest],
                step=res.step[idx_fittest],
                obj_keeper=res.obj_keeper[idx_fittest],
                grads_keeper=res.grads_keeper[idx_fittest],
            )

            metrics = {key: np.asarray(val[idx_fittest]) for (key, val) in metrics.items()}

        else:

            n_boot = res.converged.shape[0]
            idx_fittest = jnp.argmin(res.obj_keeper, axis=1)

            res = res.replace(
                params=res.params[jnp.arange(n_boot), idx_fittest, :],
                converged=res.converged[jnp.arange(n_boot), idx_fittest],
                convergence_epoch=res.convergence_epoch[jnp.arange(n_boot), idx_fittest],
                step=res.step[jnp.arange(n_boot), idx_fittest],
                obj_keeper=res.obj_keeper[jnp.arange(n_boot), idx_fittest],
                grads_keeper=res.grads_keeper[jnp.arange(n_boot), idx_fittest, :],
            )

            metrics = {key: np.asarray(val[jnp.arange(n_boot), idx_fittest, :]) for (key, val) in metrics.items()}

        return res, metrics

    @partial(jax.jit, static_argnums=(0, ))
    def _initialize_theta(self, rng):
        return self.initilization_fn(rng)

    @staticmethod
    @partial(jax.jit)
    def _get_convergence_epoch(training, epoch, converged):

        old_converged, old_convergence_epoch = training.converged, training.convergence_epoch

        return jnp.where(jnp.logical_and(old_converged, converged), old_convergence_epoch, epoch)


    @partial(jax.jit, static_argnums=(0,))
    def _check_convergence(self, training, obj_value, grads):

        obj_old = training.obj_keeper

        obj_converged = jnp.less_equal(jnp.linalg.norm(obj_value - obj_old), self.obj_threshold)

        grads_converged = jnp.less_equal(jnp.linalg.norm(grads), self.grad_threshold)

        converged = jnp.logical_or(obj_converged, grads_converged)

        convergence_epoch = self._get_convergence_epoch(training, training.step, converged)

        return converged, convergence_epoch

    @partial(jax.jit, static_argnums=(0, ))
    def _run_single(self, rng, data):

        rng, rng_init = jax.random.split(rng, 2)
        theta_init = self._initialize_theta(rng_init)

        training = TrainState.create(
            apply_fn=self.objective_fn,
            params=theta_init,
            converged=jnp.array(False, dtype=jnp.bool_),
            convergence_epoch=jnp.array(0, dtype=jnp.int32),
            obj_keeper=jnp.array(0, dtype=jnp.float32),
            grads_keeper=jnp.ones_like(theta_init, dtype=jnp.float32) * 9999,
            tx=self.optimizer
        )

        step_runner = (training, data, rng)

        (training, data, _), metrics = lax.scan(
            scan_tqdm(self.max_epochs)(self._step),
            step_runner,
            jnp.arange(self.max_epochs),
            self.max_epochs
        )

        return training, metrics

    def _bootstrap(self, rng, n_inits=1, n_boot=1_000):

        rng, rng_boot = jax.random.split(rng)

        idx_boot = jax.random.choice(
            rng_boot,
            jnp.arange(self.data[0].shape[0]),
            shape=(n_boot, self.data[0].shape[0]),
            replace=True
        )

        data_boot = tuple([jnp.take(d, idx_boot, axis=0) for d in self.data])

        if n_inits > 1:
            rngs = jax.random.split(rng, n_inits)
            res, metrics = jax.jit(jax.vmap(
                jax.vmap(self._run_single, in_axes=(0, None)),
                in_axes=(None, 0)))(rngs, data_boot)
            res, metrics = self._clean_results(res, metrics)
            return (self._collect_output(res), metrics)
        else:
            res, metrics = jax.jit(jax.vmap(self._run_single, in_axes=(None, 0)))(rng, data_boot)
            return (self._collect_output(res), metrics)

    def fit(self, rng, n_inits=1):

        rng, rng_data = jax.random.split(rng, 2)
        data, rng = self.prepare_data(rng_data)

        if n_inits > 1:
            rngs = jax.random.split(rng, n_inits)
            res, metrics = jax.jit(jax.vmap(self._run_single, in_axes=(0, None)))(rngs, data)
            res, metrics = self._clean_results(res, metrics)
            return (self._collect_output(res), metrics)
        else:
            res, metrics = jax.jit(self._run_single)(rng, data)
            return (self._collect_output(res), metrics)

    def prepare_data(self, rng):
        return self.data, rng

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, args, epoch):

        training, data, rng = args

        obj_value, grads = jax.jit(jax.value_and_grad(self.objective_fn, argnums=0, has_aux=False))(training.params, data)

        converged, convergence_epoch = self._check_convergence(training, obj_value, grads)

        training = training.apply_gradients(grads=grads)
        training = training.replace(
            obj_keeper=obj_value.squeeze(),
            grads_keeper=grads,
            converged=converged,
            convergence_epoch=convergence_epoch
        )

        metrics = {
            "objective_value": obj_value,
            "epoch": epoch,
            "converged": converged
        }

        return (training, data, rng), metrics

