import math
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from ula_stein_evt_gen.mirror import from_mirror
from ula_stein_evt_gen.stein_nn import SteinCritic


def round_up_to_multiple(value, multiple):
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def save_surrogate(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(jax.device_get(payload), f)


def load_surrogate(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def make_surrogate_model(num_hidden_layers, hidden_size):
    return SteinCritic(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        dim=1,
    )


def predict_log_cross_section(model, params, z, target_mean, target_std):
    prediction = model.apply(params, z).squeeze(-1)
    return target_mean + target_std * prediction


def build_surrogate_log_prob_and_score(payload, cut_factors_base=None, chunk_size=None):
    model = make_surrogate_model(
        payload["num_hidden_layers"],
        payload["hidden_size"],
    )
    params = payload["params"]
    target_mean = jnp.asarray(payload["target_mean"])
    target_std = jnp.asarray(payload["target_std"])
    dim = int(payload["dim"])

    def log_prob_single(z):
        log_sigma = predict_log_cross_section(
            model,
            params,
            z[None, :],
            target_mean,
            target_std,
        )[0]
        log_prob = log_sigma - 0.5 * jnp.sum(z**2, axis=-1)
        if cut_factors_base is None:
            return log_prob

        inside = jnp.all(cut_factors_base(from_mirror(z)) > 0.0)
        return jax.lax.cond(
            inside,
            lambda _: log_prob,
            lambda _: -jnp.inf,
            None,
        )

    batch_value_and_grad = jax.jit(jax.vmap(jax.value_and_grad(log_prob_single)))

    if chunk_size is None:
        return batch_value_and_grad
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    @jax.jit
    def log_prob_and_score(z):
        if z.ndim != 2 or z.shape[-1] != dim:
            raise ValueError(f"Input shape must be (n, {dim}), got {z.shape}.")

        n_samples = z.shape[0]
        pad_len = (-n_samples) % chunk_size
        if pad_len:
            pad = jnp.zeros((pad_len, dim), dtype=z.dtype)
            z_eval = jnp.concatenate([z, pad], axis=0)
        else:
            z_eval = z

        n_padded = z_eval.shape[0]
        z_chunks = z_eval.reshape((-1, chunk_size, dim))

        def scan_body(_, z_chunk):
            return None, batch_value_and_grad(z_chunk)

        _, (log_prob_chunks, score_chunks) = jax.lax.scan(
            scan_body,
            None,
            z_chunks,
        )
        log_prob_out = log_prob_chunks.reshape((n_padded,))
        score_out = score_chunks.reshape((n_padded, dim))

        return log_prob_out[:n_samples], score_out[:n_samples]

    return log_prob_and_score


def _make_batches(z, y, batch_size):
    if z.shape[0] != y.shape[0]:
        raise ValueError(
            f"z and y must have the same leading dimension, got "
            f"{z.shape[0]} and {y.shape[0]}."
        )
    if z.shape[0] % batch_size != 0:
        raise ValueError(
            f"Dataset size ({z.shape[0]}) must be a multiple of batch_size "
            f"({batch_size})."
        )
    return (
        z.reshape((-1, batch_size, z.shape[-1])),
        y.reshape((-1, batch_size, 1)),
    )


def _eval_loss(model, params, z, y, batch_size):
    z_batched, y_batched = _make_batches(z, y, batch_size)

    @jax.jit
    def eval_batched(params, z_batches, y_batches):
        def eval_batch(_, batch):
            z_batch, y_batch = batch
            prediction = model.apply(params, z_batch)
            residual = prediction - y_batch
            return None, jnp.mean(residual**2)

        _, losses = jax.lax.scan(eval_batch, None, (z_batches, y_batches))
        return jnp.mean(losses)

    return eval_batched(params, z_batched, y_batched)


def train_surrogate_adam(
    z_train,
    y_train,
    z_val,
    y_val,
    z_test,
    y_test,
    rng,
    num_hidden_layers=5,
    hidden_size=128,
    learning_rate=1e-3,
    batch_size=8192,
    max_epochs=100,
    patience=5,
    rtol=1e-3,
    status_fn=None,
):
    def report(message):
        if status_fn is not None:
            status_fn(message)

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if max_epochs is not None and max_epochs < 1:
        raise ValueError("max_epochs must be positive or None.")

    dim = z_train.shape[-1]
    model = make_surrogate_model(num_hidden_layers, hidden_size)

    target_mean = jnp.mean(y_train)
    target_std = jnp.maximum(jnp.std(y_train), 1e-6)

    y_train_std = (y_train - target_mean) / target_std
    y_val_std = (y_val - target_mean) / target_std
    y_test_std = (y_test - target_mean) / target_std

    def loss_fn(params, batch):
        z_batch, y_batch = batch
        prediction = model.apply(params, z_batch)
        return 0.5 * jnp.mean((prediction - y_batch) ** 2)

    optimizer = optax.adam(learning_rate=learning_rate)

    rng, rng_inp, rng_init = jax.random.split(rng, 3)
    params = model.init(rng_init, jax.random.normal(rng_inp, (dim,)))

    z_train_batches, y_train_batches = _make_batches(
        z_train,
        y_train_std,
        batch_size,
    )
    z_val_batches, y_val_batches = _make_batches(z_val, y_val_std, batch_size)
    z_test_batches, y_test_batches = _make_batches(z_test, y_test_std, batch_size)

    opt_state = optimizer.init(params)

    @jax.jit
    def train_epoch(params, opt_state, z_batches, y_batches):
        def step(carry, batch):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(loss_fn)(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (params, opt_state), losses = jax.lax.scan(
            step,
            (params, opt_state),
            (z_batches, y_batches),
        )
        return params, opt_state, jnp.mean(losses)

    best_val_loss = jnp.inf
    best_significant_loss = jnp.inf
    best_params = jax.tree.map(jnp.copy, params)
    epochs_without_improvement = 0
    global_step = 0
    num_steps = z_train_batches.shape[0]

    report(
        "Surrogate Adam setup: "
        f"train={z_train.shape[0]}, val={z_val.shape[0]}, "
        f"test={z_test.shape[0]}, batch_size={batch_size}, "
        f"train_steps/epoch={num_steps}, learning_rate={learning_rate:g}, "
        f"max_epochs={max_epochs}."
    )

    epoch = 0
    while max_epochs is None or epoch < max_epochs:
        epoch += 1
        rng, rng_perm = jax.random.split(rng)
        perm = jax.random.permutation(rng_perm, z_train.shape[0])
        z_train_batches, y_train_batches = _make_batches(
            z_train[perm],
            y_train_std[perm],
            batch_size,
        )

        params, opt_state, avg_loss = train_epoch(
            params,
            opt_state,
            z_train_batches,
            y_train_batches,
        )
        global_step += num_steps

        avg_loss = float(avg_loss)
        if not math.isfinite(avg_loss):
            report(
                "Surrogate Adam training loss became non-finite: "
                f"{avg_loss}. Stopping and using the best finite validation "
                "parameters."
            )
            break

        val_loss = _eval_loss(model, params, z_val, y_val_std, batch_size)
        val_loss_float = float(val_loss)
        if not math.isfinite(val_loss_float):
            report(
                f"Surrogate validation loss became non-finite: {val_loss_float}. "
                "Stopping and using the best finite validation parameters."
            )
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = jax.tree.map(jnp.copy, params)

        threshold = (
            best_significant_loss * (1.0 - rtol)
            if jnp.isfinite(best_significant_loss)
            else best_significant_loss
        )
        if val_loss < threshold:
            best_significant_loss = val_loss
            epochs_without_improvement = 0
            marker = "*"
        else:
            epochs_without_improvement += 1
            marker = ""

        patience_bar = "█" * epochs_without_improvement + "░" * (
            patience - epochs_without_improvement
        )
        report(
            f"epoch {epoch:3d}  |  loss {avg_loss:.6f}  |  "
            f"val mse {val_loss_float:.6f}  |  lr {learning_rate:.3g}  |  "
            f"patience [{patience_bar}] {epochs_without_improvement}/{patience} "
            f"{marker}"
        )

        if epochs_without_improvement >= patience:
            break
    else:
        report(f"Surrogate Adam reached max_epochs={max_epochs}.")

    test_mse_std = _eval_loss(model, best_params, z_test, y_test_std, batch_size)
    val_mse_std = _eval_loss(model, best_params, z_val, y_val_std, batch_size)
    metrics = {
        "val_mse_standardized": float(val_mse_std),
        "test_mse_standardized": float(test_mse_std),
        "val_rmse_log": float(jnp.sqrt(val_mse_std) * target_std),
        "test_rmse_log": float(jnp.sqrt(test_mse_std) * target_std),
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "optimizer": "adam",
        "learning_rate": float(learning_rate),
        "max_epochs": max_epochs,
        "epochs": epoch,
        "global_steps": global_step,
    }

    payload = {
        "params": best_params,
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "dim": dim,
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "metrics": metrics,
    }
    return payload


def train_surrogate_kfac(
    z_train,
    y_train,
    z_val,
    y_val,
    z_test,
    y_test,
    rng,
    num_hidden_layers=5,
    hidden_size=128,
    batch_size=8192,
    max_epochs=100,
    patience=5,
    rtol=1e-3,
    status_fn=None,
):
    import kfac_jax

    def report(message):
        if status_fn is not None:
            status_fn(message)

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if max_epochs is not None and max_epochs < 1:
        raise ValueError("max_epochs must be positive or None.")

    dim = z_train.shape[-1]
    model = make_surrogate_model(num_hidden_layers, hidden_size)

    target_mean = jnp.mean(y_train)
    target_std = jnp.maximum(jnp.std(y_train), 1e-6)

    y_train_std = (y_train - target_mean) / target_std
    y_val_std = (y_val - target_mean) / target_std
    y_test_std = (y_test - target_mean) / target_std

    def loss_fn(params, batch):
        z_batch, y_batch = batch
        prediction = model.apply(params, z_batch)
        kfac_jax.register_squared_error_loss(
            prediction=prediction,
            targets=y_batch,
        )
        return 0.5 * jnp.mean((prediction - y_batch) ** 2)

    optimizer = kfac_jax.Optimizer(
        value_and_grad_func=jax.value_and_grad(loss_fn),
        l2_reg=0.0,
        value_func_has_rng=False,
        use_adaptive_learning_rate=True,
        use_adaptive_momentum=True,
        use_adaptive_damping=True,
        initial_damping=1.0,
    )

    rng, rng_inp, rng_init = jax.random.split(rng, 3)
    params = model.init(rng_init, jax.random.normal(rng_inp, (dim,)))

    z_train_batches, y_train_batches = _make_batches(
        z_train,
        y_train_std,
        batch_size,
    )
    num_steps = z_train_batches.shape[0]

    report(
        "Surrogate KFAC setup: "
        f"train={z_train.shape[0]}, val={z_val.shape[0]}, "
        f"test={z_test.shape[0]}, batch_size={batch_size}, "
        f"train_steps/epoch={num_steps}, max_epochs={max_epochs}."
    )

    report("Surrogate KFAC setup: initializing optimizer state.")
    rng, rng_init = jax.random.split(rng)
    first_batch = (z_train_batches[0], y_train_batches[0])
    opt_state = optimizer.init(params, rng_init, first_batch)

    def make_data_iterator(z_data, y_data, rng):
        while True:
            rng, rng_perm = jax.random.split(rng)
            perm = jax.random.permutation(rng_perm, z_data.shape[0])
            z_batches, y_batches = _make_batches(
                z_data[perm],
                y_data[perm],
                batch_size,
            )
            for i in range(z_batches.shape[0]):
                yield z_batches[i], y_batches[i]

    best_val_loss = jnp.inf
    best_significant_loss = jnp.inf
    best_params = jax.tree.map(jnp.copy, params)
    epochs_without_improvement = 0
    global_step = 0
    epoch = 0

    while max_epochs is None or epoch < max_epochs:
        epoch += 1
        report(
            f"Surrogate KFAC epoch {epoch}: training {num_steps} batches. "
            "The first optimizer step may compile KFAC."
        )
        rng, rng_iter = jax.random.split(rng)
        data_iter = make_data_iterator(z_train, y_train_std, rng_iter)

        epoch_loss = 0.0
        epoch_learning_rate = 0.0
        epoch_momentum = 0.0
        epoch_damping = 0.0
        nonfinite_step = False
        for _ in range(num_steps):
            rng, rng_step = jax.random.split(rng)
            params, opt_state, stats = optimizer.step(
                params,
                opt_state,
                rng_step,
                data_iterator=data_iter,
                global_step_int=global_step,
            )
            global_step += 1

            step_loss = float(stats["loss"])
            step_learning_rate = float(stats.get("learning_rate", 0.0))
            step_momentum = float(stats.get("momentum", 0.0))
            step_damping = float(stats.get("damping", 0.0))
            if not all(
                math.isfinite(value)
                for value in (
                    step_loss,
                    step_learning_rate,
                    step_momentum,
                    step_damping,
                )
            ):
                report(
                    "Surrogate KFAC encountered non-finite optimizer statistics "
                    f"at global_step={global_step}: loss={step_loss}, "
                    f"lr={step_learning_rate}, mom={step_momentum}, "
                    f"damp={step_damping}. Stopping and using the best finite "
                    "validation parameters."
                )
                nonfinite_step = True
                break

            epoch_loss += step_loss
            epoch_learning_rate += step_learning_rate
            epoch_momentum += step_momentum
            epoch_damping += step_damping

        if nonfinite_step:
            break

        avg_loss = epoch_loss / num_steps
        avg_learning_rate = epoch_learning_rate / num_steps
        avg_momentum = epoch_momentum / num_steps
        avg_damping = epoch_damping / num_steps

        val_loss = _eval_loss(model, params, z_val, y_val_std, batch_size)
        val_loss_float = float(val_loss)
        if not math.isfinite(val_loss_float):
            report(
                f"Surrogate KFAC validation loss became non-finite: "
                f"{val_loss_float}. Stopping and using the best finite "
                "validation parameters."
            )
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = jax.tree.map(jnp.copy, params)

        threshold = (
            best_significant_loss * (1.0 - rtol)
            if jnp.isfinite(best_significant_loss)
            else best_significant_loss
        )
        if val_loss < threshold:
            best_significant_loss = val_loss
            epochs_without_improvement = 0
            marker = "*"
        else:
            epochs_without_improvement += 1
            marker = ""

        patience_bar = "█" * epochs_without_improvement + "░" * (
            patience - epochs_without_improvement
        )
        report(
            f"epoch {epoch:3d}  |  loss {avg_loss:.6f}  |  "
            f"val mse {val_loss_float:.6f}  |  lr {avg_learning_rate:.3g}  |  "
            f"mom {avg_momentum:.3g}  |  damp {avg_damping:.3g}  |  "
            f"patience [{patience_bar}] {epochs_without_improvement}/{patience} "
            f"{marker}"
        )

        if epochs_without_improvement >= patience:
            break
    else:
        report(f"Surrogate KFAC reached max_epochs={max_epochs}.")

    test_mse_std = _eval_loss(model, best_params, z_test, y_test_std, batch_size)
    val_mse_std = _eval_loss(model, best_params, z_val, y_val_std, batch_size)
    metrics = {
        "val_mse_standardized": float(val_mse_std),
        "test_mse_standardized": float(test_mse_std),
        "val_rmse_log": float(jnp.sqrt(val_mse_std) * target_std),
        "test_rmse_log": float(jnp.sqrt(test_mse_std) * target_std),
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "optimizer": "kfac",
        "max_epochs": max_epochs,
        "epochs": epoch,
        "global_steps": global_step,
    }

    payload = {
        "params": best_params,
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "dim": dim,
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "metrics": metrics,
    }
    return payload
