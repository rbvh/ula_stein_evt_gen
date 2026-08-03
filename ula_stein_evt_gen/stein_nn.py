import jax
import jax.numpy as jnp
import math

from flax import linen as nn

import optax
import kfac_jax


class SteinCritic(nn.Module):
    """
    Simple feed-forward neural network
    """

    num_hidden_layers: int
    hidden_size: int
    dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.gelu(x)
        for _ in range(self.num_hidden_layers - 1):
            x = x + nn.gelu(nn.Dense(self.hidden_size)(x))
        x = nn.Dense(self.dim)(x)

        return x


def stein_discrepancy(
    x,
    scores,
    critic_model,
    critic_params,
    lamb=0.1,
    boundary_fn=None,
):
    """
    Learned Stein discrepancy with regularization, from 1810.03545 and 2002.05616.

    Uses the convention -lambda * ||f||^2. When boundary_fn is given, the
    Stein operator acts on h(x) f(x), but the raw critic f is still
    regularized so the grad h . f boundary term remains controlled near
    h(x) = 0.
    Set lamb=0 for unregularized evaluation (test statistic).
    """
    dim = x.shape[0]
    critic_vals = critic_model.apply(critic_params, x)

    # Jacobian trace via JVP
    jac_trace_vals = 0.0
    for i in range(dim):
        _, jvp = jax.jvp(
            lambda x: critic_model.apply(critic_params, x),
            (x,),
            (jax.nn.one_hot(i, dim),),
        )
        jac_trace_vals += jvp[i]

    sd = jnp.dot(scores, critic_vals) + jac_trace_vals

    if boundary_fn is not None:
        h_val = boundary_fn(x)
        grad_h = jax.grad(boundary_fn)(x)
        sd = h_val * sd + jnp.dot(grad_h, critic_vals)

    if lamb != 0:
        critic_reg = critic_vals
        sd -= lamb * jnp.dot(critic_reg, critic_reg)

    return sd


def _aggregate_fold_results(fold_results, fold_size):
    """Aggregate per-fold (mean, se) into overall (mean, se) via law of total variance."""
    K = len(fold_results)
    means = [r[0] for r in fold_results]
    ses = [r[1] for r in fold_results]
    N = K * fold_size
    overall_mean = sum(means) / K
    within_var = sum(se**2 * fold_size for se in ses) / K
    between_var = sum((m - overall_mean) ** 2 for m in means) / K
    overall_var = within_var + between_var
    overall_se = (max(0.0, overall_var) / N) ** 0.5
    return overall_mean, overall_se


adam_stein_train_cache = {}
kfac_stein_train_cache = {}


def _make_batches(x_data, scores_data, batch_size):
    n = x_data.shape[0]
    if x_data.shape != scores_data.shape:
        raise ValueError(
            f"x and scores must have the same shape, got {x_data.shape} and "
            f"{scores_data.shape}."
        )
    if n == 0:
        raise ValueError("Cannot create Stein batches from an empty dataset.")
    if n % batch_size != 0:
        raise ValueError(
            f"Stein dataset size ({n}) must be a multiple of batch_size "
            f"({batch_size})."
        )
    return (
        x_data.reshape(-1, batch_size, x_data.shape[-1]),
        scores_data.reshape(-1, batch_size, scores_data.shape[-1]),
    )


def _validate_stein_dataset(x, scores, batch_size):
    num_samples = x.shape[0]
    if x.shape != scores.shape:
        raise ValueError(
            f"x and scores must have the same shape, got {x.shape} and "
            f"{scores.shape}."
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if num_samples == 0:
        raise ValueError("Cannot train a Stein critic on an empty dataset.")

    required_multiple = 10 * batch_size
    if num_samples % required_multiple != 0:
        raise ValueError(
            f"Stein n_events ({num_samples}) must be a multiple of "
            f"10 * batch_size ({required_multiple})."
        )


def _split_train_val_test(x, scores):
    num_samples = x.shape[0]
    val_end = num_samples // 10
    test_end = 2 * num_samples // 10

    return (
        x[test_end:],
        x[:val_end],
        x[val_end:test_end],
        scores[test_end:],
        scores[:val_end],
        scores[val_end:test_end],
    )


def get_adam_stein_train_fns(
    dim,
    num_hidden_layers,
    hidden_size,
    learning_rate,
    lamb,
    batch_size,
    boundary_fn,
):
    """Return cached (critic_model, optimizer, train_epoch, eval_lsd), creating on first call."""
    cache_key = (
        dim,
        num_hidden_layers,
        hidden_size,
        learning_rate,
        lamb,
        batch_size,
        id(boundary_fn),
    )
    if cache_key in adam_stein_train_cache:
        return adam_stein_train_cache[cache_key]

    critic_model = SteinCritic(
        num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, dim=dim
    )
    optimizer = optax.adam(learning_rate=learning_rate)

    batch_stein_discrepancy = jax.vmap(
        stein_discrepancy,
        in_axes=(0, 0, None, None, None, None),
    )

    def loss_fn(critic_params, batch):
        x, scores = batch
        batch_loss = batch_stein_discrepancy(
            x,
            scores,
            critic_model,
            critic_params,
            lamb,
            boundary_fn,
        )
        return -batch_loss.mean()

    @jax.jit
    def train_epoch(critic_params, opt_state, batches):
        """Run one full epoch over pre-batched data via lax.scan."""

        def step(carry, batch):
            critic_params, opt_state = carry
            loss, grads = jax.value_and_grad(loss_fn)(critic_params, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            critic_params = optax.apply_updates(critic_params, updates)
            return (critic_params, opt_state), loss

        (critic_params, opt_state), losses = jax.lax.scan(
            step, (critic_params, opt_state), batches
        )
        return critic_params, opt_state, jnp.mean(losses)

    def make_eval_stein(eval_lamb):
        @jax.jit
        def eval_stein(critic_params, x_batched, scores_batched):
            def eval_batch(_, batch):
                x, scores = batch
                vals = jax.vmap(
                    stein_discrepancy,
                    in_axes=(0, 0, None, None, None, None),
                )(
                    x,
                    scores,
                    critic_model,
                    critic_params,
                    eval_lamb,
                    boundary_fn,
                )
                return _, vals

            _, all_vals = jax.lax.scan(eval_batch, None, (x_batched, scores_batched))
            all_vals = all_vals.reshape(-1)
            n = all_vals.shape[0]
            return jnp.mean(all_vals), jnp.std(all_vals) / jnp.sqrt(n)

        return eval_stein

    eval_lsd = make_eval_stein(0.0)
    eval_objective = make_eval_stein(lamb)

    result = (critic_model, optimizer, train_epoch, eval_lsd, eval_objective)

    adam_stein_train_cache[cache_key] = result

    return result


def get_kfac_stein_train_fns(
    dim,
    num_hidden_layers,
    hidden_size,
    lamb,
    batch_size,
    boundary_fn,
):
    """Return cached (critic_model, optimizer, eval_lsd) for KFAC training."""
    cache_key = (
        dim,
        num_hidden_layers,
        hidden_size,
        lamb,
        batch_size,
        id(boundary_fn),
    )
    if cache_key in kfac_stein_train_cache:
        return kfac_stein_train_cache[cache_key]

    critic_model = SteinCritic(
        num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, dim=dim
    )
    batch_sd = jax.vmap(
        stein_discrepancy,
        in_axes=(0, 0, None, None, None, None),
    )

    def loss_fn(params, batch):
        x, scores = batch
        prediction = critic_model.apply(params, x)

        kfac_jax.register_squared_error_loss(
            prediction,
            jnp.zeros_like(prediction),
            weight=lamb,
        )
        return -batch_sd(
            x,
            scores,
            critic_model,
            params,
            lamb,
            boundary_fn,
        ).mean()

    # Let KFAC choose the learning rate, but avoid adaptive momentum: for this
    # noisy Stein objective it can amplify a bad curvature estimate. The damping
    # floor is tied to the output regularization scale rather than tuned as a
    # separate training hyperparameter.
    optimizer = kfac_jax.Optimizer(
        value_and_grad_func=jax.value_and_grad(loss_fn),
        l2_reg=0.0,
        value_func_has_rng=False,
        use_adaptive_learning_rate=True,
        use_adaptive_momentum=False,
        momentum_schedule=lambda _: 0.0,
        use_adaptive_damping=True,
        initial_damping=1.0,
        min_damping=0.1 * lamb,
        max_damping=1e8,
    )

    def make_eval_stein(eval_lamb):
        @jax.jit
        def eval_stein(critic_params, x_batched, scores_batched):
            def eval_batch(_, batch_score):
                batch, scores = batch_score
                vals = jax.vmap(
                    stein_discrepancy,
                    in_axes=(0, 0, None, None, None, None),
                )(
                    batch,
                    scores,
                    critic_model,
                    critic_params,
                    eval_lamb,
                    boundary_fn,
                )
                return _, vals

            _, all_vals = jax.lax.scan(eval_batch, None, (x_batched, scores_batched))
            all_vals = all_vals.reshape(-1)
            n = all_vals.shape[0]
            return jnp.mean(all_vals), jnp.std(all_vals) / jnp.sqrt(n)

        return eval_stein

    eval_lsd = make_eval_stein(0.0)
    eval_objective = make_eval_stein(lamb)

    result = (critic_model, optimizer, eval_lsd, eval_objective)
    kfac_stein_train_cache[cache_key] = result

    return result


def train_stein_discrepancy(
    x,
    scores,
    rng,
    # Neural network parameters
    num_hidden_layers=2,
    hidden_size=64,
    # Lambda parameter for stein discrepancy regularization
    lamb=0.1,
    # Training parameters
    learning_rate=1e-3,
    batch_size=8192,
    patience=3,
    rtol=1e-3,
    cross_val=False,
    boundary_fn=None,
    status_fn=None,
    return_fold_results=False,
):
    """
    Train a Stein critic with Adam optimizer.

    Training runs until the validation Stein discrepancy has not improved
    by more than rtol (relative) for `patience` consecutive epochs.

    When cross_val=True, performs 10-fold cross-validation: each sample
    is used as a test point in exactly one fold, giving full-coverage
    LSD estimates.

    Returns (test_lsd, test_lsd_std). With return_fold_results=True, returns
    (test_lsd, test_lsd_std, fold_results), where fold_results is only set for
    cross-validation runs.
    """

    def report(message):
        if status_fn is not None:
            status_fn(message)

    _validate_stein_dataset(x, scores, batch_size)

    num_samples, dim = x.shape

    # ---- 10-fold cross-validation ----
    if cross_val:
        n_folds = 10
        fold_size = num_samples // n_folds
        rng, rng_shuffle = jax.random.split(rng)
        perm = jax.random.permutation(rng_shuffle, fold_size * n_folds)
        folds = x[: fold_size * n_folds][perm].reshape(n_folds, fold_size, dim)
        score_folds = scores[: fold_size * n_folds][perm].reshape(
            n_folds, fold_size, dim
        )

        fold_results = []

        for i in range(n_folds):
            val_i = (i + 1) % n_folds
            train_idxs = [j for j in range(n_folds) if j != i and j != val_i]
            x_fold = jnp.concatenate(
                [folds[val_i], folds[i]] + [folds[j] for j in train_idxs]
            )
            scores_fold = jnp.concatenate(
                [score_folds[val_i], score_folds[i]]
                + [score_folds[j] for j in train_idxs]
            )

            print(f"  fold {i+1}/{n_folds}")
            rng, rng_fold = jax.random.split(rng)
            lsd, lsd_se = train_stein_discrepancy(
                x=x_fold,
                scores=scores_fold,
                rng=rng_fold,
                num_hidden_layers=num_hidden_layers,
                hidden_size=hidden_size,
                lamb=lamb,
                learning_rate=learning_rate,
                batch_size=batch_size,
                patience=patience,
                rtol=rtol,
                boundary_fn=boundary_fn,
                status_fn=status_fn,
            )

            print(
                f"  fold {i+1}/{n_folds}: LSD = {float(lsd):.6f} ± {float(lsd_se):.6f}"
            )
            fold_results.append(
                {
                    "fold": i + 1,
                    "validation_fold": val_i + 1,
                    "test_fold": i + 1,
                    "lsd": float(lsd),
                    "lsd_se": float(lsd_se),
                }
            )

        fold_result_pairs = [(r["lsd"], r["lsd_se"]) for r in fold_results]
        overall_mean, overall_se = _aggregate_fold_results(
            fold_result_pairs,
            fold_size,
        )
        print(f"  overall: LSD = {overall_mean:.6f} ± {overall_se:.6f}")
        if return_fold_results:
            return overall_mean, overall_se, fold_results
        return overall_mean, overall_se

    # ---- single train/validation/test split ----
    (
        x_train,
        x_val,
        x_test,
        scores_train,
        scores_val,
        scores_test,
    ) = _split_train_val_test(x, scores)

    num_train = x_train.shape[0]
    num_steps = num_train // batch_size
    report(
        "Adam Stein setup: "
        f"train={x_train.shape[0]}, val={x_val.shape[0]}, "
        f"test={x_test.shape[0]}, batch_size={batch_size}, "
        f"train_steps/epoch={num_steps}, learning_rate={learning_rate:g}."
    )

    # ------ Get cached model, optimizer, JIT'd functions ------
    report("Adam Stein setup: building/reusing critic and optimizer functions.")
    critic_model, optimizer, train_epoch, eval_lsd, eval_objective = (
        get_adam_stein_train_fns(
            dim,
            num_hidden_layers,
            hidden_size,
            learning_rate,
            lamb,
            batch_size,
            boundary_fn,
        )
    )

    # Initialize
    report("Adam Stein setup: initializing critic parameters and optimizer state.")
    rng, rng_inp, rng_init = jax.random.split(rng, 3)
    critic_params = critic_model.init(rng_init, jax.random.normal(rng_inp, (dim,)))

    # Optimizer state
    opt_state = optimizer.init(critic_params)

    x_val_batched, scores_val_batched = _make_batches(x_val, scores_val, batch_size)
    x_test_batched, scores_test_batched = _make_batches(x_test, scores_test, batch_size)

    best_val_obj = -jnp.inf
    best_significant_obj = -jnp.inf
    best_critic_params = jax.tree.map(jnp.copy, critic_params)
    epochs_without_improvement = 0

    epoch = 0
    while True:
        epoch += 1
        report(f"Adam Stein epoch {epoch}: training {num_steps} batches.")
        # Shuffle and batch training data
        rng, rng_perm = jax.random.split(rng)
        perm_train = jax.random.permutation(rng_perm, num_train)
        x_train_batched, scores_train_batched = _make_batches(
            x_train[perm_train], scores_train[perm_train], batch_size
        )

        critic_params, opt_state, avg_loss = train_epoch(
            critic_params, opt_state, (x_train_batched, scores_train_batched)
        )
        avg_loss = float(avg_loss)

        # Select the critic by the regularized objective it is trained on.
        report(f"Adam Stein epoch {epoch}: evaluating validation objective and LSD.")
        val_obj, _ = eval_objective(critic_params, x_val_batched, scores_val_batched)
        val_lsd, _ = eval_lsd(critic_params, x_val_batched, scores_val_batched)
        val_obj_float = float(val_obj)
        val_lsd_float = float(val_lsd)
        if not (
            math.isfinite(avg_loss)
            and math.isfinite(val_obj_float)
            and math.isfinite(val_lsd_float)
        ):
            report(
                "Adam Stein encountered non-finite training statistics: "
                f"loss={avg_loss}, val_obj={val_obj_float}, "
                f"val_lsd={val_lsd_float}. Stopping critic training and "
                "using the best finite validation critic."
            )
            break

        # Always track the true best params
        if jnp.isfinite(val_obj) and val_obj > best_val_obj:
            best_val_obj = val_obj
            best_critic_params = critic_params

        # Check for significant improvement (relative tolerance)
        threshold = (
            best_significant_obj * (1 + rtol)
            if best_significant_obj > 0
            else best_significant_obj + rtol
        )
        if jnp.isfinite(val_obj) and val_obj > threshold:
            best_significant_obj = val_obj
            epochs_without_improvement = 0
            marker = "*"
        else:
            epochs_without_improvement += 1
            marker = ""

        patience_bar = "█" * epochs_without_improvement + "░" * (
            patience - epochs_without_improvement
        )
        print(
            f"    epoch {epoch:3d}  |  loss {avg_loss:.6f}  |  "
            f"val obj {val_obj_float:.6f}  |  val LSD {val_lsd_float:.6f}  |  "
            f"patience [{patience_bar}] {epochs_without_improvement}/{patience} "
            f"{marker}"
        )

        if epochs_without_improvement >= patience:
            break

    # Test evaluation using best params
    report("Adam Stein: evaluating best critic on the test split.")
    test_lsd, test_lsd_std = eval_lsd(
        best_critic_params, x_test_batched, scores_test_batched
    )

    if return_fold_results:
        return test_lsd, test_lsd_std, None
    return test_lsd, test_lsd_std


def train_stein_discrepancy_kfac(
    x,
    scores,
    rng,
    # Neural network parameters
    num_hidden_layers=2,
    hidden_size=64,
    # Lambda parameter for stein discrepancy regularization
    lamb=0.1,
    # Training parameters
    batch_size=8192,
    patience=3,
    rtol=1e-3,
    cross_val=False,
    boundary_fn=None,
    status_fn=None,
    return_fold_results=False,
):
    """
    Train a Stein critic with KFAC optimizer.

    The curvature estimator uses the regularization term as a squared-error
    proxy, while the actual gradient is computed from the full Stein
    discrepancy loss. With boundary_fn, the proxy is applied to h(x) f(x).

    When cross_val=True, performs 10-fold cross-validation: each sample
    is used as a test point in exactly one fold, giving full-coverage
    LSD estimates.

    Returns (test_lsd, test_lsd_std). With return_fold_results=True, returns
    (test_lsd, test_lsd_std, fold_results), where fold_results is only set for
    cross-validation runs.
    """

    def report(message):
        if status_fn is not None:
            status_fn(message)

    _validate_stein_dataset(x, scores, batch_size)

    num_samples, dim = x.shape

    # ---- 10-fold cross-validation ----
    if cross_val:
        n_folds = 10
        fold_size = num_samples // n_folds
        rng, rng_shuffle = jax.random.split(rng)
        perm = jax.random.permutation(rng_shuffle, fold_size * n_folds)
        folds = x[: fold_size * n_folds][perm].reshape(n_folds, fold_size, dim)
        score_folds = scores[: fold_size * n_folds][perm].reshape(
            n_folds, fold_size, dim
        )

        fold_results = []

        for i in range(n_folds):
            val_i = (i + 1) % n_folds
            train_idxs = [j for j in range(n_folds) if j != i and j != val_i]
            x_fold = jnp.concatenate(
                [folds[val_i], folds[i]] + [folds[j] for j in train_idxs]
            )
            scores_fold = jnp.concatenate(
                [score_folds[val_i], score_folds[i]]
                + [score_folds[j] for j in train_idxs]
            )

            print(f"  fold {i+1}/{n_folds}")
            rng, rng_fold = jax.random.split(rng)
            lsd, lsd_se = train_stein_discrepancy_kfac(
                x=x_fold,
                scores=scores_fold,
                rng=rng_fold,
                num_hidden_layers=num_hidden_layers,
                hidden_size=hidden_size,
                lamb=lamb,
                batch_size=batch_size,
                patience=patience,
                rtol=rtol,
                boundary_fn=boundary_fn,
                status_fn=status_fn,
            )

            print(
                f"  fold {i+1}/{n_folds}: LSD = {float(lsd):.6f} ± {float(lsd_se):.6f}"
            )
            fold_results.append(
                {
                    "fold": i + 1,
                    "validation_fold": val_i + 1,
                    "test_fold": i + 1,
                    "lsd": float(lsd),
                    "lsd_se": float(lsd_se),
                }
            )

        fold_result_pairs = [(r["lsd"], r["lsd_se"]) for r in fold_results]
        overall_mean, overall_se = _aggregate_fold_results(
            fold_result_pairs,
            fold_size,
        )
        print(f"  overall: LSD = {overall_mean:.6f} ± {overall_se:.6f}")
        if return_fold_results:
            return overall_mean, overall_se, fold_results
        return overall_mean, overall_se

    # ---- single train/validation/test split ----
    (
        x_train,
        x_val,
        x_test,
        scores_train,
        scores_val,
        scores_test,
    ) = _split_train_val_test(x, scores)

    num_train = x_train.shape[0]
    num_steps = num_train // batch_size
    report(
        "KFAC Stein setup: "
        f"train={x_train.shape[0]}, val={x_val.shape[0]}, "
        f"test={x_test.shape[0]}, batch_size={batch_size}, "
        f"train_steps/epoch={num_steps}."
    )

    # ------ Get cached model, optimizer, and JIT'd evaluation ------
    report("KFAC Stein setup: building/reusing critic and optimizer functions.")
    critic_model, optimizer, eval_lsd, eval_objective = get_kfac_stein_train_fns(
        dim, num_hidden_layers, hidden_size, lamb, batch_size, boundary_fn
    )

    # Initialize
    rng, rng_inp, rng_init = jax.random.split(rng, 3)
    critic_params = critic_model.init(rng_init, jax.random.normal(rng_inp, (dim,)))

    # Initialize KFAC state
    report("KFAC Stein setup: initializing critic parameters and optimizer state.")
    rng, rng_init = jax.random.split(rng)
    first_batch = (x_train[:batch_size], scores_train[:batch_size])
    opt_state = optimizer.init(critic_params, rng_init, first_batch)

    x_val_batched, scores_val_batched = _make_batches(x_val, scores_val, batch_size)
    x_test_batched, scores_test_batched = _make_batches(x_test, scores_test, batch_size)

    best_val_obj = -jnp.inf
    best_significant_obj = -jnp.inf
    best_critic_params = jax.tree.map(jnp.copy, critic_params)
    global_step = 0
    epochs_without_improvement = 0

    def make_data_iterator(x_data, scores_data, rng):
        while True:
            rng, rng_perm = jax.random.split(rng)
            perm = jax.random.permutation(rng_perm, x_data.shape[0])
            x_shuffled = x_data[perm]
            scores_shuffled = scores_data[perm]
            for i in range(0, x_data.shape[0] - batch_size + 1, batch_size):
                yield (
                    x_shuffled[i : i + batch_size],
                    scores_shuffled[i : i + batch_size],
                )

    epoch = 0
    while True:
        epoch += 1
        report(
            f"KFAC Stein epoch {epoch}: training {num_steps} batches. "
            "The first optimizer step may compile KFAC."
        )
        # Train
        rng, rng_iter = jax.random.split(rng)
        data_iter = make_data_iterator(x_train, scores_train, rng_iter)

        epoch_loss = 0.0
        epoch_learning_rate = 0.0
        epoch_momentum = 0.0
        epoch_damping = 0.0
        nonfinite_step = False
        for _ in range(num_steps):
            rng, rng_step = jax.random.split(rng)

            critic_params, opt_state, stats = optimizer.step(
                critic_params,
                opt_state,
                rng_step,
                data_iterator=data_iter,
                global_step_int=global_step,
            )
            step_loss = float(stats["loss"])
            step_learning_rate = float(stats["learning_rate"])
            step_momentum = float(stats["momentum"])
            step_damping = float(stats["damping"])
            if not (
                math.isfinite(step_loss)
                and math.isfinite(step_learning_rate)
                and math.isfinite(step_momentum)
                and math.isfinite(step_damping)
            ):
                report(
                    "KFAC Stein encountered non-finite optimizer statistics "
                    f"at global_step={global_step}: loss={step_loss}, "
                    f"lr={step_learning_rate}, mom={step_momentum}, "
                    f"damp={step_damping}. Stopping critic training and "
                    "using the best finite validation critic."
                )
                nonfinite_step = True
                break

            epoch_loss += step_loss
            epoch_learning_rate += step_learning_rate
            epoch_momentum += step_momentum
            epoch_damping += step_damping
            global_step += 1

        if nonfinite_step:
            break

        avg_loss = epoch_loss / max(num_steps, 1)
        avg_learning_rate = epoch_learning_rate / max(num_steps, 1)
        avg_momentum = epoch_momentum / max(num_steps, 1)
        avg_damping = epoch_damping / max(num_steps, 1)

        # Select the critic by the regularized objective it is trained on.
        report(f"KFAC Stein epoch {epoch}: evaluating validation objective and LSD.")
        params_copy = jax.tree.map(jnp.copy, critic_params)
        val_obj, _ = eval_objective(params_copy, x_val_batched, scores_val_batched)
        val_lsd, _ = eval_lsd(params_copy, x_val_batched, scores_val_batched)
        val_obj_float = float(val_obj)
        val_lsd_float = float(val_lsd)
        if not (math.isfinite(val_obj_float) and math.isfinite(val_lsd_float)):
            report(
                "KFAC Stein validation became non-finite: "
                f"val_obj={val_obj_float}, val_lsd={val_lsd_float}. "
                "Stopping critic training and using the best finite "
                "validation critic."
            )
            break

        # Always track the true best params
        if jnp.isfinite(val_obj) and val_obj > best_val_obj:
            best_val_obj = val_obj
            best_critic_params = params_copy

        # Check for significant improvement (relative tolerance)
        threshold = (
            best_significant_obj * (1 + rtol)
            if best_significant_obj > 0
            else best_significant_obj + rtol
        )
        if jnp.isfinite(val_obj) and val_obj > threshold:
            best_significant_obj = val_obj
            epochs_without_improvement = 0
            marker = "*"
        else:
            epochs_without_improvement += 1
            marker = ""

        patience_bar = "█" * epochs_without_improvement + "░" * (
            patience - epochs_without_improvement
        )
        print(
            f"    epoch {epoch:3d}  |  loss {avg_loss:.6f}  |  "
            f"val obj {val_obj_float:.6f}  |  val LSD {val_lsd_float:.6f}  |  "
            f"lr {avg_learning_rate:.3g}  |  mom {avg_momentum:.3g}  |  "
            f"damp {avg_damping:.3g}  |  patience [{patience_bar}] "
            f"{epochs_without_improvement}/{patience} {marker}"
        )

        if epochs_without_improvement >= patience:
            break

    # Test evaluation using best params
    report("KFAC Stein: evaluating best critic on the test split.")
    test_lsd, test_lsd_std = eval_lsd(
        best_critic_params, x_test_batched, scores_test_batched
    )

    if return_fold_results:
        return test_lsd, test_lsd_std, None
    return test_lsd, test_lsd_std
