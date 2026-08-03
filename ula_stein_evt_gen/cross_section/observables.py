from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ula_stein_evt_gen.cross_section.constants import vector_boson_mass
from ula_stein_evt_gen.cross_section.phase_space import map_phase_space


def _pt_y_phi(p):
    pt = jnp.sqrt(p[1] ** 2 + p[2] ** 2)
    y = 0.5 * jnp.log((p[0] + p[3]) / (p[0] - p[3]))
    phi = jnp.arctan2(p[2], p[1])
    return pt, y, phi


def _pt_y_phi_batch(p):
    pt = jnp.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2)
    y = 0.5 * jnp.log((p[..., 0] + p[..., 3]) / (p[..., 0] - p[..., 3]))
    phi = jnp.arctan2(p[..., 2], p[..., 1])
    return pt, y, phi


def _delta_phi(phi_a, phi_b):
    return jnp.mod(phi_a - phi_b + jnp.pi, 2 * jnp.pi) - jnp.pi


def _invariant_mass(p):
    mass2 = p[..., 0] ** 2 - jnp.sum(p[..., 1:] ** 2, axis=-1)
    return jnp.sqrt(jnp.maximum(mass2, 0.0))


def _pair_distances(jets, active, algorithm, radius):
    pt, y, phi = jax.vmap(_pt_y_phi)(jets)
    dphi = phi[:, None] - phi[None, :]
    dphi = jnp.mod(dphi + jnp.pi, 2 * jnp.pi) - jnp.pi
    dy = y[:, None] - y[None, :]
    dr2 = dy**2 + dphi**2

    if algorithm == "kt":
        distance = jnp.minimum(pt[:, None] ** 2, pt[None, :] ** 2) * dr2 / radius**2
    elif algorithm == "ca":
        distance = dr2 / radius**2
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    n = jets.shape[0]
    valid_pair = (
        active[:, None] & active[None, :] & jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    )
    return jnp.where(valid_pair, distance, jnp.inf)


def _beam_distances(jets, active, algorithm):
    pt, _, _ = jax.vmap(_pt_y_phi)(jets)
    if algorithm == "kt":
        distance = pt**2
    elif algorithm == "ca":
        distance = jnp.ones_like(pt)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
    return jnp.where(active, distance, jnp.inf)


@partial(jax.jit, static_argnames=("algorithm",))
def exclusive_clustering_scales(jets, algorithm="kt", radius=1.0):
    """Return d01, d12, ..., d(n-1)n for hadron-collider clustering."""
    n_jets = jets.shape[0]
    active = jnp.ones((n_jets,), dtype=bool)
    scales_by_active_count = jnp.zeros((n_jets + 1,), dtype=jets.dtype)

    def merge_step(_, state):
        jets_now, active_now, scales_now = state
        pair_distances = _pair_distances(jets_now, active_now, algorithm, radius)
        beam_distances = _beam_distances(jets_now, active_now, algorithm)
        pair_min = jnp.min(pair_distances)
        beam_min = jnp.min(beam_distances)
        scale = jnp.minimum(pair_min, beam_min)
        n_active = jnp.sum(active_now.astype(jnp.int32))

        def merge_pair(_):
            pair_idx = jnp.argmin(pair_distances)
            i = pair_idx // n_jets
            j = pair_idx % n_jets
            jets_next = jets_now.at[i].set(jets_now[i] + jets_now[j])
            active_next = active_now.at[j].set(False)
            return jets_next, active_next

        def merge_beam(_):
            i = jnp.argmin(beam_distances)
            active_next = active_now.at[i].set(False)
            return jets_now, active_next

        jets_next, active_next = jax.lax.cond(
            pair_min < beam_min, merge_pair, merge_beam, None
        )
        scales_next = scales_now.at[n_active].set(scale)
        return jets_next, active_next, scales_next

    _, _, scales_by_active_count = jax.lax.fori_loop(
        0, n_jets, merge_step, (jets, active, scales_by_active_count)
    )
    return scales_by_active_count[1:]


def compute_observables(
    x,
    vb,
    n_jets,
    sqrt_s=13000.0,
    pt_cut=20.0,
    y_cut=5.0,
    jet_radius=1.0,
    batch_size=100_000,
):
    """
    Compute validation observables for unit-cube phase-space points.

    Returns NumPy arrays. Clustering scales are included as ``d01``, ``d12``,
    ... for k_t, and ``y01``, ``y12``, ... for C/A, for ``n_jets >= 1``.
    """
    mv = vector_boson_mass(vb)
    map_batched = jax.jit(
        jax.vmap(map_phase_space, in_axes=(0, None, None, None, None))
    )
    kt_scales_batched = None
    ca_scales_batched = None
    if n_jets >= 1:
        kt_scales_batched = jax.jit(
            jax.vmap(
                lambda jets: exclusive_clustering_scales(
                    jets, algorithm="kt", radius=jet_radius
                )
            )
        )
        ca_scales_batched = jax.jit(
            jax.vmap(
                lambda jets: exclusive_clustering_scales(
                    jets, algorithm="ca", radius=jet_radius
                )
            )
        )

    chunks = {
        "pt_v": [],
        "y_v": [],
        "x_a": [],
        "x_b": [],
        "log_xaxb": [],
        "y_x": [],
    }
    kt_scale_chunks = []
    ca_scale_chunks = []
    n_total = int(x.shape[0])
    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        pa, pb, pv, jets, _ = map_batched(x[start:stop], sqrt_s, mv, pt_cut, y_cut)
        pt_v, y_v, phi_v = _pt_y_phi_batch(pv)
        x_a = 2.0 * pa[:, 0] / sqrt_s
        x_b = 2.0 * pb[:, 0] / sqrt_s
        chunks["pt_v"].append(np.asarray(jax.device_get(pt_v)))
        chunks["y_v"].append(np.asarray(jax.device_get(y_v)))
        chunks["x_a"].append(np.asarray(jax.device_get(x_a)))
        chunks["x_b"].append(np.asarray(jax.device_get(x_b)))
        chunks["log_xaxb"].append(
            np.asarray(jax.device_get(jnp.log(jnp.maximum(x_a * x_b, 1e-300))))
        )
        chunks["y_x"].append(
            np.asarray(
                jax.device_get(
                    0.5 * jnp.log(jnp.maximum(x_a, 1e-300) / jnp.maximum(x_b, 1e-300))
                )
            )
        )

        if n_jets >= 1:
            pt_j, y_j, phi_j = _pt_y_phi_batch(jets)
            order = jnp.argsort(-pt_j, axis=1)
            sorted_pt_j = jnp.take_along_axis(pt_j, order, axis=1)
            sorted_y_j = jnp.take_along_axis(y_j, order, axis=1)
            sorted_jets = jnp.take_along_axis(
                jets,
                jnp.broadcast_to(order[:, :, None], jets.shape),
                axis=1,
            )

            chunks.setdefault("h_t", []).append(
                np.asarray(jax.device_get(jnp.sum(pt_j, axis=1)))
            )
            chunks.setdefault("y_j1", []).append(
                np.asarray(jax.device_get(sorted_y_j[:, 0]))
            )
            chunks.setdefault("max_abs_y_j", []).append(
                np.asarray(jax.device_get(jnp.max(jnp.abs(y_j), axis=1)))
            )
            dr_zj = jnp.sqrt(
                (y_j - y_v[:, None]) ** 2 + _delta_phi(phi_j, phi_v[:, None]) ** 2
            )
            chunks.setdefault("min_deltaR_Zj", []).append(
                np.asarray(jax.device_get(jnp.min(dr_zj, axis=1)))
            )
            for i_jet in range(n_jets):
                chunks.setdefault(f"pt_j{i_jet + 1}", []).append(
                    np.asarray(jax.device_get(sorted_pt_j[:, i_jet]))
                )

            if n_jets >= 2:
                dphi_jj = _delta_phi(phi_j[:, :, None], phi_j[:, None, :])
                dy_jj = y_j[:, :, None] - y_j[:, None, :]
                dr_jj = jnp.sqrt(dy_jj**2 + dphi_jj**2)
                pair_mask = jnp.triu(
                    jnp.ones((n_jets, n_jets), dtype=bool),
                    k=1,
                )
                min_dr_jj = jnp.min(
                    jnp.where(pair_mask[None, :, :], dr_jj, jnp.inf),
                    axis=(1, 2),
                )
                chunks.setdefault("min_deltaR_jj", []).append(
                    np.asarray(jax.device_get(min_dr_jj))
                )
                chunks.setdefault("m_jj", []).append(
                    np.asarray(
                        jax.device_get(
                            _invariant_mass(jnp.sum(sorted_jets[:, :2], axis=1))
                        )
                    )
                )
                chunks.setdefault("m_all_jets", []).append(
                    np.asarray(jax.device_get(_invariant_mass(jnp.sum(jets, axis=1))))
                )

            kt_scales = kt_scales_batched(jets)
            ca_scales = ca_scales_batched(jets)
            kt_scale_chunks.append(
                np.asarray(jax.device_get(jnp.sqrt(jnp.maximum(kt_scales, 0.0))))
            )
            ca_scale_chunks.append(
                np.asarray(jax.device_get(jnp.sqrt(jnp.maximum(ca_scales, 0.0))))
            )

    observables = {
        name: np.concatenate(value_chunks)
        for name, value_chunks in chunks.items()
        if value_chunks
    }
    if n_jets >= 1:
        kt_scales = np.concatenate(kt_scale_chunks, axis=0)
        ca_scales = np.concatenate(ca_scale_chunks, axis=0)
        for i_scale in range(n_jets):
            observables[f"d{i_scale}{i_scale + 1}"] = kt_scales[:, i_scale]
            observables[f"y{i_scale}{i_scale + 1}"] = ca_scales[:, i_scale]
    return observables
