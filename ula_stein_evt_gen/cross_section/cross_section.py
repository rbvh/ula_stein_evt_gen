import importlib
from pathlib import Path
import contextlib

import jax
import jax.numpy as jnp
import numpy as np

from ula_stein_evt_gen.cross_section.phase_space import map_phase_space
from ula_stein_evt_gen.cross_section.alphas import load_alphas
from ula_stein_evt_gen.cross_section.constants import vector_boson_mass
from ula_stein_evt_gen.cross_section.pdf import load_pdfs
from ula_stein_evt_gen.cross_section.v_jets.processes.all_processes_jaxified import smatrix_Matrix_1_udx_wp, smatrix_Matrix_2_uux_z, smatrix_Matrix_3_udx_wpg, smatrix_Matrix_4_uux_zg, smatrix_Matrix_5_udx_wpgg, smatrix_Matrix_6_uux_zgg, smatrix_Matrix_7_udx_wpggg, smatrix_Matrix_8_uux_zggg, smatrix_Matrix_9_udx_wpgggg, smatrix_Matrix_10_uux_zgggg, smatrix_Matrix_11_udx_wpggggg, smatrix_Matrix_12_uux_zggggg

_GEV2_TO_PB = 0.389379338e9


def _default_chunk_size():
    return 1024 if jax.default_backend() == "gpu" else 4


def _float_dtype_context():
    from ula_stein_evt_gen.cross_section.v_jets.model.madjax_patch import _float_dtype

    if _float_dtype == np.float64:
        return _float_dtype, jax.experimental.enable_x64
    return _float_dtype, contextlib.nullcontext


def build_cross_section(
    sqrt_s = 13000.0, 
    vb="z", 
    n_jets=1, 
    pt_cut=20.0, 
    y_cut=5.0, 
    deltar_cut=0.4
):
    cross_section, _, dim = build_cross_section_components(
        sqrt_s=sqrt_s,
        vb=vb,
        n_jets=n_jets,
        pt_cut=pt_cut,
        y_cut=y_cut,
        deltar_cut=deltar_cut,
    )
    chunk_size = _default_chunk_size()
    _float_dtype, _x64_ctx = _float_dtype_context()

    # Define and compile kernel
    import time
    t0 = time.perf_counter()
    batch_kernel = jax.jit(jax.vmap(cross_section))
    with _x64_ctx():
        spec = jax.ShapeDtypeStruct((chunk_size, dim), _float_dtype)
        compiled_kernel = batch_kernel.lower(spec).compile()
    print("Kernel compilation time:", time.perf_counter() - t0)

    def evaluate(r):
        if r.shape[-1] != dim:
            raise ValueError(f"Input shape must be (..., {dim}), got {r.shape[-1]}")

        with _x64_ctx():
            r = jnp.asarray(r, dtype=_float_dtype)
            n_samples = r.shape[0]

            # Padding
            if n_samples % chunk_size != 0:
                pad_len = chunk_size - (n_samples % chunk_size)
                pad = jnp.zeros((pad_len,) + r.shape[1:], r.dtype)
                r = jnp.concatenate([r, pad], axis=0)

            num_chunks = r.shape[0] // chunk_size
            r_reshaped = r.reshape((num_chunks, chunk_size) + r.shape[1:])

            res = []
            for i in range(num_chunks):
                res.append(compiled_kernel(r_reshaped[i]))
            res = jnp.concatenate(res, axis=0)

        return res[:n_samples]

    return evaluate, dim


def build_cross_section_log_prob_and_score(
    sqrt_s=13000.0,
    vb="z",
    n_jets=1,
    pt_cut=20.0,
    y_cut=5.0,
    deltar_cut=0.4,
    mirror=True,
    chunk_size=None,
):
    """
    Build a chunked batched log-density-and-score function for MCMC.

    The returned function maps an ``(n, dim)`` array to
    ``(log_prob: (n,), score: (n, dim))``. Internally, the matrix-element
    evaluation is vectorized only over fixed-size chunks, so callers can pass
    large MCMC populations without vmapping over the full population at once.
    If ``mirror=True``, inputs are interpreted as mirror-space coordinates and
    scores are gradients with respect to those coordinates.
    """
    cross_section, cut_factors, dim = build_cross_section_components(
        sqrt_s=sqrt_s,
        vb=vb,
        n_jets=n_jets,
        pt_cut=pt_cut,
        y_cut=y_cut,
        deltar_cut=deltar_cut,
    )

    if chunk_size is None:
        chunk_size = _default_chunk_size()
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    _float_dtype, _x64_ctx = _float_dtype_context()

    def log_cross_section(r):
        sigma = cross_section(r)
        return jax.lax.cond(
            sigma > 0.0,
            lambda x: jnp.log(x),
            lambda _: -jnp.inf,
            sigma,
        )

    if mirror:
        from ula_stein_evt_gen.mirror import get_log_prob_mirror

        log_prob = get_log_prob_mirror(log_cross_section)
    else:
        log_prob = log_cross_section

    batch_value_and_grad = jax.jit(jax.vmap(jax.value_and_grad(log_prob)))

    def log_prob_and_score(x):
        if x.ndim != 2 or x.shape[-1] != dim:
            raise ValueError(f"Input shape must be (n, {dim}), got {x.shape}.")

        with _x64_ctx():
            x = jnp.asarray(x, dtype=_float_dtype)
            n_samples = x.shape[0]
            pad_len = (-n_samples) % chunk_size
            if pad_len:
                pad = jnp.zeros((pad_len, dim), dtype=x.dtype)
                x_eval = jnp.concatenate([x, pad], axis=0)
            else:
                x_eval = x

            n_padded = x_eval.shape[0]
            x_chunks = x_eval.reshape((-1, chunk_size, dim))

            def scan_body(_, x_chunk):
                return None, batch_value_and_grad(x_chunk)

            _, (log_prob_chunks, score_chunks) = jax.lax.scan(
                scan_body, None, x_chunks
            )
            log_prob_out = log_prob_chunks.reshape((n_padded,))
            score_out = score_chunks.reshape((n_padded, dim))

            return log_prob_out[:n_samples], score_out[:n_samples]

    return jax.jit(log_prob_and_score), cut_factors, dim


def build_cross_section_components(
    sqrt_s = 13000.0,
    vb="z",
    n_jets=1,
    pt_cut=20.0,
    y_cut=5.0,
    deltar_cut=0.4,
):
    """
    Build scalar cut-applied cross-section and cut-factor functions.

    The batched public ``build_cross_section`` wrapper compiles the scalar
    cross-section for fast evaluation. MCMC/Stein code can use this function
    directly when it needs value-level access to the scalar JAX computation.
    """

    madjax_path = f'{__package__}.{"v_jets"}'
    parameters = importlib.import_module(
        f'{madjax_path}.model.parameters'
    )

    lhapdf_path = f'{Path(__file__).resolve().parent}/lhapdf'

    alphas = load_alphas(lhapdf_path)
    pdfs = load_pdfs(lhapdf_path)

    if vb == "w":
        mv = vector_boson_mass(vb)
        
        if n_jets == 0:
            process = smatrix_Matrix_1_udx_wp
        elif n_jets == 1:
            process = smatrix_Matrix_3_udx_wpg
        elif n_jets == 2:
            process = smatrix_Matrix_5_udx_wpgg
        elif n_jets == 3:
            process = smatrix_Matrix_7_udx_wpggg
        elif n_jets == 4:
            process = smatrix_Matrix_9_udx_wpgggg
        elif n_jets == 5:
            process = smatrix_Matrix_11_udx_wpggggg
        else:
            raise ValueError(f"Unsupported number of jets: {n_jets}")
        
        pdf_indices = ((7,4), (4,7))
        
    elif vb == "z":
        mv = vector_boson_mass(vb)
        
        if n_jets == 0:
            process = smatrix_Matrix_2_uux_z
        elif n_jets == 1:
            process = smatrix_Matrix_4_uux_zg
        elif n_jets == 2:
            process = smatrix_Matrix_6_uux_zgg
        elif n_jets == 3:
            process = smatrix_Matrix_8_uux_zggg
        elif n_jets == 4:
            process = smatrix_Matrix_10_uux_zgggg
        elif n_jets == 5:
            process = smatrix_Matrix_12_uux_zggggg
        else:
            raise ValueError(f"Unsupported number of jets: {n_jets}")
                    
        pdf_indices = ((7, 3), (3, 7))
        
    else:
        raise ValueError(f"Unsupported vector boson type: {vb}")
    
    from ula_stein_evt_gen.cross_section.v_jets.model.madjax_patch import _float_dtype
    _x64_ctx = jax.experimental.enable_x64 if _float_dtype == np.float64 else contextlib.nullcontext

    with _x64_ctx():
        params = parameters.calculate_full_parameters({
            ("mass", 23 if vb == "z" else 24): mv,
            ("sminputs", 3): 1.
        })

    dim = 3 * n_jets + 1

    def _jet_y_phi(p_jets):
        y = 0.5 * jnp.log(
            (p_jets[:, 0] + p_jets[:, 3]) / (p_jets[:, 0] - p_jets[:, 3])
        )
        phi = jnp.arctan2(p_jets[:, 2], p_jets[:, 1])
        return y, phi
    
    def deltar_weight(p_jets):            
        if n_jets < 2:
            return jnp.array(True)

        y, phi = _jet_y_phi(p_jets)
        
        delta_phi = phi[:, None] - phi[None, :]
        delta_phi = jnp.mod(delta_phi + jnp.pi, 2*jnp.pi) - jnp.pi
        delta_y = y[:, None] - y[None, :]
        delta_r2 = delta_phi**2 + delta_y**2

        deltar_mask = delta_r2 < deltar_cut**2
        deltar_mask = jnp.triu(deltar_mask, k=1)
        
        return ~jnp.any(deltar_mask)

    def cut_factors(r):
        pa, pb, _, p_jets, _ = map_phase_space(r, sqrt_s, mv, pt_cut, y_cut)
        xa = pa[0] / (sqrt_s / 2)
        xb = pb[0] / (sqrt_s / 2)

        factors = [1.0 - xa, 1.0 - xb]

        if n_jets >= 2:
            y, phi = _jet_y_phi(p_jets)
            for i in range(n_jets):
                for j in range(i + 1, n_jets):
                    delta_phi = phi[i] - phi[j]
                    delta_phi = jnp.mod(delta_phi + jnp.pi, 2 * jnp.pi) - jnp.pi
                    delta_y = y[i] - y[j]
                    delta_r2 = delta_y**2 + delta_phi**2
                    factors.append(
                        1.0 - deltar_cut**2 / jnp.maximum(delta_r2, 1e-12)
                    )

        return jnp.stack(factors)

    def density_from_phase_space(pa, pb, pv, p_jets, weight_ps):
        # Compute H_T
        mtv = jnp.sqrt(pv[1]**2 + pv[2]**2 + mv**2)
        ht = (mtv + jnp.sum(jnp.sqrt(p_jets[:, 1]**2 + p_jets[:, 2]**2)))/2

        # Compute alpha_s
        weight_alphas = alphas(ht)

        # Two configurations
        momenta_1 = jnp.vstack([pa, pb, pv, p_jets])
        weight_me_1 = process(momenta_1, params)

        momenta_2 = jnp.vstack([pb, pa, pv, p_jets])
        weight_me_2 = process(momenta_2, params)

        # Compute pdfs
        xa = pa[0] / (sqrt_s / 2)
        xb = pb[0] / (sqrt_s / 2)
        pdfs_a = pdfs(xa, ht) / xa
        pdfs_b = pdfs(xb, ht) / xb

        weight_pdfs_1 = pdfs_a[pdf_indices[0][0]] * pdfs_b[pdf_indices[0][1]]
        weight_pdfs_2 = pdfs_a[pdf_indices[1][0]] * pdfs_b[pdf_indices[1][1]]

        flux = 1 / (2*xa*xb*sqrt_s**2)

        return jnp.maximum(weight_ps * weight_alphas**n_jets * flux * _GEV2_TO_PB * (
            weight_me_1 * weight_pdfs_1 + weight_me_2 * weight_pdfs_2
        ), 0.)

    def cross_section(r):
        # Phase space
        pa, pb, pv, p_jets, weight_ps = map_phase_space(r, sqrt_s, mv, pt_cut, y_cut)

        # Deltar weight - 0 or 1
        passed_deltar = deltar_weight(p_jets)
        passed_cuts = jnp.logical_and(passed_deltar, weight_ps > 0.0)

        def generate_after_cuts(_):
            return density_from_phase_space(pa, pb, pv, p_jets, weight_ps)

        return jax.lax.cond(passed_cuts, generate_after_cuts, lambda _: 0., None)

    return cross_section, cut_factors, dim
