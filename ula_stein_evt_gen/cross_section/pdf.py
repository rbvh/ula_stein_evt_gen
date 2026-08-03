import jax
import jax.numpy as jnp

from ula_stein_evt_gen.cross_section.cubic_splines import compute_slopes, hermite

def build_pdf_interp(x_vals, q_vals, table):
    logx_vals = jnp.log(x_vals)
    logq_vals = 2 * jnp.log(q_vals)
    
    _, nx, nq = table.shape
    
    # Slopes along x
    def slopes_x_one_flav(f_qx):
        return jax.vmap(lambda col: compute_slopes(logx_vals, col))(f_qx).T

    slopes_x = jax.vmap(slopes_x_one_flav)(table.transpose(0, 2, 1))

    # Slopes along q
    def slopes_q_one_flav(f_xq):
        return jax.vmap(lambda row: compute_slopes(logq_vals, row))(f_xq)

    slopes_q = jax.vmap(slopes_q_one_flav)(table)
    
    # Cross slopes: ∂/∂x of the ∂/∂Q spline
    slopes_x_of_sq = jax.vmap(lambda f_qx:
        jax.vmap(lambda col: compute_slopes(logx_vals, col))(f_qx).T
    )(slopes_q.transpose(0, 2, 1))
    
    def pdf_one_grid(x, q):
        logx = jnp.log(x)
        logq = 2 * jnp.log(q)
        
        ix = jnp.clip(jnp.searchsorted(logx_vals, logx, side='right') - 1, 0, nx - 2)
        iq = jnp.clip(jnp.searchsorted(logq_vals, logq, side='right') - 1, 0, nq - 2)
        
        def interp_along_x(table, slopes, iq):
            x0 = logx_vals[ix]
            x1 = logx_vals[ix + 1]
            y0 = table[:, ix, iq]
            y1 = table[:, ix + 1, iq]
            m0 = slopes[:, ix, iq]
            m1 = slopes[:, ix + 1, iq]

            return jax.vmap(hermite, in_axes=(None, None, None, 0, 0, 0, 0))(logx, x0, x1, y0, y1, m0, m1)

        f_q0 = interp_along_x(table, slopes_x, iq)
        f_q1 = interp_along_x(table, slopes_x, iq + 1)
        fq_q0 = interp_along_x(slopes_q, slopes_x_of_sq, iq)
        fq_q1 = interp_along_x(slopes_q, slopes_x_of_sq, iq + 1)
        
        q0 = logq_vals[iq]
        q1 = logq_vals[iq + 1]
        
        return jax.vmap(hermite, in_axes=(None, None, None, 0, 0, 0, 0))(logq, q0, q1, f_q0, f_q1, fq_q0, fq_q1)
        
    return pdf_one_grid
        
def load_pdfs(lhapdf_path):
    flav_indices = [-5, -4, -3, -2, -1, 21, 1, 2, 3, 4, 5]
    
    with open(f"{lhapdf_path}/NNPDF40_lo_as_01180.dat", "r") as f:
        lines = f.readlines()
    
    i_line = 3 
    
    x_vals = []
    q_vals = []
    grids = []
    q_max = []
    while i_line < len(lines):
        x = [float(x) for x in lines[i_line].strip().split(" ")]
        i_line += 1
        q = [float(q) for q in lines[i_line].strip().split(" ")]
        i_line += 2

        grids_now = [[[0 for _ in range(len(q))] for _ in range(len(x))] for _ in range(len(flav_indices))]

        n_xy = len(x) * len(q)
        for i in range(n_xy):
            ix = i // len(q)
            iq = i % len(q)

            vals = [float(v) for v in lines[i_line + i].strip().split()]

            for i_flav, val in enumerate(vals):
                grids_now[i_flav][ix][iq] = val
                
        i_line += n_xy + 1

        x_vals.append(jnp.array(x))
        q_vals.append(jnp.array(q))
        grids.append(jnp.array(grids_now))
        
        q_max.append(q[-1])
        
    q_max = jnp.array(q_max)
    grid_interps = tuple(build_pdf_interp(x, q, grid) for x, q, grid in zip(x_vals, q_vals, grids))
    
    @jax.jit
    def pdf(x, q):
        i_grid = jnp.searchsorted(q_max, q, side='right')
        i_grid = jnp.clip(i_grid, 0, len(grids) - 1)
        
        return jax.lax.switch(i_grid, grid_interps, x, q)
    
    return pdf