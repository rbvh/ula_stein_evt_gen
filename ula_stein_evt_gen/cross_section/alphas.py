import ast
import jax
import jax.numpy as jnp

from ula_stein_evt_gen.cross_section.cubic_splines import compute_slopes, hermite


def load_alphas(lhapdf_path):
    with open(f"{lhapdf_path}/NNPDF40_lo_as_01180.info", "r") as f:
        lines = f.readlines()

    # Load scales and values
    for line in lines:
        if ":" not in line:
            continue
        
        k, v = line.split(":", 1)
        if k.strip() == "AlphaS_Qs":
            scales = jnp.array(ast.literal_eval(v.strip()))
        elif k.strip() == "AlphaS_Vals":
            y = jnp.array(ast.literal_eval(v.strip()))

    # log Q^2
    x = 2*jnp.log(scales)
    
    # Split into subgrids at flavor tresholds and compute slopes
    subgrids = []
    start = 0
    for i in range(1, len(x)):
        if x[i] <= x[i - 1]:
            x_piece = x[start:i]
            y_piece = y[start:i]
            m_piece = compute_slopes(x_piece, y_piece)

            subgrids.append((x_piece, y_piece, m_piece))
            start = i

    # Last piece
    x_piece = x[start:]
    y_piece = y[start:]
    m_piece = compute_slopes(x_piece, y_piece)

    subgrids.append((x_piece, y_piece, m_piece))

    # First and last subgrid for extrapolation
    x_first = subgrids[0][0][0]
    y_first = subgrids[0][1][0]
    m_first = subgrids[0][2][0]
    
    x_last = subgrids[-1][0][-1]
    y_last = subgrids[-1][1][-1]
    m_last = subgrids[-1][2][-1]

    # Prepare for JAXified alphas 
    n_grids = len(subgrids)
    max_len = max(len(g[0]) for g in subgrids)

    x_tabs = jnp.stack([jnp.pad(grid[0], (0, max_len - len(grid[0])), 'edge') for grid in subgrids])
    y_tabs = jnp.stack([jnp.pad(grid[1], (0, max_len - len(grid[1])), 'edge') for grid in subgrids])
    m_tabs = jnp.stack([jnp.pad(grid[2], (0, max_len - len(grid[2])), 'edge') for grid in subgrids])

    # last x in each grid
    x_bounds  = jnp.array([grid[0][-1] for grid in subgrids])  
    lengths   = jnp.array([len(grid[0]) for grid in subgrids])
    
    @jax.jit
    def alphas(q):
        x = 2 * jnp.log(q)
        
        low_val = y_first + m_first * (x - x_first)
        high_val = y_last + m_last * (x - x_last)
        
        grid_idx = jnp.searchsorted(x_bounds, x, side='right')
        grid_idx = jnp.minimum(grid_idx, n_grids - 1)
        
        x_arr = x_tabs[grid_idx]
        y_arr = y_tabs[grid_idx]
        m_arr = m_tabs[grid_idx]
        n_arr = lengths[grid_idx]
        
        seg_idx = jnp.searchsorted(x_arr, x, side='right') - 1
        seg_idx = jnp.clip(seg_idx, 0, n_arr - 2)
        
        x0 = x_arr[seg_idx]
        x1 = x_arr[seg_idx + 1]
        y0 = y_arr[seg_idx]
        y1 = y_arr[seg_idx + 1]
        m0 = m_arr[seg_idx]
        m1 = m_arr[seg_idx + 1]

        # Hermite interpolation
        core = hermite(x, x0, x1, y0, y1, m0, m1)

        result = jnp.where(x < x_first, low_val, jnp.where(x > x_last, high_val, core))
        
        return result

    return alphas
    