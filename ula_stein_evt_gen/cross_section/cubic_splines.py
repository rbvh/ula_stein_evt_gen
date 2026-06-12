import jax
import jax.numpy as jnp

def compute_slopes(x, y):
    """
    Compute monotone‑preserving slopes m[i] for piece‑wise cubic Hermite.

    x, y  ... 1‑D arrays with strictly increasing x.
    """
    h     = jnp.diff(x)
    delta = jnp.diff(y) / h
    m     = jnp.zeros_like(y)

    def interior_rule(d_prev, d_next, h_prev, h_next):
        same_sign = jnp.where(d_prev * d_next > 0.0, 1.0, 0.0)
        w1 = 2.0 * h_next + h_prev
        w2 = h_next + 2.0 * h_prev
        # Use safe reciprocals to avoid Inf/NaN in reverse-mode autodiff
        val = (w1 + w2) / (w1 / jnp.where(d_prev == 0.0, 1.0, d_prev) + w2 / jnp.where(d_next == 0.0, 1.0, d_next))
        
        return same_sign * val

    def endpoint_slope(d0, d1, h0, h1):
        m0 = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        cond1 = jnp.where(jnp.sign(m0) != jnp.sign(d0), 0.0, m0)
        cond2 = jnp.where((jnp.sign(d0) != jnp.sign(d1)) &
                          (jnp.abs(cond1) > 3.0 * jnp.abs(d0)),
                          3.0 * d0, cond1)
        return jnp.array([cond2])

    m = [
        endpoint_slope(delta[0],  delta[1],  h[0],  h[1]),
        jax.vmap(interior_rule)(delta[:-1], delta[1:], h[:-1], h[1:]),
        endpoint_slope(delta[-1], delta[-2], h[-1], h[-2])
    ]

    m = jnp.concatenate(m)
    
    return m

def hermite(x, x0, x1, y0, y1, m0, m1):
    """
    Evaluate cubic Hermite polynomial at x.
    
    x0, x1: Interpolation points
    y0, y1: Function values at x0, x1
    m0, m1: Slopes at x0, x1
    """
    h  = x1 - x0
    t = (x - x0) / h
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2

    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1