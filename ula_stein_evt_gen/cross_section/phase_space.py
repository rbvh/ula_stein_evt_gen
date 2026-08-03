import jax
import jax.numpy as jnp

from ula_stein_evt_gen.cross_section.constants import VECTOR_BOSON_MASSES

def make_jet(r, pt2_max, pt2_min, y_cut):
    log_weight = 0
    
    # Pt2 - Samples from 1/pt2**2
    r_pt = jnp.clip(r[0], 1e-12, 1-1e-12)
    pt2 = pt2_max*pt2_min / ((1 - r_pt) * pt2_max + r_pt * pt2_min)
    pt = jnp.sqrt(pt2)
    log_weight += jnp.log(((pt2_max - pt2_min) * pt2**2) / (pt2_max * pt2_min))
    
    # Rapidity
    r_y = r[1]
    y = (2*r_y - 1) * y_cut
    log_weight += jnp.log(2*y_cut)
    
    # Azimuth
    r_phi = r[2]
    phi = 2 * jnp.pi * r_phi
     
    four_vec = jnp.array([
        pt * jnp.cosh(y),
        pt * jnp.cos(phi),
        pt * jnp.sin(phi),
        pt * jnp.sinh(y),
    ])
    
    return four_vec, log_weight

def map_phase_space(r, sqrt_s=13000.0, mv=VECTOR_BOSON_MASSES["z"], pt_cut=30.0, y_cut=6.0):
    assert (r.shape[0] - 1) % 3 == 0
    
    n_jets = (r.shape[0] - 1) // 3
    pt2_max = (sqrt_s / 2)**2 - mv**2
    pt2_cut = pt_cut**2

    log_weight = jnp.log(2*jnp.pi) - jnp.log(sqrt_s**2)
    log_weight -= n_jets * jnp.log(16. * jnp.pi**2)

    r_jets = r[:-1].reshape((n_jets, 3))
    r_v = r[-1]
    
    # Generate the jets
    jets, weights = jax.vmap(make_jet, in_axes=(0, None, None, None))(r_jets, pt2_max, pt2_cut, y_cut)
    log_weight += jnp.sum(weights)

    # Generate the vector boson
    ptv_x = -jnp.sum(jets[:, 1])
    ptv_y = -jnp.sum(jets[:, 2])
    ptv = jnp.sqrt(ptv_x**2 + ptv_y**2)
    mtv = jnp.sqrt(ptv**2 + mv**2)
    
    yv = (2*r_v - 1) * y_cut
    log_weight += jnp.log(2 * y_cut)

    pv = jnp.array([
        mtv * jnp.cosh(yv),
        ptv_x,
        ptv_y,
        mtv * jnp.sinh(yv),
    ])
    
    e_sum = jnp.sum(jets[:, 0]) + pv[0]
    pz_sum = jnp.sum(jets[:, 3]) + pv[3]
    
    xa = (e_sum + pz_sum) / sqrt_s
    xb = (e_sum - pz_sum) / sqrt_s

    pa = jnp.array([xa*sqrt_s/2, 0, 0, xa*sqrt_s/2])
    pb = jnp.array([xb*sqrt_s/2, 0, 0, -xb*sqrt_s/2])

    weight = jnp.where((xa > 0) & (xa < 1) & (xb > 0) & (xb < 1), jnp.exp(log_weight), 0.0)

    return pa, pb, pv, jets, weight

# Validation 
def test_map_phase_space():
    def compute_free_variables(r, sqrt_s=13000.0, mv=VECTOR_BOSON_MASSES["z"], pt_cut=30.0, y_cut=6.0):
        pa, pb, pv, jets, weight_ps = map_phase_space(
            r, sqrt_s=sqrt_s, mv=mv, pt_cut=pt_cut, y_cut=y_cut
        )

        # Compute jet variables
        pt2 = jets[:, 1]**2 + jets[:, 2]**2
        y = 0.5* jnp.log((jets[:, 0] + jets[:, 3]) / (jets[:, 0] - jets[:, 3]))
        phi = jnp.arctan2(jets[:, 2], jets[:, 1])
        
        # Compute vector boson variables
        yv = 0.5 * jnp.log((pv[0] + pv[3]) / (pv[0] - pv[3]))
        
        return jnp.concatenate((pt2, y, phi, yv[None]), axis=0)

    rng = jax.random.key(1)
    sqrt_s = 13000.0
    s = sqrt_s**2
    pt_cut = 30.0
    y_cut = 6.0
    mv = VECTOR_BOSON_MASSES["z"]

    n_jets = 5
    n_sample = 25
    r = jax.random.uniform(rng, shape=(n_sample, 3*n_jets + 1))
    
    for i in range(n_sample):
        per_jet = 1/(32. * jnp.pi**3)
    
        fac = 2*jnp.pi/s * per_jet**n_jets
        
        _, _, _, _, weight = map_phase_space(r[i], sqrt_s=sqrt_s, mv=mv, pt_cut=pt_cut, y_cut=y_cut)
        jac_det = jnp.abs(jnp.linalg.det(jax.jacfwd(compute_free_variables)(r[i], sqrt_s=sqrt_s, mv=mv, pt_cut=pt_cut, y_cut=y_cut)))
        
        if weight == 0.0:
            continue
            
        print(weight, fac*jac_det, fac*jac_det / weight)
