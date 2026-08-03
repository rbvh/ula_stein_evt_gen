import jax.numpy as jnp
from jax.scipy.special import erf, erfinv

"""
Functions to transform a log_prob back and forth between mirror space.
Follows ideas from 1802.10174, which proves optimal convergence rates.
We use the elementwise Gaussian CDF to ensure sub-Gaussian tails under mild assumptions for base distributions.
"""

def to_mirror(x):
    eps = 1e-7
    x_clip = jnp.clip(x, eps, 1-eps)
    return jnp.sqrt(2.) * erfinv(2. * x_clip - 1.)

def from_mirror(x):
    return (1. + erf(x / jnp.sqrt(2.))) / 2.
    
def get_log_prob_mirror(log_prob_base):
    
    def log_prob_mirror(x):
        """
        Compute log prob in mirror space, where x is also in mirror space.
        Note that the transform should include a normalizing constant, but it is not included since the rest of the code is normalization-agnostic.
        """
        return log_prob_base(from_mirror(x)) - jnp.sum(x**2) / 2

    return log_prob_mirror

def get_log_prob_base(log_prob_mirror):
    def log_prob_base(x):
        """
        Compute log prob in base space, where x is also in base space.
        Note that the transform should include a normalizing constant, but it is not included since the rest of the code is normalization-agnostic.
        """
        return log_prob_mirror(to_mirror(x)) + jnp.sum(x**2) / 2

    return log_prob_base

# def log_prob_mirror(x, log_prob_base):
#     """
#     Compute log prob in mirror space, where x is also in mirror space.
#     Note that the transform should include a normalizing constant, but it is not included since the rest of the code is normalization-agnostic.
#     """
    
#     return log_prob_base(from_mirror(x)) - jnp.sum(x**2)/2
    
# def log_prob_base(x, log_prob_mirror):
#     """
#     Compute log prob in base space, where x is also in base space.
#     Note that the transform should include a normalizing constant, but it is not included since the rest of the code is normalization-agnostic.
#     """
    
#     return log_prob_mirror(to_mirror(x)) + jnp.sum(x**2)/2